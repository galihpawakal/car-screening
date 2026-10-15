"""Four-stage, no-dummy screening flow for real grounded comparables."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable
from urllib.parse import urlsplit

from src.analyze import FinalAnalysis, InspectionChecklist, analyze_results, calculate_negotiation_prices
from src.cache import SQLiteAnalysisCache
from src.clean import CleanResult, RejectedListing, _canonical_url, clean_listings
from src.config import (
    MARKETPLACE_DOMAINS,
    MAX_LISTINGS_PER_MARKETPLACE_SEARCH,
    MAX_MANUAL_LISTING_URLS,
    MIN_COMPARABLES,
    SOLID_COMPARABLES,
)
from src.extract import (
    UNKNOWN,
    ExtractedListing,
    ListingExtraction,
    _parse_listings_leniently,
    extract_json_payload,
)
from src.page_reader import extract_listing_from_page
from src.rules import run_all_rules
from src.scoring import (
    calculate_screening_score,
    format_screening_result,
    get_market_confidence,
    negotiation_advice_text,
    price_banner_warning,
    seller_questions_for_cheap,
)
from src.search_provider import (
    GroundedSearchResult,
    GroundingSource,
    SearchProvider,
    TokenUsage,
    merge_grounded_results,
)
from src.stats import calculate_listing_percentile, calculate_statistics


@dataclass
class ScreeningPipelineResult:
    comparables: list
    rejected: list[dict]
    sources: list[GroundingSource]
    queries: list[str]
    fetched_at: str
    statistics: dict
    screening: dict
    final_analysis: FinalAnalysis
    confidence: dict
    warning: str | None
    ai_error: str | None
    cache_hit: bool
    daily_count: int
    usage: dict
    description_skipped: bool
    model_used: str | None = None
    completion_status: str = "selesai"
    analysis_id: str | None = None


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def build_marketplace_search_prompt(user_input: dict, marketplace_domain: str | list[str] = "") -> str:
    brand = user_input["brand"]
    model = user_input["model"]
    variant = user_input.get("variant") or ""
    year = int(user_input["year"])
    trans = user_input["transmission"]
    loc = user_input["location"]

    if isinstance(marketplace_domain, list):
        domains = marketplace_domain
    elif marketplace_domain:
        domains = [marketplace_domain]
    else:
        domains = MARKETPLACE_DOMAINS

    domain_list = ", ".join(domains)
    site_queries = " OR ".join(f"site:{d}" for d in domains)
    query = f"({site_queries}) {brand} {model} {variant} {year} {trans} bekas {loc}"
    total_listings = min(MAX_LISTINGS_PER_MARKETPLACE_SEARCH * len(domains), 15)

    return f"""Cari maksimal {total_listings} listing mobil bekas nyata dari marketplace ({domain_list}) yang paling mirip unit berikut.
Pencarian Google Search: "{query}"
Unit acuan: merek={brand}, model={model}, varian={variant}, tahun={year}, transmisi={trans}, lokasi={loc}.

PENTING: Tuliskan setiap listing yang ditemukan dari pencarian web nyata dalam format JSON di dalam blok ```json ```:
```json
{{
  "listings": [
    {{
      "brand": "{brand}",
      "model": "{model}",
      "variant": "nama varian",
      "year": "{year}",
      "kilometer": "kilometer (contoh: 50.000 km)",
      "transmission": "{trans}",
      "location": "lokasi kota",
      "seller_type": "dealer atau individu",
      "price": "harga nominal (contoh: Rp 150.000.000)",
      "source_url": "URL spesifik halaman listing atau nama domain marketplace"
    }}
  ]
}}
```
Sertakan juga rincian teks bernomor untuk setiap listing beserta URL/tautan spesifik dari marketplace ({domain_list}).
Dilarang menggunakan data tiruan (dummy). Hanya gunakan data nyata dari pencarian web."""


def parse_manual_listing_urls(raw: str | list[str] | None) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        parts = [line.strip() for line in raw.replace(",", "\n").splitlines()]
    else:
        parts = [str(item).strip() for item in raw]
    urls: list[str] = []
    for part in parts:
        if not part or not part.startswith(("http://", "https://")):
            continue
        if part not in urls:
            urls.append(part)
        if len(urls) >= MAX_MANUAL_LISTING_URLS:
            break
    return urls


def query_cache_key(user_input: dict, current_year: int | None = None) -> str:
    manual = parse_manual_listing_urls(user_input.get("manual_listing_urls"))
    key_parts = {
        "brand": user_input["brand"].strip().casefold(),
        "model": user_input["model"].strip().casefold(),
        "variant": user_input["variant"].strip().casefold(),
        "year": int(user_input["year"]),
        "transmission": user_input["transmission"].strip().casefold(),
        "location": user_input["location"].strip().casefold(),
        "current_year": current_year or datetime.now().year,
        "manual_urls": manual,
        "pipeline_version": 2,
    }
    canonical = json.dumps(key_parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _restore_cache(payload: dict) -> tuple[list, list[dict], list[GroundingSource], list[str], str]:
    from src.extract import ComparableListing

    listings = [ComparableListing.model_validate(item) for item in payload.get("listings", [])]
    rejected = payload.get("rejected", [])
    sources = [GroundingSource(**item) for item in payload.get("sources", [])]
    return listings, rejected, sources, payload.get("queries", []), payload.get("fetched_at", "")


def _fallback_analysis(user_input: dict, statistics: dict, rule_flags: list) -> FinalAnalysis:
    count = statistics["count"]
    price_median = statistics.get("price_median") or 0
    listing_price = int(user_input.get("listing_price") or 0)
    price_diff_pct = (listing_price - price_median) / price_median if price_median > 0 else 0

    if count < MIN_COMPARABLES:
        conclusion = (
            f"Hanya {count} pembanding valid yang berhasil dibaca dari web. "
            "Skor angka tidak dihitung; verifikasi harga dan kondisi langsung sebelum membeli."
        )
    elif count < SOLID_COMPARABLES:
        conclusion = (
            f"Hanya ada {count} pembanding nyata yang lolos (data sangat sedikit). "
            "Estimasi harga bersifat indikatif. "
            "Jangan putuskan pembelian sebelum meminta bukti harga dan inspeksi langsung."
        )
    else:
        conclusion = (
            "Gunakan rentang harga pembanding sebagai indikasi, bukan kepastian transaksi. "
            "Verifikasi seluruh dokumen dan kondisi kendaraan sebelum membayar."
        )

    # Saran negosiasi: hanya jika harga listing di atas pasar
    nego_warning = negotiation_advice_text(price_diff_pct, count)
    if nego_warning is not None or price_median <= 0:
        saran_nego = calculate_negotiation_prices(None)  # no suggestion
    else:
        saran_nego = calculate_negotiation_prices(price_median)

    questions = [
        "Bisakah Anda menunjukkan STNK, BPKB, dan faktur asli?",
        "Apakah odometer dan riwayat servis dapat diverifikasi?",
        "Apakah mobil pernah terendam banjir atau mengalami tabrakan?",
    ]
    # Tambah pertanyaan khusus harga sangat murah
    questions.extend(seller_questions_for_cheap(price_diff_pct))

    return FinalAnalysis(
        kesimpulan=conclusion,
        saran_nego=saran_nego,
        red_flag=[flag.message for flag in rule_flags],
        pertanyaan_untuk_penjual=questions,
        checklist_inspeksi=InspectionChecklist(
            dokumen=["Cocokkan STNK, BPKB, faktur, dan nomor rangka/mesin.", "Verifikasi pajak di Samsat."],
            fisik=["Periksa bekas banjir/tabrakan, kaki-kaki, AC, kebocoran, dan lakukan test drive."],
            transmisi=[f"Uji perpindahan transmisi {user_input['transmission']} saat dingin dan setelah panas."],
        ),
        catatan=["Analisa AI akhir tidak tersedia; ringkasan ini dibuat dari statistik dan aturan kode."],
    )


def _collect_listing_urls(grounded: GroundedSearchResult, manual_urls: list[str]) -> list[str]:
    urls: list[str] = []
    for source in grounded.sources:
        if source.url not in urls:
            urls.append(source.url)
        if len(urls) >= MAX_LISTINGS_PER_MARKETPLACE_SEARCH * max(1, len(MARKETPLACE_DOMAINS)):
            break
    for url in manual_urls:
        if url not in urls:
            urls.append(url)
    return urls


def _read_listings_from_urls(
    provider: SearchProvider,
    urls: list[str],
    *,
    notify: Callable[[str], None],
) -> tuple[ListingExtraction, list[RejectedListing], TokenUsage]:
    extracted: list[ExtractedListing] = []
    rejected: list[RejectedListing] = []
    usage = TokenUsage()

    def _read_worker(index: int, url: str) -> tuple[str, PageReadOutcome]:
        domain = urlsplit(url).netloc
        notify(f"Membaca halaman listing {index}/{len(urls)}: {domain}...")
        outcome = extract_listing_from_page(provider, url)
        if outcome.listing:
            notify(f"✅ Fakta listing ter-ekstrak dari {domain} ({outcome.listing.price}).")
        else:
            notify(f"⚠️ Listing dari {domain} dilewati ({outcome.reason or 'tidak lengkap'}).")
        return url, outcome

    max_workers = min(4, max(1, len(urls)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_read_worker, idx, url) for idx, url in enumerate(urls, start=1)]
        for future in futures:
            try:
                url, outcome = future.result()
                usage = TokenUsage(
                    usage.input_tokens + outcome.usage.input_tokens,
                    usage.output_tokens + outcome.usage.output_tokens,
                )
                if outcome.listing is None:
                    rejected.append(RejectedListing(source_url=url, reason=outcome.reason or "Halaman tidak dapat dibaca"))
                else:
                    extracted.append(outcome.listing)
            except Exception as read_error:
                rejected.append(RejectedListing(source_url="error", reason=str(read_error)))

    return ListingExtraction(listings=extracted), rejected, usage


def parse_grounded_listings(
    grounded: GroundedSearchResult,
    user_input: dict,
) -> tuple[ListingExtraction, list[RejectedListing]]:
    text = grounded.text or ""
    sources = grounded.sources or []
    extracted: list[ExtractedListing] = []
    rejected: list[RejectedListing] = []

    brand = str(user_input.get("brand") or "").strip()
    model = str(user_input.get("model") or "").strip()
    default_year = str(user_input.get("year") or "").strip()
    default_trans = str(user_input.get("transmission") or "").strip()
    default_loc = str(user_input.get("location") or "").strip()
    available_urls = [s.url for s in sources if getattr(s, "url", None)]

    # 1. Try parsing structured JSON in grounded.text
    try:
        payload = extract_json_payload(text)
        items = _parse_listings_leniently(payload)
        if items:
            for idx, item in enumerate(items):
                url = item.source_url.strip()
                if not url or not url.startswith(("http://", "https://")) or "google.com" in url:
                    if available_urls:
                        url = available_urls[idx % len(available_urls)]
                    else:
                        continue
                item_dict = item.model_dump()
                item_dict["source_url"] = url
                if not item_dict.get("brand") or item_dict["brand"] == UNKNOWN:
                    item_dict["brand"] = brand
                if not item_dict.get("model") or item_dict["model"] == UNKNOWN:
                    item_dict["model"] = model
                if not item_dict.get("year") or item_dict["year"] == UNKNOWN:
                    item_dict["year"] = default_year
                if not item_dict.get("transmission") or item_dict["transmission"] == UNKNOWN:
                    item_dict["transmission"] = default_trans
                extracted.append(ExtractedListing.model_validate(item_dict))
    except Exception:
        pass

    # 2. If JSON parsing didn't yield enough listings, parse numbered / bulleted blocks from text
    if len(extracted) < 3 and text:
        blocks = re.split(r"\n(?=(?:\d+[\.\)]|\*\*\d+[\.\)]\*\*)\s+)", "\n" + text)
        for idx, block in enumerate(blocks):
            if not block.strip() or len(block.strip()) < 20:
                continue
            price_match = re.search(r"(?:Harga|Price|Rp\.?)\s*[:\-]?\s*([\d\.,]+(?:\s*(?:juta|jt|miliar|milyar))?)", block, re.IGNORECASE)
            if not price_match and "juta" in block.lower():
                price_match = re.search(r"(\d+(?:[.,]\d+)?\s*(?:juta|jt|miliar|milyar))", block, re.IGNORECASE)
            if not price_match:
                continue
            price_val = price_match.group(1).strip()

            title_match = re.search(r"(?:\d+[\.\)]|\*\*\d+[\.\)]\*\*)\s+\*?\*?([^\n\*]+)\*?\*?", block)
            title = title_match.group(1).strip() if title_match else f"{brand} {model}"

            km_match = re.search(r"(?:Jarak tempuh|KM|Kilometer|Odometer)\s*[:\-]?\s*([^\n]+)", block, re.IGNORECASE)
            km_val = km_match.group(1).strip() if km_match else UNKNOWN

            trans_match = re.search(r"(?:Transmisi|Transmission)\s*[:\-]?\s*([^\n]+)", block, re.IGNORECASE)
            if trans_match:
                raw_trans = trans_match.group(1).strip()
                trans_val = "AT" if any(w in raw_trans.upper() for w in ("AT", "OTOMATIS", "AUTOMATIC")) else (
                    "MT" if any(w in raw_trans.upper() for w in ("MT", "MANUAL")) else default_trans
                )
            else:
                block_upper = (title + " " + block).upper()
                trans_val = "AT" if any(w in block_upper for w in (" AT", "AUTOMATIC", "OTOMATIS")) else (
                    "MT" if any(w in block_upper for w in (" MT", "MANUAL")) else default_trans
                )

            loc_match = re.search(r"(?:Lokasi|Location|Kota)\s*[:\-]?\s*([^\n\(\)]+)", block, re.IGNORECASE)
            loc_val = loc_match.group(1).strip() if loc_match else default_loc

            year_match = re.search(r"\b(201\d|202\d)\b", title + " " + block)
            year_val = year_match.group(1) if year_match else default_year

            url_match = re.search(r"(https?://[^\s\)\<\>]+)", block)
            if url_match and not any(bad in url_match.group(1) for bad in ("google.com", "vertexaisearch")):
                assigned_url = url_match.group(1)
            elif available_urls:
                assigned_url = available_urls[(len(extracted) + idx) % len(available_urls)]
            else:
                continue

            extracted.append(
                ExtractedListing(
                    brand=brand,
                    model=model,
                    variant=title,
                    year=year_val,
                    kilometer=km_val,
                    transmission=trans_val,
                    location=loc_val,
                    seller_type="dealer",
                    price=price_val,
                    source_url=assigned_url,
                )
            )

    unique_extracted: list[ExtractedListing] = []
    seen = set()
    for item in extracted:
        key = (item.price.strip(), item.variant.strip())
        if key not in seen:
            seen.add(key)
            unique_extracted.append(item)

    if available_urls:
        for idx, item in enumerate(unique_extracted):
            target_url = available_urls[idx % len(available_urls)]
            unique_extracted[idx] = item.model_copy(update={"source_url": target_url})

    return ListingExtraction(listings=unique_extracted), rejected


def run_marketplace_grounded_search(
    provider: SearchProvider,
    user_input: dict,
    *,
    marketplaces: list[str] | None = None,
    notify: Callable[[str], None] | None = None,
) -> GroundedSearchResult:
    domains = marketplaces or MARKETPLACE_DOMAINS
    if notify:
        notify(f"🔍 Mencari listing di {len(domains)} marketplace sekaligus ({', '.join(domains)})...")

    prompt = build_marketplace_search_prompt(user_input, domains)
    result = provider.search(prompt, marketplace=",".join(domains))

    if notify:
        notify(f"✅ Pencarian selesai: {len(result.sources)} URL listing ditemukan.")

    return result


def run_screening(
    *,
    provider: SearchProvider,
    cache: SQLiteAnalysisCache,
    user_input: dict,
    daily_limit: int,
    current_year: int | None = None,
    on_stage: Callable[[int, str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> ScreeningPipelineResult:
    """Run grounding, page extraction, cleaning/statistics, then final analysis."""
    current_year = current_year or datetime.now().year
    stage = on_stage or (lambda _progress, _text: None)
    notify = on_status or (lambda _text: None)
    daily_count = cache.claim_daily_analysis(daily_limit)
    cache_key = query_cache_key(user_input, current_year)
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    token_input = 0
    token_output = 0
    grounding_calls = 0
    cache_hit = False
    used_model = getattr(provider, "last_used_model", getattr(provider, "model", ""))
    manual_urls = parse_manual_listing_urls(user_input.get("manual_listing_urls"))

    try:
        notify("Memeriksa cache analisa lokal SQLite...")
        payload = cache.get(cache_key)
        if payload is not None:
            cache_hit = True
            stage(38, "Memuat hasil pencarian tersimpan (cache 24 jam)...")
            comparables, rejected, sources, queries, fetched_at = _restore_cache(payload)
            used_model = payload.get("model_used") or used_model or "Cache"
            notify(f"Data ditemukan di cache: memuat {len(comparables)} pembanding tersimpan.")
        else:
            grounded: GroundedSearchResult
            if manual_urls:
                stage(10, "Mode URL manual: melewati pencarian, langsung membaca halaman...")
                notify(f"Memakai {len(manual_urls)} URL listing pembanding yang ditempel pengguna.")
                sources = [GroundingSource(url=url, title=urlsplit(url).netloc) for url in manual_urls]
                grounded = GroundedSearchResult(
                    text="",
                    sources=sources,
                    queries=[],
                    usage=TokenUsage(),
                    grounding_calls=0,
                )
            else:
                stage(10, "Tahap 1/4: mencari listing per marketplace (Google Search)...")
                notify(
                    f"Mencari listing nyata di {len(MARKETPLACE_DOMAINS)} marketplace untuk "
                    f"{user_input['brand']} {user_input['model']} {user_input['year']}..."
                )
                calls_before_search = getattr(provider, "api_calls", 0)
                try:
                    grounded = run_marketplace_grounded_search(provider, user_input, notify=notify)
                except Exception as error:
                    grounding_calls += max(0, getattr(provider, "api_calls", calls_before_search) - calls_before_search)
                    raise PipelineError("Pencarian", f"Pencarian Gemini gagal setelah retry: {error}") from error
                token_input += grounded.usage.input_tokens
                token_output += grounded.usage.output_tokens
                grounding_calls += grounded.grounding_calls
                used_model = grounded.model or getattr(provider, "last_used_model", getattr(provider, "model", ""))
                sources = grounded.sources
                notify(
                    f"Pencarian selesai: {len(sources)} URL ter-resolve dari grounding "
                    f"(google.com ditolak)."
                )

            listing_urls = _collect_listing_urls(grounded, manual_urls)
            stage(30, "Tahap 2/4: mengekstrak fakta listing pembanding...")
            notify("Mengekstrak fakta listing dari hasil pencarian grounding...")
            extraction, page_rejected = parse_grounded_listings(grounded, user_input)
            page_usage = TokenUsage()

            if not extraction.listings and manual_urls:
                notify(f"Membaca {len(manual_urls)} URL listing manual...")
                try:
                    extraction, page_rejected_urls, page_usage = _read_listings_from_urls(
                        provider, manual_urls, notify=notify
                    )
                    page_rejected.extend(page_rejected_urls)
                    token_input += page_usage.input_tokens
                    token_output += page_usage.output_tokens
                except Exception as error:
                    notify(f"Pembacaan URL manual dilewati: {error}")
            elif not extraction.listings and listing_urls:
                notify("Mencoba ekstraksi langsung dari URL sumber...")
                try:
                    extraction, page_rejected_urls, page_usage = _read_listings_from_urls(
                        provider, listing_urls[:3], notify=notify
                    )
                    page_rejected.extend(page_rejected_urls)
                    token_input += page_usage.input_tokens
                    token_output += page_usage.output_tokens
                except Exception as error:
                    notify(f"Pembacaan URL sumber dilewati: {error}")

            notify(f"Ekstraksi selesai: {len(extraction.listings)} listing berhasil ter-ekstrak.")

            stage(48, "Tahap 3/4: membersihkan listing dan menghitung statistik...")
            notify("Memvalidasi merek, model, transmisi, URL unik, dan batas harga...")
            grounded_url_set = {source.url for source in sources} | set(manual_urls)
            try:
                cleaned: CleanResult = clean_listings(
                    extraction,
                    expected_brand=user_input["brand"],
                    expected_model=user_input["model"],
                    expected_transmission=user_input["transmission"],
                    current_year=current_year,
                    grounded_urls=grounded_url_set,
                )
            except Exception as error:
                raise PipelineError("Pembersihan", f"Data listing tidak dapat dibersihkan: {error}") from error
            comparables = cleaned.listings
            rejected = [asdict(item) for item in cleaned.rejected]
            for item in page_rejected:
                rejected.append(asdict(item))
            queries = grounded.queries
            notify(f"Pembersihan selesai: {len(comparables)} lolos validasi, {len(rejected)} ditolak.")
            payload = {
                "listings": [item.model_dump(mode="json") for item in comparables],
                "rejected": rejected,
                "sources": [
                    {
                        "url": source.url,
                        "title": source.title or urlsplit(source.url).netloc,
                        "original_url": getattr(source, "original_url", "") or "",
                    }
                    for source in sources
                ],
                "queries": queries,
                "fetched_at": fetched_at,
                "model_used": used_model,
            }
            cache.set(cache_key, payload)

        stage(58, "Menghitung rentang, posisi pasar, dan KM/tahun...")
        notify("Menghitung P25, median, P75, dan deviasi kilometer per tahun...")
        statistics = payload.get("statistics")
        if statistics is None:
            statistics = calculate_statistics(
                comparables,
                listing_price=None,
                target_year=int(user_input["year"]),
                current_year=current_year,
            )
            payload["statistics"] = statistics
            cache.set(cache_key, payload)
        statistics = dict(statistics)
        statistics["listing_percentile"] = calculate_listing_percentile(
            comparables, int(user_input["listing_price"])
        )
        warning = None
        n_count = statistics["count"]
        if n_count < MIN_COMPARABLES:
            warning = (
                f"Pembanding valid kurang dari {MIN_COMPARABLES} ({n_count}). "
                "Skor angka tidak dihitung; rentang harga berkeyakinan sangat rendah."
            )
        elif n_count < SOLID_COMPARABLES:
            warning = (
                f"Indikatif - data sangat sedikit ({n_count} pembanding). "
                "Rentang harga dan skor pasar berkeyakinan rendah."
            )

        # Price banner warning for extremely cheap listings
        _price_banner = price_banner_warning(
            int(user_input["listing_price"]),
            statistics["price_median"] or 0,
        )
        if _price_banner:
            if warning:
                warning = _price_banner + " " + warning
            else:
                warning = _price_banner

        description = str(user_input.get("deskripsi") or "").strip()
        rule_flags = run_all_rules(
            listing_price=int(user_input["listing_price"]),
            price_median=statistics["price_median"] or 0,
            kilometer=int(user_input["kilometer"]),
            year=int(user_input["year"]),
            model=str(user_input["model"]),
            description=description or None,
            current_year=current_year,
        )
        final_analysis = _fallback_analysis(user_input, statistics, rule_flags)
        ai_error = None
        if n_count >= MIN_COMPARABLES:
            stage(70, "Tahap 4/4: membuat kesimpulan terstruktur tanpa Search...")
            notify("Menyusun analisa akhir, saran negosiasi, dan checklist inspeksi...")
            analysis_input = {key: value for key, value in user_input.items() if key not in ("deskripsi", "manual_listing_urls")}
            if description:
                analysis_input["deskripsi"] = description
            try:
                final_analysis, usage = analyze_results(provider, analysis_input, statistics, comparables)
                token_input += usage.input_tokens
                token_output += usage.output_tokens
                notify("Analisa akhir terstruktur selesai disusun.")
            except Exception as error:
                ai_error = (
                    f"Analisa akhir Gemini gagal setelah retry; hasil statistik dan aturan tetap tersedia: {error}"
                )
        else:
            ai_error = (
                f"Kurang dari {MIN_COMPARABLES} listing valid setelah pembacaan halaman; analisa akhir LLM dilewati "
                "dan skor angka tidak dihitung."
            )

        stage(92, "Menghitung skor, label, dan tingkat keyakinan...")
        completeness_pct = _input_completeness(user_input)
        score_result = calculate_screening_score(
            listing_price=int(user_input["listing_price"]),
            price_low=statistics["price_low"] or 0,
            price_median=statistics["price_median"] or 0,
            price_high=statistics["price_high"] or 0,
            n_comparables=statistics["count"],
            kilometer=int(user_input["kilometer"]),
            year=int(user_input["year"]),
            has_description=bool(description),
            has_varian=bool(user_input.get("variant")),
            has_plate=bool(user_input.get("plate_code") and user_input["plate_code"] != "Lainnya"),
            has_location=bool(user_input.get("location")),
            rule_flags=rule_flags,
            llm_red_flags=final_analysis.red_flag,
            llm_summary=final_analysis.kesimpulan,
            llm_questions=final_analysis.pertanyaan_untuk_penjual,
            current_year=current_year,
            average_comparable_km_per_year=statistics["average_km_per_year"] or 0.0,
        )
        screening = format_screening_result(score_result)
        if statistics["count"] == 0:
            screening["price_label"] = "Data tidak cukup"
        confidence = get_market_confidence(
            statistics["count"], statistics["price_spread_ratio"], completeness_pct
        )
        if not description:
            confidence["reason"] += " Deskripsi kosong; analisa deskripsi oleh LLM dilewati."

        if n_count >= SOLID_COMPARABLES:
            completion_status = "selesai dengan data"
            stage(100, "Analisa selesai dengan data.")
        elif n_count >= MIN_COMPARABLES:
            completion_status = "selesai indikatif"
            stage(100, "Analisa selesai (mode indikatif, data terbatas).")
        else:
            completion_status = "tidak ada data valid"
            stage(100, "Analisa selesai tanpa data valid.")

        result = ScreeningPipelineResult(
            comparables=comparables,
            rejected=rejected,
            sources=sources,
            queries=queries,
            fetched_at=fetched_at,
            statistics=statistics,
            screening=screening,
            final_analysis=final_analysis,
            confidence=confidence,
            warning=warning,
            ai_error=ai_error,
            cache_hit=cache_hit,
            daily_count=daily_count,
            usage={
                "grounding_calls": grounding_calls,
                "input_tokens": token_input,
                "output_tokens": token_output,
                "total_tokens": token_input + token_output,
            },
            description_skipped=not bool(description),
            model_used=used_model,
            completion_status=completion_status,
        )

        save_fn = getattr(cache, "save_screening_record", None)
        if callable(save_fn):
            try:
                analysis_id = save_fn(user_input, result)
                result.analysis_id = analysis_id
                notify(f"💾 Hasil analisa tersimpan di database lokal (ID: {analysis_id[:8]}...).")
            except Exception as save_error:
                logger.warning("Gagal menyimpan hasil analisa ke database: %s", save_error)

        return result
    finally:
        cache.record_usage(
            grounding_calls=grounding_calls,
            input_tokens=token_input,
            output_tokens=token_output,
        )


def _input_completeness(user_input: dict) -> float:
    keys = ("brand", "model", "variant", "year", "transmission", "kilometer", "listing_price", "location", "plate_code")
    present = sum(bool(user_input.get(key)) for key in keys)
    if user_input.get("deskripsi"):
        present += 1
        return present / (len(keys) + 1) * 100
    return present / len(keys) * 100
