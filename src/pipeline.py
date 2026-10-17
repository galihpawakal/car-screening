"""Four-stage, no-dummy screening flow for real grounded comparables."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

from src.analyze import FinalAnalysis, InspectionChecklist, analyze_results, calculate_negotiation_prices
from src.cache import SQLiteAnalysisCache, compute_grounding_cache_key
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
    ComparableListing,
    ExtractedListing,
    ListingExtraction,
    _parse_listings_leniently,
    extract_json_payload,
)
from src.marketplace_config import (
    MARKETPLACE_CATALOG_CONFIGS,
    build_catalog_url,
    build_manual_search_urls,
    get_twin_models,
)
from src.marketplace_fetcher import (
    CrawlFunnelMetrics,
    crawl_marketplace_catalog,
    fetch_and_parse_details_concurrent,
)
from src.tiered_screening import TierEvaluationResult, evaluate_tiered_comparables
from src.page_reader import PageReadOutcome, extract_listing_from_page, fetch_page_html
from src.url_classifier import classify_url, extract_detail_links_from_html
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
    active_tier: int = 1
    tier_label: str = ""
    funnel_metrics: dict | None = None
    catalog_urls_opened: list[dict] | None = None
    rejected_by_reason: dict[str, list[str]] | None = None
    manual_search_urls: dict[str, str] | None = None
    debug_error: str | None = None
    grounding_sources: list[GroundingSource] | None = None
    catalog_detail_links: list[str] | None = None



class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


MAX_DETAIL_PER_LISTING = 5
MAX_TOTAL_DETAIL_FETCH = 5


def build_multi_round_search_prompt(user_input: dict, round_num: int, domains: list[str]) -> tuple[str, str]:
    brand = user_input["brand"]
    model = user_input["model"]
    variant = user_input.get("variant") or ""
    year = int(user_input["year"])
    trans = user_input["transmission"]
    loc = user_input["location"]

    domain_list = ", ".join(domains)
    site_queries = " OR ".join(f"site:{d}" for d in domains)

    if round_num == 1:
        # Round 1: Kueri spesifik sesuai input lengkap
        query = f"({site_queries}) {brand} {model} {variant} {year} {trans} bekas {loc}".strip()
    elif round_num == 2:
        # Round 2: Kueri tanpa varian atau cakupan Jabodetabek lebih luas
        query = f"({site_queries}) {brand} {model} {year} {trans} bekas Jabodetabek Jakarta".strip()
    else:
        # Round 3: Kueri kota penyangga sekitar (Tangerang, Bekasi, Depok, Bogor)
        query = f"({site_queries}) {brand} {model} {year} bekas Tangerang OR Bekasi OR Depok OR Bogor".strip()

    total_listings = min(MAX_LISTINGS_PER_MARKETPLACE_SEARCH * len(domains), 6)
    prompt = f"""Cari maksimal {total_listings} listing mobil bekas nyata dari marketplace ({domain_list}) untuk unit berikut.
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
    return prompt, query


def build_marketplace_search_prompt(user_input: dict, marketplace_domain: str | list[str] = "") -> str:
    if isinstance(marketplace_domain, list):
        domains = marketplace_domain
    elif marketplace_domain:
        domains = [marketplace_domain]
    else:
        domains = MARKETPLACE_DOMAINS
    prompt, _ = build_multi_round_search_prompt(user_input, 1, domains)
    return prompt


def fetch_and_parse_listings_from_urls(
    provider: SearchProvider,
    sources: list[GroundingSource],
    user_input: dict,
    already_fetched_urls: set[str],
    *,
    max_total_fetch: int = MAX_TOTAL_DETAIL_FETCH,
    notify: Callable[[str], None] = lambda _: None,
) -> tuple[list[ExtractedListing], list[dict]]:
    """Process grounding URLs: classify, extract detail links from listing pages, and fetch detail pages."""
    extracted: list[ExtractedListing] = []
    rejected: list[dict] = []
    detail_queue: list[str] = []

    for src in sources:
        url = src.url
        ctype = classify_url(url)
        if ctype == "detail":
            if url not in already_fetched_urls and url not in detail_queue:
                detail_queue.append(url)
        elif ctype == "listing":
            domain = urlsplit(url).netloc
            notify(f"📄 Halaman daftar terdeteksi ({domain}): mencari tautan unit detail...")
            try:
                html = fetch_page_html(url, timeout=6.0, retries=2, delay=0.2)
                links = extract_detail_links_from_html(
                    url,
                    html,
                    expected_model=user_input["model"],
                    max_links=MAX_DETAIL_PER_LISTING,
                )
                if links:
                    notify(f"🔗 Menemukan {len(links)} tautan unit detail dari {domain}.")
                    for link in links:
                        if link not in already_fetched_urls and link not in detail_queue:
                            detail_queue.append(link)
                    rejected.append({
                        "source_url": url,
                        "reason": f"Halaman daftar ({domain}): diekstrak {len(links)} tautan detail (angka daftar tidak dipakai langsung)",
                    })
                else:
                    rejected.append({
                        "source_url": url,
                        "reason": f"Halaman daftar ({domain}): tidak ditemukan tautan unit detail yang cocok",
                    })
            except Exception as err:
                rejected.append({
                    "source_url": url,
                    "reason": f"Halaman daftar ({domain}) tidak dapat diakses ({err})",
                })
        else:
            rejected.append({
                "source_url": url,
                "reason": "URL bukan halaman detail atau daftar marketplace yang didukung",
            })

    # Fetch detail pages one by one
    for detail_url in detail_queue:
        if len(already_fetched_urls) >= max_total_fetch:
            break
        already_fetched_urls.add(detail_url)
        domain = urlsplit(detail_url).netloc
        notify(f"🔍 Membaca halaman detail ({len(already_fetched_urls)}/{max_total_fetch}): {domain}...")
        try:
            outcome = extract_listing_from_page(
                provider,
                detail_url,
                expected_brand=user_input["brand"],
                expected_model=user_input["model"],
            )
            if outcome.listing:
                extracted.append(outcome.listing)
                notify(f"✅ Listing detail berhasil diekstrak ({domain}): Rp {int(outcome.listing.price):,}")
            else:
                rejected.append({
                    "source_url": detail_url,
                    "reason": outcome.reason or "Gagal membaca fakta listing dari halaman detail",
                })
        except Exception as err:
            rejected.append({
                "source_url": detail_url,
                "reason": f"Gagal membaca halaman detail ({err})",
            })

    return extracted, rejected


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

    if count == 0:
        conclusion = (
            "Belum ada pembanding valid yang berhasil dibaca dari web. "
            "Skor angka tidak dihitung; gunakan tautan pencarian manual di marketplace atau verifikasi harga transaksi langsung."
        )
    elif count < MIN_COMPARABLES:
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
    elif abs(price_diff_pct) > 0.50:
        conclusion = "Harga tidak wajar terhadap pasar; kemungkinan salah ketik, harga DP, atau penipuan. Tidak ada saran negosiasi."
    else:
        conclusion = (
            "Gunakan rentang harga pembanding sebagai indikasi, bukan kepastian transaksi. "
            "Verifikasi seluruh dokumen dan kondisi kendaraan sebelum membayar."
        )

    # Saran negosiasi: hanya jika harga listing di atas pasar dan wajar
    nego_warning = negotiation_advice_text(price_diff_pct, count)
    if nego_warning is not None or price_median <= 0 or abs(price_diff_pct) > 0.50:
        saran_nego = calculate_negotiation_prices(None)  # no suggestion
    else:
        saran_nego = calculate_negotiation_prices(price_median, listing_price)

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


def get_default_direct_marketplace_urls(brand: str, model: str, transmission: str = "") -> list[str]:
    """Generate direct marketplace catalog URLs for the specified brand and model."""
    brand_slug = brand.strip().lower().replace(" ", "-")
    model_slug = model.strip().lower().replace(" ", "-")

    trans_param = ""
    trans_upper = transmission.strip().upper()
    if trans_upper in ("AT", "AUTOMATIC", "OTOMATIS"):
        trans_param = "?transmission=automatic"
    elif trans_upper in ("MT", "MANUAL"):
        trans_param = "?transmission=manual"

    return [
        f"https://www.carsome.id/beli-mobil-bekas/{brand_slug}/{model_slug}{trans_param}",
        f"https://www.carmudi.co.id/mobil-dijual/{brand_slug}/{model_slug}/indonesia{trans_param}",
    ]


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
    grounding_succ = 0
    grounding_fail = 0
    cache_hit = False
    used_model = getattr(provider, "last_used_model", getattr(provider, "model", ""))
    manual_urls = parse_manual_listing_urls(user_input.get("manual_listing_urls"))
    debug_error: str | None = None
    grounded_research_text: str = ""
    warning: str | None = None

    # Inisialisasi awal semua variabel pipeline sebelum percabangan cache / mode pencarian
    comparables: list[ComparableListing] = []
    rejected: list[dict] = []
    sources: list[GroundingSource] = []
    queries: list[str] = []
    all_extracted: list[ExtractedListing] = []
    all_rejected: list[dict] = []
    all_sources: list[GroundingSource] = []
    all_queries: list[str] = []
    already_fetched_urls: set[str] = set()
    grounding_sources_list: list[GroundingSource] = []
    catalog_detail_links: list[str] = []
    catalog_links_list: list[str] = []
    used_comparables_list: list[ComparableListing] = []
    funnel = CrawlFunnelMetrics()
    funnel_metrics: dict[str, Any] | None = None
    catalog_urls_opened: list[dict[str, Any]] = []
    rejected_by_reason: dict[str, int] = {}
    active_tier: int = 1
    tier_label: str = "Tier 1: Identik Sempurna"
    manual_search_urls: dict[str, str] = build_manual_search_urls(
        brand=user_input.get("brand", ""),
        model=user_input.get("model", ""),
        year=int(user_input.get("year", current_year)),
        transmission=user_input.get("transmission", ""),
    )
    max_rounds = 1

    try:
        notify("Memeriksa cache analisa lokal SQLite...")
        payload = cache.get(cache_key)
        if payload is not None:
            cached_listings = payload.get("listings") or []
            if len(cached_listings) < MIN_COMPARABLES:
                payload = None
        if payload is not None:
            cache_hit = True
            stage(38, "Memuat hasil pencarian tersimpan (cache 24 jam)...")
            comparables, rejected, sources, queries, fetched_at = _restore_cache(payload)
            used_comparables_list = comparables
            grounding_sources_list = [
                GroundingSource(**item) if isinstance(item, dict) else item
                for item in payload.get("grounding_sources", [])
            ]
            catalog_detail_links = payload.get("catalog_detail_links", [])
            catalog_links_list = catalog_detail_links
            used_model = payload.get("model_used") or used_model or "Cache"
            active_tier = payload.get("active_tier", 1)
            tier_label = payload.get("tier_label", "Cache (tersimpan)")
            funnel_metrics = payload.get("funnel_metrics")
            catalog_urls_opened = payload.get("catalog_urls_opened", [])
            rejected_by_reason = payload.get("rejected_by_reason", {})
            manual_search_urls = payload.get("manual_search_urls") or manual_search_urls
            notify(f"Data ditemukan di cache: memuat {len(comparables)} pembanding tersimpan.")
        else:
            search_mode = user_input.get("search_mode", "grounding")

            def _crawl_all_marketplaces() -> tuple[list[ExtractedListing], list[dict[str, str]]]:
                nonlocal used_model
                domains = ["carmudi.co.id", "carsome.id", "oto.com", "olx.co.id", "mobil123.com", "mobilbekas.com"]
                domain_links: dict[str, list[str]] = {d: [] for d in domains}
                for d in domains:
                    cfg = MARKETPLACE_CATALOG_CONFIGS.get(d)
                    mp = cfg.max_pages if cfg else 3
                    links = crawl_marketplace_catalog(
                        d,
                        brand=user_input["brand"],
                        model=user_input["model"],
                        year=int(user_input["year"]),
                        transmission=user_input.get("transmission", ""),
                        max_pages=mp,
                        target_candidates=30,
                        notify=notify,
                        funnel=funnel,
                    )
                    domain_links[d].extend(links)
                    for lk in links:
                        all_sources.append(GroundingSource(url=lk, title=d))

                # If collected links are few (< 8), check twin models
                total_found = sum(len(v) for v in domain_links.values())
                twin_models = get_twin_models(user_input["model"])
                if total_found < 8 and twin_models:
                    notify(f"⚡ Kandidat langsung sedikit ({total_found}), menelusuri katalog model kembar ({', '.join(twin_models)})...")
                    for twin in twin_models:
                        for d in ["carmudi.co.id", "carsome.id"]:
                            twin_links = crawl_marketplace_catalog(
                                d,
                                brand="",
                                model=twin,
                                year=int(user_input["year"]),
                                transmission=user_input.get("transmission", ""),
                                max_pages=3,
                                target_candidates=15,
                                notify=notify,
                                funnel=funnel,
                            )
                            if d not in domain_links:
                                domain_links[d] = []
                            domain_links[d].extend(twin_links)
                            for lk in twin_links:
                                all_sources.append(GroundingSource(url=lk, title=d))

                # Interleave candidate links round-robin across domains so no single domain starves others
                max_len = max((len(v) for v in domain_links.values()), default=0)
                collected_links: list[str] = []
                for i in range(max_len):
                    for d in list(domain_links.keys()):
                        if i < len(domain_links[d]):
                            collected_links.append(domain_links[d][i])

                # Fetch details concurrently (max 4 workers, max 40 detail fetch)
                details, rej = fetch_and_parse_details_concurrent(
                    collected_links,
                    expected_brand=user_input["brand"],
                    expected_model=user_input["model"],
                    expected_year=int(user_input["year"]),
                    max_workers=4,
                    max_total_fetch=40,
                    notify=notify,
                    funnel=funnel,
                )
                if not used_model:
                    used_model = "Pencarian Langsung Marketplace"
                return details, rej

            if manual_urls:
                stage(10, "Mode URL manual: melewati pencarian, langsung membaca halaman...")
                notify(f"Memakai {len(manual_urls)} URL listing pembanding yang ditempel pengguna.")
                manual_sources = [GroundingSource(url=url, title=urlsplit(url).netloc) for url in manual_urls]
                all_sources.extend(manual_sources)
                extracted, round_rejected = fetch_and_parse_listings_from_urls(
                    provider,
                    manual_sources,
                    user_input,
                    already_fetched_urls,
                    max_total_fetch=MAX_MANUAL_LISTING_URLS,
                    notify=notify,
                )
                all_extracted.extend(extracted)
                all_rejected.extend(round_rejected)
            elif search_mode == "direct":
                stage(10, "Tahap 1/4: mencari listing terfilter langsung dari katalog marketplace...")
                notify("🔍 Menjalankan pencarian langsung katalog marketplace dengan filter tahun & transmisi...")
                dir_extracted, dir_rejected = _crawl_all_marketplaces()
                all_extracted.extend(dir_extracted)
                all_rejected.extend(dir_rejected)
                used_model = "Pencarian Langsung Marketplace"
                notify(f"Pencarian langsung selesai: {len(dir_extracted)} listing detail berhasil dibaca.")
            else:
                stage(10, "Tahap 1/4: menjalankan pencarian langsung marketplace & Google Search grounding paralel...")
                notify("🚀 Menjalankan crawl katalog marketplace dan Google Search grounding secara paralel...")

                grounding_res = {
                    "extracted": [],
                    "rejected": [],
                    "sources": [],
                    "queries": [],
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "calls": 0,
                    "model": "",
                    "research_text": "",
                    "error": None,
                }
                marketplace_res = {"extracted": [], "rejected": []}

                def _run_grounding_task():
                    g_extracted = []
                    g_rejected = []
                    g_sources = []
                    g_queries = []
                    g_in = 0
                    g_out = 0
                    g_calls = 0
                    g_model = ""
                    g_research_text = ""
                    g_already_fetched: set[str] = set()

                    try:
                        for round_num in range(1, max_rounds + 1):
                            prompt, query = build_multi_round_search_prompt(user_input, round_num, MARKETPLACE_DOMAINS)
                            g_queries.append(query)
                            g_cache_key = compute_grounding_cache_key(getattr(provider, "model", "gemini"), query)
                            cached_g = cache.get_grounding(g_cache_key)

                            if cached_g is not None:
                                cached_sources = [
                                    GroundingSource(url=s["url"], title=s.get("title", ""), original_url=s.get("original_url", ""))
                                    for s in cached_g.get("sources", [])
                                ]
                                grounded = GroundedSearchResult(
                                    text=cached_g.get("text", "") or "",
                                    sources=cached_sources,
                                    queries=cached_g.get("queries", [query]) or [query],
                                    usage=TokenUsage(0, 0),
                                    grounding_calls=0,
                                    model=cached_g.get("model", "") or getattr(provider, "model", ""),
                                    marketplace=",".join(MARKETPLACE_DOMAINS),
                                )
                            else:
                                grounded = provider.search(prompt, marketplace=",".join(MARKETPLACE_DOMAINS))
                                if grounded and (grounded.text or grounded.sources):
                                    cache.set_grounded_search_result(g_cache_key, grounded)

                            g_in += grounded.usage.input_tokens
                            g_out += grounded.usage.output_tokens
                            g_calls += grounded.grounding_calls
                            if not g_model and grounded.model:
                                g_model = grounded.model
                            if not g_research_text and grounded.text:
                                g_research_text = grounded.text

                            for s in grounded.sources:
                                if not any(existing.url == s.url for existing in g_sources):
                                    g_sources.append(s)

                            extr, rej = fetch_and_parse_listings_from_urls(
                                provider,
                                grounded.sources,
                                user_input,
                                g_already_fetched,
                                max_total_fetch=MAX_TOTAL_DETAIL_FETCH,
                                notify=notify,
                            )
                            g_extracted.extend(extr)
                            g_rejected.extend(rej)
                            if len(g_extracted) >= 5:
                                break
                    except Exception as g_err:
                        return {
                            "extracted": g_extracted,
                            "rejected": g_rejected,
                            "sources": g_sources,
                            "queries": g_queries,
                            "tokens_in": g_in,
                            "tokens_out": g_out,
                            "calls": g_calls,
                            "model": g_model,
                            "research_text": g_research_text,
                            "error": g_err,
                        }

                    return {
                        "extracted": g_extracted,
                        "rejected": g_rejected,
                        "sources": g_sources,
                        "queries": g_queries,
                        "tokens_in": g_in,
                        "tokens_out": g_out,
                        "calls": g_calls,
                        "model": g_model,
                        "research_text": g_research_text,
                        "error": None,
                    }

                t_start = time.time()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_market = executor.submit(_crawl_all_marketplaces)
                    fut_ground = executor.submit(_run_grounding_task)

                    # Tunggu marketplace selesai (atau max 120 detik)
                    try:
                        dir_extracted, dir_rejected = fut_market.result(timeout=120.0)
                        marketplace_res["extracted"] = dir_extracted
                        marketplace_res["rejected"] = dir_rejected
                    except Exception as m_err:
                        logger.warning("Marketplace crawl gagal atau timeout: %s", m_err)

                    crawl_elapsed = time.time() - t_start
                    # Deadline grounding 120 dtk tidak boleh menambah waktu tunggu setelah crawl selesai;
                    # pakai hasil marketplace begitu siap dan jangan menunggu grounding lebih lama dari 30 dtk setelahnya.
                    remaining_deadline = max(0.0, 120.0 - crawl_elapsed)
                    wait_grounding = min(remaining_deadline, 30.0)

                    try:
                        grounding_res = fut_ground.result(timeout=wait_grounding)
                        if grounding_res.get("error"):
                            notify(f"⚠️ Google Search grounding mengalami kendala ({grounding_res['error']}). Menggunakan data marketplace.")
                            fallback_msg = f"Pencarian web grounding tidak tersedia ({grounding_res['error']}); hasil menggunakan pencarian langsung katalog marketplace."
                            warning = f"{fallback_msg} {warning}" if warning else fallback_msg
                            debug_error = f"{type(grounding_res['error']).__name__}: {grounding_res['error']}"
                    except TimeoutError:
                        notify("⏱️ Batas waktu grounding (maks 30s setelah crawl selesai) tercapai. Melanjutkan dengan hasil marketplace.")
                        warning = "Pencarian grounding melampaui batas waktu 30s setelah crawl selesai; hasil menggunakan katalog marketplace."
                    except Exception as g_exc:
                        logger.warning("Grounding thread error: %s", g_exc)
                        fallback_msg = f"Pencarian web grounding tidak tersedia ({g_exc}); hasil menggunakan pencarian langsung katalog marketplace."
                        warning = f"{fallback_msg} {warning}" if warning else fallback_msg
                        debug_error = f"{type(g_exc).__name__}: {g_exc}"

                # Gabungkan hasil crawl dan grounding
                token_input += grounding_res["tokens_in"]
                token_output += grounding_res["tokens_out"]
                grounding_calls += grounding_res["calls"]
                if grounding_res["model"]:
                    used_model = grounding_res["model"]
                elif not used_model:
                    used_model = "Pencarian Langsung Marketplace"
                grounded_research_text = grounding_res["research_text"]
                all_queries.extend(grounding_res["queries"])
                grounding_sources_list.extend(grounding_res["sources"])

                seen_extracted_urls = set()
                for item in marketplace_res["extracted"]:
                    u = _canonical_url(item.source_url)
                    if u not in seen_extracted_urls:
                        seen_extracted_urls.add(u)
                        all_extracted.append(item)
                for item in grounding_res["extracted"]:
                    u = _canonical_url(item.source_url)
                    if u not in seen_extracted_urls:
                        seen_extracted_urls.add(u)
                        all_extracted.append(item)

                all_rejected.extend(marketplace_res["rejected"])
                all_rejected.extend(grounding_res["rejected"])

            stage(48, "Tahap 3/4: memvalidasi listing dengan tiered fallback (Tier 1-4)...")
            notify("Memvalidasi listing dan mengelompokkan ke dalam tingkatan (Tier 1-4)...")
            try:
                tier_result: TierEvaluationResult = evaluate_tiered_comparables(
                    all_extracted,
                    expected_brand=user_input["brand"],
                    expected_model=user_input["model"],
                    expected_year=int(user_input["year"]),
                    expected_transmission=user_input["transmission"],
                    expected_variant=user_input.get("variant") or "",
                    require_jabodetabek=True,
                    current_year=current_year,
                    target_count=5,
                )
            except Exception as error:
                raise PipelineError("Pembersihan", f"Data listing tidak dapat divalidasi: {error}") from error

            comparables = tier_result.listings
            active_tier = tier_result.active_tier
            tier_label = tier_result.tier_label
            rejected_by_reason = tier_result.rejected_reasons

            # Backup from SQLite market listings database if live comparables < 5
            if len(comparables) < 5:
                try:
                    db_backups = cache.get_recent_backup_listings(
                        model=user_input["model"],
                        transmission=user_input["transmission"],
                        year=int(user_input["year"]),
                        twin_models=get_twin_models(user_input["model"]),
                        max_age_days=14,
                        limit=5 - len(comparables),
                    )
                    live_urls = {c.source_url for c in comparables}
                    added_backups = 0
                    for row in db_backups:
                        if row["source_url"] in live_urls:
                            continue
                        live_urls.add(row["source_url"])
                        is_twin = row["model"].lower() != user_input["model"].lower()
                        comparables.append(ComparableListing(
                            brand=row["brand"],
                            model=row["model"],
                            variant=row["variant"],
                            year=row["year"],
                            kilometer=row["kilometer"] or 50000,
                            transmission=user_input["transmission"],
                            location=row["location"] or "Jabodetabek",
                            seller_type=row["seller_type"] or "dealer",
                            price=row["price"],
                            source_url=row["source_url"],
                            fetched_at=row["scraped_at"],
                            is_km_estimated=False,
                            is_same_variant=row["variant"].lower() == (user_input.get("variant") or "").lower(),
                            page_type="detail",
                            tier=4 if is_twin else 3,
                            weight=0.5 if is_twin else 0.8,
                            is_twin_model=is_twin,
                            notes=f"Database Pasar ({row['scraped_at'][:10]})",
                        ))
                        added_backups += 1
                    if added_backups > 0:
                        notify(f"💾 Memuat {added_backups} pembanding cadangan dari Database Listing Pasar (≤ 14 hari).")
                except Exception as db_err:
                    logger.debug("Gagal memuat listing cadangan dari DB: %s", db_err)

            # Save valid live comparables into market_listings table
            if comparables:
                try:
                    cache.save_market_listings(comparables)
                except Exception as save_err:
                    logger.debug("Gagal menyimpan listing ke database pasar: %s", save_err)

            # Add any detail fetch/parsing failures to rejected_by_reason
            for rej in all_rejected:
                r_reason = rej.get("reason") or "Gagal membaca detail halaman"
                r_url = rej.get("source_url") or "-"
                rejected_by_reason.setdefault(r_reason, []).append(r_url)

            # Invariant check: detail_terbaca == lolos + sum(dibuang per alasan)
            total_rejected_count = sum(len(urls) for urls in rejected_by_reason.values())
            total_detail_candidates = len(all_extracted) + len(all_rejected)
            if total_detail_candidates != len(comparables) + total_rejected_count:
                logger.error(
                    "INVARIANT VIOLATION: detail candidates (%d) != comparables (%d) + rejected (%d)",
                    total_detail_candidates,
                    len(comparables),
                    total_rejected_count,
                )

            # Flat list of all rejected items
            flat_rejected: list[dict] = []
            for reason, urls in rejected_by_reason.items():
                for u in urls:
                    flat_rejected.append({"source_url": u, "reason": reason})

            rejected = flat_rejected
            # 'sources' only contains valid comparables used in the analysis
            comparable_sources = [
                GroundingSource(url=c.source_url, title=urlsplit(c.source_url).netloc)
                for c in comparables
            ]
            sources = comparable_sources
            queries = all_queries
            catalog_detail_links = [s.url for s in all_sources if s not in grounding_sources_list]
            catalog_links_list = catalog_detail_links
            used_comparables_list = comparables
            logger.debug("Valid comparables: %d, catalog links: %d", len(used_comparables_list), len(catalog_links_list))
            funnel_metrics = {
                "catalog_urls_opened": len(funnel.catalog_urls_opened),
                "detail_links_found": funnel.detail_links_found,
                "detail_links_prefiltered_out": funnel.detail_links_prefiltered_out,
                "detail_pages_fetched": funnel.detail_pages_fetched,
                "detail_pages_parsed": funnel.detail_pages_parsed,
                "valid_comparables": len(comparables),
            }
            catalog_urls_opened = funnel.catalog_urls_opened

            notify(f"Validasi selesai: {len(comparables)} pembanding valid ({tier_label}), {len(rejected)} data ditolak.")
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
                "grounding_sources": [
                    {
                        "url": s.url,
                        "title": s.title or urlsplit(s.url).netloc,
                        "original_url": getattr(s, "original_url", "") or "",
                    }
                    for s in grounding_sources_list
                ],
                "catalog_detail_links": catalog_detail_links,
                "queries": queries,
                "fetched_at": fetched_at,
                "model_used": used_model,
                "active_tier": active_tier,
                "tier_label": tier_label,
                "funnel_metrics": funnel_metrics,
                "catalog_urls_opened": catalog_urls_opened,
                "rejected_by_reason": rejected_by_reason,
                "manual_search_urls": manual_search_urls,
            }
            if len(comparables) >= MIN_COMPARABLES:
                cache.set(cache_key, payload)
            else:
                notify(f"⚠️ Hanya ada {len(comparables)} pembanding valid (< {MIN_COMPARABLES}); hasil tidak disimpan ke cache agar dapat dicari ulang.")


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
            if len(comparables) >= MIN_COMPARABLES:
                cache.set(cache_key, payload)
        statistics = dict(statistics)
        statistics["listing_percentile"] = calculate_listing_percentile(
            comparables, int(user_input["listing_price"])
        )
        n_count = statistics["count"]
        stat_warning = None
        if n_count < MIN_COMPARABLES:
            stat_warning = (
                f"Pembanding valid kurang dari {MIN_COMPARABLES} ({n_count}). "
                "Skor angka tidak dihitung; rentang harga berkeyakinan sangat rendah."
            )
        elif n_count < SOLID_COMPARABLES:
            stat_warning = (
                f"Indikatif - data sangat sedikit ({n_count} pembanding). "
                "Rentang harga dan skor pasar berkeyakinan rendah."
            )
        if stat_warning:
            warning = f"{warning} {stat_warning}" if warning else stat_warning

        # Price banner warning for extremely cheap listings (only if comparables >= 3)
        _price_banner = price_banner_warning(
            int(user_input["listing_price"]),
            statistics["price_median"] or 0,
            n_comparables=n_count,
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
            n_comparables=n_count,
        )
        final_analysis = _fallback_analysis(user_input, statistics, rule_flags)
        ai_error = None
        if n_count >= SOLID_COMPARABLES:
            stage(70, "Tahap 4/4: membuat kesimpulan terstruktur tanpa Search...")
            notify("Menyusun analisa akhir, saran negosiasi, dan checklist inspeksi...")
            analysis_input = {key: value for key, value in user_input.items() if key not in ("deskripsi", "manual_listing_urls")}
            if description:
                analysis_input["deskripsi"] = description
            try:
                final_analysis, usage = analyze_results(
                    provider,
                    analysis_input,
                    statistics,
                    comparables,
                    research_text=grounded_research_text,
                )
                token_input += usage.input_tokens
                token_output += usage.output_tokens
                notify("Analisa akhir terstruktur selesai disusun.")
            except Exception as error:
                final_analysis = _fallback_analysis(user_input, statistics, rule_flags)
                ai_error = "Analisa AI tidak tersedia, memakai ringkasan aturan"
                debug_error = f"{type(error).__name__}: {error}"
                logger.warning("Analisa akhir AI gagal: %s", error)
        elif n_count >= MIN_COMPARABLES:
            # Mode indikatif: skip LLM call untuk hemat token, pakai fallback analysis
            ai_error = (
                f"Mode indikatif ({n_count} pembanding). Analisa akhir LLM dilewati untuk hemat token; "
                "skor dihitung dari statistik dan aturan kode."
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
        unique_domains = {urlsplit(c.source_url).netloc for c in comparables if getattr(c, "source_url", None)}
        confidence = get_market_confidence(
            statistics["count"],
            statistics["price_spread_ratio"],
            completeness_pct,
            unique_domains_count=len(unique_domains),
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

        attempts = getattr(provider, "attempt_history", [])
        grounding_succ = sum(1 for a in attempts if a.get("success") and a.get("http_code") == 200)
        grounding_fail = sum(1 for a in attempts if not a.get("success") and a.get("category") != "skipped")

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
                "grounding_calls": grounding_succ if grounding_succ > 0 else grounding_calls,
                "grounding_failed_calls": grounding_fail,
                "input_tokens": token_input,
                "output_tokens": token_output,
                "total_tokens": token_input + token_output,
            },
            description_skipped=not bool(description),
            model_used=used_model,
            completion_status=completion_status,
            active_tier=active_tier,
            tier_label=tier_label,
            funnel_metrics=funnel_metrics,
            catalog_urls_opened=catalog_urls_opened,
            rejected_by_reason=rejected_by_reason,
            manual_search_urls=manual_search_urls,
            debug_error=debug_error,
            grounding_sources=grounding_sources_list,
            catalog_detail_links=catalog_detail_links,
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
