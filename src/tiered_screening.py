"""Tiered validation and fallback logic for vehicle comparables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.clean import (
    JABODETABEK_KEYWORDS,
    _canonical_url,
    _known,
    _parse_integer,
    _parse_price,
    _unknown,
    is_jabodetabek,
)
from src.extract import ComparableListing, ExtractedListing, UNKNOWN
from src.marketplace_config import get_twin_models, slugify_model


@dataclass
class TierEvaluationResult:
    listings: list[ComparableListing]
    active_tier: int
    tier_label: str
    rejected_reasons: dict[str, list[str]]
    funnel_counts: dict[str, int]


def is_transmission_match(listing_trans: str | None, expected_trans: str) -> bool:
    """Strict transmission matching. Transmission is NEVER relaxed."""
    if not listing_trans or _unknown(listing_trans):
        return False
    lt = listing_trans.casefold()
    et = expected_trans.casefold()
    is_at = any(w in lt for w in ("at", "automatic", "otomatis"))
    is_mt = any(w in lt for w in ("mt", "manual"))
    if et == "at":
        return is_at and not is_mt
    elif et == "mt":
        return is_mt and not is_at
    return False


def is_variant_match(listing_variant: str | None, expected_variant: str) -> bool:
    """Check if listing matches the specified variant/trim level."""
    if not expected_variant or _unknown(expected_variant):
        return True
    if not listing_variant or _unknown(listing_variant):
        return False
    lv = listing_variant.casefold()
    ev = expected_variant.casefold()
    return ev in lv or lv in ev


def evaluate_tiered_comparables(
    candidates: list[ExtractedListing],
    *,
    expected_brand: str,
    expected_model: str,
    expected_year: int,
    expected_transmission: str,
    expected_variant: str = "",
    require_jabodetabek: bool = True,
    current_year: int | None = None,
    target_count: int = 5,
) -> TierEvaluationResult:
    """Evaluate candidates across Tiers 1-4 and select the tightest tier satisfying target_count."""
    now = current_year or datetime.now().year
    rejected: dict[str, list[str]] = {}

    def log_reject(reason: str, url: str) -> None:
        rejected.setdefault(reason, []).append(url)

    # Pre-parse and validate basic sanity across all candidates
    valid_candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in candidates:
        url = _canonical_url(item.source_url)
        if not url:
            log_reject("URL sumber tidak valid", item.source_url)
            continue
        if url in seen_urls:
            log_reject("Duplikat URL", url)
            continue
        seen_urls.add(url)

        price = _parse_price(item.price)
        if price is None:
            log_reject("Harga tidak tersedia", url)
            continue
        if not 20_000_000 <= price <= 20_000_000_000:
            log_reject("Harga di luar rentang masuk akal (Rp 20jt - Rp 20M)", url)
            continue

        year = _parse_integer(item.year)
        if year is None:
            log_reject("Tahun tidak tersedia", url)
            continue
        if not 1980 <= year <= now:
            log_reject(f"Tahun di luar rentang masuk akal ({year})", url)
            continue

        km = _parse_integer(item.kilometer)
        if km is None:
            log_reject("Kilometer tidak tersedia", url)
            continue
        if not 0 < km <= 1_500_000:
            log_reject("KM di luar rentang masuk akal", url)
            continue

        if not is_transmission_match(item.transmission, expected_transmission):
            log_reject(f"Transmisi tidak cocok ({item.transmission} vs {expected_transmission})", url)
            continue

        if require_jabodetabek and not is_jabodetabek(item.location):
            loc_str = _known(item.location) or "tidak diketahui"
            log_reject(f"Lokasi di luar Jabodetabek ({loc_str})", url)
            continue

        item_model = item.model.strip().casefold()
        target_model = expected_model.strip().casefold()
        item_brand = item.brand.strip().casefold() if not _unknown(item.brand) else ""
        target_brand = expected_brand.strip().casefold()

        canonical_target = slugify_model("default", target_brand, target_model)
        canonical_item = slugify_model("default", item_brand, item_model)
        twin_models = [m.casefold() for m in get_twin_models(canonical_target)]

        # Distinct model check: Livina vs Grand Livina are different models
        is_livina_grand_mismatch = (
            ("grand livina" in item_model and target_model == "livina")
            or ("grand livina" in target_model and item_model == "livina")
            or ("grand-livina" in canonical_item and canonical_target == "livina")
            or ("grand-livina" in canonical_target and canonical_item == "livina")
        )
        if is_livina_grand_mismatch:
            log_reject(f"Model berbeda ({item.model} vs {expected_model})", url)
            continue

        is_exact_model = (
            (item_model == target_model)
            or (canonical_item == canonical_target)
        )
        if not is_exact_model:
            tokens_target = set(re.findall(r"\w+", target_model))
            tokens_item = set(re.findall(r"\w+", item_model))
            if tokens_target and tokens_target == tokens_item:
                is_exact_model = True
            elif target_model in item_model or item_model in target_model:
                if ("grand" in tokens_item) != ("grand" in tokens_target):
                    is_exact_model = False
                else:
                    is_exact_model = True

        is_twin = not is_exact_model and any(
            tw == canonical_item or tw in item_model or item_model in tw for tw in twin_models
        )

        if not is_exact_model and not is_twin:
            log_reject(f"Model bukan {expected_model} ataupun model kembar", url)
            continue

        cand_data = {
            "item": item,
            "url": url,
            "price": price,
            "year": year,
            "kilometer": km,
            "transmission": expected_transmission.upper(),
            "location": _known(item.location) or "Jabodetabek",
            "variant": _known(item.variant) or UNKNOWN,
            "is_exact_model": is_exact_model,
            "is_twin": is_twin,
            "year_diff": abs(year - expected_year),
            "is_exact_variant": is_variant_match(item.variant, expected_variant),
        }

        # Deteksi duplikat listing / dealer yang sama (judul + harga + KM + lokasi hampir identik)
        is_dealer_dup = False
        for existing in valid_candidates:
            if (
                existing["year"] == cand_data["year"]
                and existing["transmission"] == cand_data["transmission"]
                and existing["is_exact_model"] == cand_data["is_exact_model"]
                and existing["is_exact_variant"] == cand_data["is_exact_variant"]
                and abs(existing["price"] - cand_data["price"]) / max(1, existing["price"]) <= 0.01
                and abs(existing["kilometer"] - cand_data["kilometer"]) <= 500
            ):
                is_dealer_dup = True
                break

        if is_dealer_dup:
            log_reject("Duplikat listing dari dealer yang sama (spesifikasi & harga identik)", url)
            continue

        valid_candidates.append(cand_data)

    # Helper to convert parsed candidate to ComparableListing
    def to_comparable(cand: dict[str, Any], tier: int, weight: float, note: str = "") -> ComparableListing:
        item: ExtractedListing = cand["item"]
        return ComparableListing(
            brand=item.brand if not _unknown(item.brand) else expected_brand,
            model=item.model,
            variant=cand["variant"],
            year=cand["year"],
            kilometer=cand["kilometer"],
            transmission=cand["transmission"],
            location=cand["location"],
            seller_type=item.seller_type or "dealer",
            price=cand["price"],
            source_url=cand["url"],
            fetched_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            is_km_estimated=False,
            is_same_variant=cand["is_exact_variant"],
            page_type="detail",
            tier=tier,
            weight=weight,
            is_twin_model=cand["is_twin"],
            notes=note,
        )

    # Helper to reject unselected candidates to maintain the invariant
    def record_unselected(selected: list[dict[str, Any]], active_tier_num: int) -> None:
        sel_urls = {c["url"] for c in selected}
        for c in valid_candidates:
            if c["url"] not in sel_urls:
                if active_tier_num == 1:
                    if c["year_diff"] > 0:
                        log_reject(f"Tahun di luar Tier 1 ({c['year']} vs {expected_year})", c["url"])
                    elif not c["is_exact_variant"]:
                        log_reject(f"Varian tidak cocok dengan Tier 1 ({c['variant']} vs {expected_variant})", c["url"])
                    elif c["is_twin"]:
                        log_reject("Model kembar tidak digunakan karena pembanding model utama mencukupi", c["url"])
                    else:
                        log_reject("Tidak terpilih dalam Tier 1", c["url"])
                elif active_tier_num == 2:
                    if c["year_diff"] > 0:
                        log_reject(f"Tahun di luar Tier 2 ({c['year']} vs {expected_year})", c["url"])
                    elif c["is_twin"]:
                        log_reject("Model kembar tidak digunakan karena pembanding model utama mencukupi", c["url"])
                    else:
                        log_reject("Tidak terpilih dalam Tier 2", c["url"])
                elif active_tier_num in (3, 4):
                    if c["year_diff"] > 1:
                        log_reject(f"Tahun di luar toleransi ±1 ({c['year']} vs {expected_year})", c["url"])
                    elif not c["is_twin"] and active_tier_num == 4:
                        log_reject("Tidak terpilih dalam Tier 4", c["url"])
                    else:
                        log_reject(f"Tidak terpilih dalam Tier {active_tier_num}", c["url"])
                else:
                    log_reject("Tidak terpilih dalam evaluasi tier", c["url"])

    # TIER 1: Model sama, tahun persis, varian persis, transmisi sama, Jabodetabek (Bobot 1.0)
    tier1_cands = [
        c for c in valid_candidates
        if c["is_exact_model"] and c["year_diff"] == 0 and c["is_exact_variant"]
    ]
    if len(tier1_cands) >= target_count:
        record_unselected(tier1_cands, 1)
        listings = [to_comparable(c, tier=1, weight=1.0, note="Tier 1: Tahun & varian persis") for c in tier1_cands]
        _check_invariant(len(candidates), len(listings), rejected)
        return TierEvaluationResult(
            listings=listings,
            active_tier=1,
            tier_label="Tier 1 (Model, Tahun & Varian Persis)",
            rejected_reasons=rejected,
            funnel_counts={"candidates": len(candidates), "valid": len(valid_candidates), "tier_selected": len(listings)},
        )

    # TIER 2: Varian boleh berbeda (Bobot 0.7 untuk varian beda, 1.0 untuk varian persis)
    tier2_cands = [
        c for c in valid_candidates
        if c["is_exact_model"] and c["year_diff"] == 0
    ]
    if len(tier2_cands) >= target_count:
        record_unselected(tier2_cands, 2)
        listings = [
            to_comparable(
                c,
                tier=2,
                weight=1.0 if c["is_exact_variant"] else 0.7,
                note="Tier 1 (varian persis)" if c["is_exact_variant"] else "Tier 2 (varian fleksibel)",
            )
            for c in tier2_cands
        ]
        _check_invariant(len(candidates), len(listings), rejected)
        return TierEvaluationResult(
            listings=listings,
            active_tier=2,
            tier_label="Tier 2 (Model & Tahun Persis, Varian Fleksibel)",
            rejected_reasons=rejected,
            funnel_counts={"candidates": len(candidates), "valid": len(valid_candidates), "tier_selected": len(listings)},
        )

    # TIER 3: Tahun ±1 (Bobot 0.8 untuk tahun ±1, 1.0 untuk tahun persis)
    tier3_cands = [
        c for c in valid_candidates
        if c["is_exact_model"] and c["year_diff"] <= 1
    ]
    if len(tier3_cands) >= target_count:
        record_unselected(tier3_cands, 3)
        listings = [
            to_comparable(
                c,
                tier=3,
                weight=1.0 if c["year_diff"] == 0 else 0.8,
                note=f"Tier 3 (Tahun {c['year']}, diff={c['year_diff']})",
            )
            for c in tier3_cands
        ]
        _check_invariant(len(candidates), len(listings), rejected)
        return TierEvaluationResult(
            listings=listings,
            active_tier=3,
            tier_label="Tier 3 (Tahun ±1 Toleransi)",
            rejected_reasons=rejected,
            funnel_counts={"candidates": len(candidates), "valid": len(valid_candidates), "tier_selected": len(listings)},
        )

    # TIER 4: Model kembar sekelas (Bobot 0.5), HANYA jika pembanding masih < 3
    tier4_cands = [
        c for c in valid_candidates
        if (c["is_exact_model"] or c["is_twin"]) and c["year_diff"] <= 1
    ]
    if len(tier3_cands) < 3 and any(c["is_twin"] for c in tier4_cands):
        record_unselected(tier4_cands, 4)
        listings = [
            to_comparable(
                c,
                tier=4 if c["is_twin"] else (3 if c["year_diff"] > 0 else 2),
                weight=0.5 if c["is_twin"] else (0.8 if c["year_diff"] > 0 else 1.0),
                note="Tier 4 (Model Kembar)" if c["is_twin"] else f"Tahun {c['year']}",
            )
            for c in tier4_cands
        ]
        _check_invariant(len(candidates), len(listings), rejected)
        return TierEvaluationResult(
            listings=listings,
            active_tier=4,
            tier_label="Tier 4 (Model Kembar Sekelas)",
            rejected_reasons=rejected,
            funnel_counts={"candidates": len(candidates), "valid": len(valid_candidates), "tier_selected": len(listings)},
        )

    # Jika pembanding di Tier 3 ada meski < 5, gunakan Tier 3
    if tier3_cands:
        record_unselected(tier3_cands, 3)
        listings = [
            to_comparable(
                c,
                tier=3,
                weight=1.0 if c["year_diff"] == 0 else 0.8,
                note=f"Tier 3 (Tahun {c['year']})",
            )
            for c in tier3_cands
        ]
        _check_invariant(len(candidates), len(listings), rejected)
        return TierEvaluationResult(
            listings=listings,
            active_tier=3,
            tier_label="Tier 3 (Tahun ±1 Toleransi)",
            rejected_reasons=rejected,
            funnel_counts={"candidates": len(candidates), "valid": len(valid_candidates), "tier_selected": len(listings)},
        )

    # Fallback jika tidak ada kandidat lolos
    record_unselected([], 0)
    _check_invariant(len(candidates), 0, rejected)
    return TierEvaluationResult(
        listings=[],
        active_tier=1,
        tier_label="Tidak ada pembanding lolos validasi",
        rejected_reasons=rejected,
        funnel_counts={"candidates": len(candidates), "valid": 0, "tier_selected": 0},
    )


def _check_invariant(total_candidates: int, total_passed: int, rejected_reasons: dict[str, list[str]]) -> None:
    """Invariant: total_candidates == total_passed + sum(dibuang per alasan)."""
    total_rejected = sum(len(urls) for urls in rejected_reasons.values())
    if total_candidates != total_passed + total_rejected:
        import logging
        logging.getLogger(__name__).error(
            "INVARIANT VIOLATION in evaluate_tiered_comparables: candidates (%d) != passed (%d) + rejected (%d)",
            total_candidates,
            total_passed,
            total_rejected,
        )
        assert total_candidates == total_passed + total_rejected, (
            f"Akuntansi buangan tidak cocok: {total_candidates} != {total_passed} + {total_rejected}"
        )

