"""Normalize and validate listing facts; rejected listings retain only URL and reason."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.extract import ComparableListing, ExtractedListing, ListingExtraction, UNKNOWN


@dataclass(frozen=True)
class RejectedListing:
    source_url: str
    reason: str


@dataclass
class CleanResult:
    listings: list[ComparableListing]
    rejected: list[RejectedListing]


def _unknown(value: str | None) -> bool:
    return not value or value.strip().casefold() == UNKNOWN


def _parse_integer(value: str | None) -> int | None:
    if _unknown(value):
        return None
    digits = re.sub(r"[^0-9]", "", value or "")
    return int(digits) if digits else None


def _parse_price(value: str | None) -> int | None:
    if _unknown(value):
        return None
    text = (value or "").strip().casefold().replace("idr", "").replace("rp", "").strip()
    unit_match = re.search(r"\b(miliar|milyar|juta|jt|m|b)\b", text)
    if unit_match:
        amount_match = re.search(r"\d+(?:[.,]\d+)?", text)
        if not amount_match:
            return None
        amount = float(amount_match.group(0).replace(",", "."))
        multiplier = 1_000_000_000 if unit_match.group(1) in {"miliar", "milyar", "b"} else 1_000_000
        return int(amount * multiplier)
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return ""
    host = parts.netloc.casefold().removeprefix("www.")
    path = parts.path.rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in {"ref", "tracking", "source", "fbclid", "gclid"}
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit((parts.scheme.casefold(), host, path, query, ""))


def _known(value: str) -> str | None:
    return None if _unknown(value) else value.strip()


JABODETABEK_KEYWORDS = [
    "jakarta", "dki", "bogor", "depok", "tangerang", "tangsel", "bekasi"
]


def is_jabodetabek(location: str | None) -> bool:
    if not location or _unknown(location):
        return False
    loc_lower = location.lower()
    return any(keyword in loc_lower for keyword in JABODETABEK_KEYWORDS)


def clean_listings(
    extracted: ListingExtraction | list[ExtractedListing],
    *,
    expected_brand: str,
    expected_model: str,
    expected_transmission: str,
    expected_year: int | None = None,
    expected_variant: str = "",
    require_jabodetabek: bool = False,
    current_year: int | None = None,
    fetched_at: datetime | None = None,
    grounded_urls: set[str] | None = None,
) -> CleanResult:
    """Return validated, deduplicated numeric facts and a reason for each rejection."""
    candidates = extracted.listings if isinstance(extracted, ListingExtraction) else extracted
    now = current_year or datetime.now().year
    timestamp = (fetched_at or datetime.now()).astimezone().isoformat(timespec="seconds")
    cleaned: list[ComparableListing] = []
    rejected: list[RejectedListing] = []
    seen_urls: set[str] = set()
    seen_facts: set[tuple[int, int, int]] = set()

    def reject(item: ExtractedListing, reason: str) -> None:
        rejected.append(RejectedListing(source_url=item.source_url, reason=reason))

    for item in candidates:
        url = _canonical_url(item.source_url)
        if not url:
            reject(item, "URL sumber tidak valid")
            continue
        allowed_urls = {_canonical_url(source) for source in grounded_urls or set()}
        if grounded_urls is not None and url not in allowed_urls:
            reject(item, "URL tidak tercantum dalam sitasi grounding")
            continue
        price = _parse_price(item.price)
        if price is None:
            reject(item, "Harga tidak tersedia")
            continue
        if not 20_000_000 <= price <= 20_000_000_000:
            reject(item, "Harga di luar rentang masuk akal")
            continue
        if item.model.strip().casefold() != expected_model.strip().casefold():
            reject(item, "Model tidak cocok dengan input")
            continue
        if not _unknown(item.brand) and item.brand.strip().casefold() != expected_brand.strip().casefold():
            reject(item, "Merek tidak cocok dengan input")
            continue

        year = _parse_integer(item.year)
        if year is not None and not 1980 <= year <= now:
            reject(item, "Tahun di luar rentang masuk akal")
            continue
        if expected_year is not None:
            if year is None:
                reject(item, "Tahun tidak tersedia")
                continue
            if abs(year - expected_year) > 1:
                reject(item, f"Tahun di luar rentang ±1 ({year} vs {expected_year})")
                continue

        kilometer = _parse_integer(item.kilometer)
        if kilometer is None:
            reject(item, "Kilometer tidak tersedia")
            continue
        if not 0 < kilometer <= 1_500_000:
            reject(item, "KM di luar rentang masuk akal")
            continue

        transmission = _known(item.transmission)
        if not transmission or transmission.casefold() not in {"at", "mt", "automatic", "manual", "otomatis", "manual transmission"}:
            reject(item, "Transmisi tidak dikenali")
            continue
        if expected_transmission.casefold() in {"at", "mt"}:
            is_at = transmission.casefold() in {"at", "automatic", "otomatis"}
            if is_at != (expected_transmission.casefold() == "at"):
                reject(item, "Transmisi tidak cocok dengan input")
                continue

        if require_jabodetabek:
            if not is_jabodetabek(item.location):
                loc_disp = _known(item.location) or "tidak diketahui"
                reject(item, f"Lokasi di luar Jabodetabek ({loc_disp})")
                continue

        if url in seen_urls:
            reject(item, "Duplikat URL")
            continue
        fact_key = (price, kilometer, year) if kilometer is not None and year is not None else None
        if fact_key and fact_key in seen_facts:
            reject(item, "Duplikat kombinasi harga+KM+tahun")
            continue

        seen_urls.add(url)
        if fact_key:
            seen_facts.add(fact_key)

        is_km_estimated = "estimasi" in (item.kilometer or "").lower()
        is_same_variant = True
        if expected_variant and not _unknown(item.variant):
            is_same_variant = expected_variant.lower() in item.variant.lower()

        cleaned.append(ComparableListing(
            brand=_known(item.brand) or expected_brand,
            model=item.model.strip(),
            variant=_known(item.variant) or UNKNOWN,
            year=year,
            kilometer=kilometer,
            transmission="AT" if (transmission and transmission.casefold() in {"at", "automatic", "otomatis"}) else "MT",
            location=_known(item.location) or UNKNOWN,
            seller_type=_known(item.seller_type) or UNKNOWN,
            price=price,
            source_url=url,
            fetched_at=timestamp,
            is_km_estimated=is_km_estimated,
            is_same_variant=is_same_variant,
            page_type="detail",
        ))
    return CleanResult(listings=cleaned, rejected=rejected)
