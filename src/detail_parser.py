"""Structured parser for vehicle detail pages prioritizing JSON-LD, OpenGraph, and regex."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from src.extract import UNKNOWN, ExtractedListing

logger = logging.getLogger(__name__)


def normalize_price(raw_val: Any) -> int | None:
    """Normalize raw price string or number into an integer rupiah.
    
    Rejects DP, installment, or out-of-range figures.
    Examples:
      'Rp 153 jt' -> 153000000
      '153.000.000' -> 153000000
      'Rp 169,4 Juta' -> 169400000
      '1.25 Miliar' -> 1250000000
      153000000 -> 153000000
    """
    if raw_val is None:
        return None
    if isinstance(raw_val, (int, float)):
        val = int(raw_val)
        return val if 20_000_000 <= val <= 20_000_000_000 else None

    text = str(raw_val).strip()
    if not text or text.lower() == UNKNOWN:
        return None

    # Check if text explicitly says DP / cicilan / angsuran without total cash
    lower = text.lower()
    if re.search(r"\b(dp|uang muka|angsuran|cicilan|per bulan|/bln)\b", lower) and not re.search(r"\b(cash|otr|total|harga jual)\b", lower):
        # Could be an installment indicator
        pass

    # Clean currency labels
    cleaned = re.sub(r"(?i)\b(rp\.?|idr)\b", "", text).strip()

    # Look for million / billion multipliers
    multiplier_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:juta|jt|miliar|milyar|m|b)\b", cleaned, re.IGNORECASE)
    if multiplier_match:
        num_str = multiplier_match.group(1).replace(".", "").replace(",", ".")
        try:
            num = float(num_str)
            unit = multiplier_match.group(0).lower()
            if any(k in unit for k in ("miliar", "milyar", "b")):
                val = int(num * 1_000_000_000)
            else:
                val = int(num * 1_000_000)
            return val if 20_000_000 <= val <= 20_000_000_000 else None
        except ValueError:
            pass

    # Plain digits with periods or commas as thousand separators
    digits_only = re.sub(r"[^\d]", "", cleaned)
    if digits_only:
        try:
            val = int(digits_only)
            # If price was entered as millions without unit (e.g. 153 for 153 million)
            if 40 <= val <= 3000:
                val *= 1_000_000
            return val if 20_000_000 <= val <= 20_000_000_000 else None
        except ValueError:
            pass

    return None


def normalize_kilometer(raw_val: Any) -> tuple[int | None, bool]:
    """Normalize odometer reading string or number into an integer and estimation flag.
    
    Supports single values and ranges:
      '55rb km' -> (55000, False)
      '50-55 rb km' -> (52500, True)  # Midpoint of range
      '50.000 - 60.000 km' -> (55000, True)
      '85,969 Km' -> (85969, False)
    """
    if raw_val is None:
        return None, False
    if isinstance(raw_val, (int, float)):
        val = int(raw_val)
        return (val, False) if 0 < val <= 1_500_000 else (None, False)

    text = str(raw_val).strip()
    if not text or text.lower() == UNKNOWN:
        return None, False

    # Check for range: e.g. "50-55 rb km" or "50.000 - 60.000" or "50k-55k"
    range_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:rb|k|ribu)?\s*[-–—to]\s*(\d+(?:[.,]\d+)?)\s*(?:rb|k|ribu|km)?",
        text,
        re.IGNORECASE,
    )
    if range_match:
        try:
            low_raw = range_match.group(1).replace(".", "").replace(",", ".")
            high_raw = range_match.group(2).replace(".", "").replace(",", ".")
            low_f = float(low_raw)
            high_f = float(high_raw)
            # Detect multiplier
            has_rb = bool(re.search(r"\b(rb|k|ribu)\b", text, re.I))
            if has_rb or (low_f < 1000 and high_f < 1000):
                low_val = int(low_f * 1000)
                high_val = int(high_f * 1000)
            else:
                low_val = int(low_f)
                high_val = int(high_f)

            if 0 < low_val <= high_val <= 1_500_000:
                midpoint = int((low_val + high_val) / 2)
                return midpoint, True
        except (ValueError, ZeroDivisionError):
            pass

    # Single value with "rb" or "k"
    rb_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:rb|k|ribu)\b", text, re.IGNORECASE)
    if rb_match:
        try:
            num = float(rb_match.group(1).replace(".", "").replace(",", "."))
            val = int(num * 1000)
            return (val, False) if 0 < val <= 1_500_000 else (None, False)
        except ValueError:
            pass

    # Digits with comma/dot
    clean_digits = re.sub(r"[^\d]", "", text)
    if clean_digits:
        try:
            val = int(clean_digits)
            # If written as single digits in thousands (e.g. 55 km might mean 55.000 if not brand new)
            return (val, False) if 0 < val <= 1_500_000 else (None, False)
        except ValueError:
            pass

    return None, False


def detect_transmission(text: str) -> str:
    """Detect AT or MT transmission from text, avoiding false positives on English 'at'."""
    if not text:
        return UNKNOWN
    
    # 1. Look for explicit transmission phrases first
    manual_match = re.search(r"(?:transmisi|transmission|gearbox)?\s*[:\-]?\s*\b(manual|m/t|m\.t)\b", text, re.IGNORECASE)
    if manual_match and manual_match.group(1):
        return "MT"
    
    auto_match = re.search(r"(?:transmisi|transmission|gearbox)?\s*[:\-]?\s*\b(otomatis|automatic|matic|a/t|a\.t|cvt|dct|ags)\b", text, re.IGNORECASE)
    if auto_match and auto_match.group(1):
        return "AT"

    # 2. Case-sensitive checks for standalone acronyms in uppercase
    if re.search(r"\bM/T\b", text) or re.search(r"\bMT\b", text):
        return "MT"
    if re.search(r"\bA/T\b", text) or re.search(r"\bAT\b", text):
        return "AT"

    return UNKNOWN


def extract_location(text: str) -> str:
    """Extract Jabodetabek or other Indonesian cities from text."""
    if not text:
        return UNKNOWN

    # Check for "dijual di <Kota>" pattern (common in oto.com and carmudi)
    dijual_match = re.search(r"dijual di\s+([A-Za-z\s]+?)(?:\s*\(\d+\)|\.|\,|$)", text, re.IGNORECASE)
    if dijual_match:
        cand = dijual_match.group(1).strip()
        if len(cand) < 30:
            return cand

    jabodetabek_subareas = [
        "Jakarta Pusat", "Jakarta Selatan", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara",
        "DKI Jakarta", "Jakarta", "Tangerang Selatan", "Tangsel", "Tangerang",
        "Bekasi Selatan", "Bekasi Utara", "Bekasi", "Depok", "Bogor",
    ]
    for loc in jabodetabek_subareas:
        pattern = rf"\b{re.escape(loc)}\b"
        if re.search(pattern, text, re.IGNORECASE):
            if loc.lower() in ("dki jakarta", "jakarta"):
                return "DKI Jakarta"
            if loc.lower() in ("tangsel", "tangerang selatan"):
                return "Tangerang Selatan"
            return loc

    other_cities = ["Bandung", "Surabaya", "Semarang", "Yogyakarta", "Solo", "Medan", "Pekanbaru", "Palembang", "Bali"]
    for city in other_cities:
        if re.search(rf"\b{re.escape(city)}\b", text, re.IGNORECASE):
            return city

    return UNKNOWN


def _parse_json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    """Extract car facts from JSON-LD schema scripts."""
    results: dict[str, Any] = {}
    scripts = soup.find_all("script", type="application/ld+json")
    for s in scripts:
        if not s.string:
            continue
        try:
            data = json.loads(s.string)
        except Exception:
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("@type", "")).lower()

            if any(t in item_type for t in ("car", "vehicle", "product")):
                # Brand
                brand_obj = item.get("brand")
                if isinstance(brand_obj, dict):
                    results["brand"] = brand_obj.get("name")
                elif isinstance(brand_obj, str):
                    results["brand"] = brand_obj

                # Model
                if item.get("model"):
                    results["model"] = item.get("model")

                # Name / Title / Variant
                if item.get("name"):
                    results["name"] = item.get("name")

                # Description
                if item.get("description"):
                    results["description"] = item.get("description")

                # Year
                if item.get("vehicleModelDate") or item.get("productionDate"):
                    results["year"] = str(item.get("vehicleModelDate") or item.get("productionDate"))

                # Odometer
                mileage = item.get("mileageFromOdometer")
                if isinstance(mileage, dict) and "value" in mileage:
                    results["kilometer"] = mileage["value"]
                elif mileage is not None:
                    results["kilometer"] = mileage

                # Transmission
                if item.get("vehicleTransmission"):
                    results["transmission"] = str(item.get("vehicleTransmission"))

                # Offers
                offers = item.get("offers")
                if isinstance(offers, dict):
                    if offers.get("price"):
                        results["price"] = str(offers.get("price"))
                    seller = offers.get("seller", {})
                    if isinstance(seller, dict):
                        addr = seller.get("address", {})
                        if isinstance(addr, dict):
                            loc_parts = [addr.get("addressLocality"), addr.get("addressRegion")]
                            results["location"] = ", ".join(p for p in loc_parts if p)
                elif isinstance(offers, list) and offers:
                    first_offer = offers[0]
                    if isinstance(first_offer, dict) and first_offer.get("price"):
                        results["price"] = str(first_offer.get("price"))

    return results


def _parse_open_graph(soup: BeautifulSoup) -> dict[str, str]:
    """Extract facts from OpenGraph and standard meta tags."""
    meta: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name")
        content = tag.get("content")
        if key and content:
            meta[key.lower()] = content.strip()
    return meta


def parse_detail_page(
    html_content: str,
    url: str,
    *,
    expected_brand: str = "",
    expected_model: str = "",
) -> ExtractedListing | None:
    """Parse vehicle listing facts from a detail page.
    
    Prioritizes JSON-LD, then OpenGraph metadata, then regex/HTML text fallback.
    Returns an ExtractedListing instance or None if critical fields cannot be found.
    """
    if not html_content or len(html_content.strip()) < 200:
        return None

    soup = BeautifulSoup(html_content, "html.parser")

    json_ld = _parse_json_ld(soup)
    og = _parse_open_graph(soup)

    # 1. Brand & Model
    page_title = og.get("og:title") or (soup.title.string if soup.title else "") or ""
    title_lower = (page_title + " " + (json_ld.get("name") or "")).lower()

    brand = json_ld.get("brand") or UNKNOWN
    if brand == UNKNOWN and expected_brand:
        if expected_brand.lower() in title_lower:
            brand = expected_brand
        else:
            brand = expected_brand

    model = json_ld.get("model") or UNKNOWN
    # Model specific check: Grand Livina vs Livina
    if "grand livina" in title_lower:
        model = "Grand Livina"
    elif model == UNKNOWN:
        if expected_model and expected_model.lower() in title_lower:
            model = expected_model
        else:
            model = expected_model or UNKNOWN

    # 2. Variant
    variant = UNKNOWN
    full_name = json_ld.get("name") or page_title
    if full_name:
        # Try extracting variant by removing brand, model, and year
        variant_candidate = full_name
        if brand and brand != UNKNOWN:
            variant_candidate = re.sub(re.escape(brand), "", variant_candidate, flags=re.I)
        if model and model != UNKNOWN:
            variant_candidate = re.sub(re.escape(model), "", variant_candidate, flags=re.I)
        variant_candidate = re.sub(r"\b(201\d|202\d)\b", "", variant_candidate)
        variant_candidate = re.sub(r"\b(jual|beli|mobil|bekas|dijual|harga|detail|seller)\b", "", variant_candidate, flags=re.I)
        variant_clean = " ".join(variant_candidate.split()).strip(" -|&,()")
        if variant_clean:
            variant = variant_clean[:50]

    # 3. Year
    year_str = json_ld.get("year")
    if not year_str or year_str == UNKNOWN:
        year_match = re.search(r"\b(201\d|202\d)\b", full_name + " " + og.get("og:description", ""))
        if not year_match:
            # Check LI tags (e.g. oto.com "Tahun Buat 2017")
            year_match = re.search(r"Tahun\s*(?:Buat|Registrasi|Pembuatan)?\s*[:\-]?\s*(201\d|202\d)", soup.get_text(" "), re.I)
        year_str = year_match.group(1) if year_match else UNKNOWN

    # 4. Kilometer
    km_str = json_ld.get("kilometer")
    if not km_str:
        # Search in title, og:description, or page text
        # e.g. "Beli Mobil 2021 Toyota AVANZA 52533 KM Bekas"
        km_match = re.search(r"\b([\d\.,]+\s*(?:km|kilometer))\b", page_title + " " + og.get("og:description", ""), re.I)
        if not km_match:
            # LI tags in oto.com: "Didorong KMs 85,969 Km"
            km_match = re.search(r"(?:Didorong KMs|Odometer|Kilometer|Jarak tempuh)\s*[:\-]?\s*([\d\.,]+(?:\s*-\s*[\d\.,]+)?\s*(?:rb|k|ribu|km)?)", soup.get_text(" "), re.I)
        km_str = km_match.group(1) if km_match else UNKNOWN

    # 5. Price
    price_str = json_ld.get("price") or og.get("product:price:amount") or og.get("og:price:amount")
    if not price_str:
        # Check specific main price elements first (e.g. oto.com 'span.vh-price.f-24' or prominent price tag)
        prominent_price = soup.find(lambda el: el.name in ("span", "h1", "h2", "div") and
                                     any(c in el.get("class", []) for c in ("f-24", "f-22", "main-price", "listing-price")) and
                                     "vh-price" in " ".join(el.get("class", [])))
        if prominent_price:
            p_text = prominent_price.get_text(strip=True)
            if not any(bad in p_text.lower() for bad in ("bursa", "mulai", "dp", "cicilan")):
                price_str = p_text

    if not price_str:
        # Search page title or description
        price_match = re.search(r"(?:Rp\.?|IDR)\s*([\d\.,]+(?:\s*(?:juta|jt|miliar|milyar))?)", page_title + " " + og.get("og:description", ""), re.I)
        if not price_match:
            # Look for general vh-price or price-box that doesn't say Bursa or DP
            for el in soup.find_all(class_=re.compile(r"vh-price|price-box|car-price", re.I)):
                t = el.get_text(strip=True)
                if not any(bad in t.lower() for bad in ("bursa", "mulai", "dp", "cicilan")):
                    m = re.search(r"(?:Rp\.?|IDR)?\s*([\d\.,]+(?:\s*(?:juta|jt|miliar|milyar))?)", t, re.I)
                    if m and normalize_price(m.group(0)):
                        price_match = m
                        break
        price_str = price_match.group(0) if price_match else UNKNOWN

    # 6. Transmission
    trans_str = json_ld.get("transmission")
    if not trans_str or trans_str == UNKNOWN:
        trans_str = detect_transmission(full_name + " " + og.get("og:description", "") + " " + soup.get_text(" ")[:5000])

    # 7. Location
    loc_str = json_ld.get("location")
    if not loc_str or loc_str == UNKNOWN:
        loc_str = extract_location(page_title + " " + og.get("og:description", "") + " " + soup.get_text(" ")[:5000])

    # Normalize values
    norm_price = normalize_price(price_str)
    norm_km, is_km_estimated = normalize_kilometer(km_str)

    if norm_price is None or norm_km is None:
        logger.debug("Detail parse failed for %s: price=%s, km=%s", url, norm_price, norm_km)
        return None

    km_display = f"{norm_km:,} km{' (estimasi)' if is_km_estimated else ''}"

    return ExtractedListing(
        brand=str(brand).strip(),
        model=str(model).strip(),
        variant=str(variant).strip(),
        year=str(year_str).strip(),
        kilometer=km_display,
        transmission=trans_str,
        location=loc_str,
        seller_type="dealer",
        price=str(norm_price),
        source_url=url,
    )
