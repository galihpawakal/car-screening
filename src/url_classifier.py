"""URL classification and detail link extraction for car marketplace websites."""

from __future__ import annotations

import logging
import re
from typing import Literal
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

UrlType = Literal["detail", "listing", "other"]

# Configurable domain rules for detail and listing URLs
# Patterns match the path (and optionally query) of the URL
MARKETPLACE_URL_PATTERNS: dict[str, dict[str, list[re.Pattern]]] = {
    "oto.com": {
        "detail": [
            re.compile(r"^/(?:en/)?detail-mobil-(?:bekas|baru)/[^/]+(?:-\d+)?$", re.IGNORECASE),
            re.compile(r"^/(?:en/)?mobil-(?:bekas|baru)/.*-\d+$", re.IGNORECASE),
        ],
        "listing": [
            re.compile(r"^/(?:en/)?mobil-bekas(?:/|\+|$)", re.IGNORECASE),
            re.compile(r"^/(?:en/)?mobil-baru(?:/|\+|$)", re.IGNORECASE),
            re.compile(r"^/(?:en/)?cari-mobil(?:/|\+|$)", re.IGNORECASE),
        ],
    },
    "carmudi.co.id": {
        "detail": [
            re.compile(r"^/(?:dijual|listing)/[^/]+/\d+$", re.IGNORECASE),
        ],
        "listing": [
            re.compile(r"^/(?:mobil-bekas|mobil-dijual|mobil-bekas-dijual)(?:/|$)", re.IGNORECASE),
            re.compile(r"^/(?:mobil-baru|cars-for-sale)(?:/|$)", re.IGNORECASE),
        ],
    },
    "mobil123.com": {
        "detail": [
            re.compile(r"^/(?:dijual|listing)/[^/]+/\d+$", re.IGNORECASE),
        ],
        "listing": [
            re.compile(r"^/(?:mobil-dijual|mobil-bekas)(?:/|$)", re.IGNORECASE),
            re.compile(r"^/(?:mobil-baru|cars-for-sale)(?:/|$)", re.IGNORECASE),
        ],
    },
    "carsome.id": {
        "detail": [
            # Example: /beli-mobil-bekas/toyota/avanza/2021-toyota-avanza-g-1.3/c5uh000
            re.compile(r"^/(?:en/)?(?:beli-mobil-bekas|buy-car)/[^/]+/[^/]+/[^/]+/[a-z0-9]+$", re.IGNORECASE),
            re.compile(r"^/(?:en/)?(?:beli-mobil-bekas|buy-car)/.*/[c][a-z0-9]{5,8}$", re.IGNORECASE),
        ],
        "listing": [
            re.compile(r"^/(?:en/)?(?:beli-mobil-bekas|buy-car)(?:/[^/]+){0,2}/?$", re.IGNORECASE),
        ],
    },
    "olx.co.id": {
        "detail": [
            re.compile(r"^/item/[^/]+-iid-\d+$", re.IGNORECASE),
            re.compile(r"^/item/[^/]+$", re.IGNORECASE),
        ],
        "listing": [
            re.compile(r"^/(?:[^/]+/)?mobil-bekas(?:_[a-z0-9]+)?(?:/|$)", re.IGNORECASE),
            re.compile(r"^/all-results", re.IGNORECASE),
        ],
    },
    "mobilbekas.com": {
        "detail": [
            re.compile(r"^/(?:detail|review|mobil)/.*-\d+\.html$", re.IGNORECASE),
            re.compile(r"^/iklan/.*", re.IGNORECASE),
        ],
        "listing": [
            re.compile(r"^/(?:mobil-bekas|cari-mobil)(?:/|$)", re.IGNORECASE),
        ],
    },
}


def _normalize_host(netloc: str) -> str:
    host = netloc.casefold().strip()
    if ":" in host:
        host = host.split(":")[0]
    return host.removeprefix("www.")


def classify_url_by_pattern(url: str) -> UrlType | None:
    """Classify URL based on known domain patterns without network access."""
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "other"

    host = _normalize_host(parsed.netloc)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"

    # Find matching configured domain
    for domain, patterns in MARKETPLACE_URL_PATTERNS.items():
        if host == domain or host.endswith("." + domain):
            for pattern in patterns["detail"]:
                if pattern.search(path):
                    return "detail"
            for pattern in patterns["listing"]:
                if pattern.search(path):
                    return "listing"
            # Known domain but not matching detail or listing patterns
            return "other"

    return None


def classify_url_by_content_heuristics(html_content: str) -> UrlType:
    """Fallback heuristic for unknown domains based on page content.
    
    A detail page typically has a single primary vehicle price and a single odometer.
    A listing/search page has multiple vehicle cards, multiple prices, or pagination.
    """
    if not html_content or len(html_content.strip()) < 100:
        return "other"

    soup = BeautifulSoup(html_content, "html.parser")

    # 0. Check canonical or og:url if present
    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        by_can = classify_url_by_pattern(canonical["href"])
        if by_can:
            return by_can
    og_url = soup.find("meta", property="og:url")
    if og_url and og_url.get("content"):
        by_og = classify_url_by_pattern(og_url["content"])
        if by_og:
            return by_og

    # 1. Check for structured data (@type Car, Vehicle, Product with Offer)
    json_ld_scripts = soup.find_all("script", type="application/ld+json")
    for script in json_ld_scripts:
        if not script.string:
            continue
        lower_txt = script.string.lower()
        if any(t in lower_txt for t in ('"@type": "car"', '"@type":"car"', '"@type": "vehicle"', '"@type":"vehicle"')):
            return "detail"
        if ('"@type": "product"' in lower_txt or '"@type":"product"' in lower_txt) and ("mileage" in lower_txt or "price" in lower_txt):
            return "detail"

    # 2. Check title / headings for specific single unit signals
    page_title = (soup.title.string if soup.title else "") or ""
    og_title = (soup.find("meta", property="og:title") or {}).get("content", "")
    full_title = f"{page_title} {og_title}".lower()

    if any(sig in full_title for sig in ("detail seller", "spesifikasi", "informasi mobil", "detail iklan", "detail unit")):
        return "detail"
    if any(sig in full_title for sig in ("daftar mobil", "bursa mobil", "cari mobil", "harga kredit murah di")):
        return "listing"

    # 3. Check pagination or search filters
    pagination_signs = len(soup.find_all(class_=re.compile(r"pagination|paging|page-numbers", re.I)))
    if pagination_signs > 0:
        return "listing"

    # 4. Count distinct price occurrences in text (excluding DP / cicilan)
    text = soup.get_text(" ", strip=True)
    prices = re.findall(r"(?:rp\.?|idr)\s*[\d\.,]+(?:\s*(?:juta|jt|miliar|milyar))?", text, re.IGNORECASE)
    odometers = re.findall(r"\b[\d\.,]+\s*(?:km|kilometer)\b", text, re.IGNORECASE)

    if len(prices) >= 5:
        return "listing"
    if len(prices) >= 1 and len(odometers) >= 1:
        return "detail"

    return "other"


def classify_url(url: str, html_content: str | None = None) -> UrlType:
    """Classify a URL into 'detail', 'listing', or 'other'.
    
    Uses domain patterns first. If domain is unknown, falls back to content heuristics
    if HTML is provided; otherwise inspects path characteristics.
    """
    classified = classify_url_by_pattern(url)
    if classified is not None:
        return classified

    if html_content:
        return classify_url_by_content_heuristics(html_content)

    # Path heuristics for unknown domains
    parsed = urlsplit(url.strip())
    path = parsed.path.lower()
    if re.search(r"\b(cari|search|daftar|listing|list|kategori|category|jual-beli)\b", path):
        return "listing"
    if re.search(r"\b(detail|unit|mobil|iklan|product|item|car|cars)\b", path) and re.search(r"\d+", path):
        return "detail"

    return "other"


def extract_detail_links_from_html(
    base_url: str,
    html_content: str,
    *,
    expected_model: str = "",
    max_links: int = 10,
) -> list[str]:
    """Extract distinct detail page links from a listing/category page HTML.
    
    Filters anchors by classify_url == 'detail', validates host, and optionally
    checks relevance to expected_model.
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    parsed_base = urlsplit(base_url)
    base_host = _normalize_host(parsed_base.netloc)

    found_links: list[str] = []
    seen: set[str] = set()

    model_clean = expected_model.strip().lower()

    for anchor in soup.find_all("a", href=True):
        raw_href = anchor["href"].strip()
        if not raw_href or raw_href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue

        abs_url = urljoin(base_url, raw_href)
        parsed_target = urlsplit(abs_url)
        target_host = _normalize_host(parsed_target.netloc)

        # Host must match or be a subdomain
        if target_host != base_host and not target_host.endswith("." + base_host):
            continue

        # Strip query and fragments for canonical comparison of detail pages
        clean_url = urlunsplit((parsed_target.scheme, parsed_target.netloc, parsed_target.path.rstrip("/"), "", ""))
        if clean_url in seen:
            continue

        if classify_url(clean_url) == "detail":
            # If expected model is given, verify it appears in the URL slug or anchor text
            if model_clean:
                slug_or_text = (clean_url + " " + anchor.get_text(" ")).lower()
                if model_clean not in slug_or_text:
                    # Model not mentioned, might be another vehicle on the same listing page
                    continue

            seen.add(clean_url)
            found_links.append(clean_url)
            if len(found_links) >= max_links:
                break

    return found_links
