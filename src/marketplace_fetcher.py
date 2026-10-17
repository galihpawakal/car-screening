"""High-coverage marketplace scraper with pagination, slug pre-filtering, and bounded concurrency."""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from src.extract import ExtractedListing
from src.marketplace_config import (
    MARKETPLACE_CATALOG_CONFIGS,
    MarketplaceDomainConfig,
    build_catalog_url,
    extract_year_from_slug,
    get_model_slug_variants,
    get_twin_models,
    slugify_model,
)
from src.page_reader import extract_listing_from_page, fetch_page_html, fetch_page_response
from src.url_classifier import classify_url, extract_detail_links_from_html

logger = logging.getLogger(__name__)

# Global session cache so duplicate URLs in the same run/session are never re-fetched
_SESSION_FETCH_CACHE: dict[str, Any] = {}


@dataclass
class CrawlFunnelMetrics:
    catalog_urls_opened: list[dict[str, Any]] = field(default_factory=list)
    detail_links_found: int = 0
    detail_links_prefiltered_out: int = 0
    detail_pages_fetched: int = 0
    detail_pages_parsed: int = 0
    rejected_urls: list[dict[str, str]] = field(default_factory=list)


def is_slug_year_acceptable(url: str, target_year: int, tolerance: int = 1) -> bool:
    """Pre-filter check: returns False if URL slug explicitly mentions a year outside tolerance."""
    extracted_yr = extract_year_from_slug(url)
    if extracted_yr is None:
        return True  # Year not visible in slug, allow fetch
    return abs(extracted_yr - target_year) <= tolerance


def crawl_marketplace_catalog(
    domain: str,
    brand: str,
    model: str,
    *,
    year: int,
    transmission: str,
    max_pages: int = 3,
    target_candidates: int = 15,
    notify: Callable[[str], None] = lambda _: None,
    funnel: CrawlFunnelMetrics | None = None,
) -> list[str]:
    """Crawl paginated catalog pages for a domain with slug fallback and detailed diagnostics."""
    cfg = MARKETPLACE_CATALOG_CONFIGS.get(domain)
    if not cfg:
        return []

    # 1. Check if domain is blocked or requires JavaScript
    if not getattr(cfg, "is_accessible_from_server", True):
        reason = cfg.notes or "tidak dapat diakses dari server"
        catalog_url = build_catalog_url(domain, brand=brand, model=model, year=year, transmission=transmission, page=1) or cfg.base_url
        if funnel:
            funnel.catalog_urls_opened.append({
                "url": catalog_url,
                "url_requested": catalog_url,
                "final_url": catalog_url,
                "domain": domain,
                "page": 1,
                "http_status": 403 if "403" in reason else (200 if "JavaScript" in reason else 0),
                "response_bytes": 0,
                "total_anchors": 0,
                "detail_links_passed": 0,
                "body_snippet": "",
                "status": "skipped",
                "failure_reason": f"{domain}: {reason}",
                "links_found": 0,
            })
        notify(f"⚠️ {cfg.name}: {reason}. Dilewati.")
        return []

    collected_detail_urls: list[str] = []
    seen: set[str] = set()
    seen_catalog_urls: set[str] = set()

    slug_variants = get_model_slug_variants(domain, brand, model)
    working_slug: str | None = None
    working_include_year: bool = True

    # Phase 1: Try all slug variants with year filter (if supported)
    # Phase 2: If all slug variants yielded 0 links, try relaxing year filter
    strategies = [True, False] if cfg.supports_year_filter else [False]
    for inc_year in strategies:
        filter_desc = "dengan filter tahun" if inc_year else "tanpa filter tahun"
        for slug in slug_variants:
            catalog_url = build_catalog_url(
                domain,
                brand=brand,
                model=model,
                year=year,
                min_year=year - 1,
                max_year=year + 1,
                transmission=transmission,
                page=1,
                model_slug_override=slug,
                include_year_filter=inc_year,
            )
            if not catalog_url or catalog_url in seen_catalog_urls:
                continue
            seen_catalog_urls.add(catalog_url)

            notify(f"📄 Mencoba katalog {cfg.name} (slug: {slug}, {filter_desc}): {catalog_url[:80]}...")
            resp = fetch_page_response(catalog_url, timeout=6.0, retries=1, delay=0.2)

            soup = BeautifulSoup(resp.html, "html.parser") if resp.html else None
            anchors_count = len(soup.find_all("a", href=True)) if soup else 0

            page_links: list[str] = []
            if resp.status == 200 and resp.html:
                page_links = extract_detail_links_from_html(
                    resp.final_url,
                    resp.html,
                    expected_model=model,
                    max_links=100,
                )

            body_snippet = soup.get_text(" ", strip=True)[:200] if (soup and len(page_links) == 0) else ""

            failure_reason = None
            if len(page_links) == 0:
                if resp.status == 404:
                    failure_reason = f"{domain}: 404"
                elif resp.status == 403:
                    failure_reason = f"{domain}: diblokir 403"
                elif resp.status == 200:
                    failure_reason = f"{domain}: 0 tautan (halaman kosong)"
                elif resp.error:
                    failure_reason = f"{domain}: error ({resp.error})"
                else:
                    failure_reason = f"{domain}: 0 tautan"

            new_links = [l for l in page_links if l not in seen]

            if len(new_links) > 0:
                working_slug = slug
                working_include_year = inc_year
                for l in new_links:
                    seen.add(l)
                    collected_detail_urls.append(l)

                if funnel:
                    funnel.catalog_urls_opened.append({
                        "url": catalog_url,
                        "url_requested": catalog_url,
                        "final_url": resp.final_url,
                        "domain": domain,
                        "page": 1,
                        "http_status": resp.status,
                        "response_bytes": resp.response_bytes,
                        "total_anchors": anchors_count,
                        "detail_links_passed": len(page_links),
                        "body_snippet": "",
                        "status": "success",
                        "failure_reason": None,
                        "links_found": len(new_links),
                    })
                    funnel.detail_links_found += len(new_links)

                notify(f"🔗 {cfg.name} (slug: {slug}): {len(new_links)} tautan detail ditemukan.")
                break
            else:
                if funnel:
                    funnel.catalog_urls_opened.append({
                        "url": catalog_url,
                        "url_requested": catalog_url,
                        "final_url": resp.final_url,
                        "domain": domain,
                        "page": 1,
                        "http_status": resp.status,
                        "response_bytes": resp.response_bytes,
                        "total_anchors": anchors_count,
                        "detail_links_passed": len(page_links),
                        "body_snippet": body_snippet,
                        "status": "empty" if resp.status == 200 else "error",
                        "failure_reason": failure_reason,
                        "links_found": 0,
                    })

        if working_slug:
            break

    # If page 1 found links and max_pages > 1, paginate remaining pages with the working slug
    if working_slug and len(collected_detail_urls) < target_candidates and max_pages > 1:
        for page_num in range(2, max_pages + 1):
            catalog_url = build_catalog_url(
                domain,
                brand=brand,
                model=model,
                year=year,
                min_year=year - 1,
                max_year=year + 1,
                transmission=transmission,
                page=page_num,
                model_slug_override=working_slug,
                include_year_filter=working_include_year,
            )
            if not catalog_url:
                break

            notify(f"📄 Membuka katalog {cfg.name} (hal. {page_num}/{max_pages}): {catalog_url[:80]}...")
            resp = fetch_page_response(catalog_url, timeout=6.0, retries=1, delay=0.2)
            if resp.status != 200 or not resp.html:
                break

            soup = BeautifulSoup(resp.html, "html.parser")
            anchors_count = len(soup.find_all("a", href=True))
            page_links = extract_detail_links_from_html(
                resp.final_url,
                resp.html,
                expected_model=model,
                max_links=100,
            )
            new_links = [l for l in page_links if l not in seen]
            for l in new_links:
                seen.add(l)
                collected_detail_urls.append(l)

            if funnel:
                funnel.catalog_urls_opened.append({
                    "url": catalog_url,
                    "url_requested": catalog_url,
                    "final_url": resp.final_url,
                    "domain": domain,
                    "page": page_num,
                    "http_status": resp.status,
                    "response_bytes": resp.response_bytes,
                    "total_anchors": anchors_count,
                    "detail_links_passed": len(page_links),
                    "body_snippet": "",
                    "status": "success" if new_links else "empty",
                    "failure_reason": None if new_links else f"{domain}: 0 tautan baru",
                    "links_found": len(new_links),
                })
                funnel.detail_links_found += len(new_links)

            notify(f"🔗 {cfg.name} hal. {page_num}: {len(new_links)} tautan detail ditemukan.")
            if len(new_links) == 0 or len(collected_detail_urls) >= target_candidates:
                break

    return collected_detail_urls


def fetch_and_parse_details_concurrent(
    urls: list[str],
    *,
    expected_brand: str,
    expected_model: str,
    expected_year: int,
    max_workers: int = 4,
    max_total_fetch: int = 40,
    notify: Callable[[str], None] = lambda _: None,
    funnel: CrawlFunnelMetrics | None = None,
) -> tuple[list[ExtractedListing], list[dict[str, str]]]:
    """Fetch and parse detail URLs concurrently up to max_workers with slug pre-filtering."""
    extracted: list[ExtractedListing] = []
    rejected: list[dict[str, str]] = []

    # 1. Slug pre-filtering to save network requests
    urls_to_fetch: list[str] = []
    for u in urls:
        if len(urls_to_fetch) >= max_total_fetch:
            break
        # Allow year tolerance ±1 or if year is not in slug
        if not is_slug_year_acceptable(u, expected_year, tolerance=1):
            extracted_yr = extract_year_from_slug(u)
            rej_entry = {
                "source_url": u,
                "reason": f"Tahun di slug URL di luar rentang ({extracted_yr} vs {expected_year}±1)",
            }
            rejected.append(rej_entry)
            if funnel:
                funnel.detail_links_prefiltered_out += 1
                funnel.rejected_urls.append(rej_entry)
            continue
        urls_to_fetch.append(u)

    notify(f"🚀 Membaca {len(urls_to_fetch)} halaman detail (maks {max_workers} koneksi paralel)...")

    def _fetch_one(detail_url: str):
        if detail_url in _SESSION_FETCH_CACHE:
            return _SESSION_FETCH_CACHE[detail_url]

        try:
            outcome = extract_listing_from_page(
                None,  # Primary zero-token structured extractor
                detail_url,
                expected_brand=expected_brand,
                expected_model=expected_model,
            )
            _SESSION_FETCH_CACHE[detail_url] = outcome
            return outcome
        except Exception as err:
            logger.debug("Failed detail fetch %s: %s", detail_url, err)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(_fetch_one, u): u for u in urls_to_fetch}
        for future in concurrent.futures.as_completed(future_to_url):
            u = future_to_url[future]
            domain = urlsplit(u).netloc
            if funnel:
                funnel.detail_pages_fetched += 1
            try:
                outcome = future.result()
                if outcome and outcome.listing:
                    extracted.append(outcome.listing)
                    if funnel:
                        funnel.detail_pages_parsed += 1
                    price_str = str(outcome.listing.price)
                    notify(f"✅ Listing detail terbaca ({domain}): {outcome.listing.variant} ({price_str})")
                else:
                    reason = outcome.reason if outcome else "Gagal mengunduh halaman detail"
                    rej_entry = {"source_url": u, "reason": reason}
                    rejected.append(rej_entry)
                    if funnel:
                        funnel.rejected_urls.append(rej_entry)
            except Exception as err:
                rej_entry = {"source_url": u, "reason": f"Koneksi terputus saat membaca detail: {err}"}
                rejected.append(rej_entry)
                if funnel:
                    funnel.rejected_urls.append(rej_entry)

    return extracted, rejected
