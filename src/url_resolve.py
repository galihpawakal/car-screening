"""Resolve grounding redirect URLs to final listing domains."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

USER_AGENT = "CarScreening/1.0 (Jabodetabek price research; +https://github.com/local/car-screening)"


@dataclass(frozen=True)
class ResolvedUrl:
    original: str
    final_url: str
    domain: str


def normalize_host(netloc: str) -> str:
    return netloc.casefold().removeprefix("www.")


def is_rejected_listing_domain(netloc: str) -> bool:
    host = normalize_host(netloc)
    if not host:
        return True
    if host == "google.com" or host.endswith(".google.com"):
        return True
    if host in {"google.co.id", "googleusercontent.com"} or host.endswith(".googleusercontent.com"):
        return True
    if "vertexaisearch" in host or host.endswith("goo.gl"):
        return True
    return False


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def resolve_url(
    url: str,
    *,
    timeout: float = 3.0,
    opener: Callable[..., object] | None = None,
) -> ResolvedUrl | None:
    """Follow redirects (short timeout) and return the final URL if domain is allowed."""
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None

    initial_domain = normalize_host(parts.netloc)
    # If the URL is already an allowed marketplace domain (not a Google/redirect domain), skip network
    if not is_rejected_listing_domain(initial_domain) and "redirect" not in url.lower():
        return ResolvedUrl(original=url, final_url=url, domain=initial_domain)

    final = url
    if opener is not None:
        request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        try:
            with opener(request, timeout=timeout) as response:
                final = getattr(response, "url", None) or url
        except Exception:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with opener(request, timeout=timeout) as response:
                    final = getattr(response, "url", None) or url
            except Exception as error:
                logger.debug("URL resolve failed with custom opener for %s: %s", url, error)
    else:
        # Intercept redirect header immediately (0.05s) without fetching the destination page
        fast_opener = urllib.request.build_opener(_NoRedirectHandler)
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with fast_opener.open(request, timeout=timeout) as response:
                final = getattr(response, "url", None) or url
        except urllib.error.HTTPError as error:
            loc = error.headers.get("Location") if getattr(error, "headers", None) else None
            if loc:
                final = loc
        except Exception as error:
            logger.debug("Fast URL redirect resolve failed for %s: %s", url, error)

    final_parts = urlsplit(final)
    if final_parts.scheme not in {"http", "https"} or not final_parts.netloc:
        return None
    domain = normalize_host(final_parts.netloc)
    if is_rejected_listing_domain(domain):
        logger.debug("Rejected listing domain after resolve: %s -> %s", url, final)
        return None
    return ResolvedUrl(original=url, final_url=final, domain=domain)


def resolve_grounding_urls(
    urls: list[str],
    *,
    timeout: float = 3.0,
    opener: Callable[..., object] | None = None,
) -> list[ResolvedUrl]:
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        unique_urls.append(url)

    if not unique_urls:
        return []

    def _resolve_task(u: str) -> ResolvedUrl | None:
        try:
            return resolve_url(u, timeout=timeout, opener=opener)
        except Exception:
            return None

    max_workers = min(8, len(unique_urls))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        items = list(executor.map(_resolve_task, unique_urls))

    resolved: list[ResolvedUrl] = []
    for item in items:
        if item is not None:
            canonical = item.final_url.casefold().rstrip("/")
            if canonical not in {r.final_url.casefold().rstrip("/") for r in resolved}:
                resolved.append(item)
    return resolved
