"""Tests for grounding URL resolution, page contract, scoring, and usage metadata."""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from src.clean import clean_listings
from src.config import MIN_COMPARABLES_FOR_SCORE
from src.extract import ExtractedListing, ListingExtraction, extract_listings
from src.scoring import calculate_screening_score, format_screening_result
from src.search_provider import (
    GroundedSearchResult,
    GroundingSource,
    StructuredResponse,
    TokenUsage,
    _grounding_metadata,
    _usage,
)
from src.url_resolve import ResolvedUrl, is_rejected_listing_domain, resolve_url


class _MockResponse:
    def __init__(self, url: str):
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b""


def test_google_domain_rejected_after_redirect():
    assert is_rejected_listing_domain("www.google.com")
    assert is_rejected_listing_domain("google.com")

    def opener(_request, timeout=0):
        return _MockResponse("https://www.olx.co.id/item/123")

    resolved = resolve_url("https://www.google.com/url?q=https://olx.co.id/item/123", opener=opener)
    assert resolved is not None
    assert "olx.co.id" in resolved.domain


def test_grounding_metadata_resolves_and_skips_google(monkeypatch):
    def fake_resolve(urls, timeout=5.0, opener=None):
        out: list[ResolvedUrl] = []
        for url in urls:
            if "google.com" in url:
                continue
            if "redirect" in url:
                out.append(
                    ResolvedUrl(
                        original=url,
                        final_url="https://mobil123.com/mobil/toyota-avanza-123",
                        domain="mobil123.com",
                    )
                )
            else:
                host = urlsplit(url).netloc
                out.append(ResolvedUrl(original=url, final_url=url, domain=host.removeprefix("www.")))
        return out

    monkeypatch.setattr("src.search_provider.resolve_grounding_urls", fake_resolve)
    response = SimpleNamespace(
        steps=[],
        grounding_metadata={
            "web_search_queries": ["Toyota Avanza 2020"],
            "grounding_chunks": [
                {"web": {"uri": "https://vertex.example/redirect/1", "title": "OLX"}},
                {"web": {"uri": "https://www.google.com/search?q=avanza", "title": "Google"}},
            ],
        },
    )
    sources, queries = _grounding_metadata(response, marketplace="mobil123.com")
    assert queries == ["Toyota Avanza 2020"]
    assert len(sources) == 1
    assert sources[0].url == "https://mobil123.com/mobil/toyota-avanza-123"


def test_extract_rejects_urls_not_in_grounding_chunks():
    class FakeProvider:
        def generate_structured(self, prompt, response_model):
            payload = ListingExtraction(
                listings=[
                    ExtractedListing(
                        brand="Toyota",
                        model="Avanza",
                        variant="G",
                        year="2020",
                        kilometer="50.000",
                        transmission="AT",
                        location="Jakarta",
                        seller_type="dealer",
                        price="Rp 200.000.000",
                        source_url="https://olx.co.id/invented-listing",
                    )
                ]
            ).model_dump_json()
            return StructuredResponse(payload, TokenUsage(3, 4))

    search = GroundedSearchResult(
        text="teks model menulis URL palsu",
        sources=[GroundingSource("https://olx.co.id/real-listing", "OLX")],
        queries=[],
        usage=TokenUsage(1, 1),
    )
    result, _usage = extract_listings(FakeProvider(), search)
    assert result.listings == []


def test_clean_rejects_listing_without_grounded_url():
    extraction = ListingExtraction(
        listings=[
            ExtractedListing(
                brand="Toyota",
                model="Avanza",
                variant="G",
                year="2020",
                kilometer="50000",
                transmission="AT",
                location="Jakarta",
                seller_type="dealer",
                price="Rp 200.000.000",
                source_url="https://olx.co.id/no-grounding",
            )
        ]
    )
    cleaned = clean_listings(
        extraction,
        expected_brand="Toyota",
        expected_model="Avanza",
        expected_transmission="AT",
        grounded_urls={"https://olx.co.id/grounded-only"},
    )
    assert cleaned.listings == []
    assert cleaned.rejected[0].reason == "URL tidak tercantum dalam sitasi grounding"


def test_score_not_assessable_when_fewer_than_min_comparables():
    from src.rules import run_all_rules

    flags = run_all_rules(220_000_000, 0, 80_000, 2020, "Avanza", None, current_year=2026)
    result = calculate_screening_score(
        listing_price=220_000_000,
        price_low=0,
        price_median=0,
        price_high=0,
        n_comparables=MIN_COMPARABLES_FOR_SCORE - 1,
        kilometer=80_000,
        year=2020,
        has_description=False,
        has_varian=True,
        has_plate=True,
        has_location=True,
        rule_flags=flags,
    )
    formatted = format_screening_result(result)
    assert result.total_score < 0
    assert formatted["total_score_display"] == "Tidak dapat dinilai"
    assert formatted["score_assessable"] is False
    desc = next(item for item in result.components if item.name == "Risiko Deskripsi")
    # Empty description -> assessed=False, score=-1
    assert desc.assessed is False
    assert desc.score == -1


def test_usage_metadata_prompt_and_candidate_counts():
    response = SimpleNamespace(
        usage_metadata={"prompt_token_count": 120, "candidates_token_count": 45}
    )
    usage = _usage(response)
    assert usage.input_tokens == 120
    assert usage.output_tokens == 45
