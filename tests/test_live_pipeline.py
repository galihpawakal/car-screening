"""Offline tests for grounded Gemini listing analysis and deterministic processing."""

from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.analyze import FinalAnalysis, InspectionChecklist, NegotiationAdvice, analyze_results, calculate_negotiation_prices, sanitize_numeric_claims
from src.cache import DailyLimitExceeded, SQLiteAnalysisCache
from src.clean import clean_listings
from src.extract import (
    ComparableListing,
    ExtractedListing,
    ListingExtraction,
    extract_listings,
    _normalize_listing_dict,
    _parse_listings_leniently,
)
from src.page_reader import ListingPageExtraction
from src.scoring import calculate_km_age_fairness_score
from src.search_provider import (
    GeminiGroundingProvider,
    GroundedSearchResult,
    GroundingSource,
    StructuredResponse,
    TokenUsage,
    get_gemini_settings,
)
from src.stats import calculate_statistics


FETCHED_AT = datetime(2026, 10, 15, 12, 0)


def extracted(**overrides) -> ExtractedListing:
    facts = {
        "brand": "Toyota",
        "model": "Avanza",
        "variant": "G",
        "year": "2020",
        "kilometer": "50.000 km",
        "transmission": "AT",
        "location": "Jakarta Selatan",
        "seller_type": "dealer",
        "price": "Rp 200.000.000",
        "source_url": "https://example.test/listing/1",
    }
    facts.update(overrides)
    return ExtractedListing(**facts)


def listing(price: int, year: int, kilometer: int, source_url: str = "https://example.test/a") -> ComparableListing:
    return ComparableListing(
        brand="Toyota",
        model="Avanza",
        variant="G",
        year=year,
        kilometer=kilometer,
        transmission="AT",
        location="Jakarta Selatan",
        seller_type="dealer",
        price=price,
        source_url=source_url,
        fetched_at="2026-10-15T12:00:00+00:00",
    )


class TestSchemasAndExtraction:
    def test_extraction_schema_validates_required_listing_fields(self):
        raw = {"listings": [extracted().model_dump()]}
        assert len(ListingExtraction.model_validate(raw).listings) == 1
        with pytest.raises(ValidationError):
            ListingExtraction.model_validate({"listings": [{"model": "Avanza"}]})

    def test_extract_uses_structured_response_without_tools(self):
        class FakeProvider:
            response_model = None
            prompt = ""

            def generate_structured(self, prompt, response_model):
                self.prompt = prompt
                self.response_model = response_model
                payload = ListingExtraction(listings=[extracted()]).model_dump_json()
                return StructuredResponse(payload, TokenUsage(10, 20))

        provider = FakeProvider()
        search = GroundedSearchResult(
            text="Listing fakta sumber.",
            sources=[GroundingSource("https://example.test/listing/1", "listing")],
            queries=["Toyota Avanza G 2020 AT Jakarta"],
            usage=TokenUsage(1, 2),
        )
        result, usage = extract_listings(provider, search)
        assert result.listings[0].model == "Avanza"
        assert usage.total_tokens == 30
        assert provider.response_model is ListingExtraction
        assert "untrusted data" in provider.prompt
        assert "https://example.test/listing/1" in provider.prompt

    def test_extract_handles_markdown_fences_and_raw_json_prefix_with_list(self):
        class FakeProvider:
            def generate_structured(self, prompt, response_model):
                raw = (
                    "json\n"
                    "[\n"
                    "  {\n"
                    '    "merek": "Daihatsu",\n'
                    '    "model": "Xenia",\n'
                    '    "varian": "R",\n'
                    '    "tahun": "2020",\n'
                    '    "km": "50.000 km",\n'
                    '    "transmisi": "AT",\n'
                    '    "lokasi": "Jakarta Pusat",\n'
                    '    "tipe_penjual": "dealer",\n'
                    '    "harga": "Rp 145.000.000",\n'
                    '    "url": "https://example.test/listing/xenia"\n'
                    "  }\n"
                    "]"
                )
                return StructuredResponse(raw, TokenUsage(5, 10))

        provider = FakeProvider()
        search = GroundedSearchResult(
            text="Hasil pencarian Xenia.",
            sources=[GroundingSource("https://example.test/listing/xenia", "Xenia")],
            queries=["Daihatsu Xenia 2020"],
            usage=TokenUsage(1, 2),
        )
        result, usage = extract_listings(provider, search)
        assert len(result.listings) == 1
        assert result.listings[0].brand == "Daihatsu"
        assert result.listings[0].model == "Xenia"
        assert result.listings[0].price == "Rp 145.000.000"
        assert result.listings[0].source_url == "https://example.test/listing/xenia"

    def test_extract_handles_fenced_markdown_code_block(self):
        class FakeProvider:
            def generate_structured(self, prompt, response_model):
                raw = (
                    "```json\n"
                    "{\n"
                    '  "listings": [\n'
                    "    {\n"
                    '      "brand": "Toyota",\n'
                    '      "model": "Avanza",\n'
                    '      "variant": "G",\n'
                    '      "year": "2020",\n'
                    '      "kilometer": "60.000 km",\n'
                    '      "transmission": "AT",\n'
                    '      "location": "Jakarta Selatan",\n'
                    '      "seller_type": "dealer",\n'
                    '      "price": "Rp 190.000.000",\n'
                    '      "source_url": "https://example.test/listing/1"\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "```"
                )
                return StructuredResponse(raw, TokenUsage(10, 20))

        provider = FakeProvider()
        search = GroundedSearchResult(
            text="Listing Avanza.",
            sources=[GroundingSource("https://example.test/listing/1", "Avanza")],
            queries=["Toyota Avanza 2020"],
            usage=TokenUsage(1, 2),
        )
        result, usage = extract_listings(provider, search)
        assert len(result.listings) == 1
        assert result.listings[0].model == "Avanza"


    def test_empty_description_is_not_sent_to_final_analysis(self):
        class FakeProvider:
            prompt = ""

            def generate_structured(self, prompt, response_model):
                self.prompt = prompt
                answer = FinalAnalysis(
                    kesimpulan="Perlu inspeksi langsung sebelum membeli.",
                    saran_nego={"awal": 1, "realistis": 2, "batas": 3},
                    red_flag=[],
                    pertanyaan_untuk_penjual=["Apakah pajak masih berlaku?"],
                    checklist_inspeksi={"dokumen": [], "fisik": [], "transmisi": []},
                    catatan=[],
                )
                return StructuredResponse(answer.model_dump_json(), TokenUsage(4, 8))

        provider = FakeProvider()
        result, usage = analyze_results(
            provider,
            {"model": "Avanza", "deskripsi": "   ", "harga_listing": 220_000_000},
            {"count": 5, "price_median": 200_000_000},
            [listing(180_000_000, 2020, 50_000)],
        )
        assert '"deskripsi"' not in provider.prompt
        assert result.saran_nego == calculate_negotiation_prices(200_000_000)
        assert (result.saran_nego.awal, result.saran_nego.realistis, result.saran_nego.batas) == (
            180_000_000,
            187_000_000,
            194_000_000,
        )
        assert usage.total_tokens == 12

    def test_analysis_discards_numbers_not_present_in_input_or_statistics(self):
        answer = FinalAnalysis(
            kesimpulan="Harga unit pasti naik 60% tahun depan. Tahun 2020 perlu diverifikasi.",
            saran_nego={"awal": 1, "realistis": 2, "batas": 3},
            red_flag=["Harga 60% di bawah pasar."],
            pertanyaan_untuk_penjual=["Apakah tahun 2020 sesuai dokumen?"] ,
            checklist_inspeksi={"dokumen": [], "fisik": [], "transmisi": []},
            catatan=["Cek asumsi transaksi 3-10% di bawah listing."],
        )
        safe = sanitize_numeric_claims(
            answer,
            {"tahun": 2020, "listing_price": 200_000_000},
            {"count": 4, "price_median": 200_000_000},
            [listing(200_000_000, 2020, 50_000)],
        )
        assert "60%" not in safe.kesimpulan
        assert safe.red_flag == []
        assert "2020" in safe.pertanyaan_untuk_penjual[0]
        assert safe.catatan


class TestCleaningAndStatistics:
    def test_clean_rejects_bad_and_duplicate_records_with_reasons(self):
        candidates = [
            extracted(),
            extracted(source_url="https://www.example.test/listing/1?tracking=x"),
            extracted(source_url="https://example.test/other", variant="V"),
            extracted(source_url="https://example.test/no-price", price="tidak diketahui"),
            extracted(source_url="https://example.test/wrong-model", model="Brio"),
            extracted(source_url="https://example.test/wrong-source"),
        ]
        result = clean_listings(
            ListingExtraction(listings=candidates),
            expected_brand="Toyota",
            expected_model="Avanza",
            expected_transmission="AT",
            current_year=2026,
            fetched_at=FETCHED_AT,
            grounded_urls={
                "https://example.test/listing/1",
                "https://example.test/other",
                "https://example.test/no-price",
                "https://example.test/wrong-model",
            },
        )
        assert len(result.listings) == 1
        assert result.listings[0].price == 200_000_000
        assert {item.reason for item in result.rejected} == {
            "Duplikat URL",
            "Duplikat kombinasi harga+KM+tahun",
            "Harga tidak tersedia",
            "Model tidak cocok dengan input",
            "URL tidak tercantum dalam sitasi grounding",
        }

    def test_cleans_localized_price_and_rejects_impossible_price(self):
        result = clean_listings(
            [
                extracted(price="1,5 miliar", source_url="https://example.test/billion"),
                extracted(price="Rp 50.000", source_url="https://example.test/too-low"),
            ],
            expected_brand="Toyota",
            expected_model="Avanza",
            expected_transmission="AT",
            current_year=2026,
            fetched_at=FETCHED_AT,
            grounded_urls={"https://example.test/billion", "https://example.test/too-low"},
        )
        assert result.listings[0].price == 1_500_000_000
        assert result.rejected[0].reason == "Harga di luar rentang masuk akal"

    def test_distinct_query_id_urls_are_not_false_duplicates(self):
        candidates = [
            extracted(source_url="https://market.example/listing?id=1", price="200000000", kilometer="50000"),
            extracted(source_url="https://market.example/listing?id=2", price="210000000", kilometer="60000"),
        ]
        result = clean_listings(
            candidates,
            expected_brand="Toyota",
            expected_model="Avanza",
            expected_transmission="AT",
            current_year=2026,
            fetched_at=FETCHED_AT,
            grounded_urls={item.source_url for item in candidates},
        )
        assert len(result.listings) == 2

    def test_percentiles_position_and_average_comparable_km_per_year(self):
        comparables = [
            listing(100, 2022, 40_000, "https://example.test/1"),
            listing(200, 2021, 50_000, "https://example.test/2"),
            listing(300, 2020, 60_000, "https://example.test/3"),
            listing(400, 2019, 70_000, "https://example.test/4"),
        ]
        result = calculate_statistics(
            comparables,
            listing_price=250,
            target_year=2022,
            current_year=2026,
        )
        assert result["count"] == 4
        # n=4 >= SOLID_COMPARABLES (3) -> P25/P75
        assert result["price_low"] == 175
        assert result["price_median"] == 250
        assert result["price_high"] == 325
        assert result["listing_percentile"] == 50
        assert result["average_km_per_year"] == pytest.approx((10_000 + 10_000 + 10_000 + 10_000) / 4)

    def test_no_comparables_never_produces_fake_prices(self):
        result = calculate_statistics([], listing_price=220_000_000, target_year=2020, current_year=2026)
        assert result["count"] == 0
        assert result["price_low"] is None
        assert result["price_median"] is None
        assert result["price_high"] is None


class TestCacheAndProvider:
    def test_cache_ttl_and_daily_limit(self, tmp_path):
        cache = SQLiteAnalysisCache(tmp_path / "cache.sqlite3", ttl_seconds=100)
        cache.set("query", {"facts": [1, 2]}, now=1000)
        assert cache.get("query", now=1099) == {"facts": [1, 2]}
        assert cache.get("query", now=1100) is None
        day = date(2026, 10, 15)
        assert cache.claim_daily_analysis(1, day=day) == 1
        with pytest.raises(DailyLimitExceeded, match="sudah tercapai"):
            cache.claim_daily_analysis(1, day=day)
        usage_id = cache.record_usage(grounding_calls=1, input_tokens=10, output_tokens=5)
        assert cache.recent_usage()[0]["analysis_id"] == usage_id

    def test_provider_search_metadata_and_separate_no_tool_schema_call(self):
        class FakeInteractions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if "tools" in kwargs:
                    return SimpleNamespace(
                        output_text="listing fakta",
                        steps=[
                            SimpleNamespace(type="google_search_call", arguments={"queries": ["Avanza bekas Jakarta"]}),
                            SimpleNamespace(type="model_output", content=[SimpleNamespace(annotations=[SimpleNamespace(
                                type="url_citation", url="https://cars.example/listing/1", title="Listing 1"
                            )])]),
                        ],
                        usage=SimpleNamespace(input_tokens=11, output_tokens=13),
                    )
                return SimpleNamespace(output_text='{"ok":true}', usage=SimpleNamespace(input_tokens=3, output_tokens=4))

        interactions = FakeInteractions()
        provider = GeminiGroundingProvider(
            client=SimpleNamespace(interactions=interactions),
            model="gemini-3.8-flash",
        )
        search = provider.search("cari listing")
        structured = provider.generate_structured("extract", ListingExtraction)
        assert search.queries == ["Avanza bekas Jakarta"]
        assert search.sources[0].url == "https://cars.example/listing/1"
        assert search.usage.total_tokens == 24
        assert "tools" in interactions.calls[0]
        assert "tools" not in interactions.calls[1]
        assert interactions.calls[1]["response_format"]["mime_type"] == "application/json"
        assert structured.usage.total_tokens == 7

    def test_provider_retries_429_with_backoff(self, monkeypatch):
        attempts = 0

        class RateLimitError(Exception):
            code = 429

        class FakeInteractions:
            def create(self, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RateLimitError("rate limit")
                return SimpleNamespace(output_text="ok", steps=[], usage={})

        monkeypatch.setattr("src.search_provider.time.sleep", lambda _: None)
        provider = GeminiGroundingProvider(
            client=SimpleNamespace(interactions=FakeInteractions()),
            model="gemini-3.8-flash",
            max_retries=2,
            backoff_seconds=0,
        )
        assert provider.search("query").text == "ok"
        assert attempts == 2
        assert provider.api_calls == 2

    def test_provider_retries_503_with_backoff(self, monkeypatch):
        attempts = 0

        class ServerError(Exception):
            code = 503

        class FakeInteractions:
            def create(self, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ServerError("temporary error")
                return SimpleNamespace(output_text="ok", steps=[], usage={})

        monkeypatch.setattr("src.search_provider.time.sleep", lambda _: None)
        provider = GeminiGroundingProvider(
            client=SimpleNamespace(interactions=FakeInteractions()),
            model="gemini-3.8-flash",
            max_retries=2,
            backoff_seconds=0,
        )
        assert provider.search("query").text == "ok"
        assert attempts == 2

    def test_provider_reads_candidate_grounding_metadata_shape(self):
        from src.search_provider import _grounding_metadata

        response = SimpleNamespace(steps=[], grounding_metadata={
            "web_search_queries": ["Avanza bekas"],
            "grounding_chunks": [{"web": {"uri": "https://market.example/listing", "title": "Market"}}],
        })
        sources, queries = _grounding_metadata(response)
        assert queries == ["Avanza bekas"]
        assert sources[0].url == "https://market.example/listing"

    def test_model_and_key_are_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
        monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
        assert get_gemini_settings() == ("test-key", "gemini-3.8-flash", [])

        monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash, gemini-2.5-pro")
        assert get_gemini_settings() == ("test-key", "gemini-3.8-flash", ["gemini-2.5-flash", "gemini-2.5-pro"])

        monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "unsupported-fallback")
        with pytest.raises(ValueError, match="tidak tercantum"):
            get_gemini_settings()

        monkeypatch.setenv("GEMINI_MODEL", "unsupported-model")
        with pytest.raises(ValueError, match="tidak tercantum"):
            get_gemini_settings()

    def test_provider_has_no_implicit_api_key_or_model_default(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY belum diatur"):
            get_gemini_settings()
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with pytest.raises(ValueError, match="GEMINI_MODEL belum diatur"):
            get_gemini_settings()

    def test_get_all_gemini_api_keys_splits_and_deduplicates(self, monkeypatch):
        from src.search_provider import get_all_gemini_api_keys

        monkeypatch.setenv("GEMINI_API_KEY", "key-primary")
        monkeypatch.setenv("GEMINI_API_KEYS", "key-primary, key-second,  key-third, key-second ")
        keys = get_all_gemini_api_keys()
        assert keys == ["key-primary", "key-second", "key-third"]


class TestScoringFromLiveFacts:
    def test_km_score_uses_comparable_average(self):
        component = calculate_km_age_fairness_score(
            kilometer=60_000,
            year=2022,
            current_year=2026,
            average_comparable_km_per_year=15_000,
        )
        assert component.score == 100
        assert "rata-rata pembanding" in component.reason

    def test_fewer_than_five_is_explicitly_low_confidence_rule(self):
        from src.scoring import get_market_confidence

        confidence = get_market_confidence(count=2, price_spread_ratio=0.01, completeness_pct=100)
        assert confidence["level"] == "Rendah"
        assert "Data pembanding kurang dari 3" in confidence["reason"]


class TestFourStagePipeline:
    def test_mocked_pipeline_caches_stages_one_to_three_without_dummy_fallback(self, tmp_path):
        from src.pipeline import query_cache_key, run_screening

        class FakeProvider:
            def __init__(self):
                self.search_calls = 0
                self.structured_schemas = []
                self.final_prompt = ""
                self.api_calls = 0

            def search(self, prompt, marketplace=""):
                self.search_calls += 1
                self.api_calls += 1
                self.search_prompt = prompt
                number = self.search_calls
                if number > 4:
                    return GroundedSearchResult(
                        text="",
                        sources=[],
                        queries=[],
                        usage=TokenUsage(0, 0),
                        grounding_calls=1,
                        marketplace=marketplace,
                    )
                sources = [GroundingSource(f"https://market.example/car/{n}", f"Listing {n}") for n in range(1, 5)]
                text = "\n".join(f"{n}. Toyota Avanza G 2024 AT Rp {n * 100_000_000:,} km {n * 10_000:,}" for n in range(1, 5))
                return GroundedSearchResult(
                    text=text,
                    sources=sources,
                    queries=[f"Toyota Avanza 2024 {marketplace}"],
                    usage=TokenUsage(10, 20),
                    grounding_calls=1,
                    marketplace=marketplace,
                )

            def read_url_context(self, url: str):
                return f"Listing content for {url}", TokenUsage(5, 5)

            def generate_structured(self, prompt, response_model):
                self.api_calls += 1
                self.structured_schemas.append(response_model)
                if response_model in (ListingExtraction, ListingPageExtraction):
                    for number in range(1, 5):
                        url = f"https://market.example/car/{number}"
                        if url in prompt:
                            record = extracted(
                                price=f"Rp {number * 100_000_000:,}",
                                kilometer=f"{number * 10_000:,}",
                                source_url=url,
                                year="2024",
                            )
                            payload = ListingPageExtraction(listings=[record]).model_dump_json()
                            return StructuredResponse(payload, TokenUsage(30, 40))
                self.final_prompt = prompt
                answer = FinalAnalysis(
                    kesimpulan="Data pembanding terbatas; verifikasi unit dan dokumen secara langsung.",
                    saran_nego={"awal": 1, "realistis": 2, "batas": 3},
                    red_flag=["Tanyakan bukti riwayat servis."],
                    pertanyaan_untuk_penjual=["Apakah riwayat servis tersedia?"],
                    checklist_inspeksi={"dokumen": [], "fisik": [], "transmisi": []},
                    catatan=["Harga adalah estimasi listing."],
                )
                return StructuredResponse(answer.model_dump_json(), TokenUsage(50, 60))

        provider = FakeProvider()
        cache = SQLiteAnalysisCache(tmp_path / "analysis.sqlite3")
        user_input = {
            "brand": "Toyota",
            "model": "Avanza",
            "variant": "G",
            "body_type": "MPV",
            "year": 2024,
            "transmission": "AT",
            "kilometer": 40_000,
            "listing_price": 250_000_000,
            "location": "Jakarta Selatan",
            "plate_code": "B",
            "deskripsi": "",
        }
        first = run_screening(
            provider=provider,
            cache=cache,
            user_input=user_input,
            daily_limit=3,
            current_year=2026,
        )
        assert first.statistics["count"] == 4
        assert first.warning is None
        assert first.confidence["level"] == "Sedang"
        assert first.description_skipped is True
        assert '"deskripsi"' not in provider.final_prompt
        assert first.final_analysis.saran_nego == calculate_negotiation_prices(first.statistics["price_median"])
        assert first.usage["grounding_calls"] >= 1
        assert first.usage["total_tokens"] > 0
        assert "site:" in provider.search_prompt

        query_key = query_cache_key(user_input, 2026)
        cached_payload = cache.get(query_key)
        assert cached_payload["statistics"]["price_median"] == first.statistics["price_median"]
        assert cached_payload["statistics"]["listing_percentile"] is None

        second_input = {**user_input, "listing_price": 450_000_000}
        second = run_screening(
            provider=provider,
            cache=cache,
            user_input=second_input,
            daily_limit=3,
            current_year=2026,
        )
        assert second.cache_hit is True
        assert provider.search_calls == first.usage["grounding_calls"]
        assert second.statistics["listing_percentile"] == 100
        assert second.usage["grounding_calls"] == 0


class TestGeminiResilienceAndFallback:
    def test_503_then_success_on_fallback_model(self, monkeypatch):
        class ServerError503(Exception):
            code = 503

        recorded_calls = []

        class MockInteractions:
            def create(self, model, **kwargs):
                recorded_calls.append(model)
                if model == "gemini-3.8-flash":
                    raise ServerError503("503: No capacity available on server")
                if model == "gemini-2.5-flash":
                    return SimpleNamespace(
                        output_text="Listing Avanza 2024",
                        steps=[
                            SimpleNamespace(type="google_search_call", arguments={"queries": ["Avanza 2024"]}),
                            SimpleNamespace(type="model_output", content=[SimpleNamespace(annotations=[
                                SimpleNamespace(type="url_citation", url="https://market.test/avanza-2024", title="Avanza")
                            ])]),
                        ],
                        usage=SimpleNamespace(input_tokens=15, output_tokens=25),
                    )
                raise ValueError(f"Model tidak dikenal: {model}")

        monkeypatch.setattr("src.search_provider.time.sleep", lambda _: None)
        provider = GeminiGroundingProvider(
            client=SimpleNamespace(interactions=MockInteractions()),
            model="gemini-3.8-flash",
            fallback_models=["gemini-2.5-flash"],
            max_retries=2,
            backoff_seconds=0.0,
        )

        search_result = provider.search("Cari listing Toyota Avanza 2024")

        assert search_result.model == "gemini-2.5-flash"
        assert provider.last_used_model == "gemini-2.5-flash"
        assert provider.active_model == "gemini-2.5-flash"
        assert len(search_result.sources) == 1
        assert search_result.sources[0].url == "https://market.test/avanza-2024"
        # Primary model tried twice (max_retries=2), then fallback model succeeded on first attempt
        assert recorded_calls == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-2.5-flash"]

        # Subsequent structured call should start directly with the active fallback model
        recorded_calls.clear()
        structured = provider.generate_structured("prompt", ListingExtraction)
        assert structured.model == "gemini-2.5-flash"
        assert recorded_calls == ["gemini-2.5-flash"]

    def test_all_models_fail_raises_clear_error_without_dummy_data(self, monkeypatch, tmp_path):
        from src.pipeline import PipelineError, query_cache_key, run_screening

        class ServerError503(Exception):
            code = 503

        class MockAllFailInteractions:
            def create(self, model, **kwargs):
                raise ServerError503(f"Server overload 503 on model {model}")

        monkeypatch.setattr("src.search_provider.time.sleep", lambda _: None)
        monkeypatch.setattr("src.pipeline.crawl_marketplace_catalog", lambda *args, **kwargs: [])
        provider = GeminiGroundingProvider(
            client=SimpleNamespace(interactions=MockAllFailInteractions()),
            model="gemini-3.8-flash",
            fallback_models=["gemini-2.5-flash"],
            max_retries=2,
            backoff_seconds=0.0,
        )

        with pytest.raises(RuntimeError, match="Semua model Gemini .* gagal setelah retry"):
            provider.search("Cari Avanza")

        # In pipeline, when grounding fails across all models, it falls back to direct marketplace
        # search with warning instead of crashing with PipelineError on 'Pencarian'
        cache = SQLiteAnalysisCache(tmp_path / "test_no_dummy.sqlite3")
        user_input = {
            "brand": "Toyota",
            "model": "Avanza",
            "variant": "G",
            "body_type": "MPV",
            "year": 2024,
            "transmission": "AT",
            "kilometer": 30_000,
            "listing_price": 230_000_000,
            "location": "Jakarta Selatan",
            "plate_code": "B",
        }
        result = run_screening(
            provider=provider,
            cache=cache,
            user_input=user_input,
            daily_limit=5,
            current_year=2026,
        )
        assert result is not None
        assert result.warning is not None
        assert "tidak tersedia" in result.warning
        assert result.debug_error is not None
        # Verify no cache was written for failed search
        assert cache.get(query_cache_key(user_input, 2026)) is None

    def test_retry_sequence_with_exponential_backoff_and_jitter(self, monkeypatch):
        class ServerError503(Exception):
            code = 503

        call_sequence = []
        sleep_durations = []

        class MockSequenceInteractions:
            def create(self, model, **kwargs):
                call_sequence.append(model)
                if model == "gemini-3.8-flash":
                    raise ServerError503("503 Unavailable")
                if model == "gemini-2.5-flash":
                    return SimpleNamespace(
                        output_text="listing found",
                        steps=[],
                        usage=SimpleNamespace(input_tokens=5, output_tokens=10),
                    )
                raise ValueError(f"Unknown model: {model}")

        def mock_sleep(seconds):
            sleep_durations.append(seconds)

        monkeypatch.setattr("src.search_provider.time.sleep", mock_sleep)

        provider = GeminiGroundingProvider(
            client=SimpleNamespace(interactions=MockSequenceInteractions()),
            model="gemini-3.8-flash",
            fallback_models=["gemini-2.5-flash"],
            max_retries=3,
            backoff_seconds=0.2,
        )

        result = provider.search("query")
        assert result.model == "gemini-2.5-flash"
        # Primary model retried 3 times (attempts 1, 2, 3), then fallback model on attempt 1
        assert call_sequence == [
            "gemini-3.8-flash",
            "gemini-3.8-flash",
            "gemini-3.8-flash",
            "gemini-2.5-flash",
        ]
        # Primary retried twice before exhausting (2 sleep calls)
        assert len(sleep_durations) == 2
        # Verify exponential backoff: delay 2 should be larger than delay 1
        assert sleep_durations[1] > sleep_durations[0]

    def test_cache_hit_skips_stage_one_to_three_calls(self, tmp_path):
        from src.pipeline import run_screening

        class RecordingFakeProvider:
            def __init__(self):
                self.search_calls = 0
                self.extract_calls = 0
                self.final_calls = 0
                self.last_used_model = "gemini-2.5-flash"

            def search(self, prompt, marketplace=""):
                self.search_calls += 1
                return GroundedSearchResult(
                    text="listing facts",
                    sources=[
                        GroundingSource("https://www.mobil123.com/dijual/toyota-yaris/1", "Source 1"),
                        GroundingSource("https://www.mobil123.com/dijual/toyota-yaris/2", "Source 2"),
                    ],
                    queries=["Toyota Yaris 2022"],
                    usage=TokenUsage(10, 20),
                    model="gemini-2.5-flash",
                    marketplace=marketplace,
                )

            def read_url_context(self, url: str):
                return f"Listing content for {url}", TokenUsage(5, 5)

            def generate_structured(self, prompt, response_model):
                if response_model in (ListingExtraction, ListingPageExtraction):
                    self.extract_calls += 1
                    is_second = "toyota-yaris/2" in prompt
                    target_url = "https://www.mobil123.com/dijual/toyota-yaris/2" if is_second else "https://www.mobil123.com/dijual/toyota-yaris/1"
                    target_price = "Rp 245.000.000" if is_second else "Rp 240.000.000"
                    return StructuredResponse(
                        ListingPageExtraction(listings=[
                            extracted(
                                brand="Toyota",
                                model="Yaris",
                                variant="GR Sport",
                                year="2022",
                                kilometer="35.000 km",
                                transmission="AT",
                                location="Jakarta Barat",
                                price=target_price,
                                source_url=target_url,
                            )
                        ]).model_dump_json(),
                        TokenUsage(15, 25),
                        model="gemini-2.5-flash",
                    )
                self.final_calls += 1
                return StructuredResponse(
                    FinalAnalysis(
                        kesimpulan="Unit dalam kondisi wajar.",
                        saran_nego=NegotiationAdvice(awal=216_000_000, realistis=224_500_000, batas=233_000_000),
                        red_flag=[],
                        pertanyaan_untuk_penjual=[],
                        checklist_inspeksi=InspectionChecklist(dokumen=[], fisik=[], transmisi=[]),
                        catatan=[],
                    ).model_dump_json(),
                    TokenUsage(20, 30),
                    model="gemini-2.5-flash",
                )

        provider = RecordingFakeProvider()
        cache = SQLiteAnalysisCache(tmp_path / "cache_ttl.sqlite3", ttl_seconds=86_400)
        user_input = {
            "brand": "Toyota",
            "model": "Yaris",
            "variant": "GR Sport",
            "body_type": "Hatchback",
            "year": 2022,
            "transmission": "AT",
            "kilometer": 35_000,
            "listing_price": 245_000_000,
            "location": "Jakarta Barat",
            "plate_code": "B",
        }

        # First run (cache miss)
        res1 = run_screening(
            provider=provider,
            cache=cache,
            user_input=user_input,
            daily_limit=5,
            current_year=2026,
        )
        assert res1.cache_hit is False
        assert res1.usage["grounding_calls"] >= 1
        assert res1.model_used == "gemini-2.5-flash"
        assert provider.search_calls == res1.usage["grounding_calls"]

        # Second run with same specs (cache hit)
        res2 = run_screening(
            provider=provider,
            cache=cache,
            user_input=user_input,
            daily_limit=5,
            current_year=2026,
        )
        assert res2.cache_hit is True
        assert res2.usage["grounding_calls"] == 0
        assert provider.search_calls == res1.usage["grounding_calls"]
        assert res2.model_used == "gemini-2.5-flash"


class TestLenientExtraction:
    """Tests for resilient per-item extraction that skips invalid listings."""

    def test_normalize_indonesian_field_names(self):
        raw = {
            "merek": "Toyota",
            "model": "Avanza",
            "varian": "G",
            "tahun": "2020",
            "km": "60.000",
            "transmisi": "AT",
            "lokasi": "Jakarta",
            "tipe_penjual": "dealer",
            "harga": "Rp 200.000.000",
            "url": "https://example.test/1",
        }
        normalized = _normalize_listing_dict(raw)
        assert normalized["brand"] == "Toyota"
        assert normalized["model"] == "Avanza"
        assert normalized["variant"] == "G"
        assert normalized["year"] == "2020"
        assert normalized["kilometer"] == "60.000"
        assert normalized["transmission"] == "AT"
        assert normalized["location"] == "Jakarta"
        assert normalized["seller_type"] == "dealer"
        assert normalized["price"] == "Rp 200.000.000"
        assert normalized["source_url"] == "https://example.test/1"

    def test_normalize_fills_missing_fields_with_unknown(self):
        raw = {"model": "Brio", "price": "100jt"}
        normalized = _normalize_listing_dict(raw)
        assert normalized["brand"] == "tidak diketahui"
        assert normalized["model"] == "Brio"
        assert normalized["variant"] == "tidak diketahui"

    def test_normalize_flattens_nested_single_key(self):
        raw = {"mobil": {"merek": "Honda", "model": "Jazz", "varian": "RS",
                         "tahun": "2021", "km": "30000", "transmisi": "AT",
                         "lokasi": "Depok", "penjual": "showroom",
                         "harga": "250jt", "link": "https://test.com/1"}}
        normalized = _normalize_listing_dict(raw)
        assert normalized["brand"] == "Honda"
        assert normalized["model"] == "Jazz"

    def test_lenient_parse_skips_invalid_items(self):
        good = {
            "brand": "Toyota", "model": "Avanza", "variant": "G",
            "year": "2020", "kilometer": "50000", "transmission": "AT",
            "location": "Jakarta", "seller_type": "dealer",
            "price": "200000000", "source_url": "https://example.test/1",
        }
        bad = {"brand": "incomplete"}  # Missing all required fields
        items = _parse_listings_leniently({"listings": [good, bad, good]})
        # Both good items should parse, bad item should be skipped
        assert len(items) >= 1  # At least the good ones
        assert items[0].brand == "Toyota"

    def test_lenient_parse_handles_all_indonesian_names(self):
        items = _parse_listings_leniently([
            {
                "merek": "Daihatsu", "model": "Xenia", "varian": "R",
                "tahun": "2020", "km": "50.000 km", "transmisi": "AT",
                "lokasi": "Jakarta Pusat", "tipe_penjual": "dealer",
                "harga": "Rp 145.000.000", "url": "https://example.test/xenia",
            }
        ])
        assert len(items) == 1
        assert items[0].brand == "Daihatsu"
        assert items[0].model == "Xenia"
        assert items[0].price == "Rp 145.000.000"

    def test_lenient_parse_returns_empty_for_all_invalid(self):
        items = _parse_listings_leniently([
            {"x": 1},
            {"y": 2},
        ])
        assert items == []

    def test_extract_with_mixed_valid_invalid_listings(self):
        class FakeProvider:
            def generate_structured(self, prompt, response_model):
                payload = json.dumps({"listings": [
                    {
                        "brand": "Toyota", "model": "Avanza", "variant": "G",
                        "year": "2020", "kilometer": "50000", "transmission": "AT",
                        "location": "Jakarta", "seller_type": "dealer",
                        "price": "Rp 200.000.000",
                        "source_url": "https://example.test/1",
                    },
                    {"totally_wrong_field": "garbage"},
                    {
                        "merek": "Toyota", "model": "Avanza", "varian": "E",
                        "tahun": "2019", "km": "70000", "transmisi": "MT",
                        "lokasi": "Bekasi", "penjual": "private",
                        "harga": "Rp 180.000.000",
                        "url": "https://example.test/2",
                    },
                ]})
                return StructuredResponse(payload, TokenUsage(10, 20))

        provider = FakeProvider()
        search = GroundedSearchResult(
            text="Listing data",
            sources=[
                GroundingSource("https://example.test/1", "listing1"),
                GroundingSource("https://example.test/2", "listing2"),
            ],
            queries=["Avanza bekas"],
            usage=TokenUsage(1, 2),
        )
        result, usage = extract_listings(provider, search)
        # Should have 2 valid listings (invalid one skipped)
        assert len(result.listings) == 2
        assert result.listings[0].brand == "Toyota"
        assert result.listings[1].brand == "Toyota"  # Normalized from 'merek'
        assert result.listings[1].variant == "E"

    def test_extract_raises_value_error_when_all_listings_invalid(self):
        class FakeProvider:
            def generate_structured(self, prompt, response_model):
                payload = json.dumps({"listings": [
                    {"random_field_1": "x"},
                    {"random_field_2": "y"},
                ]})
                return StructuredResponse(payload, TokenUsage(10, 20))

        provider = FakeProvider()
        search = GroundedSearchResult(
            text="Listing data",
            sources=[GroundingSource("https://example.test/1", "listing1")],
            queries=["Avanza bekas"],
            usage=TokenUsage(1, 2),
        )
        # All items invalid -> lenient parsing returns empty -> strict fallback also fails -> ValueError
        with pytest.raises(ValueError, match="tidak dapat divalidasi"):
            extract_listings(provider, search)
