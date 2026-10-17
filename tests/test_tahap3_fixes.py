import pytest
from pydantic import ValidationError

from src.analyze import FinalAnalysis, analyze_results
from src.marketplace_config import (
    get_model_slug_variants,
    slugify_model,
)
from src.pipeline import run_screening, _fallback_analysis
from src.search_provider import SearchProvider, StructuredResponse, TokenUsage
from src.cache import SQLiteAnalysisCache


def test_slugify_model():
    # Domain specific mappings
    assert slugify_model("carmudi.co.id", "Nissan", "Grand Livina") == "grand-livina"
    assert slugify_model("carsome.id", "Nissan", "Grand Livina") == "livina"
    assert slugify_model("carmudi.co.id", "Toyota", "All New Avanza") == "avanza"
    assert slugify_model("carsome.id", "Honda", "Brio Satya") == "brio"
    assert slugify_model("oto.com", "Toyota", "Agya") == "agya"
    assert slugify_model("carmudi.co.id", "Toyota", "Calya") == "calya"

    # Marketing prefixes stripped
    assert slugify_model("carmudi.co.id", "Toyota", "All New Rush") == "rush"
    assert slugify_model("carsome.id", "Daihatsu", "The New Terios") == "terios"

    # Variants list
    carmudi_variants = get_model_slug_variants("carmudi.co.id", "Nissan", "Grand Livina")
    assert "grand-livina" in carmudi_variants
    assert "livina" in carmudi_variants
    assert "grandlivina" in carmudi_variants

    carsome_variants = get_model_slug_variants("carsome.id", "Nissan", "Grand Livina")
    assert carsome_variants[0] == "livina"
    assert "grand-livina" in carsome_variants


def test_final_analysis_validator_unwrap_and_catatan():
    # 1. Output wrapped in 'hasil_analisis' with dict 'catatan'
    payload_wrapped = {
        "hasil_analisis": {
            "kesimpulan": "Kondisi unit layak dipertimbangkan. Lakukan inspeksi fisik sebelum pembayaran.",
            "saran_nego": {"awal": 100000000, "realistis": 105000000, "batas": 110000000},
            "red_flag": ["Pajak lewat 1 bulan"],
            "pertanyaan_untuk_penjual": ["Apakah BPKB asli ready?"],
            "checklist_inspeksi": {
                "dokumen": ["Cek STNK", "Cek BPKB"],
                "fisik": ["Cek oli"],
                "transmisi": ["Cek perpindahan D ke R"],
            },
            "catatan": {"pajak": "Pajak hidup", "estimasi": "Biaya mutasi Rp 2.500.000"},
        }
    }
    model = FinalAnalysis.model_validate(payload_wrapped)
    assert model.kesimpulan.startswith("Kondisi unit layak")
    assert isinstance(model.catatan, list)
    assert len(model.catatan) == 2
    assert any("pajak: Pajak hidup" in item for item in model.catatan)
    assert any("Biaya mutasi" in item for item in model.catatan)

    # 2. Catatan as string
    payload_str_catatan = {
        "kesimpulan": "Harga berada di bawah median pasar Jabodetabek.",
        "catatan": "Periksa kondisi kaki-kaki dan riwayat servis berkala."
    }
    model2 = FinalAnalysis.model_validate(payload_str_catatan)
    assert model2.catatan == ["Periksa kondisi kaki-kaki dan riwayat servis berkala."]

    # 3. Catatan as None
    payload_none_catatan = {
        "kesimpulan": "Harga sesuai estimasi pasar.",
        "catatan": None
    }
    model3 = FinalAnalysis.model_validate(payload_none_catatan)
    assert model3.catatan == []

    # 4. Wrapped in 'analisis'
    payload_analisis = {
        "analisis": {
            "kesimpulan": "Unit bagus.",
            "catatan": ["Catatan 1", "Catatan 2"]
        }
    }
    model4 = FinalAnalysis.model_validate(payload_analisis)
    assert model4.kesimpulan == "Unit bagus."
    assert model4.catatan == ["Catatan 1", "Catatan 2"]


class MockSmartRetryProvider(SearchProvider):
    def __init__(self, succeed_on_attempt: int = 2):
        self.model = "gemini-2.5-flash"
        self.call_count = 0
        self.prompts_received = []
        self.succeed_on_attempt = succeed_on_attempt

    def search(self, prompt: str, *, marketplace: str = ""):
        raise NotImplementedError

    def generate_structured(self, prompt: str, response_model):
        self.call_count += 1
        self.prompts_received.append(prompt)
        if self.call_count < self.succeed_on_attempt:
            # Return invalid JSON missing 'kesimpulan'
            bad_json = '{"saran_nego": {"awal": 100000000}, "catatan": "cek mesin"}'
            return StructuredResponse(bad_json, TokenUsage(100, 50), model=self.model)
        else:
            # Return valid JSON
            good_json = '{"kesimpulan": "Unit terawat dan layak beli.", "catatan": ["Cek dokumen"]}'
            return StructuredResponse(good_json, TokenUsage(120, 60), model=self.model)


def test_analyze_results_smart_retry():
    provider = MockSmartRetryProvider(succeed_on_attempt=2)
    user_input = {"brand": "Nissan", "model": "Grand Livina", "year": 2020}
    statistics = {"price_median": 120000000, "count": 5}
    listings = []

    res, usage = analyze_results(provider, user_input, statistics, listings, research_text="Hasil search riset")
    assert provider.call_count == 2
    assert "gagal divalidasi dengan error berikut" in provider.prompts_received[1]
    assert res.kesimpulan == "Unit terawat dan layak beli."
    assert usage.input_tokens == 220
    assert usage.output_tokens == 110


def test_analyze_results_max_retries_failure():
    provider = MockSmartRetryProvider(succeed_on_attempt=99)
    user_input = {"brand": "Nissan", "model": "Grand Livina", "year": 2020}
    statistics = {"price_median": 120000000, "count": 5}
    listings = []

    with pytest.raises(ValidationError):
        analyze_results(provider, user_input, statistics, listings)
    # Attempt 1 + 2 retries = 3 calls total
    assert provider.call_count == 3


def test_pipeline_fallback_when_gemini_fails(tmp_path, monkeypatch):
    from src.extract import ExtractedListing

    class AlwaysFailingProvider(SearchProvider):
        def __init__(self):
            self.model = "gemini-2.5-flash"
        def search(self, prompt: str, *, marketplace: str = ""):
            raise RuntimeError("Search error")
        def generate_structured(self, prompt: str, response_model):
            # Invalid output causing validation failure
            return StructuredResponse('{"invalid": "data"}', TokenUsage(10, 10), model=self.model)

    # Mock catalog crawl and detail fetch so test runs offline instantly
    monkeypatch.setattr(
        "src.pipeline.crawl_marketplace_catalog",
        lambda *args, **kwargs: ["https://example.com/detail/1", "https://example.com/detail/2", "https://example.com/detail/3"]
    )
    mock_listings = [
        ExtractedListing(
            brand="Toyota",
            model="Avanza",
            variant="G",
            year="2020",
            transmission="AT",
            kilometer="50000",
            price="Rp 175.000.000",
            location="Jakarta Selatan",
            seller_type="dealer",
            source_url="https://example.com/detail/1",
        ),
        ExtractedListing(
            brand="Toyota",
            model="Avanza",
            variant="G",
            year="2020",
            transmission="AT",
            kilometer="55000",
            price="Rp 170.000.000",
            location="Jakarta Barat",
            seller_type="dealer",
            source_url="https://example.com/detail/2",
        ),
        ExtractedListing(
            brand="Toyota",
            model="Avanza",
            variant="G",
            year="2020",
            transmission="AT",
            kilometer="60000",
            price="Rp 180.000.000",
            location="Jakarta Timur",
            seller_type="dealer",
            source_url="https://example.com/detail/3",
        ),
    ]
    monkeypatch.setattr(
        "src.pipeline.fetch_and_parse_details_concurrent",
        lambda *args, **kwargs: (mock_listings, [])
    )

    cache = SQLiteAnalysisCache(str(tmp_path / "test.db"))
    user_input = {
        "brand": "Toyota",
        "model": "Avanza",
        "variant": "G",
        "year": 2020,
        "transmission": "AT",
        "body_type": "MPV",
        "kilometer": 60000,
        "location": "Jakarta Selatan",
        "plate_code": "B",
        "listing_price": 170000000,
        "search_mode": "direct",
    }

    result = run_screening(
        provider=AlwaysFailingProvider(),
        cache=cache,
        user_input=user_input,
        daily_limit=50,
    )

    # When Gemini fails, it should cleanly fallback to code rules
    assert result.final_analysis is not None
    assert result.final_analysis.kesimpulan != ""
    assert result.ai_error == "Analisa AI tidak tersedia, memakai ringkasan aturan"
    assert result.debug_error is not None
    assert "ValidationError" in result.debug_error


def test_classify_gemini_error():
    from src.search_provider import classify_gemini_error

    class CustomCodeError(Exception):
        def __init__(self, msg, code):
            super().__init__(msg)
            self.code = code

    # 1. Quota 429
    err_429 = CustomCodeError("Resource has been exhausted (e.g. check quota).", 429)
    info_429 = classify_gemini_error(err_429)
    assert info_429.http_code == 429
    assert info_429.category == "quota"
    assert info_429.is_retryable is True

    # 2. Auth 403
    err_403 = CustomCodeError("Permission denied on resource.", 403)
    info_403 = classify_gemini_error(err_403)
    assert info_403.http_code == 403
    assert info_403.category == "auth"
    assert info_403.is_retryable is False

    # 3. Auth 401 / API_KEY_INVALID
    err_401 = Exception("API key not valid. Please pass a valid API key (401).")
    info_401 = classify_gemini_error(err_401)
    assert info_401.http_code == 401
    assert info_401.category == "auth"
    assert info_401.is_retryable is False

    # 4. Timeout
    import concurrent.futures
    err_timeout = concurrent.futures.TimeoutError()
    info_timeout = classify_gemini_error(err_timeout)
    assert info_timeout.category == "timeout"
    assert info_timeout.is_retryable is True


def test_mask_api_keys_in_logs_and_errors():
    from src.search_provider import mask_api_keys, classify_gemini_error

    dummy_key = "AIzaSyB1234567890abcdefghijklmnopqrstuv"
    raw_text = f"Gagal mengakses https://generativelanguage.googleapis.com?key={dummy_key} dengan pesan error"
    masked = mask_api_keys(raw_text, [dummy_key])
    assert dummy_key not in masked
    assert "[REDACTED_API_KEY]" in masked

    err = Exception(f"Invalid call for key {dummy_key}")
    info = classify_gemini_error(err, [dummy_key])
    assert dummy_key not in info.message
    assert "[REDACTED_API_KEY]" in info.message


def test_gemini_provider_empirical_shared_project_stops_key_rotation():
    from src.search_provider import GeminiGroundingProvider

    call_log = []

    class MockClient:
        def __init__(self, key_id):
            self.key_id = key_id
            self.interactions = self

        def create(self, model, **kwargs):
            call_log.append((model, self.key_id))
            # Always raise 429
            raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded for project")

    provider = GeminiGroundingProvider(
        client=MockClient("c0"),
        model="gemini-2.5-flash",
        fallback_models=["gemini-2.5-pro"],
        max_retries=1,
    )
    # Inject 3 distinct mock clients
    provider.clients = [MockClient("c0"), MockClient("c1"), MockClient("c2")]
    provider.api_keys = ["key0", "key1", "key2"]

    with pytest.raises(RuntimeError) as exc_info:
        provider.search("cari mobil")

    assert "Semua model Gemini" in str(exc_info.value)
    # Model 1 (flash): c0 terkena 429, c1 terkena 429 -> terdeteksi shared project! Stop rotasi, jangan panggil c2.
    flash_calls = [c for c in call_log if c[0] == "gemini-2.5-flash"]
    assert len(flash_calls) == 2
    assert flash_calls == [("gemini-2.5-flash", "c0"), ("gemini-2.5-flash", "c1")]
    # Pindah ke model 2 (pro)
    pro_calls = [c for c in call_log if c[0] == "gemini-2.5-pro"]
    assert len(pro_calls) >= 1


def test_gemini_provider_grounding_disabled_vs_key_invalid():
    from src.search_provider import GeminiGroundingProvider

    class MockClient:
        def __init__(self, key_id):
            self.key_id = key_id
            self.interactions = self

        def create(self, model, **kwargs):
            if "tools" in kwargs:
                # 403 Grounding permission denied
                raise RuntimeError("403 PERMISSION_DENIED: Google Search grounding not allowed")
            else:
                # Structured call succeeds
                return type("Resp", (), {"output_text": '{"result": "ok"}', "usage": None})()

    provider = GeminiGroundingProvider(
        client=MockClient("c0"),
        model="gemini-2.5-flash",
        fallback_models=[],
        max_retries=1,
    )

    with pytest.raises(RuntimeError):
        provider.search("cari mobil")

    # 403 saat grounding menonaktifkan grounding untuk client 0, BUKAN key invalid
    assert provider._is_grounding_disabled(0) is True
    assert provider._is_key_invalid(0) is False

    # Client berikutnya / panggilan structured tanpa tools masih bisa menggunakan client yang sama
    next_client_grounding = provider._next_available_client(is_grounding=True)
    assert next_client_grounding is None  # Grounding disabled

    next_client_structured = provider._next_available_client(is_grounding=False)
    assert next_client_structured is not None  # Structured call tetap bisa pakai key ini!


def test_grounding_cache_sqlite_and_skip_api_call(tmp_path):
    from src.cache import SQLiteAnalysisCache, compute_grounding_cache_key
    from src.search_provider import GroundedSearchResult, GroundingSource, TokenUsage

    db_path = tmp_path / "cache_test.db"
    cache = SQLiteAnalysisCache(db_path, ttl_seconds=86400)

    model = "gemini-2.5-flash"
    query = "toyota avanza 2020 bekas jakarta"
    key = compute_grounding_cache_key(model, query)

    # 1. Deterministic hashing & case normalization
    key2 = compute_grounding_cache_key("gemini-2.5-flash", "   TOYOTA   avanza 2020  BEKAS jakarta ")
    assert key == key2

    # 2. Cache miss initially
    assert cache.get_grounding(key) is None

    # 3. Do not cache empty result
    empty_result = GroundedSearchResult(text="", sources=[], queries=[], usage=TokenUsage())
    cache.set_grounded_search_result(key, empty_result)
    assert cache.get_grounding(key) is None

    # 4. Cache valid result
    valid_result = GroundedSearchResult(
        text="Ditemukan beberapa unit",
        sources=[GroundingSource(url="https://carmudi.co.id/detail/1", title="Carmudi")],
        queries=[query],
        usage=TokenUsage(100, 50),
        model=model,
    )
    cache.set_grounded_search_result(key, valid_result)

    # 5. Cache hit
    cached = cache.get_grounding(key)
    assert cached is not None
    assert cached["text"] == "Ditemukan beberapa unit"
    assert len(cached["sources"]) == 1
    assert cached["sources"][0]["url"] == "https://carmudi.co.id/detail/1"

    # 6. TTL expiry
    now_future = 9999999999.0
    assert cache.get_grounding(key, now=now_future) is None


def test_pipeline_fallback_to_marketplace_when_grounding_fails(tmp_path, monkeypatch):
    from src.extract import ExtractedListing

    class GroundingFailingProvider(SearchProvider):
        def __init__(self):
            self.model = "gemini-2.5-flash"
            self.api_calls = 0

        def search(self, prompt: str, *, marketplace: str = ""):
            self.api_calls += 1
            # Grounding completely fails with 429
            raise RuntimeError("429 RESOURCE_EXHAUSTED: Google Search grounding quota exceeded")

        def generate_structured(self, prompt: str, response_model):
            # Final analysis succeeds
            return StructuredResponse(
                '{"kesimpulan": "Unit terawat dan layak beli.", "catatan": ["Cek dokumen"]}',
                TokenUsage(50, 50),
                model=self.model,
            )

    # Mock catalog crawl and details so it returns 4 valid listings
    monkeypatch.setattr(
        "src.pipeline.crawl_marketplace_catalog",
        lambda *args, **kwargs: ["https://example.com/detail/1", "https://example.com/detail/2", "https://example.com/detail/3", "https://example.com/detail/4"]
    )
    mock_listings = [
        ExtractedListing(
            brand="Toyota",
            model="Avanza",
            variant="G",
            year="2020",
            transmission="AT",
            kilometer=f"{45000 + i*5000}",
            price=f"Rp {170000000 + i*5000000}",
            location="Jakarta",
            seller_type="dealer",
            source_url=f"https://example.com/detail/{i+1}",
        )
        for i in range(4)
    ]
    monkeypatch.setattr(
        "src.pipeline.fetch_and_parse_details_concurrent",
        lambda *args, **kwargs: (mock_listings, [])
    )

    cache = SQLiteAnalysisCache(str(tmp_path / "test_fallback.db"))
    user_input = {
        "brand": "Toyota",
        "model": "Avanza",
        "variant": "G",
        "year": 2020,
        "transmission": "AT",
        "body_type": "MPV",
        "kilometer": 50000,
        "location": "Jakarta",
        "plate_code": "B",
        "listing_price": 175000000,
        "search_mode": "grounding",
    }

    # Should NOT raise PipelineError("Pencarian")!
    result = run_screening(
        provider=GroundingFailingProvider(),
        cache=cache,
        user_input=user_input,
        daily_limit=50,
    )

    assert result is not None
    # Pipeline continues using direct marketplace results
    assert len(result.comparables) >= 3
    assert result.warning is not None
    assert "tidak tersedia" in result.warning
    assert result.debug_error is not None
    assert "429" in result.debug_error
    assert result.final_analysis.kesimpulan == "Unit terawat dan layak beli."


def test_gemini_provider_global_deadline_exceeded(monkeypatch):
    import time
    from src.search_provider import GeminiGroundingProvider, GroundingDeadlineExceeded

    class SlowClient:
        def __init__(self):
            self.interactions = self

        def create(self, model, **kwargs):
            return type("Resp", (), {"output_text": "ok", "usage": None})()

    provider = GeminiGroundingProvider(
        client=SlowClient(),
        model="gemini-2.5-flash",
        fallback_models=[],
    )

    # Simulate deadline being exceeded immediately
    # First call sets deadline = t0 + 120. Second call check is simulated as t0 + 125
    t0 = 1000.0
    times = [t0, t0 + 125.0]

    def mock_time():
        if times:
            return times.pop(0)
        return t0 + 130.0

    monkeypatch.setattr(time, "time", mock_time)

    with pytest.raises(GroundingDeadlineExceeded) as exc_info:
        provider.search("cari mobil")

    assert "Deadline global" in str(exc_info.value)


def test_all_keys_exhausted_resets_if_older_than_60s_and_formats_detailed_error():
    import time
    from src.search_provider import GeminiGroundingProvider

    calls_made = []

    class MockClient:
        def __init__(self, key_id):
            self.key_id = key_id
            self.interactions = self

        def create(self, model, **kwargs):
            calls_made.append(self.key_id)
            raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded")

    provider = GeminiGroundingProvider(
        client=MockClient("c0"),
        model="gemini-2.5-flash",
        fallback_models=[],
        max_retries=1,
    )
    provider.clients = [MockClient("c0"), MockClient("c1")]
    provider.api_keys = ["AIzaSyKeyZero12345678901234567890123", "AIzaSyKeyOne123456789012345678901234"]

    now = time.time()
    # Tandai kedua key sebagai exhausted 75 detik yang lalu (> 60 detik)
    old_time = now - 75.0
    provider._exhausted_clients_meta[0] = {
        "timestamp": old_time,
        "expiry": old_time + 300.0,
        "reason": "429 quota",
        "http_code": 429,
        "error_message": "Quota 429 habis",
    }
    provider._exhausted_clients_meta[1] = {
        "timestamp": old_time,
        "expiry": old_time + 300.0,
        "reason": "429 quota",
        "http_code": 429,
        "error_message": "Quota 429 habis",
    }

    with pytest.raises(RuntimeError) as exc_info:
        provider.search("cari mobil")

    err_str = str(exc_info.value)
    # 1. Pastikan TIDAK PERNAH memuat "Error terakhir: None"
    assert "Error terakhir: None" not in err_str
    assert "None" not in err_str or "Rincian per percobaan" in err_str

    # 2. Pastikan ada reset dan retry nyata karena flag > 60s
    assert len(calls_made) > 0

    # 3. Pastikan pesan error akhir memuat rincian terstruktur
    assert "Semua model Gemini" in err_str
    assert "Rincian per percobaan:" in err_str
    assert "gemini-2.5-flash" in err_str
    assert "key #1" in err_str
    assert "HTTP: 429" in err_str

    # 4. Pastikan tidak ada raw API key yang bocor
    assert "AIzaSyKeyZero" not in err_str
    assert "AIzaSyKeyOne" not in err_str

    # 5. Pastikan attempt_history mencatat detail percobaan
    assert len(provider.attempt_history) > 0
    assert any(att.get("http_code") == 429 for att in provider.attempt_history)


def test_keys_exhausted_recent_records_skipped_reason_with_timestamp_and_code():
    import time
    from src.search_provider import GeminiGroundingProvider

    class NoCallClient:
        def __init__(self):
            self.interactions = self

        def create(self, model, **kwargs):
            raise AssertionError("Client seharusnya tidak dipanggil karena baru saja exhausted!")

    provider = GeminiGroundingProvider(
        client=NoCallClient(),
        model="gemini-2.5-flash",
        fallback_models=[],
    )
    provider.clients = [NoCallClient()]
    provider.api_keys = ["AIzaSyKeyRecent1234567890123456789012"]

    now = time.time()
    # Tandai exhausted baru 10 detik yang lalu (< 60 detik)
    provider._exhausted_clients_meta[0] = {
        "timestamp": now - 10.0,
        "expiry": now + 290.0,
        "reason": "429 quota",
        "http_code": 429,
        "error_message": "Rate limit 429",
    }

    with pytest.raises(RuntimeError) as exc_info:
        provider.search("cari mobil")

    err_str = str(exc_info.value)
    # Error tidak boleh "None"
    assert "Error terakhir: None" not in err_str
    assert "dilewati: ditandai habis pada" in err_str
    assert "kode 429" in err_str
    assert "Rincian per percobaan:" in err_str

    # Cek attempt history mencatat format yang diminta
    assert len(provider.attempt_history) >= 1
    skipped_entry = provider.attempt_history[0]
    assert skipped_entry["category"] == "skipped"
    assert "dilewati: ditandai habis pada" in skipped_entry["error"]
    assert "kode 429" in skipped_entry["error"]



