"""Tests for environment diagnostics panel and API key privacy."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.cache import SQLiteAnalysisCache
from src.diagnostics import (
    check_domain_connectivity,
    check_marketplace_connectivity,
    format_diagnostics_as_text,
    get_daily_limit_info,
    get_full_diagnostics_summary,
    get_git_info,
    get_model_and_config_info,
    get_python_and_package_versions,
    get_server_time_info,
    get_sqlite_info,
    render_environment_diagnostics,
)


@pytest.fixture
def temp_cache(tmp_path):
    db_file = tmp_path / "test_diag_cache.sqlite3"
    cache = SQLiteAnalysisCache(db_file)
    return cache


def test_git_info_returns_non_empty_values():
    info = get_git_info()
    assert "commit_hash" in info
    assert "build_date" in info
    assert isinstance(info["commit_hash"], str)
    assert len(info["commit_hash"]) > 0


def test_python_and_packages_info():
    info = get_python_and_package_versions()
    assert "python_version" in info
    assert "packages" in info
    pkgs = info["packages"]
    for required in ["streamlit", "google-genai", "pydantic", "requests", "httpx", "beautifulsoup4", "lxml"]:
        assert required in pkgs
        assert isinstance(pkgs[required], str)
    # Streamlit and pydantic must have concrete versions in test env
    assert pkgs["streamlit"] != "(tidak terinstall)"
    assert pkgs["pydantic"] != "(tidak terinstall)"


def test_server_time_utc_and_wib():
    time_info = get_server_time_info()
    assert "utc" in time_info
    assert "wib" in time_info
    assert "UTC" in time_info["utc"]
    assert "WIB" in time_info["wib"]


def test_sqlite_info_and_persistence(tmp_path):
    db_path = tmp_path / "persistence_test.sqlite3"
    # First creation: should be newly created
    cache1 = SQLiteAnalysisCache(db_path)
    info1 = get_sqlite_info(cache1)
    assert "Baru dibuat" in info1["persistence_status"]
    assert info1["exists"] is True
    assert "riwayat (screening_history)" in info1["counts"]
    assert "grounding_cache" in info1["counts"]
    assert "listing pasar (market_listings)" in info1["counts"]

    # Re-opening existing DB: should be persistent
    cache2 = SQLiteAnalysisCache(db_path)
    info2 = get_sqlite_info(cache2)
    assert "Persisten" in info2["persistence_status"]


def test_daily_limit_and_counter(temp_cache, monkeypatch):
    monkeypatch.setenv("DAILY_ANALYSIS_LIMIT", "15")
    limit_info = get_daily_limit_info(temp_cache)
    assert limit_info["limit"] == 15
    assert limit_info["current_count"] == 0
    assert limit_info["remaining"] == 15

    # Claim 2 analyses
    temp_cache.claim_daily_analysis(15)
    temp_cache.claim_daily_analysis(15)

    limit_info2 = get_daily_limit_info(temp_cache)
    assert limit_info2["current_count"] == 2
    assert limit_info2["remaining"] == 13


def test_connectivity_check_mocked():
    with patch("requests.head") as mock_head:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_head.return_value = mock_resp

        res = check_domain_connectivity("carmudi.co.id", timeout=2.0)
        assert res["domain"] == "carmudi.co.id"
        assert res["status_code"] == "200"
        assert res["is_ok"] is True
        assert res["is_blocked"] is False
        assert res["response_time_ms"] >= 0.0

    with patch("requests.head") as mock_head, patch("requests.get") as mock_get:
        mock_resp_403 = MagicMock()
        mock_resp_403.status_code = 403
        mock_head.return_value = mock_resp_403
        mock_get.return_value = mock_resp_403

        res_blocked = check_domain_connectivity("mobil123.com", timeout=2.0)
        assert res_blocked["status_code"] == "403"
        assert res_blocked["is_blocked"] is True
        assert res_blocked["is_ok"] is False


def test_connectivity_parallel_mocked():
    with patch("requests.head") as mock_head:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_head.return_value = mock_resp

        results = check_marketplace_connectivity(["carmudi.co.id", "carsome.id"], timeout=1.0)
        assert len(results) == 2
        assert all(r["is_ok"] for r in results)


def test_diagnostics_never_leaks_api_keys(temp_cache, monkeypatch):
    """Critical test: Panel text and data structures NEVER contain secret keys or partial prefixes."""
    secret_key_1 = "dummy_mock_secret_key_testing_alpha_123"
    secret_key_2 = "dummy_mock_secret_key_testing_beta_456"
    secret_key_3 = "dummy_mock_secret_key_testing_gamma_789"

    monkeypatch.setenv("GEMINI_API_KEY", secret_key_1)
    monkeypatch.setenv("GEMINI_API_KEYS", f"{secret_key_2},{secret_key_3}")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite")

    model_info = get_model_and_config_info()
    # It must report count of keys (3), but NO key strings
    assert model_info["api_key_count"] == 3

    # Generate full summary and text representation
    summary = get_full_diagnostics_summary(temp_cache)
    summary_text = format_diagnostics_as_text(summary)

    # Check that neither key1, key2, key3 nor their prefixes/suffixes appear anywhere
    all_keys = [secret_key_1, secret_key_2, secret_key_3]
    for key in all_keys:
        assert key not in str(summary), f"Leak detected for key in summary dict: {key[:8]}..."
        assert key not in summary_text, f"Leak detected for key in text: {key[:8]}..."
        # Even partial fragments (first 8 chars or last 8 chars) must not appear
        prefix = key[:8]
        suffix = key[-8:]
        assert prefix not in summary_text, f"Leak detected for prefix {prefix} in text"
        assert suffix not in summary_text, f"Leak detected for suffix {suffix} in text"


def test_render_diagnostics_ui_calls_never_pass_secrets(temp_cache, monkeypatch):
    """Verify that Streamlit render functions never receive API keys in arguments."""
    secret_key = "dummy_mock_secret_key_testing_ui_secure_999"
    monkeypatch.setenv("GEMINI_API_KEY", secret_key)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")

    recorded_calls = []

    mock_st = MagicMock()
    # Intercept all calls to st methods
    def record_call(name):
        def _wrapper(*args, **kwargs):
            text_repr = f"{name}: {args} {kwargs}"
            recorded_calls.append(text_repr)
            if name == "columns":
                return [MagicMock(), MagicMock()]
            if name == "expander":
                return MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            return MagicMock()
        return _wrapper

    for method in ["write", "markdown", "caption", "dataframe", "columns", "expander", "divider", "button", "info", "code"]:
        setattr(mock_st, method, MagicMock(side_effect=record_call(method)))

    mock_st.session_state = {}

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        render_environment_diagnostics(temp_cache)

    for call_str in recorded_calls:
        assert secret_key not in call_str, f"Leak in Streamlit UI call: {call_str}"
        assert secret_key[:8] not in call_str
