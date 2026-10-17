"""Modul Diagnostik Lingkungan (Environment Diagnostics).

Menampilkan informasi sistem, dependensi, waktu, model, status database,
konektivitas marketplace, dan batas harian TANPA nilai rahasia apa pun.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import requests

from src.cache import SQLiteAnalysisCache
from src.config import CACHE_DB_PATH, MARKETPLACE_DOMAINS, PROJECT_ROOT
from src.page_reader import USER_AGENT
from src.search_provider import get_all_gemini_api_keys

logger = logging.getLogger(__name__)


def get_git_info() -> dict[str, str]:
    """Baca hash commit dan tanggal build dari git atau fallback env/file versi."""
    commit_hash = "Tidak diketahui"
    build_date = "Tidak diketahui"

    # 1. Coba perintah git
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        if res.returncode == 0 and res.stdout.strip():
            commit_hash = res.stdout.strip()

        res_date = subprocess.run(
            ["git", "show", "-s", "--format=%cd", "--date=iso", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        if res_date.returncode == 0 and res_date.stdout.strip():
            build_date = res_date.stdout.strip()
    except Exception:
        pass

    # 2. Fallback ke Environment Variables (sering diset saat build/deploy di cloud)
    if commit_hash == "Tidak diketahui":
        for var in ("GIT_COMMIT", "COMMIT_HASH", "GIT_SHA", "RENDER_GIT_COMMIT", "VERCEL_GIT_COMMIT_SHA"):
            val = os.environ.get(var)
            if val and val.strip():
                commit_hash = val.strip()[:7]
                break

    if build_date == "Tidak diketahui":
        for var in ("BUILD_DATE", "DEPLOY_DATE", "BUILD_TIME", "RELEASE_DATE"):
            val = os.environ.get(var)
            if val and val.strip():
                build_date = val.strip()
                break

    # 3. Fallback ke file versi (misal version.json)
    version_file = PROJECT_ROOT / "version.json"
    if (commit_hash == "Tidak diketahui" or build_date == "Tidak diketahui") and version_file.exists():
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if commit_hash == "Tidak diketahui":
                    commit_hash = str(data.get("commit", data.get("git_hash", commit_hash)))
                if build_date == "Tidak diketahui":
                    build_date = str(data.get("build_date", data.get("date", build_date)))
        except Exception:
            pass

    return {
        "commit_hash": commit_hash,
        "build_date": build_date,
    }


def get_python_and_package_versions() -> dict[str, Any]:
    """Baca versi Python runtime dan paket-paket kunci."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    packages_to_check = [
        "streamlit",
        "google-genai",
        "pydantic",
        "requests",
        "httpx",
        "beautifulsoup4",
        "lxml",
    ]

    pkg_versions: dict[str, str] = {}
    for pkg in packages_to_check:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            # Coba varian nama alternatif (misal google_genai atau bs4)
            alt_name = pkg.replace("-", "_")
            try:
                pkg_versions[pkg] = importlib.metadata.version(alt_name)
            except importlib.metadata.PackageNotFoundError:
                if pkg == "beautifulsoup4":
                    try:
                        pkg_versions[pkg] = importlib.metadata.version("bs4")
                    except Exception:
                        pkg_versions[pkg] = "(tidak terinstall)"
                else:
                    pkg_versions[pkg] = "(tidak terinstall)"

    return {
        "python_version": py_ver,
        "packages": pkg_versions,
    }


def get_server_time_info() -> dict[str, str]:
    """Zona waktu dan waktu server saat ini dalam UTC dan WIB."""
    now_utc = datetime.now(timezone.utc)
    wib_tz = timezone(timedelta(hours=7))
    now_wib = now_utc.astimezone(wib_tz)

    server_local = datetime.now().astimezone()
    local_tz = server_local.tzname() or str(server_local.tzinfo or "Local")

    return {
        "utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "wib": now_wib.strftime("%Y-%m-%d %H:%M:%S WIB"),
        "local_timezone": local_tz,
        "local_time": server_local.strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_model_and_config_info() -> dict[str, Any]:
    """Baca nama model utama dan fallback, sumber config, serta jumlah key.
    
    PERINGATAN KEAMANAN: Fungsi ini HANYA mengembalikan jumlah key (angka bulat),
    sama sekali tidak mengembalikan nilai key atau prefix key.
    """
    # Deteksi sumber GEMINI_MODEL
    model_source = "default"
    primary_model = "Belum diatur"
    if os.environ.get("GEMINI_MODEL"):
        primary_model = os.environ.get("GEMINI_MODEL", "").strip()
        model_source = "env"
    else:
        try:
            import streamlit as st
            secret_model = st.secrets.get("GEMINI_MODEL")
            if secret_model:
                primary_model = str(secret_model).strip()
                model_source = "st.secrets"
        except Exception:
            pass

    # Deteksi sumber GEMINI_FALLBACK_MODELS
    fallback_source = "tidak ada"
    fallback_models: list[str] = []
    raw_fallback = None
    if os.environ.get("GEMINI_FALLBACK_MODELS"):
        raw_fallback = os.environ.get("GEMINI_FALLBACK_MODELS", "").strip()
        fallback_source = "env"
    else:
        try:
            import streamlit as st
            secret_fb = st.secrets.get("GEMINI_FALLBACK_MODELS")
            if secret_fb:
                raw_fallback = str(secret_fb).strip()
                fallback_source = "st.secrets"
        except Exception:
            pass

    if raw_fallback:
        fallback_models = [item.strip() for item in raw_fallback.split(",") if item.strip()]

    # Deteksi jumlah dan sumber API key
    key_sources: set[str] = set()
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS"):
        key_sources.add("env")
    try:
        import streamlit as st
        if st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEYS"):
            key_sources.add("st.secrets")
    except Exception:
        pass

    if not key_sources:
        api_keys_source = "tidak diatur"
    elif len(key_sources) == 1:
        api_keys_source = next(iter(key_sources))
    else:
        api_keys_source = "campuran (env & st.secrets)"

    try:
        keys = get_all_gemini_api_keys()
        key_count = len(keys)
    except Exception:
        key_count = 0

    return {
        "primary_model": primary_model,
        "primary_model_source": model_source,
        "fallback_models": fallback_models,
        "fallback_models_source": fallback_source,
        "api_key_count": key_count,
        "api_keys_source": api_keys_source,
    }


def get_sqlite_info(cache: SQLiteAnalysisCache | None = None) -> dict[str, Any]:
    """Info file SQLite: lokasi, ukuran file, jumlah baris per tabel, status persistensi."""
    db_path = cache.path if cache else CACHE_DB_PATH
    db_path = Path(db_path)

    exists = db_path.exists()
    file_size_bytes = db_path.stat().st_size if exists else 0

    if file_size_bytes < 1024:
        size_formatted = f"{file_size_bytes} B"
    elif file_size_bytes < 1024 * 1024:
        size_formatted = f"{file_size_bytes / 1024:.1f} KB"
    else:
        size_formatted = f"{file_size_bytes / (1024 * 1024):.2f} MB"

    # Status persistensi
    if cache and hasattr(cache, "existed_before_session"):
        if cache.existed_before_session:
            persistence_status = "Persisten (file sudah ada sebelum sesi aplikasi ini)"
        else:
            persistence_status = "Baru dibuat pada sesi ini"
    else:
        persistence_status = "Persisten" if exists else "Belum dibuat"

    # Baris per tabel
    if cache:
        table_counts = cache.get_table_counts()
    else:
        temp_cache = SQLiteAnalysisCache(db_path)
        table_counts = temp_cache.get_table_counts()

    return {
        "db_path": str(db_path.resolve()),
        "exists": exists,
        "file_size_bytes": file_size_bytes,
        "file_size_formatted": size_formatted,
        "persistence_status": persistence_status,
        "counts": {
            "riwayat (screening_history)": table_counts.get("screening_history", 0),
            "grounding_cache": table_counts.get("grounding_cache", 0),
            "listing pasar (market_listings)": table_counts.get("market_listings", 0),
            "query_cache": table_counts.get("query_cache", 0),
            "daily_limits": table_counts.get("daily_limits", 0),
            "usage_logs": table_counts.get("usage_logs", 0),
        },
    }


def check_domain_connectivity(domain: str, timeout: float = 5.0) -> dict[str, Any]:
    """Uji konektivitas ringan ke domain marketplace (HEAD/GET timeout 5s)."""
    clean_domain = domain.strip().lower()
    url = f"https://www.{clean_domain}" if not clean_domain.startswith("www.") else f"https://{clean_domain}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    t0 = time.perf_counter()
    try:
        # Coba HEAD terlebih dahulu
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        # Jika server marketplace menolak HEAD (403/405), coba GET stream ringan
        if response.status_code in {403, 405}:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        status_code = response.status_code

        if 200 <= status_code < 400:
            status_desc = f"{status_code} OK / Accessible"
            is_blocked = False
            is_ok = True
        elif status_code == 403:
            status_desc = "403 Forbidden (Kemungkinan Cloudflare/WAF block)"
            is_blocked = True
            is_ok = False
        elif status_code == 429:
            status_desc = "429 Too Many Requests (Rate limited)"
            is_blocked = True
            is_ok = False
        else:
            status_desc = f"HTTP {status_code}"
            is_blocked = status_code in {401, 403, 429}
            is_ok = False

        return {
            "domain": clean_domain,
            "url": url,
            "status_code": str(status_code),
            "status_desc": status_desc,
            "response_time_ms": elapsed_ms,
            "is_ok": is_ok,
            "is_blocked": is_blocked,
        }
    except requests.exceptions.Timeout:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "domain": clean_domain,
            "url": url,
            "status_code": "Timeout",
            "status_desc": f"Timeout (> {timeout}s)",
            "response_time_ms": elapsed_ms,
            "is_ok": False,
            "is_blocked": False,
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "domain": clean_domain,
            "url": url,
            "status_code": "Error",
            "status_desc": f"Gagal koneksi ({exc.__class__.__name__})",
            "response_time_ms": elapsed_ms,
            "is_ok": False,
            "is_blocked": False,
        }


def check_marketplace_connectivity(
    domains: list[str] | None = None,
    timeout: float = 5.0,
) -> list[dict[str, Any]]:
    """Uji konektivitas beberapa domain marketplace secara paralel."""
    target_domains = domains or MARKETPLACE_DOMAINS
    results: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_domains) or 1) as executor:
        future_to_domain = {
            executor.submit(check_domain_connectivity, domain, timeout): domain
            for domain in target_domains
        }
        for future in concurrent.futures.as_completed(future_to_domain):
            try:
                results.append(future.result())
            except Exception as e:
                domain = future_to_domain[future]
                results.append({
                    "domain": domain,
                    "url": f"https://{domain}",
                    "status_code": "Error",
                    "status_desc": f"Error: {e}",
                    "response_time_ms": 0.0,
                    "is_ok": False,
                    "is_blocked": False,
                })

    results.sort(key=lambda item: item["domain"])
    return results


def get_daily_limit_info(cache: SQLiteAnalysisCache) -> dict[str, Any]:
    """Ambil konfigurasi DAILY_ANALYSIS_LIMIT dan penggunaan harian saat ini."""
    from src.ui import _daily_limit

    try:
        limit = _daily_limit()
    except Exception:
        limit = 10

    current_count = cache.get_daily_count()
    remaining = max(0, limit - current_count)

    return {
        "limit": limit,
        "current_count": current_count,
        "remaining": remaining,
    }


def get_full_diagnostics_summary(
    cache: SQLiteAnalysisCache,
    connectivity_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Mengumpulkan seluruh data diagnostik sebagai dictionary terstruktur (bebas rahasia)."""
    return {
        "git": get_git_info(),
        "runtime": get_python_and_package_versions(),
        "server_time": get_server_time_info(),
        "model_config": get_model_and_config_info(),
        "sqlite": get_sqlite_info(cache),
        "daily_limit": get_daily_limit_info(cache),
        "connectivity": connectivity_results or [],
    }


def format_diagnostics_as_text(data: dict[str, Any]) -> str:
    """Mengonversi data diagnostik ke teks markdown untuk rendering atau audit tes."""
    git = data.get("git", {})
    runtime = data.get("runtime", {})
    time_info = data.get("server_time", {})
    model = data.get("model_config", {})
    sqlite = data.get("sqlite", {})
    limit = data.get("daily_limit", {})
    conn = data.get("connectivity", [])

    lines = [
        "### Diagnostik Lingkungan",
        f"- **Git Commit**: `{git.get('commit_hash')}` (Build date: `{git.get('build_date')}`)",
        f"- **Python**: `{runtime.get('python_version')}`",
        "- **Paket Kunci**:",
    ]
    for pkg, ver in runtime.get("packages", {}).items():
        lines.append(f"  - `{pkg}`: `{ver}`")

    lines.extend([
        f"- **Waktu Server (UTC)**: `{time_info.get('utc')}`",
        f"- **Waktu Server (WIB)**: `{time_info.get('wib')}` (Lokal: `{time_info.get('local_timezone')}`)",
        f"- **Model Utama**: `{model.get('primary_model')}` (Sumber: `{model.get('primary_model_source')}`)",
        f"- **Model Fallback**: `{', '.join(model.get('fallback_models', [])) or 'Tidak ada'}` (Sumber: `{model.get('fallback_models_source')}`)",
        f"- **Jumlah Gemini API Key Terbaca**: `{model.get('api_key_count')}` key (Sumber: `{model.get('api_keys_source')}`)",
        f"- **File SQLite**: `{sqlite.get('db_path')}` ({sqlite.get('file_size_formatted')})",
        f"- **Status SQLite**: {sqlite.get('persistence_status')}",
        "- **Jumlah Baris Tabel**:",
    ])
    for tbl, count in sqlite.get("counts", {}).items():
        lines.append(f"  - `{tbl}`: {count} baris")

    lines.extend([
        f"- **Batas Harian Analisa**: {limit.get('current_count')} / {limit.get('limit')} terpakai (Tersisa: {limit.get('remaining')})",
    ])

    if conn:
        lines.append("- **Konektivitas Marketplace**:")
        for c in conn:
            lines.append(f"  - `{c.get('domain')}`: {c.get('status_desc')} ({c.get('response_time_ms')} ms)")

    return "\n".join(lines)


def render_environment_diagnostics(
    cache: SQLiteAnalysisCache,
    as_expander: bool = False,
    key_prefix: str = "",
) -> None:
    """Render panel Diagnostik Lingkungan di Streamlit UI."""
    import streamlit as st

    def _render_inner():
        # 1. Info Git & Versi
        git_info = get_git_info()
        time_info = get_server_time_info()
        runtime_info = get_python_and_package_versions()
        model_info = get_model_and_config_info()
        sqlite_info = get_sqlite_info(cache)
        limit_info = get_daily_limit_info(cache)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📌 Versi & Git")
            st.write(f"**Git Commit:** `{git_info['commit_hash']}`")
            st.write(f"**Tanggal Build:** `{git_info['build_date']}`")
            st.write(f"**Python Runtime:** `{runtime_info['python_version']}`")

            st.markdown("##### 🕒 Waktu Server")
            st.write(f"**UTC:** `{time_info['utc']}`")
            st.write(f"**WIB:** `{time_info['wib']}`")
            st.caption(f"Server timezone: {time_info['local_timezone']} ({time_info['local_time']})")

        with col2:
            st.markdown("##### 🤖 Model AI & Konfigurasi")
            st.write(f"**Model Utama:** `{model_info['primary_model']}` (Sumber: `{model_info['primary_model_source']}`)")
            fallback_str = ", ".join(model_info["fallback_models"]) if model_info["fallback_models"] else "Tidak ada"
            st.write(f"**Model Fallback:** `{fallback_str}` (Sumber: `{model_info['fallback_models_source']}`)")
            st.write(f"**Jumlah Key Terbaca:** `{model_info['api_key_count']}` key (Sumber: `{model_info['api_keys_source']}`)")
            st.caption("🔒 *Nilai API key, prefix, dan suffix dirahasiakan.*")

            st.markdown("##### ⏱️ Kuota Analisa Harian")
            st.write(
                f"**DAILY_ANALYSIS_LIMIT:** `{limit_info['limit']}` | "
                f"**Terpakai Hari Ini:** `{limit_info['current_count']}` "
                f"(Sisa: **{limit_info['remaining']}**)"
            )

        st.divider()

        col_pkg, col_db = st.columns(2)
        with col_pkg:
            st.markdown("##### 📦 Paket Dependensi Kunci")
            pkg_data = [{"Paket": k, "Versi": v} for k, v in runtime_info["packages"].items()]
            st.dataframe(pkg_data, use_container_width=True, hide_index=True)

        with col_db:
            st.markdown("##### 🗄️ Database SQLite Lokal")
            st.write(f"**Lokasi:** `{sqlite_info['db_path']}`")
            st.write(f"**Ukuran:** `{sqlite_info['file_size_formatted']}` ({sqlite_info['file_size_bytes']} bytes)")
            st.write(f"**Status File:** {sqlite_info['persistence_status']}")
            
            counts_data = [{"Tabel": k, "Jumlah Baris": v} for k, v in sqlite_info["counts"].items()]
            st.dataframe(counts_data, use_container_width=True, hide_index=True)

        st.divider()

        # Konektivitas Marketplace
        st.markdown("##### 🌐 Uji Konektivitas Marketplace (HEAD/GET timeout 5s)")
        st.caption("Pemeriksaan ringan untuk mendeteksi apakah server/cloud Anda diblokir oleh domain marketplace.")

        # Tombol jalankan tes konektivitas dengan key unik
        connectivity_key = "marketplace_connectivity_results"
        button_key = f"{key_prefix}btn_run_connectivity_test" if key_prefix else "btn_run_connectivity_test"
        run_test = st.button("🔄 Jalankan Uji Konektivitas Marketplace", key=button_key)

        if run_test or connectivity_key in st.session_state:
            if run_test or connectivity_key not in st.session_state:
                with st.spinner("Menguji konektivitas ke 5 domain marketplace..."):
                    conn_results = check_marketplace_connectivity(timeout=5.0)
                    st.session_state[connectivity_key] = conn_results
            else:
                conn_results = st.session_state[connectivity_key]

            # Tampilkan tabel status
            table_display = []
            for r in conn_results:
                icon = "✅" if r["is_ok"] else ("⛔" if r["is_blocked"] else "⚠️")
                table_display.append({
                    "Status": f"{icon} {r['status_code']}",
                    "Domain": r["domain"],
                    "Keterangan": r["status_desc"],
                    "Latency (ms)": f"{r['response_time_ms']} ms",
                })
            st.dataframe(table_display, use_container_width=True, hide_index=True)
        else:
            st.info("Tekan tombol di atas untuk menjalankan uji konektivitas ringan.")

    if as_expander:
        with st.expander("🩺 Diagnostik Lingkungan", expanded=False):
            _render_inner()
    else:
        _render_inner()

