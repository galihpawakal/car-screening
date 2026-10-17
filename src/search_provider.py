"""Official google-genai provider for Search grounding and schema-constrained calls."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.config import SUPPORTED_SEARCH_MODELS
from src.url_resolve import ResolvedUrl, normalize_host, resolve_grounding_urls

logger = logging.getLogger(__name__)
DEBUG_LOG = os.environ.get("CAR_SCREENING_DEBUG") == "1"


@dataclass(frozen=True)
class GeminiErrorInfo:
    http_code: int | None
    category: str  # "quota" | "auth" | "timeout" | "server" | "unknown"
    message: str   # Cleaned message without raw API key
    is_retryable: bool


class GroundingDeadlineExceeded(TimeoutError):
    """Raised when global grounding deadline (120s) is exceeded across models/keys."""
    pass


def mask_api_keys(text: str, api_keys: list[str] | None = None) -> str:
    """Mask Google API keys in error strings and logs."""
    if not text:
        return ""
    masked = str(text)
    masked = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", masked)
    if api_keys:
        for k in api_keys:
            if k and len(k) > 6:
                masked = masked.replace(k, "[REDACTED_API_KEY]")
    return masked


def classify_gemini_error(error: Exception, api_keys: list[str] | None = None) -> GeminiErrorInfo:
    """Classify Gemini API errors with extracted HTTP code, category, and sanitized message."""
    raw_str = str(error)
    cleaned_msg = mask_api_keys(raw_str, api_keys)

    # 1. Extract HTTP code
    http_code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if callable(http_code):
        try:
            http_code = http_code()
        except Exception:
            http_code = None
    try:
        http_code = int(http_code)
    except (TypeError, ValueError):
        http_code = None

    if http_code is None:
        code_match = re.search(r"\b(400|401|403|404|429|500|502|503|504)\b", raw_str)
        if code_match:
            http_code = int(code_match.group(1))

    upper_msg = raw_str.upper()

    # 2. Categorize
    is_timeout = (
        isinstance(error, (concurrent.futures.TimeoutError, TimeoutError))
        or "DEADLINE_EXCEEDED" in upper_msg
        or "TIMEOUT" in upper_msg
        or http_code == 504
    )
    is_quota = (
        http_code == 429
        or "RESOURCE_EXHAUSTED" in upper_msg
        or "RATE_LIMIT" in upper_msg
        or "QUOTA" in upper_msg
        or "TOO_MANY_REQUESTS" in upper_msg
        or "PER DAY" in upper_msg
    )
    is_auth = (
        http_code in (401, 403)
        or "API_KEY_INVALID" in upper_msg
        or "PERMISSION_DENIED" in upper_msg
        or "UNAUTHENTICATED" in upper_msg
        or "KEY NOT VALID" in upper_msg
    )
    is_server = (
        (http_code is not None and 500 <= http_code <= 599 and http_code != 504)
        or "UNAVAILABLE" in upper_msg
        or "INTERNAL" in upper_msg
    )

    if is_timeout:
        category = "timeout"
        is_retryable = True
    elif is_quota:
        category = "quota"
        is_retryable = True
    elif is_auth:
        category = "auth"
        is_retryable = False
    elif is_server:
        category = "server"
        is_retryable = True
    else:
        category = "unknown"
        is_retryable = False

    return GeminiErrorInfo(
        http_code=http_code,
        category=category,
        message=cleaned_msg,
        is_retryable=is_retryable,
    )


@dataclass(frozen=True)
class GroundingSource:
    url: str
    title: str
    original_url: str = ""


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class GroundedSearchResult:
    text: str
    sources: list[GroundingSource]
    queries: list[str]
    usage: TokenUsage
    grounding_calls: int = 1
    model: str = ""
    marketplace: str = ""


@dataclass(frozen=True)
class StructuredResponse:
    output_text: str
    usage: TokenUsage
    model: str = ""


class SearchProvider(ABC):
    @property
    def last_used_model(self) -> str:
        return getattr(self, "_last_used_model", getattr(self, "model", ""))

    @last_used_model.setter
    def last_used_model(self, value: str) -> None:
        self._last_used_model = value

    @abstractmethod
    def search(self, prompt: str, *, marketplace: str = "") -> GroundedSearchResult:
        """Run one Search-grounded call."""

    @abstractmethod
    def generate_structured(self, prompt: str, response_model) -> StructuredResponse:
        """Run one JSON-schema call without tools."""


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip()
    try:
        import streamlit as st

        secret = st.secrets.get(name)
        return str(secret).strip() if secret else None
    except Exception:
        return None


def parse_fallback_models(raw: str | list[str] | tuple[str, ...] | None, primary_model: str) -> list[str]:
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(item).strip() for item in raw if str(item).strip()]
    else:
        items = []

    supported = ", ".join(sorted(SUPPORTED_SEARCH_MODELS))
    fallback_models: list[str] = []
    for item in items:
        if item not in SUPPORTED_SEARCH_MODELS:
            raise ValueError(
                f"Model cadangan '{item}' di GEMINI_FALLBACK_MODELS tidak tercantum sebagai model Google Search yang didukung. Pilihan terdokumentasi: {supported}."
            )
        if item != primary_model and item not in fallback_models:
            fallback_models.append(item)
    return fallback_models


def get_all_gemini_api_keys() -> list[str]:
    keys: list[str] = []
    keys_raw = _secret("GEMINI_API_KEYS")
    if keys_raw:
        for item in keys_raw.split(","):
            cleaned = item.strip()
            if cleaned and cleaned not in keys:
                keys.append(cleaned)
    single_key = _secret("GEMINI_API_KEY")
    if single_key and single_key.strip():
        cleaned = single_key.strip()
        if cleaned not in keys:
            keys.insert(0, cleaned)
    return keys


def get_gemini_settings() -> tuple[str, str, list[str]]:
    all_keys = get_all_gemini_api_keys()
    api_key = all_keys[0] if all_keys else None
    if not api_key:
        raise ValueError("GEMINI_API_KEY belum diatur di environment atau Streamlit secrets.")
    model = _secret("GEMINI_MODEL")
    if not model:
        raise ValueError("GEMINI_MODEL belum diatur. Gunakan model Gemini yang mendukung Google Search.")
    if model not in SUPPORTED_SEARCH_MODELS:
        supported = ", ".join(sorted(SUPPORTED_SEARCH_MODELS))
        raise ValueError(
            f"GEMINI_MODEL '{model}' tidak tercantum sebagai model Google Search yang didukung. Pilihan terdokumentasi: {supported}."
        )
    fallback_raw = _secret("GEMINI_FALLBACK_MODELS")
    fallback_models = parse_fallback_models(fallback_raw, primary_model=model)
    return api_key, model, fallback_models


def _value(item: Any, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _plain(item: Any) -> Any:
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return item


def _usage(response: Any) -> TokenUsage:
    usage = _value(response, "usage") or _value(response, "usage_metadata") or {}
    input_tokens = (
        _value(usage, "input_tokens")
        or _value(usage, "prompt_token_count")
        or _value(usage, "promptTokenCount")
        or 0
    )
    output_tokens = (
        _value(usage, "output_tokens")
        or _value(usage, "candidates_token_count")
        or _value(usage, "candidatesTokenCount")
        or 0
    )
    return TokenUsage(int(input_tokens or 0), int(output_tokens or 0))


def _text(response: Any) -> str:
    text = _value(response, "output_text")
    if text:
        return str(text)
    steps = _value(response, "steps", []) or []
    parts: list[str] = []
    for step in steps:
        if _value(step, "type") == "model_output":
            for block in _value(step, "content", []) or []:
                val = _value(block, "text")
                if val:
                    parts.append(str(val))
    if parts:
        return "\n".join(parts)
    candidates = _value(response, "candidates", []) or []
    for candidate in candidates:
        content = _value(candidate, "content")
        for part in _value(content, "parts", []) or []:
            value = _value(part, "text")
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def _log_grounding_debug(response: Any, label: str = "search") -> None:
    if not DEBUG_LOG:
        return
    try:
        logger.debug("RAW_%s_RESPONSE_TEXT: %s", label.upper(), _text(response))
    except Exception as error:
        logger.debug("RAW_%s_TEXT_FAILED: %s", label.upper(), error)
    try:
        logger.debug("RAW_%s_RESPONSE: %s", label.upper(), json.dumps(_plain(response), ensure_ascii=False, indent=2))
    except Exception as error:
        logger.debug("RAW_%s_SERIALIZE_FAILED: %s", label.upper(), error)
    gm = _value(response, "grounding_metadata") or _value(response, "groundingMetadata")
    try:
        if gm is not None:
            logger.debug("GROUNDING_METADATA_FULL: %s", json.dumps(_plain(gm), ensure_ascii=False, indent=2))
        else:
            logger.debug("GROUNDING_METADATA_FULL: null")
        gm_plain = _plain(gm) if gm is not None else {}
        if isinstance(gm_plain, dict):
            for key in ("grounding_chunks", "groundingChunks", "grounding_supports", "groundingSupports", "web_search_queries", "webSearchQueries"):
                if key in gm_plain:
                    logger.debug("GROUNDING_%s: %s", key.upper(), json.dumps(gm_plain[key], ensure_ascii=False, indent=2))
    except Exception as error:
        logger.debug("GROUNDING_METADATA_LOG_FAILED: %s", error)


def _is_retryable(error: Exception) -> bool:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if callable(code):
        try:
            code = code()
        except Exception:
            code = None
    try:
        status = int(code)
        if status in (429, 500, 503) or (500 <= status <= 599):
            return True
    except (TypeError, ValueError):
        pass

    error_str = str(error).upper()
    retryable_markers = (
        "429",
        "500",
        "503",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "NO CAPACITY",
        "CAPACITY",
        "OVERLOADED",
        "RATE_LIMIT",
        "DEADLINE_EXCEEDED",
    )
    return any(marker in error_str for marker in retryable_markers)


class GeminiGroundingProvider(SearchProvider):
    def __init__(
        self,
        client=None,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        on_status=None,
        url_resolver=None,
    ):
        self.on_status = on_status
        self.url_resolver = url_resolver
        if client is None:
            api_key, configured_model, configured_fallbacks = get_gemini_settings()
            self.api_keys = get_all_gemini_api_keys() or [api_key]
            http_opts = None
            try:
                http_opts = types.HttpOptions(
                    timeout=90_000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                )
            except Exception:
                pass
            self.clients = [
                genai.Client(api_key=k, http_options=http_opts) if http_opts else genai.Client(api_key=k)
                for k in self.api_keys
            ]
            self.client_index = 0
            self.client = self.clients[0]
            self.model = model or configured_model
            self.fallback_models = fallback_models if fallback_models is not None else configured_fallbacks
        else:
            self.client = client
            self.clients = [client]
            self.client_index = 0
            self.api_keys = []
            self.model = model or _secret("GEMINI_MODEL")
            if not self.model:
                raise ValueError("GEMINI_MODEL belum diatur.")
            if fallback_models is not None:
                self.fallback_models = parse_fallback_models(fallback_models, primary_model=self.model)
            else:
                fallback_raw = _secret("GEMINI_FALLBACK_MODELS")
                self.fallback_models = parse_fallback_models(fallback_raw, primary_model=self.model)

        supported = ", ".join(sorted(SUPPORTED_SEARCH_MODELS))
        if self.model not in SUPPORTED_SEARCH_MODELS:
            raise ValueError(
                f"Model '{self.model}' tidak didukung untuk Google Search grounding. Pilihan terdokumentasi: {supported}."
            )
        for fallback in self.fallback_models:
            if fallback not in SUPPORTED_SEARCH_MODELS:
                raise ValueError(
                    f"Model cadangan '{fallback}' tidak didukung untuk Google Search grounding. Pilihan terdokumentasi: {supported}."
                )

        self.active_model: str = self.model
        self.last_used_model: str = self.model
        self.max_retries = max(1, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.api_calls = 0
        self.attempt_history: list[dict] = []
        self._lock = threading.RLock()
        self._exhausted_clients_meta: dict[int, dict] = {}
        self._key_invalid_meta: dict[int, dict] = {}
        self._grounding_disabled_meta: dict[int, dict] = {}
        self._quota_exhausted_keys: dict[str, set[int]] = {}
        shared_env = os.environ.get("GEMINI_KEYS_SHARED_PROJECT", "").strip().lower()
        self.shared_project: bool = shared_env in ("1", "true", "yes")

    def _clean_expired_client_states(self, now: float | None = None) -> None:
        t = now if now is not None else time.time()
        with self._lock:
            expired_exhausted = [i for i, m in self._exhausted_clients_meta.items() if t >= m.get("expiry", 0.0)]
            for i in expired_exhausted:
                del self._exhausted_clients_meta[i]
            expired_invalid = [i for i, m in self._key_invalid_meta.items() if t >= m.get("expiry", 0.0)]
            for i in expired_invalid:
                del self._key_invalid_meta[i]
            expired_grounding = [i for i, m in self._grounding_disabled_meta.items() if t >= m.get("expiry", 0.0)]
            for i in expired_grounding:
                del self._grounding_disabled_meta[i]

    @property
    def _exhausted_client_indices(self) -> set[int]:
        self._clean_expired_client_states()
        with self._lock:
            return set(self._exhausted_clients_meta.keys())

    def _is_key_invalid(self, idx: int, now: float | None = None) -> bool:
        t = now if now is not None else time.time()
        with self._lock:
            if idx in self._key_invalid_meta:
                if t < self._key_invalid_meta[idx].get("expiry", 0.0):
                    return True
                del self._key_invalid_meta[idx]
            return False

    def _is_grounding_disabled(self, idx: int, now: float | None = None) -> bool:
        t = now if now is not None else time.time()
        with self._lock:
            if idx in self._grounding_disabled_meta:
                if t < self._grounding_disabled_meta[idx].get("expiry", 0.0):
                    return True
                del self._grounding_disabled_meta[idx]
            return False

    def _mark_key_invalid(self, idx: int, http_code: int | None = 401, error_message: str = "", ttl: float = 3600.0) -> None:
        now = time.time()
        with self._lock:
            self._key_invalid_meta[idx] = {
                "timestamp": now,
                "expiry": now + ttl,
                "http_code": http_code,
                "error_message": error_message or "API key tidak valid",
            }

    def _mark_grounding_disabled(self, idx: int, http_code: int | None = 403, error_message: str = "", ttl: float = 3600.0) -> None:
        now = time.time()
        with self._lock:
            self._grounding_disabled_meta[idx] = {
                "timestamp": now,
                "expiry": now + ttl,
                "http_code": http_code,
                "error_message": error_message or "Permission denied on search grounding",
            }

    def _mark_client_exhausted(
        self,
        idx: int,
        reason: str = "exhausted",
        http_code: int | None = None,
        error_message: str = "",
        ttl: float | None = None,
    ) -> None:
        now = time.time()
        if ttl is None:
            if http_code == 429 or "quota" in reason.lower() or "429" in reason:
                ttl = 300.0  # 5 menit untuk 429
            elif "timeout" in reason.lower():
                ttl = 60.0   # 60 detik untuk timeout
            else:
                ttl = 60.0
        with self._lock:
            self._exhausted_clients_meta[idx] = {
                "timestamp": now,
                "expiry": now + ttl,
                "reason": reason,
                "http_code": http_code,
                "error_message": error_message or reason,
            }
            total = len(getattr(self, "clients", [1]))
            if total > 1:
                self.client_index = (idx + 1) % total

    def _get_client_skip_info(self, idx: int, is_grounding: bool = False, now: float | None = None) -> tuple[str, int | None] | None:
        t = now if now is not None else time.time()
        with self._lock:
            if idx in self._key_invalid_meta:
                meta = self._key_invalid_meta[idx]
                if t < meta.get("expiry", 0.0):
                    dt = datetime.fromtimestamp(meta["timestamp"]).strftime("%H:%M:%S")
                    code = meta.get("http_code", 401)
                    err_msg = meta.get("error_message") or "API key tidak valid"
                    return f"dilewati: ditandai tidak valid pada {dt}, kode {code}: {err_msg}", code
                else:
                    del self._key_invalid_meta[idx]

            if is_grounding and idx in self._grounding_disabled_meta:
                meta = self._grounding_disabled_meta[idx]
                if t < meta.get("expiry", 0.0):
                    dt = datetime.fromtimestamp(meta["timestamp"]).strftime("%H:%M:%S")
                    code = meta.get("http_code", 403)
                    err_msg = meta.get("error_message") or "Permission denied on grounding"
                    return f"dilewati: grounding dinonaktifkan pada {dt}, kode {code}: {err_msg}", code
                else:
                    del self._grounding_disabled_meta[idx]

            if idx in self._exhausted_clients_meta:
                meta = self._exhausted_clients_meta[idx]
                if t < meta.get("expiry", 0.0):
                    dt = datetime.fromtimestamp(meta["timestamp"]).strftime("%H:%M:%S")
                    code = meta.get("http_code")
                    code_str = str(code) if code is not None else "-"
                    err_msg = meta.get("error_message") or meta.get("reason") or "kuota habis"
                    return f"dilewati: ditandai habis pada {dt}, kode {code_str}: {err_msg}", code
                else:
                    del self._exhausted_clients_meta[idx]

            return None

    def _next_available_client(self, is_grounding: bool = False) -> tuple[Any, int] | None:
        self._clean_expired_client_states()
        with self._lock:
            clients = getattr(self, "clients", None)
            if not clients:
                return self.client, 0
            total = len(clients)
            now = time.time()
            for offset in range(total):
                idx = (self.client_index + offset) % total
                if idx in self._exhausted_clients_meta:
                    continue
                if self._is_key_invalid(idx, now):
                    continue
                if is_grounding and self._is_grounding_disabled(idx, now):
                    continue
                return clients[idx], idx
            return None

    def find_first_available_key(self, model: str = "", is_grounding: bool = False) -> int | None:
        self._clean_expired_client_states()
        clients = getattr(self, "clients", None)
        now = time.time()
        if not clients or len(clients) <= 1:
            if 0 in self._exhausted_clients_meta or self._is_key_invalid(0, now):
                return None
            if is_grounding and self._is_grounding_disabled(0, now):
                return None
            return 0

        total = len(clients)
        for offset in range(total):
            idx = (self.client_index + offset) % total
            if idx in self._exhausted_clients_meta:
                continue
            if self._is_key_invalid(idx, now):
                continue
            if is_grounding and self._is_grounding_disabled(idx, now):
                continue
            return idx
        return None

    def _notify(self, message: str) -> None:
        if self.on_status and callable(self.on_status):
            try:
                self.on_status(mask_api_keys(message, getattr(self, "api_keys", None)))
            except Exception:
                pass

    def _candidate_models(self) -> list[str]:
        candidates = [self.active_model]
        for candidate in [self.model] + self.fallback_models:
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _request(self, **kwargs):
        candidates = self._candidate_models()
        last_error: Exception | None = None
        total_clients = len(getattr(self, "clients", [1]))
        tools = kwargs.get("tools")
        is_grounding = bool(tools)

        default_call_timeout = 90.0 if is_grounding else 60.0
        global_deadline = time.time() + 120.0 if is_grounding else None

        for pass_num in range(2):
            real_api_calls_in_request = 0

            for candidate_model in candidates:
                if global_deadline is not None and time.time() >= global_deadline:
                    raise GroundingDeadlineExceeded("Deadline global tahap grounding (120 detik) terlampaui.")

                start_index = self.client_index
                for offset in range(total_clients):
                    if global_deadline is not None and time.time() >= global_deadline:
                        raise GroundingDeadlineExceeded("Deadline global tahap grounding (120 detik) terlampaui.")

                    client_idx = (start_index + offset) % total_clients
                    active_client = self.clients[client_idx] if getattr(self, "clients", None) else self.client

                    # Periksa apakah key ini dilewati
                    skip_info = self._get_client_skip_info(client_idx, is_grounding=is_grounding)
                    if skip_info is not None:
                        skip_msg, skip_code = skip_info
                        self.attempt_history.append({
                            "model": candidate_model,
                            "attempt": 0,
                            "key_index": client_idx,
                            "success": False,
                            "http_code": skip_code,
                            "category": "skipped",
                            "error": skip_msg,
                            "retryable": False,
                        })
                        if last_error is None:
                            last_error = RuntimeError(skip_msg)
                        continue

                    key_tag = f" (key #{client_idx + 1}/{total_clients})" if total_clients > 1 else ""
                    action_desc = "sedang menelusuri listing di web" if is_grounding else "memproses data terstruktur"
                    self._notify(
                        f"🌐 Menghubungi model `{candidate_model}`{key_tag} ({action_desc})..."
                    )

                    key_failed = False
                    timeout_retries_on_this_key = 0
                    max_timeout_retries = 2

                    for attempt in range(self.max_retries):
                        if global_deadline is not None and time.time() >= global_deadline:
                            raise GroundingDeadlineExceeded("Deadline global tahap grounding (120 detik) terlampaui.")

                        self.api_calls += 1
                        real_api_calls_in_request += 1
                        call_timeout = default_call_timeout
                        if global_deadline is not None:
                            remaining = global_deadline - time.time()
                            call_timeout = min(call_timeout, max(1.0, remaining))

                        try:
                            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                            future = executor.submit(active_client.interactions.create, model=candidate_model, **kwargs)
                            try:
                                response = future.result(timeout=call_timeout)
                            finally:
                                executor.shutdown(wait=False)

                            self.active_model = candidate_model
                            self.last_used_model = candidate_model
                            self.attempt_history.append(
                                {
                                    "model": candidate_model,
                                    "attempt": attempt + 1,
                                    "key_index": client_idx,
                                    "success": True,
                                    "http_code": 200,
                                    "category": "success",
                                    "error": None,
                                }
                            )
                            self._notify(f"✅ Model `{candidate_model}` berhasil merespons via key #{client_idx + 1}.")
                            with self._lock:
                                self.client_index = client_idx
                            return response, candidate_model

                        except concurrent.futures.TimeoutError as timeout_err:
                            last_error = timeout_err
                            err_info = classify_gemini_error(timeout_err, getattr(self, "api_keys", None))
                            self.attempt_history.append(
                                {
                                    "model": candidate_model,
                                    "attempt": attempt + 1,
                                    "key_index": client_idx,
                                    "success": False,
                                    "http_code": err_info.http_code,
                                    "category": err_info.category,
                                    "error": err_info.message,
                                    "retryable": True,
                                }
                            )

                            if timeout_retries_on_this_key < max_timeout_retries:
                                timeout_retries_on_this_key += 1
                                delay = self.backoff_seconds * (2 ** (timeout_retries_on_this_key - 1))
                                self._notify(
                                    f"⚠️ Key #{client_idx + 1} tidak merespons (timeout {int(call_timeout)}s). Retry {timeout_retries_on_this_key}/{max_timeout_retries} pada key yang sama (jeda {delay:.1f}s)..."
                                )
                                if delay > 0:
                                    time.sleep(delay)
                                continue
                            else:
                                self._mark_client_exhausted(
                                    client_idx,
                                    reason="timeout",
                                    http_code=err_info.http_code or 504,
                                    error_message=f"Timeout {int(call_timeout)}s",
                                    ttl=60.0,
                                )
                                remaining_keys = total_clients - len(self._exhausted_client_indices)
                                self._notify(
                                    f"⚠️ Key #{client_idx + 1} berulang kali timeout ({max_timeout_retries}x retry gagal). Beralih ke key berikutnya ({remaining_keys} key tersisa)..."
                                )
                                key_failed = True
                                break

                        except Exception as error:
                            last_error = error
                            err_info = classify_gemini_error(error, getattr(self, "api_keys", None))
                            self.attempt_history.append(
                                {
                                    "model": candidate_model,
                                    "attempt": attempt + 1,
                                    "key_index": client_idx,
                                    "success": False,
                                    "http_code": err_info.http_code,
                                    "category": err_info.category,
                                    "error": err_info.message,
                                    "retryable": err_info.is_retryable,
                                }
                            )

                            if err_info.category == "quota":
                                if attempt + 1 < self.max_retries and err_info.is_retryable:
                                    delay = self.backoff_seconds * (2 ** attempt)
                                    self._notify(
                                        f"⚠️ Key #{client_idx + 1} kuota/rate limit (429). Menunggu jeda {delay:.1f} detik untuk retry..."
                                    )
                                    if delay > 0:
                                        time.sleep(delay)
                                    continue

                                self._mark_client_exhausted(
                                    client_idx,
                                    reason="429 quota",
                                    http_code=429,
                                    error_message=err_info.message or "Quota 429",
                                    ttl=300.0,
                                )
                                self._quota_exhausted_keys.setdefault(candidate_model, set()).add(client_idx)
                                shared_detected = (
                                    self.shared_project
                                    or len(self._quota_exhausted_keys[candidate_model]) >= 2
                                )
                                if shared_detected:
                                    self._notify(
                                        f"⚠️ Kuota habis (429) pada model `{candidate_model}` (terdeteksi project bersama). Menghentikan rotasi key, langsung beralih ke model cadangan..."
                                    )
                                    key_failed = True
                                    break

                                remaining_keys = total_clients - len(self._exhausted_client_indices)
                                if remaining_keys > 0:
                                    self._notify(
                                        f"⚠️ API key #{client_idx + 1} kuota habis/429. Beralih ke key berikutnya ({remaining_keys} key tersisa)..."
                                    )
                                else:
                                    self._notify(
                                        f"⚠️ Seluruh {total_clients} API key telah habis kuota pada model `{candidate_model}`."
                                    )
                                key_failed = True
                                break

                            if err_info.category == "auth":
                                if is_grounding and err_info.http_code == 403:
                                    self._mark_grounding_disabled(
                                        client_idx,
                                        http_code=403,
                                        error_message=err_info.message or "Permission denied on search grounding",
                                        ttl=3600.0,
                                    )
                                    self._notify(
                                        f"⚠️ API key #{client_idx + 1} tidak memiliki izin grounding (403). Fitur grounding dinonaktifkan sementara untuk key ini..."
                                    )
                                else:
                                    self._mark_key_invalid(
                                        client_idx,
                                        http_code=err_info.http_code or 401,
                                        error_message=err_info.message or "Auth error",
                                        ttl=3600.0,
                                    )
                                    self._mark_client_exhausted(
                                        client_idx,
                                        reason="auth invalid",
                                        http_code=err_info.http_code or 401,
                                        error_message=err_info.message or "Auth error",
                                        ttl=3600.0,
                                    )
                                    self._notify(
                                        f"⚠️ API key #{client_idx + 1} tidak valid/tertolak ({err_info.http_code or 'auth error'}). Menonaktifkan key ini sementara..."
                                    )
                                key_failed = True
                                break

                            if attempt + 1 < self.max_retries and err_info.is_retryable:
                                delay = self.backoff_seconds * (2 ** attempt)
                                self._notify(
                                    f"⚠️ Key #{client_idx + 1} terkendala ({err_info.category}: {err_info.http_code or ''}). Menunggu jeda {delay:.1f} detik untuk retry..."
                                )
                                if delay > 0:
                                    time.sleep(delay)
                                continue
                            else:
                                key_failed = True
                                break

                    if key_failed:
                        if len(self._quota_exhausted_keys.get(candidate_model, set())) >= 2 or self.shared_project:
                            break
                        continue

                if candidates.index(candidate_model) + 1 < len(candidates):
                    next_model = candidates[candidates.index(candidate_model) + 1]
                    self._notify(
                        f"🔄 Model `{candidate_model}` tidak tersedia. Beralih ke model cadangan: `{next_model}`..."
                    )

            # Jika SEMUA key dilewati tanpa satu pun panggilan nyata pada pass 0:
            if real_api_calls_in_request == 0 and pass_num == 0:
                now = time.time()
                reset_count = 0
                with self._lock:
                    keys_to_reset = [
                        idx for idx, meta in self._exhausted_clients_meta.items()
                        if (now - meta["timestamp"]) >= 60.0 or now >= meta["expiry"]
                    ]
                    for idx in keys_to_reset:
                        del self._exhausted_clients_meta[idx]
                        reset_count += 1
                if reset_count > 0:
                    self._notify(
                        f"🔄 Seluruh key dilewati tanpa panggilan nyata. Mereset {reset_count} key yang habis > 60s lalu dan mencoba ulang..."
                    )
                    continue
            break

        models_tried = ", ".join(candidates)
        error_lines = [f"Semua model Gemini ({models_tried}) gagal setelah retry."]
        if self.attempt_history:
            error_lines.append("Rincian per percobaan:")
            for entry in self.attempt_history:
                m = entry.get("model", "-")
                k_idx = entry.get("key_index", 0)
                k_str = f"key #{k_idx + 1}"
                code_str = str(entry.get("http_code")) if entry.get("http_code") is not None else "-"
                msg = entry.get("error") or "Unknown error"
                msg_clean = mask_api_keys(msg, getattr(self, "api_keys", None))
                error_lines.append(f"- [{m}, {k_str}, HTTP: {code_str}]: {msg_clean}")
        else:
            cleaned_last_err = mask_api_keys(str(last_error or "Tidak ada panggilan yang dieksekusi"), getattr(self, "api_keys", None))
            error_lines.append(f"Error: {cleaned_last_err}")

        final_err_msg = "\n".join(error_lines)
        raise RuntimeError(final_err_msg) from last_error

    def search(self, prompt: str, *, marketplace: str = "") -> GroundedSearchResult:
        self._notify("🔍 Menjalankan pencarian web Google Search grounding...")
        calls_before = self.api_calls
        response, used_model = self._request(input=prompt, tools=[{"type": "google_search"}])
        _log_grounding_debug(response, label=f"search_{marketplace or 'general'}")
        text = _text(response)
        sources, queries = _grounding_metadata(response, url_resolver=self.url_resolver, marketplace=marketplace)
        call_count = max(1, self.api_calls - calls_before)
        self._notify(f"📄 Ditemukan {len(sources)} URL listing dari {len(queries)} query pencarian.")
        return GroundedSearchResult(
            text,
            sources,
            queries,
            _usage(response),
            grounding_calls=call_count,
            model=used_model,
            marketplace=marketplace,
        )

    def read_url_context(self, url: str) -> tuple[str, TokenUsage]:
        self._notify(f"📰 Membaca halaman listing via url_context: {url[:80]}...")
        response, _used_model = self._request(
            input=f"Baca konten halaman listing mobil ini dan ringkas fakta harga, tahun, KM, transmisi, varian, lokasi yang terlihat: {url}",
            tools=[{"type": "url_context"}],
        )
        _log_grounding_debug(response, label="url_context")
        return _text(response), _usage(response)

    def generate_structured(self, prompt: str, response_model) -> StructuredResponse:
        self._notify(
            f"📑 Mengekstrak data terstruktur menggunakan skema {getattr(response_model, '__name__', 'Pydantic')}..."
        )
        response, used_model = self._request(
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": response_model.model_json_schema(),
            },
        )
        return StructuredResponse(_text(response), _usage(response), model=used_model)


def _grounding_metadata(
    response: Any,
    *,
    url_resolver=None,
    marketplace: str = "",
) -> tuple[list[GroundingSource], list[str]]:
    sources: dict[str, GroundingSource] = {}
    raw_urls: list[str] = []
    queries: list[str] = []
    steps = _value(response, "steps", []) or []
    for step in steps:
        step_type = _value(step, "type", "")
        if step_type == "google_search_call":
            arguments = _plain(_value(step, "arguments", {})) or {}
            for query in arguments.get("queries", []) if isinstance(arguments, dict) else []:
                if query and query not in queries:
                    queries.append(str(query))
        if step_type == "model_output":
            for block in _value(step, "content", []) or []:
                for annotation in _value(block, "annotations", []) or []:
                    if _value(annotation, "type") != "url_citation":
                        continue
                    url = _value(annotation, "url")
                    if url:
                        raw_urls.append(str(url))
        if step_type == "google_search_result":
            result = _plain(_value(step, "result", {})) or {}
            _collect_urls(result, raw_urls)

    metadata = _value(response, "grounding_metadata") or _value(response, "groundingMetadata")
    if not metadata:
        candidates = _value(response, "candidates", []) or []
        if candidates:
            metadata = _value(candidates[0], "grounding_metadata") or _value(candidates[0], "groundingMetadata")
    metadata = _plain(metadata) or {}
    if isinstance(metadata, dict):
        for query in metadata.get("web_search_queries", metadata.get("webSearchQueries", [])):
            if query and query not in queries:
                queries.append(str(query))
        chunks = metadata.get("grounding_chunks", metadata.get("groundingChunks", []))
        for chunk in chunks:
            web = chunk.get("web", {}) if isinstance(chunk, dict) else {}
            url = web.get("uri") or web.get("url")
            if url:
                raw_urls.append(str(url))
        entry_point = metadata.get("search_entry_point") or metadata.get("searchEntryPoint") or {}
        if isinstance(entry_point, dict):
            rendered = str(entry_point.get("rendered_content") or entry_point.get("renderedContent") or "")
            for href in re.findall(r'href=[\'"](https?://[^\'"]+)[\'"]', rendered):
                if href and href not in raw_urls:
                    raw_urls.append(href)

    resolved = _resolve_sources(raw_urls, url_resolver=url_resolver, marketplace=marketplace)
    for item in resolved:
        sources[item.url] = item
    return list(sources.values()), queries


def _collect_urls(value: Any, raw_urls: list[str]) -> None:
    if isinstance(value, dict):
        url = value.get("url") or value.get("uri")
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            raw_urls.append(url)
        for nested in value.values():
            _collect_urls(nested, raw_urls)
    elif isinstance(value, list):
        for nested in value:
            _collect_urls(nested, raw_urls)


def _resolve_sources(
    raw_urls: list[str],
    *,
    url_resolver=None,
    marketplace: str = "",
) -> list[GroundingSource]:
    if url_resolver is not None:
        resolved_items = url_resolver(raw_urls)
    else:
        resolved_items = resolve_grounding_urls(raw_urls)

    target_domains = [normalize_host(d.strip()) for d in marketplace.split(",") if d.strip()] if marketplace else []
    output: list[GroundingSource] = []
    for item in resolved_items:
        if isinstance(item, GroundingSource):
            output.append(item)
            continue
        if not isinstance(item, ResolvedUrl):
            continue
        domain = normalize_host(item.domain)
        if target_domains and not any(domain == td or domain.endswith("." + td) for td in target_domains):
            continue
        output.append(
            GroundingSource(
                url=item.final_url,
                title=domain,
                original_url=item.original,
            )
        )
    return output


def merge_grounded_results(results: list[GroundedSearchResult]) -> GroundedSearchResult:
    if not results:
        return GroundedSearchResult("", [], [], TokenUsage(), grounding_calls=0)
    text_parts = [item.text for item in results if item.text]
    source_map: dict[str, GroundingSource] = {}
    queries: list[str] = []
    usage = TokenUsage()
    calls = 0
    model = results[-1].model
    for item in results:
        calls += item.grounding_calls
        usage = TokenUsage(usage.input_tokens + item.usage.input_tokens, usage.output_tokens + item.usage.output_tokens)
        for query in item.queries:
            if query not in queries:
                queries.append(query)
        for source in item.sources:
            source_map.setdefault(source.url, source)
    return GroundedSearchResult(
        "\n\n".join(text_parts),
        list(source_map.values()),
        queries,
        usage,
        grounding_calls=calls,
        model=model,
    )
