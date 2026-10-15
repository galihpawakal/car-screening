"""Official google-genai provider for Search grounding and schema-constrained calls."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from google import genai

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
            self.clients = [genai.Client(api_key=k) for k in self.api_keys]
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
        self._lock = threading.Lock()
        self._exhausted_client_indices: set[int] = set()

    def _next_client(self) -> tuple[Any, int]:
        with self._lock:
            clients = getattr(self, "clients", None)
            if not clients:
                return self.client, 0
            total = len(clients)
            if total <= 1:
                return clients[0], 0
            for offset in range(total):
                idx = (self.client_index + offset) % total
                if idx not in self._exhausted_client_indices:
                    self.client_index = (idx + 1) % total
                    return clients[idx], idx
            # If all are marked exhausted, reset to allow retry across keys
            self._exhausted_client_indices.clear()
            idx = self.client_index % total
            self.client_index = (idx + 1) % total
            return clients[idx], idx

    def _mark_client_exhausted(self, idx: int) -> None:
        with self._lock:
            self._exhausted_client_indices.add(idx)

    def _notify(self, message: str) -> None:
        if self.on_status and callable(self.on_status):
            try:
                self.on_status(message)
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
        for candidate_model in candidates:
            for attempt in range(self.max_retries):
                self.api_calls += 1
                tools = kwargs.get("tools")
                action_desc = "sedang menelusuri listing di web" if tools else "memproses data terstruktur"
                active_client, client_idx = self._next_client()
                total_clients = len(getattr(self, "clients", [1]))
                key_tag = f" (key #{client_idx + 1}/{total_clients})" if total_clients > 1 else ""
                self._notify(
                    f"🌐 Menghubungi model `{candidate_model}`{key_tag} ({action_desc}, percobaan {attempt + 1}/{self.max_retries})..."
                )
                try:
                    response = active_client.interactions.create(model=candidate_model, **kwargs)
                    self.active_model = candidate_model
                    self.last_used_model = candidate_model
                    self.attempt_history.append(
                        {"model": candidate_model, "attempt": attempt + 1, "success": True, "error": None}
                    )
                    self._notify(f"✅ Model `{candidate_model}` berhasil merespons.")
                    return response, candidate_model
                except Exception as error:
                    last_error = error
                    err_msg = str(error)
                    is_daily_exhausted = "RESOURCE_EXHAUSTED" in err_msg and (
                        "limit:" in err_msg or "free_tier" in err_msg or "quota" in err_msg.lower()
                    )
                    if is_daily_exhausted:
                        self._mark_client_exhausted(client_idx)
                        if total_clients > 1 and len(self._exhausted_client_indices) < total_clients:
                            self._notify(
                                f"⚠️ API key #{client_idx + 1} kehabisan kuota harian. Beralih ke API key lain..."
                            )
                            continue
                        raise RuntimeError(
                            f"Semua API key Gemini ({total_clients} key) telah mencapai batas kuota harian free tier (20 request/hari). "
                            "Silakan gunakan API key baru / Pay-as-you-go atau tunggu reset kuota harian dari Google."
                        )

                    is_retry = _is_retryable(error)
                    self.attempt_history.append(
                        {
                            "model": candidate_model,
                            "attempt": attempt + 1,
                            "success": False,
                            "error": err_msg,
                            "retryable": is_retry,
                        }
                    )

                    if attempt + 1 < self.max_retries and is_retry:
                        delay = self.backoff_seconds * (2**attempt) + random.uniform(
                            0, max(0.01, self.backoff_seconds / 2) if self.backoff_seconds > 0 else 0
                        )
                        self._notify(
                            f"⚠️ Model `{candidate_model}` terkendala ({error}). Menunggu jeda {delay:.1f} detik untuk retry..."
                        )
                        if delay > 0:
                            time.sleep(delay)
                        continue
                    if candidates.index(candidate_model) + 1 < len(candidates):
                        next_model = candidates[candidates.index(candidate_model) + 1]
                        self._notify(
                            f"🔄 Percobaan pada `{candidate_model}` selesai. Beralih ke model cadangan: `{next_model}`..."
                        )
                    break

        models_tried = ", ".join(candidates)
        raise RuntimeError(f"Semua model Gemini ({models_tried}) gagal setelah retry. Error terakhir: {last_error}") from last_error

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
