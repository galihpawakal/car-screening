"""SQLite cache and daily analysis/token accounting."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class DailyLimitExceeded(RuntimeError):
    pass


class SQLiteAnalysisCache:
    def __init__(self, path: str | Path, ttl_seconds: int = 86_400):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS query_cache (
                cache_key TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                payload_json TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS daily_limits (
                day TEXT PRIMARY KEY,
                analysis_count INTEGER NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS usage_logs (
                analysis_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                grounding_calls INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS screening_history (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                variant TEXT NOT NULL,
                year INTEGER NOT NULL,
                transmission TEXT NOT NULL,
                location TEXT NOT NULL,
                kilometer INTEGER NOT NULL,
                listing_price INTEGER NOT NULL,
                price_median INTEGER,
                p25 INTEGER,
                p75 INTEGER,
                comparables_count INTEGER NOT NULL,
                screening_score INTEGER,
                recommendation TEXT,
                market_confidence TEXT,
                completion_status TEXT NOT NULL,
                model_used TEXT,
                result_json TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS market_listings (
                id TEXT PRIMARY KEY,
                screening_id TEXT,
                source_url TEXT NOT NULL UNIQUE,
                domain TEXT NOT NULL,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                variant TEXT NOT NULL,
                year INTEGER,
                kilometer INTEGER,
                transmission TEXT NOT NULL,
                location TEXT NOT NULL,
                price INTEGER NOT NULL,
                seller_type TEXT NOT NULL,
                scraped_at TEXT NOT NULL,
                FOREIGN KEY(screening_id) REFERENCES screening_history(id)
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_history_created_at ON screening_history(created_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_market_listings_car ON market_listings(brand, model, year)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_market_listings_domain ON market_listings(domain)")

    def get(self, cache_key: str, *, now: float | None = None) -> dict | None:
        moment = now if now is not None else datetime.now(timezone.utc).timestamp()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT created_at, payload_json FROM query_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            if moment - row["created_at"] >= self.ttl_seconds:
                connection.execute("DELETE FROM query_cache WHERE cache_key = ?", (cache_key,))
                return None
            return json.loads(row["payload_json"])

    def set(self, cache_key: str, payload: dict, *, now: float | None = None) -> None:
        moment = now if now is not None else datetime.now(timezone.utc).timestamp()
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO query_cache(cache_key, created_at, payload_json) VALUES (?, ?, ?)",
                (cache_key, moment, encoded),
            )

    def claim_daily_analysis(self, limit: int, *, day: date | None = None) -> int:
        if limit < 1:
            raise ValueError("DAILY_ANALYSIS_LIMIT harus minimal 1.")
        day_key = (day or datetime.now().date()).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT analysis_count FROM daily_limits WHERE day = ?", (day_key,)).fetchone()
            count = int(row["analysis_count"]) if row else 0
            if count >= limit:
                connection.rollback()
                raise DailyLimitExceeded(f"Batas analisa harian ({limit}) sudah tercapai. Coba lagi besok.")
            count += 1
            connection.execute(
                "INSERT OR REPLACE INTO daily_limits(day, analysis_count) VALUES (?, ?)",
                (day_key, count),
            )
            connection.commit()
            return count

    def record_usage(self, *, grounding_calls: int, input_tokens: int, output_tokens: int) -> str:
        analysis_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO usage_logs VALUES (?, ?, ?, ?, ?)",
                (analysis_id, datetime.now(timezone.utc).isoformat(), grounding_calls, input_tokens, output_tokens),
            )
        return analysis_id

    def recent_usage(self, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM usage_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def save_screening_record(self, user_input: dict, result: Any) -> str:
        analysis_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        brand = str(user_input.get("brand") or "").strip()
        model = str(user_input.get("model") or "").strip()
        variant = str(user_input.get("variant") or "").strip()
        year = int(user_input.get("year") or 0)
        transmission = str(user_input.get("transmission") or "").strip()
        location = str(user_input.get("location") or "").strip()
        kilometer = int(user_input.get("kilometer") or 0)
        listing_price = int(user_input.get("listing_price") or 0)

        stats = getattr(result, "statistics", {}) or {}
        price_median = stats.get("price_median")
        p25 = stats.get("p25")
        p75 = stats.get("p75")
        comparables = getattr(result, "comparables", []) or []
        comparables_count = len(comparables)

        screening = getattr(result, "screening", {}) or {}
        screening_score = screening.get("score")
        if isinstance(screening_score, (int, float)):
            score_val = int(screening_score)
        else:
            score_val = None

        recommendation = screening.get("recommendation") or screening.get("label") or ""
        confidence = getattr(result, "confidence", {}) or {}
        market_confidence = confidence.get("level") or ""
        completion_status = getattr(result, "completion_status", "selesai")
        model_used = getattr(result, "model_used", "") or ""

        final_obj = getattr(result, "final_analysis", None)
        checklist_obj = getattr(final_obj, "checklist_inspeksi", None)

        def _to_plain(val: Any) -> Any:
            if hasattr(val, "model_dump"):
                return val.model_dump(mode="json")
            if isinstance(val, dict):
                return val
            try:
                return asdict(val)
            except Exception:
                return str(val)

        summary_data = {
            "user_input": user_input,
            "statistics": stats,
            "screening": screening,
            "confidence": confidence,
            "completion_status": completion_status,
            "model_used": model_used,
            "fetched_at": getattr(result, "fetched_at", ""),
            "comparables_count": comparables_count,
            "warning": getattr(result, "warning", None),
            "ai_error": getattr(result, "ai_error", None),
            "kesimpulan": getattr(final_obj, "kesimpulan", "") if final_obj else "",
            "saran_nego": _to_plain(getattr(final_obj, "saran_nego", {})) if final_obj else {},
            "red_flag": getattr(final_obj, "red_flag", []) if final_obj else [],
            "checklist": {
                "dokumen": getattr(checklist_obj, "dokumen", []) if checklist_obj else [],
                "fisik": getattr(checklist_obj, "fisik", []) if checklist_obj else [],
                "transmisi": getattr(checklist_obj, "transmisi", []) if checklist_obj else [],
            },
            "comparables": [_to_plain(item) for item in comparables],
        }
        result_json = json.dumps(summary_data, ensure_ascii=False)

        with self._connect() as connection:
            connection.execute(
                """INSERT INTO screening_history (
                    id, created_at, brand, model, variant, year, transmission,
                    location, kilometer, listing_price, price_median, p25, p75,
                    comparables_count, screening_score, recommendation,
                    market_confidence, completion_status, model_used, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id, created_at, brand, model, variant, year, transmission,
                    location, kilometer, listing_price, price_median, p25, p75,
                    comparables_count, score_val, recommendation,
                    market_confidence, completion_status, model_used, result_json,
                ),
            )

            for comp in comparables:
                source_url = getattr(comp, "source_url", None) or (comp.get("source_url") if isinstance(comp, dict) else "")
                if not source_url:
                    continue
                listing_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_url))
                domain = urlsplit(source_url).netloc
                c_brand = getattr(comp, "brand", None) or (comp.get("brand") if isinstance(comp, dict) else brand)
                c_model = getattr(comp, "model", None) or (comp.get("model") if isinstance(comp, dict) else model)
                c_variant = getattr(comp, "variant", None) or (comp.get("variant") if isinstance(comp, dict) else "")
                c_year = getattr(comp, "year", None) or (comp.get("year") if isinstance(comp, dict) else None)
                c_km = getattr(comp, "kilometer", None) or (comp.get("kilometer") if isinstance(comp, dict) else None)
                c_trans = getattr(comp, "transmission", None) or (comp.get("transmission") if isinstance(comp, dict) else "")
                c_loc = getattr(comp, "location", None) or (comp.get("location") if isinstance(comp, dict) else "")
                c_price = getattr(comp, "price", 0) or (comp.get("price", 0) if isinstance(comp, dict) else 0)
                c_seller = getattr(comp, "seller_type", "") or (comp.get("seller_type", "") if isinstance(comp, dict) else "")
                c_scraped = getattr(comp, "fetched_at", "") or created_at

                connection.execute(
                    """INSERT INTO market_listings (
                        id, screening_id, source_url, domain, brand, model, variant,
                        year, kilometer, transmission, location, price, seller_type, scraped_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_url) DO UPDATE SET
                        screening_id = excluded.screening_id,
                        price = excluded.price,
                        kilometer = COALESCE(excluded.kilometer, market_listings.kilometer),
                        scraped_at = excluded.scraped_at
                    """,
                    (
                        listing_id, analysis_id, source_url, domain, str(c_brand),
                        str(c_model), str(c_variant), c_year, c_km, str(c_trans),
                        str(c_loc), int(c_price), str(c_seller), str(c_scraped),
                    ),
                )

        return analysis_id

    def list_screening_history(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, created_at, brand, model, variant, year, transmission,
                          location, kilometer, listing_price, price_median, p25, p75,
                          comparables_count, screening_score, recommendation,
                          market_confidence, completion_status, model_used
                   FROM screening_history
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_screening_record(self, record_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM screening_history WHERE id = ?", (record_id,)
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("result_json"):
                data["result_data"] = json.loads(data["result_json"])
            return data

    def delete_screening_record(self, record_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM screening_history WHERE id = ?", (record_id,))

    def list_market_listings(
        self,
        *,
        brand: str | None = None,
        model: str | None = None,
        domain: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        query = "SELECT * FROM market_listings WHERE 1=1"
        params: list[Any] = []
        if brand:
            query += " AND LOWER(brand) = LOWER(?)"
            params.append(brand.strip())
        if model:
            query += " AND LOWER(model) = LOWER(?)"
            params.append(model.strip())
        if domain:
            query += " AND LOWER(domain) LIKE LOWER(?)"
            params.append(f"%{domain.strip()}%")
        query += " ORDER BY scraped_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def count_market_listings(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM market_listings").fetchone()
            return int(row["total"]) if row else 0
