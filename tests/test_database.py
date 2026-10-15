"""Tests for SQLite database storage: screening history and market listings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from src.analyze import FinalAnalysis, InspectionChecklist, NegotiationAdvice
from src.cache import SQLiteAnalysisCache
from src.extract import ComparableListing
from src.pipeline import ScreeningPipelineResult
from src.search_provider import GroundingSource


def make_dummy_result(
    comparables: list[ComparableListing],
    score: int | None = 85,
    recommendation: str = "Layak dipertimbangkan",
) -> ScreeningPipelineResult:
    return ScreeningPipelineResult(
        comparables=comparables,
        rejected=[],
        sources=[GroundingSource("https://olx.co.id/item/1", "olx.co.id")],
        queries=["Toyota Avanza 2020"],
        fetched_at=datetime.now().isoformat(),
        statistics={
            "count": len(comparables),
            "price_low": 180_000_000,
            "price_median": 200_000_000,
            "price_high": 220_000_000,
            "p25": 190_000_000,
            "p75": 210_000_000,
            "average_km_per_year": 15000.0,
            "price_spread_ratio": 0.15,
        },
        screening={
            "score": score,
            "label": "Wajar",
            "recommendation": recommendation,
        },
        final_analysis=FinalAnalysis(
            kesimpulan="Unit dalam kondisi harga wajar pasar.",
            saran_nego=NegotiationAdvice(awal=190_000_000, realistis=200_000_000, batas=210_000_000),
            red_flag=[],
            pertanyaan_untuk_penjual=["Apakah riwayat servis ada?"],
            checklist_inspeksi=InspectionChecklist(dokumen=["STNK", "BPKB"], fisik=["Ban", "Mesin"], transmisi=["Oli matic"]),
            catatan=["Catatan pengujian."],
        ),
        confidence={"level": "Tinggi", "reason": "Data cukup"},
        warning=None,
        ai_error=None,
        cache_hit=False,
        daily_count=1,
        usage={"grounding_calls": 2, "input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        description_skipped=False,
        model_used="gemini-2.5-flash",
        completion_status="selesai dengan data",
    )


class TestDatabaseStorage:
    def test_save_and_retrieve_screening_record(self, tmp_path: Path):
        cache = SQLiteAnalysisCache(tmp_path / "test_db.sqlite3")
        user_input = {
            "brand": "Toyota",
            "model": "Avanza",
            "variant": "G",
            "year": 2020,
            "transmission": "AT",
            "kilometer": 50000,
            "listing_price": 205_000_000,
            "location": "Jakarta Pusat",
        }
        comp1 = ComparableListing(
            brand="Toyota",
            model="Avanza",
            variant="G",
            year=2020,
            kilometer=48000,
            transmission="AT",
            location="Jakarta Pusat",
            seller_type="Individu",
            price=200_000_000,
            source_url="https://olx.co.id/item/101",
            fetched_at=datetime.now().isoformat(),
        )
        comp2 = ComparableListing(
            brand="Toyota",
            model="Avanza",
            variant="G",
            year=2020,
            kilometer=55000,
            transmission="AT",
            location="Jakarta Selatan",
            seller_type="Dealer",
            price=210_000_000,
            source_url="https://carmudi.co.id/item/202",
            fetched_at=datetime.now().isoformat(),
        )
        dummy_result = make_dummy_result([comp1, comp2], score=82)

        analysis_id = cache.save_screening_record(user_input, dummy_result)
        assert analysis_id is not None
        assert len(analysis_id) > 10

        # Retrieve history
        history = cache.list_screening_history(limit=10)
        assert len(history) == 1
        record = history[0]
        assert record["id"] == analysis_id
        assert record["brand"] == "Toyota"
        assert record["model"] == "Avanza"
        assert record["year"] == 2020
        assert record["screening_score"] == 82
        assert record["listing_price"] == 205_000_000
        assert record["comparables_count"] == 2
        assert record["completion_status"] == "selesai dengan data"

        # Retrieve detail with parsed JSON
        detail = cache.get_screening_record(analysis_id)
        assert detail is not None
        assert "result_data" in detail
        data = detail["result_data"]
        assert data["user_input"]["model"] == "Avanza"
        assert len(data["comparables"]) == 2
        assert data["kesimpulan"] == "Unit dalam kondisi harga wajar pasar."

        # Market listings table check
        assert cache.count_market_listings() == 2
        listings = cache.list_market_listings()
        assert len(listings) == 2
        domains = {item["domain"] for item in listings}
        assert "olx.co.id" in domains
        assert "carmudi.co.id" in domains

    def test_market_listings_upsert_on_conflict(self, tmp_path: Path):
        cache = SQLiteAnalysisCache(tmp_path / "test_db.sqlite3")
        user_input = {
            "brand": "Toyota",
            "model": "Yaris",
            "variant": "GR",
            "year": 2022,
            "transmission": "AT",
            "kilometer": 30000,
            "listing_price": 250_000_000,
            "location": "Jakarta Barat",
        }
        url = "https://olx.co.id/mobil/yaris-2022-super-mulus"
        comp1 = ComparableListing(
            brand="Toyota",
            model="Yaris",
            variant="GR",
            year=2022,
            kilometer=28000,
            transmission="AT",
            location="Jakarta Barat",
            seller_type="Individu",
            price=245_000_000,
            source_url=url,
            fetched_at="2026-10-14T10:00:00",
        )
        res1 = make_dummy_result([comp1])
        cache.save_screening_record(user_input, res1)
        assert cache.count_market_listings() == 1

        # Save again with same URL but updated price and timestamp
        comp2 = ComparableListing(
            brand="Toyota",
            model="Yaris",
            variant="GR",
            year=2022,
            kilometer=29000,
            transmission="AT",
            location="Jakarta Barat",
            seller_type="Individu",
            price=240_000_000,
            source_url=url,
            fetched_at="2026-10-15T10:00:00",
        )
        res2 = make_dummy_result([comp2])
        cache.save_screening_record(user_input, res2)

        # Still 1 unique listing, but price updated to 240_000_000
        assert cache.count_market_listings() == 1
        listings = cache.list_market_listings(brand="Toyota", model="Yaris")
        assert len(listings) == 1
        assert listings[0]["price"] == 240_000_000
        assert listings[0]["scraped_at"] == "2026-10-15T10:00:00"

    def test_filter_market_listings(self, tmp_path: Path):
        cache = SQLiteAnalysisCache(tmp_path / "test_db.sqlite3")
        user_input = {"brand": "Honda", "model": "Civic", "variant": "Turbo", "year": 2019, "transmission": "AT", "kilometer": 60000, "listing_price": 350_000_000, "location": "Jakarta Selatan"}
        comp_civic = ComparableListing(
            brand="Honda",
            model="Civic",
            variant="Turbo",
            year=2019,
            kilometer=58000,
            transmission="AT",
            location="Jakarta Selatan",
            seller_type="Individu",
            price=340_000_000,
            source_url="https://carmudi.co.id/civic-turbo-2019",
            fetched_at="2026-10-15T10:00:00",
        )
        comp_brio = ComparableListing(
            brand="Honda",
            model="Brio",
            variant="RS",
            year=2021,
            kilometer=30000,
            transmission="AT",
            location="Jakarta Barat",
            seller_type="Dealer",
            price=160_000_000,
            source_url="https://mobil123.com/brio-rs-2021",
            fetched_at="2026-10-15T10:00:00",
        )
        res = make_dummy_result([comp_civic, comp_brio])
        cache.save_screening_record(user_input, res)

        # Filter by model
        civic_only = cache.list_market_listings(model="Civic")
        assert len(civic_only) == 1
        assert civic_only[0]["model"] == "Civic"

        # Filter by domain
        mobil123_only = cache.list_market_listings(domain="mobil123.com")
        assert len(mobil123_only) == 1
        assert mobil123_only[0]["model"] == "Brio"

    def test_delete_screening_record(self, tmp_path: Path):
        cache = SQLiteAnalysisCache(tmp_path / "test_db.sqlite3")
        user_input = {"brand": "Daihatsu", "model": "Xenia", "variant": "R", "year": 2020, "transmission": "MT", "kilometer": 70000, "listing_price": 140_000_000, "location": "Depok"}
        res = make_dummy_result([])
        analysis_id = cache.save_screening_record(user_input, res)
        assert len(cache.list_screening_history()) == 1

        cache.delete_screening_record(analysis_id)
        assert len(cache.list_screening_history()) == 0
        assert cache.get_screening_record(analysis_id) is None
