"""Unit tests for tiered comparable screening, catalog URL building, slug prefiltering, and low comparables logic."""

import pytest
from src.extract import ExtractedListing
from src.marketplace_config import (
    MARKETPLACE_CATALOG_CONFIGS,
    build_catalog_url,
    build_manual_search_urls,
    extract_year_from_slug,
    get_twin_models,
)
from src.marketplace_fetcher import is_slug_year_acceptable
from src.pipeline import _fallback_analysis
from src.scoring import negotiation_advice_text
from src.stats import calculate_statistics, calculate_weighted_percentile
from src.tiered_screening import evaluate_tiered_comparables, is_transmission_match


def test_build_catalog_url_per_domain():
    # Carmudi: min_year, max_year, transmission, page_number
    carmudi_url = build_catalog_url(
        "carmudi.co.id",
        brand="Daihatsu",
        model="Xenia",
        year=2020,
        min_year=2019,
        max_year=2021,
        transmission="AT",
        page=2,
    )
    assert "carmudi.co.id/mobil-bekas-dijual/daihatsu/xenia/indonesia" in carmudi_url
    assert "transmission=automatic" in carmudi_url
    assert "min_year=2019" in carmudi_url
    assert "max_year=2021" in carmudi_url
    assert "page_number=2" in carmudi_url

    # Carsome: year, transmission, pageNo
    carsome_url = build_catalog_url(
        "carsome.id",
        brand="Daihatsu",
        model="Xenia",
        year=2020,
        transmission="AT",
        page=3,
    )
    assert "carsome.id/beli-mobil-bekas/daihatsu/xenia" in carsome_url
    assert "year=2020" in carsome_url
    assert "transmission=automatic" in carsome_url
    assert "pageNo=3" in carsome_url

    # Oto.com
    oto_url = build_catalog_url("oto.com", brand="Toyota", model="Avanza", year=2020)
    assert "oto.com/mobil-bekas/toyota-avanza-2020" in oto_url

    # OLX
    olx_url = build_catalog_url("olx.co.id", brand="Honda", model="Brio", year=2020)
    assert "olx.co.id/mobil-bekas_c198/q-honda-brio-2020" in olx_url

    # Mobil123
    m123_url = build_catalog_url("mobil123.com", brand="Daihatsu", model="Xenia", transmission="MT", page=1)
    assert "mobil123.com/mobil-dijual/daihatsu/xenia/indonesia" in m123_url
    assert "transmission=manual" in m123_url

    # Mobilbekas.com
    mb_url = build_catalog_url("mobilbekas.com", brand="Daihatsu", model="Xenia", year=2020)
    assert "mobilbekas.com/search?keyword=daihatsu+xenia+2020" in mb_url


def test_slug_year_extraction_and_prefiltering():
    # Slug with year
    url1 = "https://www.carsome.id/beli-mobil-bekas/daihatsu/xenia/2020-daihatsu-xenia-r-1.3/abc"
    assert extract_year_from_slug(url1) == 2020
    assert is_slug_year_acceptable(url1, target_year=2020, tolerance=1) is True

    # Slug with different year within tolerance (2021 vs 2020)
    url2 = "https://www.carmudi.co.id/dijual/daihatsu-xenia-r-deluxe-2021/12345"
    assert extract_year_from_slug(url2) == 2021
    assert is_slug_year_acceptable(url2, target_year=2020, tolerance=1) is True

    # Slug with out-of-range year (2016 vs 2020)
    url3 = "https://www.carmudi.co.id/dijual/daihatsu-xenia-r-deluxe-2016/12345"
    assert extract_year_from_slug(url3) == 2016
    assert is_slug_year_acceptable(url3, target_year=2020, tolerance=1) is False

    # Slug without explicit year
    url4 = "https://www.olx.co.id/item/daihatsu-xenia-istimewa-iid-123456"
    assert extract_year_from_slug(url4) is None
    assert is_slug_year_acceptable(url4, target_year=2020, tolerance=1) is True


def test_transmission_strict_matching():
    # Strict matching - transmission is never relaxed
    assert is_transmission_match("Automatic", "AT") is True
    assert is_transmission_match("AT", "AT") is True
    assert is_transmission_match("Otomatis", "AT") is True
    assert is_transmission_match("Manual", "AT") is False
    assert is_transmission_match("MT", "AT") is False
    assert is_transmission_match("Manual", "MT") is True
    assert is_transmission_match("Automatic", "MT") is False
    assert is_transmission_match("", "AT") is False
    assert is_transmission_match(None, "AT") is False


def test_twin_models_map():
    assert "avanza" in get_twin_models("xenia")
    assert "xenia" in get_twin_models("avanza")
    assert "calya" in get_twin_models("sigra")
    assert "agya" in get_twin_models("brio")


def test_tiered_fallback_logic():
    # Helper to generate dummy ExtractedListing
    def _make_candidate(url, model, variant, year, trans, price="150000000", km="50000", loc="Jakarta Selatan"):
        return ExtractedListing(
            brand="Daihatsu" if "xenia" in model.lower() else "Toyota",
            model=model,
            variant=variant,
            year=str(year),
            kilometer=km,
            transmission=trans,
            location=loc,
            seller_type="dealer",
            price=price,
            source_url=url,
        )

    # 1. Tier 1 Test: 5 candidates with exact model, year, variant, trans
    tier1_candidates = [
        _make_candidate(f"http://test.com/t1_{i}", "Xenia", "E", 2020, "AT")
        for i in range(5)
    ]
    res1 = evaluate_tiered_comparables(
        tier1_candidates,
        expected_brand="Daihatsu",
        expected_model="Xenia",
        expected_year=2020,
        expected_transmission="AT",
        expected_variant="E",
        target_count=5,
    )
    assert res1.active_tier == 1
    assert len(res1.listings) == 5
    assert all(item.weight == 1.0 for item in res1.listings)

    # 2. Tier 2 Test: exact model and year, but variants differ
    tier2_candidates = [
        _make_candidate(f"http://test.com/t2_{i}", "Xenia", f"Variant_{i}", 2020, "AT")
        for i in range(5)
    ]
    res2 = evaluate_tiered_comparables(
        tier2_candidates,
        expected_brand="Daihatsu",
        expected_model="Xenia",
        expected_year=2020,
        expected_transmission="AT",
        expected_variant="E",
        target_count=5,
    )
    assert res2.active_tier == 2
    assert len(res2.listings) == 5
    assert all(item.weight == 0.7 for item in res2.listings)

    # 3. Tier 3 Test: exact model, but year is +/- 1
    tier3_candidates = [
        _make_candidate(f"http://test.com/t3_{i}", "Xenia", "E", 2019 if i % 2 == 0 else 2021, "AT")
        for i in range(5)
    ]
    res3 = evaluate_tiered_comparables(
        tier3_candidates,
        expected_brand="Daihatsu",
        expected_model="Xenia",
        expected_year=2020,
        expected_transmission="AT",
        expected_variant="E",
        target_count=5,
    )
    assert res3.active_tier == 3
    assert len(res3.listings) == 5
    assert all(item.weight == 0.8 for item in res3.listings)

    # 4. Tier 4 Test: twin model (Avanza for Xenia) when primary model < 3
    tier4_candidates = [
        _make_candidate("http://test.com/xenia_1", "Xenia", "E", 2020, "AT"),
        _make_candidate("http://test.com/avanza_1", "Avanza", "G", 2020, "AT"),
        _make_candidate("http://test.com/avanza_2", "Avanza", "E", 2020, "AT"),
    ]
    res4 = evaluate_tiered_comparables(
        tier4_candidates,
        expected_brand="Daihatsu",
        expected_model="Xenia",
        expected_year=2020,
        expected_transmission="AT",
        expected_variant="E",
        target_count=5,
    )
    assert res4.active_tier == 4
    assert any(item.is_twin_model for item in res4.listings)
    twin_item = next(item for item in res4.listings if item.is_twin_model)
    assert twin_item.weight == 0.5

    # 5. Transmission non-negotiability: MT candidates are excluded
    mt_candidates = [
        _make_candidate(f"http://test.com/mt_{i}", "Xenia", "E", 2020, "MT")
        for i in range(5)
    ]
    res_mt = evaluate_tiered_comparables(
        mt_candidates,
        expected_brand="Daihatsu",
        expected_model="Xenia",
        expected_year=2020,
        expected_transmission="AT",
        expected_variant="E",
    )
    assert len(res_mt.listings) == 0


def test_weighted_statistics_and_iqr():
    from src.extract import ComparableListing

    def _comp(price, weight=1.0):
        return ComparableListing(
            brand="Toyota",
            model="Avanza",
            variant="G",
            year=2020,
            kilometer=50000,
            transmission="AT",
            location="Jakarta Pusat",
            seller_type="dealer",
            price=price,
            source_url=f"http://test.com/{price}",
            fetched_at="2026-10-16T12:00:00Z",
            weight=weight,
        )


    # For n < 5, IQR outlier removal must not be applied
    small_list = [_comp(100_000_000), _comp(150_000_000), _comp(160_000_000), _comp(500_000_000)]
    stats_small = calculate_statistics(small_list, target_year=2020)
    assert stats_small["count"] == 4  # 500M is not removed because n < 5

    # For n >= 5, extreme outlier is removed by IQR
    large_list = [
        _comp(150_000_000),
        _comp(152_000_000),
        _comp(155_000_000),
        _comp(158_000_000),
        _comp(160_000_000),
        _comp(1_500_000_000),  # Extreme outlier
    ]
    stats_large = calculate_statistics(large_list, target_year=2020)
    assert stats_large["count"] == 5
    assert stats_large["price_median"] < 200_000_000


def test_negotiation_advice_text_for_low_comparables():
    # When n < 3, must return exact requirement text
    for n in (0, 1, 2):
        advice = negotiation_advice_text(price_diff_pct=0.0, n_comparables=n)
        assert advice is not None
        assert "Belum ada cukup pembanding untuk saran tawar" in advice
        assert "3-10% di bawah listing" in advice

    # When n >= 3 and price_diff_pct <= 0 (cheap)
    advice_cheap = negotiation_advice_text(price_diff_pct=-0.15, n_comparables=5)
    assert "Harga sudah di bawah pembanding" in advice_cheap

    # When n >= 3 and price_diff_pct > 0 (expensive)
    assert negotiation_advice_text(price_diff_pct=0.10, n_comparables=5) is None


def test_fallback_analysis_for_zero_comparables():
    user_input = {"listing_price": 100_000_000, "transmission": "AT"}
    stats = {"count": 0, "price_median": 0}
    analysis = _fallback_analysis(user_input, stats, rule_flags=[])
    assert "Belum ada pembanding valid" in analysis.kesimpulan
    assert analysis.saran_nego.awal is None


def test_manual_search_urls_builder():
    urls = build_manual_search_urls("Daihatsu", "Xenia", 2020, "AT")
    assert "Carmudi" in urls
    assert "Carsome" in urls
    assert "Oto.com" in urls
    assert "OLX" in urls
    assert "Mobil123" in urls
    assert "Mobilbekas" in urls
    assert "2020" in urls["Carsome"]
    assert "min_year=2019" in urls["Carmudi"]
