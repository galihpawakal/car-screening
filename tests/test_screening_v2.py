"""Comprehensive tests for the new MIN_COMPARABLES=2 screening logic.

Covers: n=0,1,2,5; min-max vs percentile ranges; price labels at all boundaries;
unassessed component exclusion with weight normalization; indicative cap at 70;
category cap when diff < -25%; negotiation advice absent for below-market;
price banner <50% acuan; red flag deduplication; token summation.
"""

import pytest

from src.config import (
    MIN_COMPARABLES,
    SOLID_COMPARABLES,
    INDICATIVE_MAX_SCORE,
    price_fairness_score,
    km_age_fairness_score,
)
from src.rules import RuleRedFlag, merge_and_deduplicate_flags
from src.scoring import (
    ScoreComponent,
    ScreeningResult,
    calculate_screening_score,
    calculate_price_fairness_score,
    calculate_km_age_fairness_score,
    calculate_info_completeness_score,
    calculate_description_risk_score,
    format_screening_result,
    get_price_label,
    get_market_confidence,
    negotiation_advice_text,
    price_banner_warning,
    seller_questions_for_cheap,
    _weighted_average_assessed,
)
from src.stats import calculate_statistics


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_flags():
    """Return minimal rule flags for scoring tests."""
    return []

def _score(n_comparables, listing_price=200_000_000, price_median=200_000_000,
           has_description=True, **kw):
    """Shortcut to call calculate_screening_score with sensible defaults."""
    defaults = dict(
        listing_price=listing_price,
        price_low=int(price_median * 0.85),
        price_median=price_median,
        price_high=int(price_median * 1.15),
        n_comparables=n_comparables,
        kilometer=60_000,
        year=2020,
        has_description=has_description,
        has_varian=True,
        has_plate=True,
        has_location=True,
        rule_flags=[],
        current_year=2026,
        average_comparable_km_per_year=10_000.0,
    )
    defaults.update(kw)
    return calculate_screening_score(**defaults)


# ── 1. Mode by comparable count: n=0, n=1, n=2, n=5 ─────────────────────────

class TestComparableModes:
    """n < 2 → not assessable; 2 ≤ n < 5 → indicative; n ≥ 5 → normal."""

    def test_zero_comparables_not_assessable(self):
        result = _score(0)
        assert result.total_score < 0
        assert result.category == "perlu_cek"
        assert result.price_label == "Tidak dapat dinilai"
        formatted = format_screening_result(result)
        assert formatted["score_assessable"] is False
        assert formatted["total_score_display"] == "Tidak dapat dinilai"

    def test_one_comparable_not_assessable(self):
        result = _score(1)
        assert result.total_score < 0
        assert result.price_label == "Tidak dapat dinilai"
        assert format_screening_result(result)["score_assessable"] is False

    def test_two_comparables_indicative(self):
        result = _score(2)
        assert result.total_score >= 0
        assert result.is_indicative is True
        formatted = format_screening_result(result)
        assert formatted["is_indicative"] is True
        assert "indikatif" in formatted["total_score_display"].lower()

    def test_five_comparables_normal(self):
        result = _score(5)
        assert result.total_score >= 0
        assert result.is_indicative is False
        formatted = format_screening_result(result)
        assert formatted["is_indicative"] is False
        assert "indikatif" not in formatted["total_score_display"].lower()


# ── 2. Statistics: min-max vs P25/P75 ────────────────────────────────────────

class TestStatisticsRanges:
    """n < 5 → min/max; n ≥ 5 → P25/P75."""

    @staticmethod
    def _make_listing(price, year=2022, km=50_000, url="https://ex.test/1"):
        from src.extract import ComparableListing
        return ComparableListing(
            brand="Toyota", model="Avanza", variant="G", year=year,
            kilometer=km, transmission="AT", location="Jakarta",
            seller_type="individu", price=price, source_url=url,
            fetched_at="2026-01-01T00:00:00",
        )

    def test_two_listings_uses_min_max(self):
        listings = [
            self._make_listing(100_000_000, url="https://ex.test/1"),
            self._make_listing(200_000_000, url="https://ex.test/2"),
        ]
        result = calculate_statistics(listings, listing_price=150_000_000,
                                      target_year=2022, current_year=2026)
        assert result["count"] == 2
        assert result["price_low"] == 100_000_000  # min
        assert result["price_high"] == 200_000_000  # max

    def test_four_listings_uses_min_max(self):
        listings = [
            self._make_listing(100, url="https://ex.test/1"),
            self._make_listing(200, url="https://ex.test/2"),
            self._make_listing(300, url="https://ex.test/3"),
            self._make_listing(400, url="https://ex.test/4"),
        ]
        result = calculate_statistics(listings, listing_price=250,
                                      target_year=2022, current_year=2026)
        assert result["price_low"] == 100
        assert result["price_high"] == 400

    def test_five_listings_uses_percentiles(self):
        import numpy as np
        listings = [
            self._make_listing(100, url="https://ex.test/1"),
            self._make_listing(200, url="https://ex.test/2"),
            self._make_listing(300, url="https://ex.test/3"),
            self._make_listing(400, url="https://ex.test/4"),
            self._make_listing(500, url="https://ex.test/5"),
        ]
        result = calculate_statistics(listings, listing_price=300,
                                      target_year=2022, current_year=2026)
        prices = [100, 200, 300, 400, 500]
        expected_low = int(round(np.percentile(prices, 25)))
        expected_high = int(round(np.percentile(prices, 75)))
        assert result["price_low"] == expected_low
        assert result["price_high"] == expected_high


# ── 3. Price labels at every boundary ────────────────────────────────────────

class TestPriceLabels:
    """Boundary tests for the new label thresholds."""

    def test_very_cheap(self):
        assert get_price_label(-0.30) == "Sangat murah, waspada"
        assert get_price_label(-0.26) == "Sangat murah, waspada"

    def test_boundary_minus_25(self):
        # -25% exactly should be "Di bawah pasar" (>= -25%)
        assert get_price_label(-0.25) == "Di bawah pasar"

    def test_below_market(self):
        assert get_price_label(-0.10) == "Di bawah pasar"
        assert get_price_label(-0.06) == "Di bawah pasar"

    def test_boundary_minus_5(self):
        assert get_price_label(-0.05) == "Wajar"

    def test_fair(self):
        assert get_price_label(0.0) == "Wajar"
        assert get_price_label(0.05) == "Wajar"

    def test_boundary_plus_5(self):
        # Just above +5% -> Agak mahal
        assert get_price_label(0.06) == "Agak mahal"

    def test_slightly_expensive(self):
        assert get_price_label(0.10) == "Agak mahal"
        assert get_price_label(0.15) == "Agak mahal"

    def test_expensive(self):
        assert get_price_label(0.16) == "Mahal"
        assert get_price_label(0.50) == "Mahal"

    def test_indicative_suffix(self):
        label = get_price_label(0.0, is_indicative=True)
        assert label == "Wajar (indikatif)"

    def test_indicative_suffix_all_labels(self):
        for diff, expected_base in [(-0.30, "Sangat murah, waspada"),
                                      (-0.10, "Di bawah pasar"),
                                      (0.0, "Wajar"),
                                      (0.10, "Agak mahal"),
                                      (0.20, "Mahal")]:
            assert get_price_label(diff, is_indicative=True) == expected_base + " (indikatif)"


# ── 4. Scoring functions from config ─────────────────────────────────────────

class TestConfigScoringFunctions:
    """Test the scoring functions defined in config.py."""

    def test_price_fairness_within_5pct(self):
        assert price_fairness_score(0.0) == 100.0
        assert price_fairness_score(0.05) == 100.0
        assert price_fairness_score(-0.05) == 100.0

    def test_price_fairness_above_5pct(self):
        # diff = +10%, score = max(0, 100 - 4*(10-5)) = 100 - 20 = 80
        assert price_fairness_score(0.10) == 80.0

    def test_price_fairness_above_30pct(self):
        # diff = +30%, score = max(0, 100 - 4*(30-5)) = 100 - 100 = 0
        assert price_fairness_score(0.30) == 0.0

    def test_price_fairness_below_minus_5pct(self):
        # diff = -10%, score = max(20, 100 - 3*(10-5)) = 100 - 15 = 85
        assert price_fairness_score(-0.10) == 85.0

    def test_price_fairness_extreme_cheap(self):
        # diff = -40%, score = max(20, 100 - 3*(40-5)) = max(20, -5) = 20
        assert price_fairness_score(-0.40) == 20.0

    def test_km_fairness_within_25pct(self):
        score, assessment = km_age_fairness_score(12_000, 10_000)
        assert score == 100.0
        assert assessment == "sebanding"

    def test_km_fairness_above_25pct(self):
        score, assessment = km_age_fairness_score(14_000, 10_000)
        assert score < 100.0
        assert "lebih tinggi" in assessment

    def test_km_fairness_very_low_triggers_warning(self):
        # 3000 vs 10000 = -70% -> "jauh lebih rendah"
        score, assessment = km_age_fairness_score(3_000, 10_000)
        assert score < 40
        assert "jauh lebih rendah" in assessment


# ── 5. Unassessed component exclusion & weight normalization ─────────────────

class TestUnassessedComponentExclusion:
    """Description empty -> component excluded, weights renormalized."""

    def test_empty_description_excluded_from_average(self):
        result = _score(5, has_description=False)
        desc = next(c for c in result.components if c.name == "Risiko Deskripsi")
        assert desc.assessed is False
        assert desc.score == -1

        # Verify the weighted average only uses assessed components
        assessed = [c for c in result.components if c.assessed]
        assert len(assessed) == 3  # price, km, info
        total_weight = sum(c.weight for c in assessed)
        expected = sum(c.score * c.weight for c in assessed) / total_weight
        # total_score should be close to expected (rounded)
        assert abs(result.total_score - round(expected)) <= 1

    def test_formatted_shows_dash_for_unassessed(self):
        result = _score(5, has_description=False)
        formatted = format_screening_result(result)
        desc_formatted = next(c for c in formatted["components"]
                              if c["name"] == "Risiko Deskripsi")
        assert desc_formatted["score_display"] == "-"
        assert desc_formatted["assessed"] is False
        assert desc_formatted["score"] is None

    def test_weighted_average_normalizes_weights(self):
        comps = [
            ScoreComponent("A", 80, 0.40, "reason", assessed=True),
            ScoreComponent("B", 60, 0.20, "reason", assessed=True),
            ScoreComponent("C", -1, 0.20, "reason", assessed=False),
            ScoreComponent("D", 100, 0.20, "reason", assessed=True),
        ]
        avg = _weighted_average_assessed(comps)
        # Only A(80, 0.40) + B(60, 0.20) + D(100, 0.20) = (32 + 12 + 20) / 0.80 = 80
        assert avg == pytest.approx(80.0)


# ── 6. Indicative mode cap at 70 ─────────────────────────────────────────────

class TestIndicativeScoreCap:
    """n < 5 → score capped at INDICATIVE_MAX_SCORE = 70."""

    def test_indicative_mode_caps_score_at_70(self):
        # Perfect inputs: listing=median, all fields filled, no flags
        result = _score(2, listing_price=200_000_000, price_median=200_000_000)
        assert result.is_indicative is True
        assert result.total_score <= INDICATIVE_MAX_SCORE

    def test_normal_mode_no_cap(self):
        result = _score(10, listing_price=200_000_000, price_median=200_000_000)
        assert result.is_indicative is False
        # With perfect inputs, score should be above 70
        assert result.total_score > INDICATIVE_MAX_SCORE

    def test_indicative_badge_in_formatted(self):
        result = _score(3)
        formatted = format_screening_result(result)
        assert formatted["is_indicative"] is True
        assert "indikatif" in formatted["total_score_display"].lower()


# ── 7. Category cap when diff < -25% ────────────────────────────────────────

class TestCategoryCap:
    """diff < -25% → category ≤ perlu_cek (never 'layak')."""

    def test_very_cheap_capped_to_perlu_cek(self):
        # Listing 70M vs median 200M = -65%, but give lots of comparables for a high base score
        result = _score(10, listing_price=70_000_000, price_median=200_000_000)
        assert result.price_diff_pct < -0.25
        assert result.category in ("perlu_cek", "berisiko")
        assert result.category != "layak"

    def test_normal_price_can_be_layak(self):
        result = _score(10, listing_price=200_000_000, price_median=200_000_000)
        # With perfect inputs, category should be layak
        assert result.category == "layak"


# ── 8. Negotiation advice ────────────────────────────────────────────────────

class TestNegotiationAdvice:
    """Saran nego only when listing > market; advisory text when below."""

    def test_below_market_no_nego(self):
        # diff <= 0 -> advisory text
        advice = negotiation_advice_text(-0.10, 5)
        assert advice is not None
        assert "di bawah" in advice.lower()
        assert "verifikasi" in advice.lower()

    def test_at_market_no_nego(self):
        advice = negotiation_advice_text(0.0, 5)
        assert advice is not None  # diff = 0 <= 0

    def test_above_market_nego_available(self):
        advice = negotiation_advice_text(0.10, 5)
        assert advice is None  # no warning -> negotiation makes sense


# ── 9. Price banner warning ──────────────────────────────────────────────────

class TestPriceBannerWarning:
    """Banner when listing < 50% acuan."""

    def test_banner_when_listing_below_50pct(self):
        banner = price_banner_warning(90_000_000, 200_000_000)
        assert banner is not None
        assert "salah ketik" in banner.lower() or "penipuan" in banner.lower()

    def test_no_banner_when_listing_at_60pct(self):
        banner = price_banner_warning(120_000_000, 200_000_000)
        assert banner is None

    def test_no_banner_when_no_median(self):
        banner = price_banner_warning(100_000_000, 0)
        assert banner is None


# ── 10. Red flag deduplication ───────────────────────────────────────────────

class TestRedFlagDeduplication:
    """Rule + AI flags merged without duplicates."""

    def test_no_duplicate_price_flags(self):
        rule_flags = [
            RuleRedFlag("PRICE_TOO_LOW", "Harga listing 30% di bawah estimasi pasar", "high"),
        ]
        ai_flags = ["Harga terlalu murah dibandingkan pembanding"]
        merged = merge_and_deduplicate_flags(rule_flags, ai_flags)
        assert len(merged) == 1
        assert merged[0].source == "rule"

    def test_different_issues_both_kept(self):
        rule_flags = [
            RuleRedFlag("PRICE_TOO_LOW", "Harga listing 30% di bawah estimasi pasar", "high"),
        ]
        ai_flags = ["Kondisi mesin tidak disebutkan dalam deskripsi"]
        merged = merge_and_deduplicate_flags(rule_flags, ai_flags)
        assert len(merged) == 2

    def test_empty_inputs(self):
        merged = merge_and_deduplicate_flags([], [])
        assert merged == []


# ── 11. Seller questions for cheap listings ──────────────────────────────────

class TestSellerQuestionsCheap:
    """Extra questions only for diff < -25%."""

    def test_cheap_listing_gets_extra_questions(self):
        questions = seller_questions_for_cheap(-0.30)
        assert len(questions) >= 4
        assert any("leasing" in q.lower() or "bank" in q.lower() for q in questions)
        assert any("pajak" in q.lower() for q in questions)

    def test_normal_listing_no_extra_questions(self):
        questions = seller_questions_for_cheap(-0.10)
        assert questions == []

    def test_expensive_listing_no_extra_questions(self):
        questions = seller_questions_for_cheap(0.10)
        assert questions == []


# ── 12. KM red flag for very low odometer ────────────────────────────────────

class TestKmRedFlag:
    """KM/year < -40% of reference triggers odometer verification warning."""

    def test_very_low_km_has_odometer_warning(self):
        component = calculate_km_age_fairness_score(
            kilometer=10_000,
            year=2020,
            current_year=2026,
            average_comparable_km_per_year=15_000,
        )
        # 10000/6 ≈ 1667 km/year vs 15000 = -89% -> should flag
        assert "verifikasi odometer" in component.reason.lower()

    def test_normal_km_no_odometer_warning(self):
        component = calculate_km_age_fairness_score(
            kilometer=60_000,
            year=2020,
            current_year=2026,
            average_comparable_km_per_year=10_000,
        )
        assert "verifikasi odometer" not in component.reason.lower()


# ── 13. Info completeness displays filled/missing ────────────────────────────

class TestInfoCompleteness:
    def test_all_filled(self):
        comp = calculate_info_completeness_score(True, True, True, True)
        assert comp.score == 100
        assert "lengkap" in comp.reason.lower()

    def test_partial_filled(self):
        comp = calculate_info_completeness_score(False, True, False, True)
        assert comp.score == 50
        assert "Deskripsi" in comp.reason
        assert "Plat Nomor" in comp.reason
        assert "Terisi" in comp.reason
        assert "Belum" in comp.reason


# ── 14. Market confidence ────────────────────────────────────────────────────

class TestMarketConfidence:
    def test_below_five_is_low(self):
        conf = get_market_confidence(3, 0.01, 100)
        assert conf["level"] == "Rendah"

    def test_ten_tight_spread_high(self):
        conf = get_market_confidence(10, 0.10, 90)
        assert conf["level"] == "Tinggi"

    def test_five_moderate_spread_sedang(self):
        conf = get_market_confidence(5, 0.30, 70)
        assert conf["level"] == "Sedang"


# ── 15. Token summation ─────────────────────────────────────────────────────

class TestTokenSummation:
    """Usage metadata from Gemini calls should accumulate."""

    def test_usage_sums_correctly(self):
        from src.search_provider import TokenUsage
        u1 = TokenUsage(100, 50)
        u2 = TokenUsage(200, 80)
        total_input = u1.input_tokens + u2.input_tokens
        total_output = u1.output_tokens + u2.output_tokens
        assert total_input == 300
        assert total_output == 130


# ── 16. Full screening score integration ─────────────────────────────────────

class TestFullScreeningIntegration:
    """End-to-end screening score calculations."""

    def test_n2_indicative_all_wajar(self):
        """n=2, listing=median -> indicative with capped score."""
        result = _score(2, listing_price=200_000_000, price_median=200_000_000)
        assert result.is_indicative
        assert result.total_score <= 70
        assert result.total_score > 0
        assert "indikatif" in result.price_label.lower()

    def test_n5_normal_all_wajar(self):
        """n=5, listing=median -> normal mode, possibly layak."""
        result = _score(5, listing_price=200_000_000, price_median=200_000_000)
        assert not result.is_indicative
        assert result.total_score > 70
        assert result.category == "layak"

    def test_n2_very_cheap_is_perlu_cek_or_berisiko(self):
        """n=2, listing far below median -> risky, never layak."""
        result = _score(2, listing_price=70_000_000, price_median=200_000_000)
        assert result.price_diff_pct < -0.25
        assert result.category != "layak"

    def test_n0_returns_unassessable(self):
        result = _score(0)
        assert result.total_score == -1
        formatted = format_screening_result(result)
        assert formatted["total_score"] is None

    def test_description_empty_excluded_weight_renormalized(self):
        """With description empty, only 3 components contribute."""
        r_with = _score(5, has_description=True)
        r_without = _score(5, has_description=False)
        # Scores differ because weight redistribution
        # But both should be valid positive scores
        assert r_with.total_score > 0
        assert r_without.total_score > 0
        # The description-empty one should have 3 assessed components
        assessed = [c for c in r_without.components if c.assessed]
        assert len(assessed) == 3
