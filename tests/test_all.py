"""Deterministic red-flag and screening-score tests."""

from src.rules import (
    RuleRedFlag,
    check_description_length,
    check_km_per_year,
    check_price_below_estimate,
    merge_and_deduplicate_flags,
    run_all_rules,
)
from src.scoring import (
    calculate_description_risk_score,
    calculate_km_age_fairness_score,
    calculate_screening_score,
    get_market_confidence,
    get_price_label,
)


def test_rules_detect_price_km_and_empty_description():
    assert check_price_below_estimate(150_000_000, 250_000_000).code == "PRICE_TOO_LOW"
    assert check_km_per_year(5_000, 2020, current_year=2026).code == "KM_TOO_LOW"
    assert check_description_length("").code == "DESC_EMPTY"


def test_rule_and_ai_flags_merge_once_with_source():
    rules = [RuleRedFlag("PRICE_TOO_LOW", "Harga listing terlalu rendah dibanding pasar", "high")]
    merged = merge_and_deduplicate_flags(rules, ["Harga terlalu murah dibandingkan pembanding"])
    assert len(merged) == 1
    assert merged[0].source == "rule"


def test_duplicate_ai_flag_does_not_add_second_score_penalty():
    rule_flag = RuleRedFlag("PRICE_TOO_LOW", "Harga listing 30% di bawah estimasi", "high")
    arguments = {
        "listing_price": 150_000_000,
        "price_low": 180_000_000,
        "price_median": 220_000_000,
        "price_high": 250_000_000,
        "n_comparables": 10,
        "kilometer": 60_000,
        "year": 2020,
        "has_description": True,
        "has_varian": True,
        "has_plate": True,
        "has_location": True,
    }
    base = calculate_screening_score(**arguments, rule_flags=[rule_flag])
    duplicate = calculate_screening_score(
        **arguments,
        rule_flags=[rule_flag],
        llm_red_flags=["Harga terlalu murah dibanding pasar"],
    )
    assert duplicate.total_score == base.total_score


def test_km_score_compares_against_observed_comparable_average():
    component = calculate_km_age_fairness_score(
        60_000,
        2022,
        current_year=2026,
        average_comparable_km_per_year=15_000,
    )
    assert component.score == 100
    assert "rata-rata pembanding" in component.reason


def test_market_confidence_is_low_below_solid_comparables():
    confidence = get_market_confidence(2, 0.01, 100)
    assert confidence["level"] == "Rendah"
    assert "kurang dari 3" in confidence["reason"]


def test_price_labels_are_based_on_market_median():
    # New label names
    assert get_price_label(-0.30) == "Sangat murah, waspada"
    assert get_price_label(-0.25) == "Di bawah pasar"  # boundary: -25% is in [-25%, -5%) range
    assert get_price_label(-0.10) == "Di bawah pasar"
    assert get_price_label(0.02) == "Wajar"
    assert get_price_label(0.10) == "Agak mahal"
    assert get_price_label(0.20) == "Mahal"


def test_empty_description_is_an_explicit_rule_flag():
    flags = run_all_rules(220_000_000, 220_000_000, 80_000, 2020, "Avanza", "", current_year=2026)
    assert any(flag.code == "DESC_EMPTY" for flag in flags)
