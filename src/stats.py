"""Deterministic price and mileage statistics from cleaned live comparables."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.config import SOLID_COMPARABLES
from src.extract import ComparableListing


def filter_price_outliers_iqr(prices: list[int]) -> list[int]:
    """Remove price outliers using standard IQR rule only if n >= 5."""
    if len(prices) < 5:
        return prices
    q25, q75 = np.percentile(prices, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return prices
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    filtered = [p for p in prices if lower_bound <= p <= upper_bound]
    return filtered if len(filtered) >= 2 else prices


def calculate_weighted_percentile(data: list[float | int], weights: list[float], percentile: float) -> float:
    """Compute weighted percentile. Delegates to np.percentile when all weights are equal."""
    if not data:
        return 0.0
    if len(set(weights)) <= 1:
        return float(np.percentile(data, percentile))
    arr_data = np.array(data, dtype=float)
    arr_weights = np.array(weights, dtype=float)
    sorter = np.argsort(arr_data)
    sorted_data = arr_data[sorter]
    sorted_weights = arr_weights[sorter]
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    target = (percentile / 100.0) * total_weight
    idx = int(np.searchsorted(cum_weights, target))
    return float(sorted_data[min(idx, len(sorted_data) - 1)])


def calculate_statistics(
    listings: list[ComparableListing],
    *,
    listing_price: int | None = None,
    target_year: int,
    current_year: int | None = None,
) -> dict:

    current_year = current_year or datetime.now().year
    valid_pairs = [(l.price, float(getattr(l, "weight", 1.0) or 1.0)) for l in listings if l.price > 0]
    count = len(valid_pairs)
    if not count:
        return {
            "count": 0,
            "price_low": None,
            "price_median": None,
            "price_high": None,
            "listing_percentile": None,
            "average_km_per_year": None,
            "price_spread_ratio": None,
        }

    raw_prices = [p for p, _ in valid_pairs]
    raw_weights = [w for _, w in valid_pairs]

    # Apply IQR outlier filtering only if n >= 5
    if len(raw_prices) >= 5:
        q25, q75 = np.percentile(raw_prices, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            filtered_pairs = [(p, w) for p, w in valid_pairs if lower <= p <= upper]
            if len(filtered_pairs) >= 2:
                valid_pairs = filtered_pairs
                raw_prices = [p for p, _ in valid_pairs]
                raw_weights = [w for _, w in valid_pairs]

    median = calculate_weighted_percentile(raw_prices, raw_weights, 50.0)
    if len(raw_prices) >= SOLID_COMPARABLES:
        # Mode normal: P25 / P75
        low = calculate_weighted_percentile(raw_prices, raw_weights, 25.0)
        high = calculate_weighted_percentile(raw_prices, raw_weights, 75.0)
    else:
        # Mode indikatif: min / max
        low = float(min(raw_prices))
        high = float(max(raw_prices))

    km_rates = [
        listing.kilometer / max(1, current_year - listing.year)
        for listing in listings
        if listing.kilometer is not None and listing.year is not None and listing.year <= current_year
    ]
    return {
        "count": len(valid_pairs),
        "price_low": int(round(low)),
        "price_median": int(round(median)),
        "price_high": int(round(high)),
        "listing_percentile": (
            sum(price <= listing_price for price in raw_prices) / len(raw_prices) * 100
            if listing_price is not None else None
        ),
        "average_km_per_year": sum(km_rates) / len(km_rates) if km_rates else None,
        "price_spread_ratio": float(np.std(raw_prices) / median) if len(raw_prices) > 1 and median else 0.0,
    }


def calculate_listing_percentile(listings: list[ComparableListing], listing_price: int) -> float | None:
    prices = [listing.price for listing in listings if listing.price > 0]
    if not prices:
        return None
    return sum(price <= listing_price for price in prices) / len(prices) * 100
