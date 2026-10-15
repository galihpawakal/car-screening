"""Deterministic price and mileage statistics from cleaned live comparables."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.config import SOLID_COMPARABLES
from src.extract import ComparableListing


def calculate_statistics(
    listings: list[ComparableListing],
    *,
    listing_price: int | None,
    target_year: int,
    current_year: int | None = None,
) -> dict:
    current_year = current_year or datetime.now().year
    prices = [listing.price for listing in listings if listing.price > 0]
    count = len(prices)
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

    median = float(np.median(prices))
    if count >= SOLID_COMPARABLES:
        # Mode normal: P25 / P75
        low, high = np.percentile(prices, [25, 75]).tolist()
    else:
        # Mode indikatif (n < 5): min / max
        low = float(min(prices))
        high = float(max(prices))

    km_rates = [
        listing.kilometer / max(1, current_year - listing.year)
        for listing in listings
        if listing.kilometer is not None and listing.year is not None and listing.year <= current_year
    ]
    return {
        "count": count,
        "price_low": int(round(low)),
        "price_median": int(round(median)),
        "price_high": int(round(high)),
        "listing_percentile": (
            sum(price <= listing_price for price in prices) / count * 100
            if listing_price is not None else None
        ),
        "average_km_per_year": sum(km_rates) / len(km_rates) if km_rates else None,
        "price_spread_ratio": float(np.std(prices) / median) if count > 1 and median else 0.0,
    }


def calculate_listing_percentile(listings: list[ComparableListing], listing_price: int) -> float | None:
    prices = [listing.price for listing in listings if listing.price > 0]
    if not prices:
        return None
    return sum(price <= listing_price for price in prices) / len(prices) * 100
