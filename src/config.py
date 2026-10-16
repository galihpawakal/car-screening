"""Konfigurasi aplikasi: daftar merek/model/lokasi, bobot skor, ambang batas."""

from typing import Literal
from pathlib import Path

# Model Gemini yang didukung untuk Google Search grounding
SUPPORTED_SEARCH_MODELS: frozenset[str] = frozenset({
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
})

# Merek dan model populer di Jabodetabek
MODELS_BY_BRAND: dict[str, list[str]] = {
    "Toyota": ["Avanza", "Innova", "Fortuner", "Rush", "Agya", "Yaris", "Calya", "Veloz"],
    "Daihatsu": ["Xenia", "Ayla", "Sigra", "Rocky", "Terios"],
    "Honda": ["Brio", "Mobilio", "Jazz", "City", "CR-V", "HR-V", "BR-V"],
    "Suzuki": ["Ertiga", "XL7", "Ignis", "Swift", "Jimny", "Carry"],
    "Mitsubishi": ["Xpander", "Pajero Sport", "Mirage"],
    "Nissan": ["Livina", "Grand Livina", "Kicks", "Terra"],
    "Wuling": ["Confero", "Cortez", "Almaz", "Air EV"],
}

# Flatten untuk dropdown
ALL_MODELS: list[str] = sorted(
    [model for models in MODELS_BY_BRAND.values() for model in models]
)

# Mapping model ke merek untuk validasi
MODEL_TO_BRAND: dict[str, str] = {}
for brand, models in MODELS_BY_BRAND.items():
    for model in models:
        MODEL_TO_BRAND[model] = brand

BODY_TYPES: list[str] = ["MPV", "SUV", "Sedan", "Hatchback", "LCV", "Minibus"]

TRANSMISSION_TYPES: list[Literal["AT", "MT"]] = ["AT", "MT"]

LOCATIONS: list[str] = [
    "Jakarta Pusat",
    "Jakarta Utara",
    "Jakarta Selatan",
    "Jakarta Barat",
    "Jakarta Timur",
    "Bogor",
    "Depok",
    "Tangerang",
    "Tangerang Selatan",
    "Bekasi",
    "Bekasi Utara",
    "Bekasi Selatan",
]

PLATE_CODES: list[str] = ["B", "F", "D", "T", "Lainnya"]

# Bobot skor screening
SCORING_WEIGHTS: dict[str, float] = {
    "price_fairness": 0.40,
    "km_age_fairness": 0.20,
    "info_completeness": 0.20,
    "description_risk": 0.20,
}

# Ambang batas label harga (batas atas tiap label, dalam fraksi diff%)
PRICE_LABEL_THRESHOLDS: dict[str, float] = {
    "sangat_murah": -0.25,     # diff < -25%
    "di_bawah_pasar": -0.05,   # -25% <= diff < -5%
    "wajar": 0.05,             # -5% <= diff <= +5%
    "agak_mahal": 0.15,        # +5% < diff <= +15%
    "mahal": float("inf"),     # diff > +15%
}

# Ambang batas red flag berbasis aturan
RULE_THRESHOLDS: dict[str, float] = {
    "price_below_estimate_pct": 0.25,    # >25% di bawah estimasi
    "km_per_year_min": 3000,             # KM/tahun minimum
    "km_per_year_max": 30000,            # KM/tahun maksimum
    "description_min_length": 50,        # Deskripsi terlalu pendek
}

# Kata kunci mencurigakan dalam deskripsi
SUSPICIOUS_KEYWORDS: list[str] = [
    "BU",
    "butuh uang",
    "dp dulu",
    "transfer dulu",
    "tanpa surat",
    "cash jalan",
    "urgent",
    "nggak perlu cek",
    "as is",
    "pasal uang",
    "cepat dijual",
]

# Kategori skor
SCORE_CATEGORIES: dict[str, tuple[int, int]] = {
    "layak": (75, 100),
    "perlu_cek": (50, 74),
    "berisiko": (0, 49),
}

# Runtime cache: only normalized facts, source URLs, query metadata, and analysis usage.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DB_PATH = PROJECT_ROOT / "data" / "analysis_cache.sqlite3"

# Marketplace domains — one grounded search call each (max 5 listing URLs per call).
MARKETPLACE_DOMAINS: list[str] = [
    "olx.co.id",
    "mobil123.com",
    "carmudi.co.id",
    "oto.com",
    "carsome.id",
]

# --- Comparable thresholds ---
# Minimum: n < MIN_COMPARABLES -> "Tidak dapat dinilai" (sangat rendah)
MIN_COMPARABLES = 2
# Solid: n >= SOLID_COMPARABLES -> mode normal; 2 <= n < 3 -> mode indikatif
SOLID_COMPARABLES = 3
# Backward-compat alias used by existing tests
MIN_COMPARABLES_FOR_SCORE = MIN_COMPARABLES

MAX_MANUAL_LISTING_URLS = 10
MAX_LISTINGS_PER_MARKETPLACE_SEARCH = 3

# --- Scoring functions (di config agar mudah diubah) ---

def price_fairness_score(diff_pct: float) -> float:
    """Hitung skor kewajaran harga berdasarkan diff% terhadap median.

    |diff| <= 5% -> 100
    diff > 5%   -> max(0, 100 - 4*(diff-5))
    diff < -5%  -> max(20, 100 - 3*(|diff|-5))  (murah ekstrem = risiko)
    """
    abs_diff = abs(diff_pct * 100)  # convert to percentage points
    if abs_diff <= 5:
        return 100.0
    if diff_pct > 0:
        # Mahal: skor turun 4 poin per persen di atas 5%
        return max(0.0, 100.0 - 4.0 * (diff_pct * 100 - 5))
    # Murah: skor turun 3 poin per persen di bawah -5% (murah ekstrem = risiko)
    return max(20.0, 100.0 - 3.0 * (abs_diff - 5))


def km_age_fairness_score(km_per_year: float, reference_km_per_year: float) -> tuple[float, str]:
    """Hitung skor KM vs umur terhadap acuan (pembanding atau rentang umum).

    Dalam +/-25% -> 100, makin jauh makin turun.
    < -40% -> red flag 'verifikasi odometer'.

    Returns (score, assessment_text).
    """
    if reference_km_per_year <= 0:
        return 50.0, "Acuan KM/tahun tidak tersedia"
    diff = (km_per_year - reference_km_per_year) / reference_km_per_year
    abs_diff = abs(diff)
    if abs_diff <= 0.25:
        score = 100.0
        assessment = "sebanding"
    elif abs_diff <= 0.50:
        score = max(40.0, 100.0 - (abs_diff - 0.25) * 240)  # drops to 40 at 50%
        assessment = "lebih tinggi" if diff > 0 else "lebih rendah"
    else:
        score = max(20.0, 40.0 - (abs_diff - 0.50) * 80)
        assessment = "jauh lebih tinggi" if diff > 0 else "jauh lebih rendah"
    return score, assessment


# Batas skor maksimal di mode indikatif
INDICATIVE_MAX_SCORE = 70

# Disclaimer
DISCLAIMER = (
    "Hasil ini adalah estimasi berbasis data listing, bukan penilaian pasti. "
    "Harga transaksi aktual biasanya 3-10% di bawah harga listing. "
    "Selalu lakukan cek fisik dan bengkel sebelum membeli."
)