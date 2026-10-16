"""Sistem scoring screening 0-100 dengan alasan bahasa Indonesia."""

from dataclasses import dataclass
from typing import Literal
from src.config import (
    MIN_COMPARABLES,
    SOLID_COMPARABLES,
    SCORING_WEIGHTS,
    SCORE_CATEGORIES,
    RULE_THRESHOLDS,
    PRICE_LABEL_THRESHOLDS,
    INDICATIVE_MAX_SCORE,
    price_fairness_score as _cfg_price_score,
    km_age_fairness_score as _cfg_km_score,
)
from src.rules import RuleRedFlag, get_rule_risk_score


@dataclass
class ScoreComponent:
    """Komponen skor dengan alasan."""
    name: str
    score: float  # 0-100, or -1 for "not assessed"
    weight: float
    reason: str
    assessed: bool = True  # False = excluded from weighted average


@dataclass
class ScreeningResult:
    """Hasil screening lengkap."""
    total_score: int
    category: Literal["layak", "perlu_cek", "berisiko"]
    components: list[ScoreComponent]
    price_label: str
    price_diff_pct: float
    rule_flags: list[RuleRedFlag]
    is_indicative: bool = False
    llm_risk_score: float = 0.0
    llm_summary: str = ""
    llm_red_flags: list[str] = None
    llm_questions: list[str] = None

    def __post_init__(self):
        if self.llm_red_flags is None:
            self.llm_red_flags = []
        if self.llm_questions is None:
            self.llm_questions = []


def get_price_label(price_diff_pct: float, is_indicative: bool = False) -> str:
    """Dapatkan label harga berdasarkan persentase selisih.

    Label baru:
      diff < -25%          -> "Sangat murah, waspada"
      -25% <= diff < -5%   -> "Di bawah pasar"
      -5% <= diff <= +5%   -> "Wajar"
      +5% < diff <= +15%   -> "Agak mahal"
      diff > +15%          -> "Mahal"

    Jika is_indicative, tambahkan "(indikatif)" di akhir.
    """
    if price_diff_pct < PRICE_LABEL_THRESHOLDS["sangat_murah"]:
        label = "Sangat murah, waspada"
    elif price_diff_pct < PRICE_LABEL_THRESHOLDS["di_bawah_pasar"]:
        label = "Di bawah pasar"
    elif price_diff_pct <= PRICE_LABEL_THRESHOLDS["wajar"]:
        label = "Wajar"
    elif price_diff_pct <= PRICE_LABEL_THRESHOLDS["agak_mahal"]:
        label = "Agak mahal"
    else:
        label = "Mahal"
    if is_indicative:
        label += " (indikatif)"
    return label


def get_market_confidence(
    count: int,
    price_spread_ratio: float | None,
    completeness_pct: float,
) -> dict[str, str]:
    """Classify estimate confidence from source count, price spread, and input completeness.

    Ambang tingkat keyakinan:
      < 2   -> Sangat rendah (skor tidak dihitung)
      2     -> Rendah (indikatif)
      3-5   -> Sedang
      >= 6  -> Tinggi
    """
    spread = price_spread_ratio if price_spread_ratio is not None else 1.0
    if count < MIN_COMPARABLES:  # < 2
        return {
            "level": "Sangat rendah",
            "reason": f"Data pembanding kurang dari {MIN_COMPARABLES} ({count}); skor tidak dihitung dan rentang harga berkeyakinan sangat rendah.",
        }
    if count < SOLID_COMPARABLES:  # 2
        return {
            "level": "Rendah",
            "reason": f"Data pembanding kurang dari {SOLID_COMPARABLES} ({count}); estimasi bersifat indikatif (data sedikit).",
        }
    if count < 6:  # 3-5
        return {
            "level": "Sedang",
            "reason": (
                f"Berdasarkan {count} pembanding, sebaran harga {spread:.0%}, "
                f"dan kelengkapan input {completeness_pct:.0f}%."
            ),
        }
    # >= 6
    return {
        "level": "Tinggi",
        "reason": (
            f"Berdasarkan {count} pembanding, sebaran harga {spread:.0%}, "
            f"dan kelengkapan input {completeness_pct:.0f}%."
        ),
    }


def calculate_price_fairness_score(
    listing_price: int, price_low: int, price_median: int, price_high: int, n_comparables: int
) -> ScoreComponent:
    """Hitung skor kewajaran harga (40% bobot)."""
    if price_median <= 0:
        return ScoreComponent(
            name="Kewajaran Harga",
            score=50,
            weight=SCORING_WEIGHTS["price_fairness"],
            reason="Tidak ada data pembanding untuk estimasi harga",
        )

    diff_pct = (listing_price - price_median) / price_median
    score = _cfg_price_score(diff_pct)

    abs_pct = abs(diff_pct * 100)
    if abs(diff_pct) <= 0.05:
        reason = f"Harga listing {diff_pct*100:+.1f}% dari median {n_comparables} pembanding (wajar)"
    elif diff_pct < -0.25:
        reason = f"Harga listing {abs_pct:.1f}% di bawah median {n_comparables} pembanding (sangat murah, waspada)"
    elif diff_pct < -0.05:
        reason = f"Harga listing {abs_pct:.1f}% di bawah median {n_comparables} pembanding (di bawah pasar)"
    elif diff_pct <= 0.15:
        reason = f"Harga listing {diff_pct*100:+.1f}% di atas median {n_comparables} pembanding (agak mahal)"
    else:
        reason = f"Harga listing {diff_pct*100:+.1f}% di atas median {n_comparables} pembanding (mahal)"

    return ScoreComponent(
        name="Kewajaran Harga",
        score=round(score, 1),
        weight=SCORING_WEIGHTS["price_fairness"],
        reason=reason,
    )


def calculate_km_age_fairness_score(
    kilometer: int,
    year: int,
    current_year: int = 2024,
    average_comparable_km_per_year: float | None = None,
) -> ScoreComponent:
    """Hitung skor kewajaran KM terhadap umur (20% bobot)."""
    age = current_year - year
    if age <= 0:
        return ScoreComponent(
            name="Kewajaran KM vs Umur",
            score=50,
            weight=SCORING_WEIGHTS["km_age_fairness"],
            reason="Umur mobil tidak valid",
        )

    km_per_year = kilometer / age
    min_km = RULE_THRESHOLDS["km_per_year_min"]
    max_km = RULE_THRESHOLDS["km_per_year_max"]

    # Determine reference
    if average_comparable_km_per_year is not None and average_comparable_km_per_year > 0:
        reference = average_comparable_km_per_year
        ref_label = f"rata-rata pembanding {reference:,.0f} km/tahun"
    else:
        # Pakai tengah rentang umum
        reference = (min_km + max_km) / 2
        ref_label = f"rentang umum {min_km:,.0f}-{max_km:,.0f} km/tahun"

    score, assessment = _cfg_km_score(km_per_year, reference)
    diff_ratio = (km_per_year - reference) / reference if reference > 0 else 0

    reason = (
        f"Unit {km_per_year:,.0f} km/tahun, {ref_label} "
        f"({assessment}, {diff_ratio:+.1%})"
    )

    # Red flag: < -40% dari acuan
    if diff_ratio < -0.40:
        reason += "; verifikasi odometer disarankan"

    return ScoreComponent(
        name="Kewajaran KM vs Umur",
        score=round(score, 1),
        weight=SCORING_WEIGHTS["km_age_fairness"],
        reason=reason,
    )


def calculate_info_completeness_score(
    has_description: bool,
    has_varian: bool,
    has_plate: bool,
    has_location: bool,
) -> ScoreComponent:
    """Hitung skor kelengkapan informasi (20% bobot)."""
    fields = {
        "Deskripsi": has_description,
        "Varian": has_varian,
        "Plat Nomor": has_plate,
        "Lokasi": has_location,
    }
    filled = sum(fields.values())
    total = len(fields)
    pct = filled / total
    score = pct * 100

    missing = [k for k, v in fields.items() if not v]
    present = [k for k, v in fields.items() if v]

    if missing:
        reason = f"Terisi: {', '.join(present)}. Belum: {', '.join(missing)}"
    else:
        reason = "Semua informasi wajib dan opsional terisi lengkap"

    return ScoreComponent(
        name="Kelengkapan Informasi",
        score=round(score, 1),
        weight=SCORING_WEIGHTS["info_completeness"],
        reason=reason,
    )


def calculate_description_risk_score(
    rule_flags: list[RuleRedFlag],
    llm_risk_score: float,
    llm_red_flags: list[str],
    *,
    has_description: bool = True,
) -> ScoreComponent:
    """Hitung skor risiko deskripsi (20% bobot).

    Jika deskripsi kosong -> komponen 'tidak dinilai' (assessed=False).
    Dikeluarkan dari rata-rata, bobot dialihkan proporsional ke komponen lain.
    """
    if not has_description:
        return ScoreComponent(
            name="Risiko Deskripsi",
            score=-1,
            weight=SCORING_WEIGHTS["description_risk"],
            reason="Deskripsi kosong; komponen tidak dinilai",
            assessed=False,
        )
    rule_flags = [flag for flag in rule_flags if flag.code not in {"DESC_EMPTY", "DESC_TOO_SHORT"}]
    rule_risk = get_rule_risk_score(rule_flags)
    residual_llm_risk = max(0, llm_risk_score - len(llm_red_flags) * 10)
    combined_risk = min(100, rule_risk * 0.6 + residual_llm_risk * 0.4)
    score = 100 - combined_risk

    reasons = []
    rule_source_flags = [flag for flag in rule_flags if flag.source == "rule"]
    ai_source_flags = [flag for flag in rule_flags if flag.source == "ai"]
    if rule_source_flags:
        high_flags = [f for f in rule_source_flags if f.severity == "high"]
        med_flags = [f for f in rule_source_flags if f.severity == "medium"]
        if high_flags:
            reasons.append(f"{len(high_flags)} red flag tingkat tinggi (aturan)")
        if med_flags:
            reasons.append(f"{len(med_flags)} red flag tingkat sedang (aturan)")
    if ai_source_flags:
        reasons.append(f"{len(ai_source_flags)} red flag unik dari AI")

    reason = "; ".join(reasons) if reasons else "Tidak terdeteksi red flag signifikan"

    return ScoreComponent(
        name="Risiko Deskripsi",
        score=round(max(0, score), 1),
        weight=SCORING_WEIGHTS["description_risk"],
        reason=reason,
    )


def insufficient_data_screening_result(
    rule_flags: list[RuleRedFlag],
    *,
    has_description: bool,
    has_varian: bool,
    has_plate: bool,
    has_location: bool,
    llm_red_flags: list[str] | None = None,
) -> ScreeningResult:
    """Return a non-numeric screening outcome when comparables are below MIN_COMPARABLES."""
    from src.rules import merge_and_deduplicate_flags

    # Red flag "harga di bawah pasar / mencurigakan" TIDAK boleh muncul jika pembanding valid < 3
    rule_flags = [f for f in rule_flags if f.code != "PRICE_TOO_LOW" and "mencurigakan" not in f.message.lower()]
    llm_red_flags = [f for f in (llm_red_flags or []) if "harga" not in f.lower() and "murah" not in f.lower() and "mencurigakan" not in f.lower()]
    all_flags = merge_and_deduplicate_flags(rule_flags, llm_red_flags)
    info_comp = calculate_info_completeness_score(has_description, has_varian, has_plate, has_location)
    desc_comp = calculate_description_risk_score(all_flags, 0.0, llm_red_flags, has_description=has_description)
    components = [
        ScoreComponent(
            name="Kewajaran Harga",
            score=-1,
            weight=SCORING_WEIGHTS["price_fairness"],
            reason="Data pembanding terlalu sedikit untuk menilai kewajaran harga.",
            assessed=False,
        ),
        ScoreComponent(
            name="Kewajaran KM vs Umur",
            score=-1,
            weight=SCORING_WEIGHTS["km_age_fairness"],
            reason="Data pembanding terlalu sedikit untuk menilai kewajaran KM vs umur.",
            assessed=False,
        ),
        info_comp,
        desc_comp,
    ]
    return ScreeningResult(
        total_score=-1,
        category="perlu_cek",
        components=components,
        price_label="Tidak dapat dinilai",
        price_diff_pct=0.0,
        rule_flags=all_flags,
        llm_red_flags=llm_red_flags,
    )


def _weighted_average_assessed(components: list[ScoreComponent]) -> float:
    """Compute weighted average over assessed components only, re-normalizing weights."""
    assessed = [c for c in components if c.assessed]
    total_weight = sum(c.weight for c in assessed)
    if total_weight <= 0:
        return 0.0
    return sum(c.score * c.weight for c in assessed) / total_weight


def calculate_screening_score(
    listing_price: int,
    price_low: int,
    price_median: int,
    price_high: int,
    n_comparables: int,
    kilometer: int,
    year: int,
    has_description: bool,
    has_varian: bool,
    has_plate: bool,
    has_location: bool,
    rule_flags: list[RuleRedFlag],
    llm_risk_score: float = 0.0,
    llm_red_flags: list[str] = None,
    llm_summary: str = "",
    llm_questions: list[str] = None,
    current_year: int = 2024,
    average_comparable_km_per_year: float | None = None,
) -> ScreeningResult:
    """Hitung total skor screening."""
    llm_red_flags = llm_red_flags or []
    llm_questions = llm_questions or []

    if n_comparables < MIN_COMPARABLES:
        return insufficient_data_screening_result(
            rule_flags,
            has_description=has_description,
            has_varian=has_varian,
            has_plate=has_plate,
            has_location=has_location,
            llm_red_flags=llm_red_flags,
        )

    from src.rules import merge_and_deduplicate_flags
    all_flags = merge_and_deduplicate_flags(rule_flags, llm_red_flags)

    is_indicative = n_comparables < SOLID_COMPARABLES

    price_comp = calculate_price_fairness_score(
        listing_price, price_low, price_median, price_high, n_comparables
    )
    km_comp = calculate_km_age_fairness_score(
        kilometer, year, current_year, average_comparable_km_per_year
    )
    info_comp = calculate_info_completeness_score(
        has_description, has_varian, has_plate, has_location
    )
    # Gunakan semua flag yang sudah digabung untuk risiko deskripsi
    desc_comp = calculate_description_risk_score(
        all_flags,
        llm_risk_score,
        llm_red_flags,
        has_description=has_description,
    )

    components = [price_comp, km_comp, info_comp, desc_comp]

    # Total skor: rata-rata berbobot komponen yang dinilai saja
    raw_score = _weighted_average_assessed(components)
    total_score = round(raw_score)

    # Batasi skor maksimal 70 di mode indikatif
    if is_indicative and total_score > INDICATIVE_MAX_SCORE:
        total_score = INDICATIVE_MAX_SCORE

    # Label harga
    price_diff_pct = (listing_price - price_median) / price_median if price_median > 0 else 0

    # Kategori
    if total_score >= SCORE_CATEGORIES["layak"][0]:
        category = "layak"
    elif total_score >= SCORE_CATEGORIES["perlu_cek"][0]:
        category = "perlu_cek"
    else:
        category = "berisiko"

    # Jika diff < -25%, kategori tidak boleh lebih tinggi dari perlu_cek
    if price_diff_pct < -0.25 and category == "layak":
        category = "perlu_cek"

    price_label = get_price_label(price_diff_pct, is_indicative=is_indicative)

    return ScreeningResult(
        total_score=total_score,
        category=category,
        components=components,
        price_label=price_label,
        price_diff_pct=price_diff_pct,
        rule_flags=all_flags,
        is_indicative=is_indicative,
        llm_risk_score=llm_risk_score,
        llm_summary=llm_summary,
        llm_red_flags=llm_red_flags,
        llm_questions=llm_questions,
    )


def format_screening_result(result: ScreeningResult) -> dict:
    """Format hasil untuk tampilan UI."""
    category_labels = {
        "layak": "✅ Layak Dipertimbangkan",
        "perlu_cek": "⚠️ Perlu Cek Lebih Lanjut",
        "berisiko": "🚨 Berisiko Tinggi",
    }
    if result.total_score < 0:
        score_display = "Tidak dapat dinilai"
        category_label = "Tidak dapat dinilai"
    elif result.is_indicative:
        score_display = f"{result.total_score}/100 (indikatif)"
        category_label = category_labels[result.category]
    else:
        score_display = f"{result.total_score}/100"
        category_label = category_labels[result.category]

    # Pisahkan flag berdasarkan sumber
    rule_flags = [f for f in result.rule_flags if f.source == "rule"]
    ai_flags = [f for f in result.rule_flags if f.source == "ai"]

    formatted_components = []
    for c in result.components:
        formatted_components.append({
            "name": c.name,
            "score": c.score if c.assessed else None,
            "score_display": f"{c.score:.0f}" if c.assessed else "-",
            "weight": c.weight,
            "reason": c.reason,
            "assessed": c.assessed,
        })

    return {
        "total_score": None if result.total_score < 0 else result.total_score,
        "total_score_display": score_display,
        "score_assessable": result.total_score >= 0,
        "is_indicative": result.is_indicative,
        "category": result.category,
        "category_label": category_label,
        "price_label": result.price_label,
        "price_diff_pct": round(result.price_diff_pct * 100, 1),
        "components": formatted_components,
        "rule_flags": [
            {"code": f.code, "message": f.message, "severity": f.severity, "source": f.source}
            for f in rule_flags
        ],
        "ai_flags": [
            {"code": f.code, "message": f.message, "severity": f.severity, "source": f.source}
            for f in ai_flags
        ],
        "llm_summary": result.llm_summary,
        "llm_red_flags": result.llm_red_flags,
        "llm_questions": result.llm_questions,
    }


def negotiation_advice_text(price_diff_pct: float, n_comparables: int) -> str | None:
    """Return advisory text when comparables < 3 or listing is below market; None when negotiation makes sense."""
    if n_comparables < 3:
        return (
            "Belum ada cukup pembanding untuk saran tawar. "
            "Gunakan harga transaksi aktual 3-10% di bawah listing sebagai acuan umum dan verifikasi langsung."
        )
    if price_diff_pct <= 0:
        return (
            "Harga sudah di bawah pembanding; tidak ada saran tawar turun. "
            "Prioritaskan verifikasi dokumen, kondisi, dan alasan harga rendah."
        )
    return None



def price_banner_warning(listing_price: int, price_median: int, n_comparables: int = 10) -> str | None:
    """Return a banner warning if listing price is <50% of median and comparables >= 3, else None."""
    if n_comparables >= MIN_COMPARABLES and price_median > 0 and listing_price < price_median * 0.50:
        return (
            "⚠️ Harga yang Anda masukkan jauh di bawah pasar. "
            "Periksa apakah sudah benar (salah ketik?) atau waspadai penipuan."
        )
    return None


def seller_questions_for_cheap(price_diff_pct: float) -> list[str]:
    """Return extra seller questions specific to very cheap listings."""
    if price_diff_pct >= -0.25:
        return []
    return [
        "Apa alasan harga di bawah pasar?",
        "Bagaimana status pajak dan apakah ada tunggakan?",
        "Apakah BPKB sedang di leasing/bank?",
        "Mengapa dijual cepat?",
        "Apakah bersedia COD di lokasi publik dan cek di bengkel independen?",
    ]


if __name__ == "__main__":
    # Test
    from src.rules import run_all_rules

    flags = run_all_rules(150_000_000, 250_000_000, 5000, 2020, "Avanza", "BU butuh uang")
    result = calculate_screening_score(
        listing_price=150_000_000,
        price_low=200_000_000,
        price_median=250_000_000,
        price_high=300_000_000,
        n_comparables=10,
        kilometer=5000,
        year=2020,
        has_description=True,
        has_varian=True,
        has_plate=True,
        has_location=True,
        rule_flags=flags,
        llm_risk_score=50,
        llm_red_flags=["Harga terlalu murah", "Kata 'BU'"],
        llm_summary="Mobil dijual cepat dengan harga jauh di bawah pasar.",
        llm_questions=["Mengapa harga begitu murah?", "Apakah ada kerusakan tersembunyi?"],
    )
    print(f"Total: {result.total_score} ({result.category})")
    print(f"Price: {result.price_label} ({result.price_diff_pct*100:+.1f}%)")
    for c in result.components:
        print(f"  {c.name}: {c.score} ({c.weight*100:.0f}%) - {c.reason}")