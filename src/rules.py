"""Red flag berbasis aturan untuk screening mobil."""

from dataclasses import dataclass
from typing import Literal
from src.config import RULE_THRESHOLDS, SUSPICIOUS_KEYWORDS


@dataclass
class RuleRedFlag:
    """Red flag hasil deteksi aturan."""
    code: str
    message: str
    severity: Literal["high", "medium", "low"]
    source: str = "rule"  # "rule" atau "ai"


def check_price_below_estimate(
    listing_price: int, price_median: int, n_comparables: int = 10
) -> RuleRedFlag | None:
    """Cek apakah harga listing >25% di bawah estimasi median (hanya jika pembanding >= 3)."""
    if price_median <= 0 or n_comparables < 3:
        return None
    diff_pct = (listing_price - price_median) / price_median
    threshold = -RULE_THRESHOLDS["price_below_estimate_pct"]
    if diff_pct <= threshold:
        return RuleRedFlag(
            code="PRICE_TOO_LOW",
            message=f"Harga listing {abs(diff_pct)*100:.0f}% di bawah estimasi pasar (mencurigakan)",
            severity="high",
        )
    return None


def check_km_per_year(
    kilometer: int, year: int, current_year: int = 2024
) -> RuleRedFlag | None:
    """Cek KM per tahun di luar batas wajar (3k-30k)."""
    age = current_year - year
    if age <= 0:
        return None
    km_per_year = kilometer / age
    min_km = RULE_THRESHOLDS["km_per_year_min"]
    max_km = RULE_THRESHOLDS["km_per_year_max"]
    
    if km_per_year < min_km:
        return RuleRedFlag(
            code="KM_TOO_LOW",
            message=f"KM per tahun {km_per_year:,.0f} terlalu rendah (mencurigakan rollback KM)",
            severity="high",
        )
    if km_per_year > max_km:
        return RuleRedFlag(
            code="KM_TOO_HIGH",
            message=f"KM per tahun {km_per_year:,.0f} terlalu tinggi (pemakaian berat)",
            severity="medium",
        )
    return None


def check_description_length(description: str | None) -> RuleRedFlag | None:
    """Cek deskripsi kosong atau terlalu pendek."""
    if not description or not description.strip():
        return RuleRedFlag(
            code="DESC_EMPTY",
            message="Deskripsi penjual kosong",
            severity="medium",
        )
    if len(description.strip()) < RULE_THRESHOLDS["description_min_length"]:
        return RuleRedFlag(
            code="DESC_TOO_SHORT",
            message=f"Deskripsi terlalu pendek ({len(description.strip())} karakter)",
            severity="low",
        )
    return None


def check_suspicious_keywords(description: str | None) -> list[RuleRedFlag]:
    """Cek kata kunci mencurigakan dalam deskripsi."""
    if not description:
        return []
    
    desc_lower = description.lower()
    flags = []
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword.lower() in desc_lower:
            flags.append(RuleRedFlag(
                code="SUSPICIOUS_KEYWORD",
                message=f"Terdapat kata mencurigakan: '{keyword}'",
                severity="high",
            ))
    return flags


def check_year_range(model: str, year: int, current_year: int = 2024) -> RuleRedFlag | None:
    """Cek tahun di luar rentang wajar untuk model (placeholder)."""
    # Untuk MVP, hanya cek batas umum
    if year < 2010 or year > current_year:
        return RuleRedFlag(
            code="YEAR_OUT_OF_RANGE",
            message=f"Tahun {year} di luar rentang wajar untuk model {model}",
            severity="medium",
        )
    return None


def run_all_rules(
    listing_price: int,
    price_median: int,
    kilometer: int,
    year: int,
    model: str,
    description: str | None,
    current_year: int = 2024,
    n_comparables: int = 10,
) -> list[RuleRedFlag]:
    """Jalankan semua aturan red flag dan kembalikan daftar flag."""
    flags = []
    
    # Harga terlalu murah (hanya jika n_comparables >= 3)
    flag = check_price_below_estimate(listing_price, price_median, n_comparables=n_comparables)
    if flag:
        flags.append(flag)
    
    # KM per tahun
    flag = check_km_per_year(kilometer, year, current_year)
    if flag:
        flags.append(flag)
    
    # Deskripsi
    flag = check_description_length(description)
    if flag:
        flags.append(flag)
    
    # Kata kunci mencurigakan
    flags.extend(check_suspicious_keywords(description))
    
    # Tahun
    flag = check_year_range(model, year, current_year)
    if flag:
        flags.append(flag)
    
    return flags


def get_rule_risk_score(flags: list[RuleRedFlag]) -> float:
    """Hitung skor risiko dari red flag aturan (0-100, semakin tinggi semakin berisiko)."""
    if not flags:
        return 0.0
    
    severity_weights = {"high": 30, "medium": 15, "low": 5}
    total = sum(severity_weights.get(f.severity, 0) for f in flags)
    return min(total, 100.0)


def merge_and_deduplicate_flags(
    rule_flags: list[RuleRedFlag],
    ai_red_flags: list[str],
) -> list[RuleRedFlag]:
    """
    Gabungkan red flag aturan dan AI, hapus duplikat berdasarkan kode/message.
    Satu masalah = satu pengurang skor.
    """
    merged = []
    seen_issues = set()

    def issue_key(message: str) -> str:
        normalized = " ".join(message.lower().split())
        issue_groups = {
            "harga": ("harga", "murah", "mahal", "pasar"),
            "kilometer": ("km", "kilometer", "odometer"),
            "dokumen": ("surat", "stnk", "bpkb", "faktur"),
            "banjir": ("banjir", "terendam"),
            "tabrakan": ("tabrakan", "kecelakaan", "body repair"),
            "mesin": ("mesin", "engine"),
            "pajak": ("pajak", "stnk mati"),
            "deskripsi": ("deskripsi kosong", "deskripsi tidak disediakan"),
            "pembayaran": ("bu", "butuh uang", "transfer", "dp dulu", "urgent"),
        }
        for issue, terms in issue_groups.items():
            if any(term in normalized for term in terms):
                return issue
        return normalized.rstrip(".!? ")
    
    # Prioritaskan rule flags (lebih deterministik)
    for flag in rule_flags:
        key = issue_key(flag.message)
        if key not in seen_issues:
            seen_issues.add(key)
            merged.append(flag)
    
    # Tambah AI flags yang belum ada
    for ai_flag in ai_red_flags:
        key = issue_key(ai_flag)
        if key not in seen_issues:
            seen_issues.add(key)
            merged.append(RuleRedFlag(
                code="AI_FLAG",
                message=ai_flag,
                severity="medium",  # default severity untuk AI flag
                source="ai",
            ))
    
    return merged


if __name__ == "__main__":
    # Test
    test_flags = run_all_rules(
        listing_price=150_000_000,
        price_median=250_000_000,
        kilometer=5000,
        year=2020,
        model="Avanza",
        description="BU butuh uang cepat DP dulu transfer dulu",
    )
    for f in test_flags:
        print(f"[{f.severity.upper()}] {f.code}: {f.message}")
    print(f"Risk score: {get_rule_risk_score(test_flags)}")
    
    # Test merge
    ai_flags = ["Harga terlalu murah", "Kondisi mesin tidak disebutkan"]
    merged = merge_and_deduplicate_flags(test_flags, ai_flags)
    print("\nMerged flags:")
    for f in merged:
        print(f"  [{f.source}] {f.code}: {f.message}")