"""Verification script for testing the 3 vehicle models requested in Criterion 6.

1. Daihatsu Xenia E 2020, AT, 80.000 km, Jakarta Pusat, Rp 100 jt
2. Toyota Avanza G 2020, AT, 60.000 km, Jakarta Selatan, Rp 175 jt
3. Honda Brio RS 2020, AT, 45.000 km, Jakarta Barat, Rp 160 jt
"""

import json
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.cache import SQLiteAnalysisCache

from src.pipeline import run_screening
from src.search_provider import SearchProvider


class DirectOnlyProvider(SearchProvider):
    def __init__(self):
        self.model = "gemini-2.5-flash"

    def search(self, prompt: str, *, marketplace: str = ""):
        raise RuntimeError("Direct mode fallback")

    def generate_structured(self, prompt: str, response_model):
        raise RuntimeError("LLM disabled for direct mode")




def test_model(title: str, user_input: dict):
    print("=" * 70, flush=True)
    print(f"RUNNING SCREENING TEST: {title}", flush=True)
    print(f"Input: {user_input['brand']} {user_input['model']} {user_input.get('variant')} {user_input['year']} "
          f"{user_input['transmission']} {user_input['kilometer']}km Rp {user_input['listing_price']:,}", flush=True)
    print("=" * 70, flush=True)

    cache = SQLiteAnalysisCache("data/test_verification.db")
    provider = DirectOnlyProvider()

    def on_status(msg: str):
        print(f"  [Status] {msg}", flush=True)


    result = run_screening(
        provider=provider,
        cache=cache,
        user_input=user_input,
        daily_limit=50,
        on_status=on_status,
    )

    print("\n--- HASIL SCREENING ---")
    fm = result.funnel_metrics or {}
    print(f"1. URL Daftar Dibuka        : {fm.get('catalog_urls_opened', 0)}")
    if result.catalog_urls_opened:
        for c in result.catalog_urls_opened:
            print(f"   - {c.get('domain')} (hal {c.get('page')}): {c.get('url')} [{c.get('status')}, {c.get('links_found')} links]")
    print(f"2. Tautan Detail Ditemukan : {fm.get('detail_links_found', 0)}")
    print(f"3. Detail Terbaca          : {fm.get('detail_pages_parsed', 0)}")
    print(f"4. Lolos Validasi          : {len(result.comparables)}")
    print(f"5. Tier yang Dipakai       : {result.tier_label} (Tier {result.active_tier})")
    
    print("\n6. Listing yang Dibuang per Alasan:")
    if result.rejected_by_reason:
        for reason, urls in result.rejected_by_reason.items():
            print(f"   - {reason}: {len(urls)}")
    else:
        print("   (Tidak ada data yang dibuang)")

    print(f"\n7. Ringkasan Skor & Teks:")
    print(f"   - Skor Screening : {result.screening.get('total_score_display')}")
    print(f"   - Label Harga    : {result.screening.get('price_label')}")
    print(f"   - Kesimpulan     : {result.final_analysis.kesimpulan}")
    saran = result.final_analysis.saran_nego
    if saran.awal:
        print(f"   - Saran Tawar    : Awal Rp {saran.awal:,} | Target Rp {saran.realistis:,} | Batas Rp {saran.batas:,}")
    else:
        from src.scoring import negotiation_advice_text
        med = result.statistics.get("price_median") or 0
        diff = (user_input["listing_price"] - med) / med if med else 0
        print(f"   - Saran Tawar    : {negotiation_advice_text(diff, len(result.comparables))}")

    print(f"\n8. Daftar Pembanding yang Lolos ({len(result.comparables)} unit):")
    for idx, c in enumerate(result.comparables, 1):
        twin_str = " [Model Kembar]" if getattr(c, "is_twin_model", False) else ""
        print(f"   {idx}. {c.brand} {c.model} {c.variant} ({c.year}) - Rp {c.price:,} | KM: {c.kilometer:,} | Tier {c.tier} (bobot {c.weight}) | {c.source_url[:60]}...{twin_str}")

    return result


if __name__ == "__main__":
    # Model 1: Nissan Grand Livina E 2020 AT (Bug A case)
    test_model("1. Nissan Grand Livina E 2020 AT", {
        "brand": "Nissan",
        "model": "Grand Livina",
        "variant": "E",
        "year": 2020,
        "transmission": "AT",
        "body_type": "MPV",
        "kilometer": 60000,
        "location": "Jakarta Pusat",
        "plate_code": "B",
        "listing_price": 125_000_000,
        "deskripsi": "Kondisi sangat mulus, siap pakai, mesin kering.",
        "search_mode": "direct",
    })

    # Model 2: Toyota All New Avanza G 2020 AT
    test_model("2. Toyota All New Avanza G 2020 AT", {
        "brand": "Toyota",
        "model": "All New Avanza",
        "variant": "G",
        "year": 2020,
        "transmission": "AT",
        "body_type": "MPV",
        "kilometer": 60000,
        "location": "Jakarta Selatan",
        "plate_code": "B",
        "listing_price": 175_000_000,
        "deskripsi": "Pajak hidup, service record resmi, bukan bekas tabrak atau banjir.",
        "search_mode": "direct",
    })

    # Model 3: Honda Brio Satya 2020 AT
    test_model("3. Honda Brio Satya 2020 AT", {
        "brand": "Honda",
        "model": "Brio Satya",
        "variant": "E",
        "year": 2020,
        "transmission": "AT",
        "body_type": "Hatchback",
        "kilometer": 45000,
        "location": "Jakarta Barat",
        "plate_code": "B",
        "listing_price": 140_000_000,
        "deskripsi": "Mobil mulus tangan pertama dari baru, terawat.",
        "search_mode": "direct",
    })
