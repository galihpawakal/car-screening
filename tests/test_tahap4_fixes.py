import pytest
from src.analyze import calculate_negotiation_prices, NegotiationAdvice
from src.extract import ExtractedListing
from src.scoring import get_market_confidence, negotiation_advice_text
from src.tiered_screening import evaluate_tiered_comparables
from src.ui import _summary_text
from src.pipeline import ScreeningPipelineResult
from src.analyze import FinalAnalysis
from src.search_provider import GroundingSource


def test_guard_negotiation_unreasonable_price_50pct():
    """Jika |harga listing - median| / median > 50%, negosiasi harus None."""
    median = 200_000_000
    # Kasus Jazz 2020 Rp 10 jt (jauh di bawah 50%)
    cheap_listing = 10_000_000
    nego_cheap = calculate_negotiation_prices(median, cheap_listing)
    assert nego_cheap.awal is None
    assert nego_cheap.realistis is None
    assert nego_cheap.batas is None

    # Kasus harga kemahalan > 50% (350 jt vs 200 jt)
    expensive_listing = 350_000_000
    nego_expensive = calculate_negotiation_prices(median, expensive_listing)
    assert nego_expensive.awal is None
    assert nego_expensive.realistis is None
    assert nego_expensive.batas is None

    # Kasus wajar (210 jt vs 200 jt, diff 5%)
    normal_listing = 210_000_000
    nego_normal = calculate_negotiation_prices(median, normal_listing)
    assert nego_normal.awal is not None
    assert nego_normal.realistis is not None
    assert nego_normal.batas is not None

    # Pesan saran negosiasi untuk selisih > 50%
    diff_pct = (cheap_listing - median) / median
    text = negotiation_advice_text(diff_pct, 5)
    assert "Harga tidak wajar terhadap pasar" in text
    assert "kemungkinan salah ketik, harga DP, atau penipuan" in text


def test_summary_text_unreasonable_price_consistent():
    """Di ringkasan teks salin, tidak boleh ada nominal nego dan teks harus konsisten."""
    user_input = {
        "brand": "Honda",
        "model": "Jazz",
        "variant": "E",
        "year": 2020,
        "transmission": "AT",
        "kilometer": 50000,
        "listing_price": 10_000_000,  # 10 jt vs 200 jt
        "location": "Jakarta Selatan",
    }
    stats = {
        "count": 5,
        "price_low": 180_000_000,
        "price_median": 200_000_000,
        "price_high": 220_000_000,
        "price_spread_ratio": 0.20,
        "average_km_per_year": 10000,
    }
    final_analysis = FinalAnalysis(
        kesimpulan="Harga tidak wajar terhadap pasar; kemungkinan salah ketik, harga DP, atau penipuan. Tidak ada saran negosiasi.",
        saran_nego=NegotiationAdvice(awal=None, realistis=None, batas=None),
    )
    from src.extract import ComparableListing
    comp = ComparableListing(
        brand="Honda", model="Jazz", variant="E", year=2020, kilometer=50000,
        transmission="AT", location="Jakarta", seller_type="dealer", price=200_000_000,
        source_url="https://carmudi.co.id/dijual/honda-jazz/1", fetched_at="2026-10-17T00:00:00Z",
    )
    dummy_source = GroundingSource(url="https://google.com/search?q=jazz", title="Google")
    result = ScreeningPipelineResult(
        comparables=[comp],
        rejected=[],
        sources=[dummy_source],
        queries=[],
        fetched_at="2026-10-17",
        statistics=stats,
        screening={"total_score": 50, "category_label": "Perlu Cek"},
        final_analysis=final_analysis,
        confidence={"level": "Sedang", "reason": "Berdasarkan 5 pembanding"},
        warning=None,
        ai_error=None,
        cache_hit=False,
        daily_count=1,
        usage={"grounding_calls": 1, "input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        description_skipped=True,
    )

    report = _summary_text(result, user_input)
    assert "Negosiasi awal/target/batas:" not in report
    assert "Harga tidak wajar terhadap pasar; kemungkinan salah ketik, harga DP, atau penipuan. Tidak ada saran negosiasi." in report
    # Sumber hanya pembanding yang lolos
    assert "Sumber: https://carmudi.co.id/dijual/honda-jazz/1" in report
    assert "https://google.com/search" not in report


def test_invariant_accounting_rejected_listings():
    """Invariant: detail_terbaca == lolos + sum(dibuang per alasan)."""
    candidates = [
        ExtractedListing(
            brand="Honda", model="Jazz", variant="RS", year="2020", kilometer="40000",
            transmission="AT", location="Jakarta", price="210000000", seller_type="dealer", source_url="https://site.com/1"
        ),
        ExtractedListing(
            brand="Honda", model="Jazz", variant="RS", year="2020", kilometer="50000",
            transmission="MT", location="Jakarta", price="190000000", seller_type="dealer", source_url="https://site.com/2"  # transmisi beda
        ),
        ExtractedListing(
            brand="Honda", model="Jazz", variant="RS", year="2018", kilometer="60000",
            transmission="AT", location="Jakarta", price="170000000", seller_type="dealer", source_url="https://site.com/3"  # tahun di luar
        ),
        ExtractedListing(
            brand="Honda", model="Jazz", variant="RS", year="2020", kilometer="45000",
            transmission="AT", location="Surabaya", price="205000000", seller_type="dealer", source_url="https://site.com/4"  # lokasi luar
        ),
        ExtractedListing(
            brand="Toyota", model="Yaris", variant="G", year="2020", kilometer="45000",
            transmission="AT", location="Jakarta", price="195000000", seller_type="dealer", source_url="https://site.com/5"  # model beda
        ),
    ]

    res = evaluate_tiered_comparables(
        candidates,
        expected_brand="Honda",
        expected_model="Jazz",
        expected_year=2020,
        expected_transmission="AT",
        expected_variant="RS",
        require_jabodetabek=True,
        target_count=1,
    )

    total_candidates = len(candidates)
    passed = len(res.listings)
    rejected_total = sum(len(urls) for urls in res.rejected_reasons.values())

    assert passed == 1
    assert passed + rejected_total == total_candidates
    assert "Transmisi tidak cocok" in str(res.rejected_reasons)
    assert "Lokasi di luar Jabodetabek" in str(res.rejected_reasons)


def test_dealer_duplicate_detection():
    """Dua listing dengan judul, harga, KM, dan lokasi identik harus didedup."""
    cand1 = ExtractedListing(
        brand="Honda", model="Jazz", variant="RS", year="2020", kilometer="40000",
        transmission="AT", location="Jakarta Pusat", price="210000000", seller_type="dealer", source_url="https://site.com/jazz-1"
    )
    cand2 = ExtractedListing(
        brand="Honda", model="Jazz", variant="RS", year="2020", kilometer="40000",
        transmission="AT", location="Jakarta Pusat", price="210000000", seller_type="dealer", source_url="https://site.com/jazz-2"
    )

    res = evaluate_tiered_comparables(
        [cand1, cand2],
        expected_brand="Honda",
        expected_model="Jazz",
        expected_year=2020,
        expected_transmission="AT",
        expected_variant="RS",
        target_count=1,
    )
    assert len(res.listings) == 1
    total_rej = sum(len(urls) for urls in res.rejected_reasons.values())
    assert total_rej == 1
    assert any("Duplikat listing dari dealer yang sama" in k for k in res.rejected_reasons.keys())


def test_confidence_single_domain_penalty_and_spread_note():
    """Jika semua pembanding berasal dari 1 domain unik, keyakinan turun 1 level."""
    # Normal 4 pembanding -> Sedang
    conf_multi = get_market_confidence(count=4, price_spread_ratio=0.15, completeness_pct=85, unique_domains_count=2)
    assert conf_multi["level"] == "Sedang"
    assert "Semua pembanding berasal dari 1 sumber" not in conf_multi["reason"]

    # 1 domain unik -> Turun dari Sedang ke Rendah
    conf_single = get_market_confidence(count=4, price_spread_ratio=0.15, completeness_pct=85, unique_domains_count=1)
    assert conf_single["level"] == "Rendah"
    assert "Semua pembanding berasal dari 1 sumber." in conf_single["reason"]

    # 1 domain unik pada 6 pembanding -> Turun dari Tinggi ke Sedang
    conf_single_high = get_market_confidence(count=6, price_spread_ratio=0.15, completeness_pct=85, unique_domains_count=1)
    assert conf_single_high["level"] == "Sedang"
    assert "Semua pembanding berasal dari 1 sumber." in conf_single_high["reason"]

    # Sebaran harga sangat rapat (< 5%) pada n <= 5
    conf_tight = get_market_confidence(count=4, price_spread_ratio=0.03, completeness_pct=85, unique_domains_count=2)
    assert "sebaran sangat rapat, periksa kemungkinan listing dari dealer yang sama" in conf_tight["reason"]
