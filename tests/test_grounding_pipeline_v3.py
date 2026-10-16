from pathlib import Path
import pytest

from src.url_classifier import classify_url, extract_detail_links_from_html
from src.detail_parser import normalize_price, normalize_kilometer, parse_detail_page
from src.clean import clean_listings, is_jabodetabek
from src.stats import calculate_statistics, filter_price_outliers_iqr
from src.rules import run_all_rules
from src.scoring import get_market_confidence, insufficient_data_screening_result
from src.extract import ExtractedListing

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestUrlClassification:
    def test_classify_url_oto_com(self):
        detail_url = "https://www.oto.com/detail-mobil-bekas/toyota-avanza-1-3g-at-659460"
        detail_url_alt = "https://www.oto.com/mobil-bekas/toyota/avanza/1-3l-g-mt-668836"
        listing_url = "https://www.oto.com/mobil-bekas/toyota-avanza+2020"
        other_url = "https://www.oto.com/berita/tips-beli-mobil-bekas"

        assert classify_url(detail_url) == "detail"
        assert classify_url(detail_url_alt) == "detail"
        assert classify_url(listing_url) == "listing"
        assert classify_url(other_url) == "other"

    def test_classify_url_carmudi_and_mobil123(self):
        carmudi_detail = "https://www.carmudi.co.id/dijual/toyota-avanza-g-jakarta/123456"
        carmudi_listing = "https://www.carmudi.co.id/mobil-bekas/toyota/avanza"
        m123_detail = "https://www.mobil123.com/dijual/toyota-avanza-g-jakarta/987654"
        m123_listing = "https://www.mobil123.com/mobil-bekas/toyota/avanza"

        assert classify_url(carmudi_detail) == "detail"
        assert classify_url(carmudi_listing) == "listing"
        assert classify_url(m123_detail) == "detail"
        assert classify_url(m123_listing) == "listing"

    def test_classify_url_carsome(self):
        carsome_detail = "https://www.carsome.id/beli-mobil-bekas/toyota/avanza/2021-toyota-avanza-g-1.3/c5uh000"
        carsome_listing = "https://www.carsome.id/beli-mobil-bekas/toyota/avanza"

        assert classify_url(carsome_detail) == "detail"
        assert classify_url(carsome_listing) == "listing"

    def test_classify_url_olx(self):
        olx_detail = "https://www.olx.co.id/item/toyota-avanza-g-2020-at-istimewa-iid-918273645"
        olx_listing = "https://www.olx.co.id/jakarta-dki_g2000007/mobil-bekas_c198"

        assert classify_url(olx_detail) == "detail"
        assert classify_url(olx_listing) == "listing"

    def test_classify_fallback_content_heuristic(self):
        oto_detail_html = (FIXTURES_DIR / "oto_detail.html").read_text(encoding="utf-8", errors="ignore")
        oto_listing_html = (FIXTURES_DIR / "oto_listing.html").read_text(encoding="utf-8", errors="ignore")

        # Unknown domain fallback
        assert classify_url("https://unknown-auto.test/page1", html_content=oto_detail_html) == "detail"
        assert classify_url("https://unknown-auto.test/search", html_content=oto_listing_html) == "listing"

    def test_extract_detail_links_from_oto_listing_fixture(self):
        listing_html = (FIXTURES_DIR / "oto_listing.html").read_text(encoding="utf-8", errors="ignore")
        links = extract_detail_links_from_html(
            html_content=listing_html,
            base_url="https://www.oto.com/mobil-bekas/toyota-avanza+2020",
            max_links=5,
        )
        assert len(links) > 0
        assert len(links) <= 5
        for link in links:
            assert classify_url(link) == "detail"
            assert "oto.com" in link


class TestNormalization:
    def test_normalize_price_variations(self):
        assert normalize_price("Rp 153 jt") == 153_000_000
        assert normalize_price("153.000.000") == 153_000_000
        assert normalize_price("Rp 153,5 jt") == 153_500_000
        assert normalize_price("Rp 153.500.000,-") == 153_500_000
        assert normalize_price("153 jt nego") == 153_000_000
        # DP / installment must be rejected / ignored
        assert normalize_price("DP 15 jt") is None
        assert normalize_price("invalid price") is None

    def test_normalize_kilometer_variations(self):
        km, est = normalize_kilometer("55rb km")
        assert km == 55_000
        assert est is False

        km, est = normalize_kilometer("55.000 km")
        assert km == 55_000
        assert est is False

        km, est = normalize_kilometer("50-55 rb km")
        assert km == 52_500
        assert est is True

        km, est = normalize_kilometer("50.000 - 60.000 km")
        assert km == 55_000
        assert est is True

        km, est = normalize_kilometer("50000-55000")
        assert km == 52_500
        assert est is True

        km, est = normalize_kilometer("baru")
        assert km is None


class TestDetailParsingWithFixtures:
    def test_parse_oto_detail_fixture(self):
        html = (FIXTURES_DIR / "oto_detail.html").read_text(encoding="utf-8", errors="ignore")
        url = "https://www.oto.com/detail-mobil-bekas/toyota-avanza-1-3g-mt-668836"
        res = parse_detail_page(html, url, expected_brand="Toyota", expected_model="Avanza")
        assert res is not None
        assert res.brand.lower() == "toyota"
        assert "avanza" in res.model.lower()
        assert normalize_price(res.price) == 152_000_000
        km_num, _ = normalize_kilometer(res.kilometer)
        assert km_num == 85_969
        assert res.year == "2017"
        assert res.transmission == "MT"
        assert res.location == "Pekanbaru"

    def test_parse_carmudi_detail_fixture(self):
        html = (FIXTURES_DIR / "carmudi_detail.html").read_text(encoding="utf-8", errors="ignore")
        url = "https://www.carmudi.co.id/dijual/toyota-avanza-g-jakarta/12345"
        res = parse_detail_page(html, url, expected_brand="Toyota", expected_model="Kijang Innova")
        assert res is not None
        assert res.brand.lower() == "toyota"
        assert "innova" in res.model.lower()
        assert normalize_price(res.price) == 390_000_000
        assert res.year == "2020"
        assert res.transmission == "AT"
        assert "Jakarta" in res.location

    def test_parse_carsome_detail_fixture(self):
        html = (FIXTURES_DIR / "carsome_detail.html").read_text(encoding="utf-8", errors="ignore")
        url = "https://www.carsome.id/beli-mobil-bekas/toyota/avanza/c123456"
        res = parse_detail_page(html, url, expected_brand="Toyota", expected_model="Avanza")
        assert res is not None
        assert res.brand.lower() == "toyota"
        assert "avanza" in res.model.lower()
        assert normalize_price(res.price) == 177_000_000
        assert res.year == "2021"
        assert res.transmission == "MT"
        assert "Bogor" in res.location


class TestComparableValidation:
    def test_clean_listings_validations_and_rejections(self):
        listings = [
            # 1. Valid exact match
            ExtractedListing(
                brand="Toyota", model="Avanza", variant="G", year="2020",
                transmission="AT", kilometer="80.000 km", price="Rp 160.000.000",
                location="Jakarta Selatan", seller_type="dealer", source_url="https://oto.com/1"
            ),
            # 2. Valid different variant (marked lower weight)
            ExtractedListing(
                brand="Toyota", model="Avanza", variant="Veloz", year="2021",
                transmission="AT", kilometer="70.000 km", price="Rp 175.000.000",
                location="Bekasi", seller_type="dealer", source_url="https://oto.com/2"
            ),
            # 3. Rejected: Model mismatch
            ExtractedListing(
                brand="Toyota", model="Innova", variant="G", year="2020",
                transmission="AT", kilometer="80.000 km", price="Rp 260.000.000",
                location="Jakarta Pusat", seller_type="dealer", source_url="https://oto.com/3"
            ),
            # 4. Rejected: Year difference > 1
            ExtractedListing(
                brand="Toyota", model="Avanza", variant="G", year="2017",
                transmission="AT", kilometer="80.000 km", price="Rp 120.000.000",
                location="Jakarta Pusat", seller_type="dealer", source_url="https://oto.com/4"
            ),
            # 5. Rejected: Transmission mismatch (MT vs AT)
            ExtractedListing(
                brand="Toyota", model="Avanza", variant="G", year="2020",
                transmission="MT", kilometer="80.000 km", price="Rp 150.000.000",
                location="Jakarta Pusat", seller_type="dealer", source_url="https://oto.com/5"
            ),
            # 6. Rejected: Non-Jabodetabek location
            ExtractedListing(
                brand="Toyota", model="Avanza", variant="G", year="2020",
                transmission="AT", kilometer="80.000 km", price="Rp 155.000.000",
                location="Pekanbaru, Riau", seller_type="dealer", source_url="https://oto.com/6"
            ),
            # 7. Rejected: Price too low (DP/cicilan)
            ExtractedListing(
                brand="Toyota", model="Avanza", variant="G", year="2020",
                transmission="AT", kilometer="80.000 km", price="Rp 15.000.000",
                location="Jakarta Pusat", seller_type="dealer", source_url="https://oto.com/7"
            ),
        ]

        clean_res = clean_listings(
            extracted=listings,
            expected_model="Avanza",
            expected_brand="Toyota",
            expected_year=2020,
            expected_transmission="AT",
            expected_variant="G",
            require_jabodetabek=True,
        )
        cleaned = clean_res.listings
        rejected = clean_res.rejected

        assert len(cleaned) == 2
        assert cleaned[0].is_same_variant is True
        assert cleaned[1].is_same_variant is False

        assert len(rejected) == 5
        reasons = [r.reason for r in rejected]
        assert any("Model tidak cocok" in r for r in reasons)
        assert any("Tahun di luar rentang" in r for r in reasons)
        assert any("Transmisi tidak cocok" in r for r in reasons)
        assert any("Lokasi di luar Jabodetabek" in r for r in reasons)
        assert any("Harga di luar rentang" in r for r in reasons)

    def test_is_jabodetabek_helper(self):
        assert is_jabodetabek("Jakarta Pusat") is True
        assert is_jabodetabek("Tangerang Selatan, Banten") is True
        assert is_jabodetabek("Bekasi Barat") is True
        assert is_jabodetabek("Bogor Kota") is True
        assert is_jabodetabek("Depok") is True
        assert is_jabodetabek("Surabaya, Jawa Timur") is False
        assert is_jabodetabek("Medan") is False
        assert is_jabodetabek("Bandung") is False


class TestRedFlagsAndScoring:
    def test_price_too_low_red_flag_suppressed_when_under_3_comparables(self):
        # Median is 160 jt, target 100 jt is 37.5% below median.
        # If n_comparables == 1, PRICE_TOO_LOW must NOT trigger.
        rules_under_3 = run_all_rules(
            listing_price=100_000_000,
            price_median=160_000_000,
            kilometer=80_000,
            year=2020,
            model="Avanza",
            description="Kondisi terawat, siap pakai.",
            current_year=2026,
            n_comparables=1,
        )
        rule_codes_under_3 = [r.code for r in rules_under_3]
        assert "PRICE_TOO_LOW" not in rule_codes_under_3

        # If n_comparables == 4, PRICE_TOO_LOW should trigger.
        rules_ge_3 = run_all_rules(
            listing_price=100_000_000,
            price_median=160_000_000,
            kilometer=80_000,
            year=2020,
            model="Avanza",
            description="Kondisi terawat, siap pakai.",
            current_year=2026,
            n_comparables=4,
        )
        rule_codes_ge_3 = [r.code for r in rules_ge_3]
        assert "PRICE_TOO_LOW" in rule_codes_ge_3

    def test_confidence_levels_and_neutral_notes(self):
        # < 2
        conf_0 = get_market_confidence(0, 0.1, 100.0)
        assert conf_0["level"] == "Sangat rendah"
        conf_1 = get_market_confidence(1, 0.1, 100.0)
        assert conf_1["level"] == "Sangat rendah"

        # 2 (indikatif)
        conf_2 = get_market_confidence(2, 0.1, 100.0)
        assert conf_2["level"] == "Rendah"

        # 3-5
        conf_3 = get_market_confidence(3, 0.1, 100.0)
        assert conf_3["level"] == "Sedang"
        conf_5 = get_market_confidence(5, 0.1, 100.0)
        assert conf_5["level"] == "Sedang"

        # >= 6
        conf_6 = get_market_confidence(6, 0.1, 100.0)
        assert conf_6["level"] == "Tinggi"

    def test_outlier_filtering_iqr(self):
        # Normal prices around 150m, plus extreme outliers (10m and 900m)
        prices = [10_000_000, 145_000_000, 150_000_000, 152_000_000, 155_000_000, 160_000_000, 900_000_000]
        filtered = filter_price_outliers_iqr(prices)
        assert 10_000_000 not in filtered
        assert 900_000_000 not in filtered
        assert len(filtered) == 5

    def test_insufficient_data_screening_result_neutral_note(self):
        res = insufficient_data_screening_result(
            rule_flags=[],
            has_description=True,
            has_varian=True,
            has_plate=True,
            has_location=True,
            llm_red_flags=["Harga listing 35% di bawah estimasi pasar (mencurigakan)"],
        )
        assert res.total_score == -1
        assert res.price_label == "Tidak dapat dinilai"
        # The suspicious red flag must be purged
        assert not any("mencurigakan" in f.lower() for f in res.llm_red_flags)
        # Neutral note present
        price_comp = next(c for c in res.components if c.name == "Kewajaran Harga")
        assert "Data pembanding terlalu sedikit untuk menilai kewajaran harga." in price_comp.reason
        assert price_comp.assessed is False
