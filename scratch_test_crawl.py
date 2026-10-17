from src.marketplace_config import build_catalog_url
from src.page_reader import fetch_page_html
from src.url_classifier import extract_detail_links_from_html

# Test carmudi with Livina vs Grand Livina
for url in [
    "https://www.carmudi.co.id/mobil-bekas-dijual/nissan/grand-livina/indonesia?transmission=automatic",
    "https://www.carmudi.co.id/mobil-bekas-dijual/nissan/livina/indonesia?transmission=automatic",
]:
    html = fetch_page_html(url)
    links_livina = extract_detail_links_from_html(url, html, expected_model="Grand Livina", max_links=20)
    print(url, "-> links with expected_model='Grand Livina':", len(links_livina))
    if links_livina:
        print("  Sample link:", links_livina[0])

# Test carsome
for url in [
    "https://www.carsome.id/beli-mobil-bekas/nissan/livina?transmission=automatic",
    "https://www.carsome.id/beli-mobil-bekas/nissan/grand-livina",
]:
    html = fetch_page_html(url)
    links_carsome = extract_detail_links_from_html(url, html, expected_model="Grand Livina", max_links=20)
    print(url, "-> links with expected_model='Grand Livina':", len(links_carsome))
    if links_carsome:
        print("  Sample link:", links_carsome[0])
