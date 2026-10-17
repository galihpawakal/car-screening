from src.marketplace_config import build_catalog_url, get_model_slug_variants
from src.page_reader import fetch_page_response
from src.url_classifier import extract_detail_links_from_html

domain = "carmudi.co.id"
brand = "Nissan"
model = "Grand Livina"
year = 2020

variants = get_model_slug_variants(domain, brand, model)
print("Variants:", variants)

for slug in variants:
    url = build_catalog_url(domain, brand, model, year=year, min_year=2019, max_year=2021, transmission="AT", page=1, model_slug_override=slug, include_year_filter=True)
    resp = fetch_page_response(url)
    links = extract_detail_links_from_html(resp.final_url, resp.html, expected_model=model, max_links=25)
    print(f"Slug: {slug} (with year filter 2019-2021) -> {len(links)} links")
    if links:
        print("  First link:", links[0])
