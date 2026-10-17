from bs4 import BeautifulSoup
from src.marketplace_config import (
    MARKETPLACE_CATALOG_CONFIGS,
    build_catalog_url,
    get_model_slug_variants,
    slugify_model,
)
from src.page_reader import fetch_page_response
from src.url_classifier import extract_detail_links_from_html

def test_crawl(domain, brand, model, year=2020, trans="AT"):
    print(f"\n=== CRAWL TEST: {domain} for {brand} {model} {year} {trans} ===")
    cfg = MARKETPLACE_CATALOG_CONFIGS.get(domain)
    if not cfg:
        print("No config")
        return []
    if not cfg.is_accessible_from_server:
        print(f"Skipping inaccessible domain: {domain} ({cfg.notes})")
        return []

    variants = get_model_slug_variants(domain, brand, model)
    print("Slug variants to try:", variants)
    
    working_slug = None
    working_include_year = True
    collected = []
    
    for slug in variants:
        for inc_year in (True, False):
            url = build_catalog_url(
                domain, brand, model, year=year, min_year=year-1, max_year=year+1,
                transmission=trans, page=1, model_slug_override=slug, include_year_filter=inc_year
            )
            print(f"  Trying URL (slug={slug}, inc_year={inc_year}): {url}")
            resp = fetch_page_response(url)
            soup = BeautifulSoup(resp.html, "html.parser")
            anchors = len(soup.find_all("a", href=True))
            links = extract_detail_links_from_html(resp.final_url, resp.html, expected_model=model, max_links=50)
            print(f"    Status: {resp.status}, Bytes: {resp.response_bytes}, Anchors: {anchors}, Links found: {len(links)}")
            if len(links) > 0:
                working_slug = slug
                working_include_year = inc_year
                collected.extend(links)
                print(f"    --> Found {len(links)} links using slug '{slug}' (year_filter={inc_year})!")
                break
            else:
                if resp.status == 404:
                    print(f"    --> 404 for slug '{slug}'")
                elif resp.status == 200:
                    first_200 = soup.get_text(" ", strip=True)[:200]
                    print(f"    --> 0 links (200 OK), snippet: {first_200[:100]}...")
        if working_slug:
            break
            
    return collected

# Test Grand Livina
test_crawl("carmudi.co.id", "Nissan", "Grand Livina")
test_crawl("carsome.id", "Nissan", "Grand Livina")
test_crawl("oto.com", "Nissan", "Grand Livina")
test_crawl("mobil123.com", "Nissan", "Grand Livina")
