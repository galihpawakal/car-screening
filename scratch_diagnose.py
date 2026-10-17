import urllib.request
import urllib.error
from bs4 import BeautifulSoup
from src.url_classifier import classify_url, extract_detail_links_from_html
from src.marketplace_config import MARKETPLACE_CATALOG_CONFIGS, build_catalog_url

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 (Jabodetabek car screening research)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "id,en-US;q=0.9,en;q=0.8",
}

def inspect_url(url, expected_model=""):
    print(f"\n--- Fetching: {url} (model: {expected_model}) ---")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            final_url = resp.geturl()
            status = getattr(resp, "status", 200)
            body = resp.read()
            size = len(body)
            html = body.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.find_all("a", href=True)
            detail_links = extract_detail_links_from_html(final_url, html, expected_model=expected_model, max_links=50)
            
            # also test without expected_model
            all_details = extract_detail_links_from_html(final_url, html, expected_model="", max_links=50)
            
            print(f"Status: {status}, Final URL: {final_url}, Size: {size} bytes, Total <a>: {len(anchors)}")
            print(f"Detail links (with model filter '{expected_model}'): {len(detail_links)}")
            print(f"Detail links (WITHOUT model filter): {len(all_details)}")
            if all_details[:3]:
                print(f"Sample detail links without model filter: {all_details[:3]}")
            if len(detail_links) == 0:
                # Print title and 200 chars body text
                title = (soup.title.string if soup.title else "") or ""
                print(f"Page Title: {title.strip()}")
                first_200 = soup.get_text(" ", strip=True)[:200]
                print(f"First 200 chars text: {first_200}")
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code} for {url}")
        try:
            body = e.read()
            html = body.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            print(f"Title: {soup.title.string if soup.title else ''}")
            print(f"First 200 chars: {soup.get_text(' ', strip=True)[:200]}")
        except Exception:
            pass
    except Exception as e:
        print(f"Error: {e}")

models_to_test = [
    ("Nissan", "Grand Livina", 2020, "AT"),
    ("Toyota", "All New Avanza", 2020, "AT"),
    ("Honda", "Brio Satya", 2020, "AT"),
    ("Toyota", "Agya", 2020, "AT"),
    ("Toyota", "Calya", 2020, "AT"),
]

for brand, model, year, trans in models_to_test:
    print(f"\n==================== {brand} {model} ====================")
    for domain in ["carmudi.co.id", "carsome.id", "oto.com", "mobil123.com", "olx.co.id", "mobilbekas.com"]:
        url = build_catalog_url(domain, brand=brand, model=model, year=year, min_year=year-1, max_year=year+1, transmission=trans)
        if url:
            inspect_url(url, expected_model=model)
