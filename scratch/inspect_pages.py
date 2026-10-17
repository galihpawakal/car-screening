import urllib.request
import re
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

def inspect_carmudi_location():
    print("=== INSPECTING CARMUDI LOCATION ===")
    test_urls = [
        "https://www.carmudi.co.id/mobil-bekas-dijual/honda/jazz?city=jakarta",
        "https://www.carmudi.co.id/mobil-bekas-dijual/honda/jazz?location=jakarta",
        "https://www.carmudi.co.id/mobil-bekas-dijual/honda/jazz?region=jakarta",
        "https://www.carmudi.co.id/mobil-bekas-dijual/indonesia/jakarta/honda/jazz",
        "https://www.carmudi.co.id/cars-for-sale/jakarta/honda/jazz",
        "https://www.carmudi.co.id/mobil-dijual/jakarta/honda/jazz",
        "https://www.carmudi.co.id/mobil-bekas-dijual/honda/jazz/indonesia/jakarta",
        "https://www.carmudi.co.id/mobil-bekas-dijual/honda/jazz?city_id=1",
    ]
    for url in test_urls:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html, 'html.parser')
                detail_links = [a.get('href') for a in soup.find_all('a', href=True) if re.search(r'/(?:dijual|listing)/[^/]+/\d+', a.get('href', ''))]
                print(f"Status 200: {url} -> {len(detail_links)} detail links")
        except urllib.error.HTTPError as e:
            print(f"Status {e.code}: {url}")
        except Exception as e:
            print(f"Error {url}: {e}")

def inspect_carmudi_filter_form():
    print("\n=== INSPECTING CARMUDI FILTER FORM ===")
    url = "https://www.carmudi.co.id/mobil-bekas-dijual/honda/jazz"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')
            # Look for location select/inputs
            loc_inputs = soup.find_all(['select', 'input'], attrs={'name': re.compile(r'loc|city|reg|prov|area|state', re.I)})
            print(f"Found {len(loc_inputs)} location inputs:")
            for inp in loc_inputs:
                print(" ", inp.get('name'), inp.get('id'), inp.get('type'))
            # Look for location filter links in sidebar
            loc_links = [a.get('href') for a in soup.find_all('a', href=True) if any(k in a.get('href', '').lower() for k in ['jakarta', 'banten', 'jawa-barat'])]
            print(f"Found {len(loc_links)} links containing jakarta/banten/jawa-barat:")
            for lk in loc_links[:10]:
                print("  ", lk)
    except Exception as e:
        print("Carmudi form error:", e)

if __name__ == '__main__':
    inspect_carmudi_location()
    inspect_carmudi_filter_form()
