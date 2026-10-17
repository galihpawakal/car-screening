import urllib.request
import re
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
url = "https://www.carmudi.co.id/mobil-bekas-dijual/honda/jazz?state=jabodetabek&min_year=2019&max_year=2021"
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        soup = BeautifulSoup(resp.read().decode('utf-8', errors='ignore'), 'html.parser')
        detail_links = [a.get('href') for a in soup.find_all('a', href=True) if re.search(r'/(?:dijual|listing)/[^/]+/\d+', a.get('href', ''))]
        print(f"Status {resp.status}: {len(detail_links)} detail links with ?state=jabodetabek")
        for lk in detail_links[:5]:
            print(" ", lk)
except Exception as e:
    print("Error:", e)
