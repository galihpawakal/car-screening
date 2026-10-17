import urllib.request
import re

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
url = "https://www.carsome.id/beli-mobil-bekas/honda/jazz"
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        # Cari semua pattern URL mobil Carsome di seluruh HTML, termasuk script tags
        # Format Carsome: /beli-mobil-bekas/<brand>/<model>/<slug>/<id>
        matches = re.findall(r'(/beli-mobil-bekas/[a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_.]+/c[a-z0-9]+)', html)
        print("Regex matches in entire HTML:", len(matches))
        for m in set(matches):
            print("  ", m)
        
        # Cek apakah ada kata 2020 di html
        y2020 = re.findall(r'.{0,50}2020.{0,50}', html)
        print(f"Total occurrences of '2020': {len(y2020)}")
        for y in y2020[:5]:
            print("  snippet:", y.strip())
except Exception as e:
    print("Error:", e)
