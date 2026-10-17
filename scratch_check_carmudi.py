import urllib.request
from bs4 import BeautifulSoup
from src.url_classifier import extract_detail_links_from_html

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

url = "https://www.carmudi.co.id/mobil-bekas-dijual/nissan/grand-livina/indonesia"
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode("utf-8", errors="ignore")

soup = BeautifulSoup(html, "html.parser")
for a in soup.find_all("a", href=True):
    if "/dijual/" in a["href"]:
        print("href:", a["href"][:80], "| text:", repr(a.get_text(" ")[:50]))
        break

links_with_model = extract_detail_links_from_html(url, html, expected_model="Grand Livina")
print("extract_detail_links_from_html with 'Grand Livina':", len(links_with_model))
links_without_model = extract_detail_links_from_html(url, html, expected_model="")
print("extract_detail_links_from_html without model:", len(links_without_model))
if links_without_model:
    print("first link:", links_without_model[0])
