import urllib.request
import urllib.error
from bs4 import BeautifulSoup
from src.url_classifier import classify_url, extract_detail_links_from_html

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 (Jabodetabek car screening research)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "id,en-US;q=0.9,en;q=0.8",
}

urls_to_test = [
    # Carmudi slug variants and parameters
    "https://www.carmudi.co.id/mobil-bekas-dijual/nissan/grand-livina/indonesia",
    "https://www.carmudi.co.id/mobil-bekas-dijual/nissan/grandlivina/indonesia",
    "https://www.carmudi.co.id/mobil-bekas-dijual/nissan/livina/indonesia",
    "https://www.carmudi.co.id/mobil-bekas-dijual/nissan/grand-livina/indonesia?transmission=automatic",
    "https://www.carmudi.co.id/mobil-bekas-dijual/nissan/grand-livina/indonesia?min_year=2015&max_year=2021",
    "https://www.carmudi.co.id/mobil-bekas-dijual/nissan/livina/indonesia?min_year=2019&max_year=2021&transmission=automatic",
    
    # Carsome slug variants
    "https://www.carsome.id/beli-mobil-bekas/nissan/grand-livina",
    "https://www.carsome.id/beli-mobil-bekas/nissan/livina",
    "https://www.carsome.id/beli-mobil-bekas/nissan/livina?year=2020",
    "https://www.carsome.id/beli-mobil-bekas/nissan/livina?transmission=automatic",
]

for url in urls_to_test:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            final_url = resp.geturl()
            status = getattr(resp, "status", 200)
            body = resp.read()
            html = body.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            # find all /dijual/ or /beli-mobil-bekas/ links
            links = [a["href"] for a in soup.find_all("a", href=True) if "/dijual/" in a["href"] or "/beli-mobil-bekas/" in a["href"]]
            details = extract_detail_links_from_html(final_url, html, expected_model="", max_links=50)
            print(f"URL: {url}")
            print(f"  Final: {final_url} | Status: {status} | Size: {len(body)} | Raw matched links: {len(links)} | Classified details: {len(details)}")
            if details:
                print(f"  First 2 details: {details[:2]}")
            else:
                title = soup.title.string.strip() if soup.title and soup.title.string else ""
                print(f"  Title: {title}")
    except urllib.error.HTTPError as e:
        print(f"URL: {url} -> HTTPError: {e.code}")
    except Exception as e:
        print(f"URL: {url} -> Error: {e}")
