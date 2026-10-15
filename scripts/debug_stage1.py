"""Debug Tahap 1: log raw search text and groundingMetadata for one unit."""

from __future__ import annotations

import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv()

from src.pipeline import build_marketplace_search_prompt
from src.search_provider import GeminiGroundingProvider


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    user_input = {
        "brand": "Toyota",
        "model": "Avanza",
        "variant": "G",
        "year": 2020,
        "transmission": "AT",
        "location": "Jakarta Pusat",
    }
    os.environ["CAR_SCREENING_DEBUG"] = "1"
    provider = GeminiGroundingProvider()
    domain = "olx.co.id"
    prompt = build_marketplace_search_prompt(user_input, domain)
    result = provider.search(prompt, marketplace=domain)
    print("=== TEKS RESPON TAHAP 1 ===")
    print(result.text)
    print("=== URL TER-RESOLVE ===")
    for source in result.sources:
        print(source.url, "|", source.title, "| original:", getattr(source, "original_url", ""))
    print("=== QUERIES ===")
    for query in result.queries:
        print("-", query)
    print("=== TOKEN ===")
    print(result.usage.input_tokens, result.usage.output_tokens)


if __name__ == "__main__":
    main()
