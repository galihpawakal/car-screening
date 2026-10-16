"""Domain catalog configuration, filter parameters, and twin model mappings for car screening."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote_plus


@dataclass(frozen=True)
class MarketplaceDomainConfig:
    name: str
    base_url: str
    catalog_template: str
    detail_url_pattern: re.Pattern
    pagination_param: str
    max_pages: int = 5
    items_per_page: int = 25
    supports_transmission: bool = True
    supports_year_filter: bool = True
    notes: str = ""


# Peta model kembar sekelas (Tier 4 fallback)
# Jika pembanding < 3, model kembar boleh dipakai dengan bobot 0.5
TWIN_MODELS_MAP: dict[str, list[str]] = {
    "xenia": ["avanza"],
    "avanza": ["xenia"],
    "sigra": ["calya"],
    "calya": ["sigra"],
    "agya": ["ayla", "brio"],
    "ayla": ["agya", "brio"],
    "brio": ["agya", "ayla"],
    "rush": ["terios"],
    "terios": ["rush"],
    "raize": ["rocky"],
    "rocky": ["raize"],
    "ertiga": ["xl7"],
    "xl7": ["ertiga"],
    "mobilio": ["br-v"],
    "br-v": ["mobilio"],
    "innova": ["innova reborn", "innova zenix"],
}


def get_twin_models(model: str) -> list[str]:
    """Return list of twin/equivalent models in the same vehicle class."""
    m_clean = model.strip().lower()
    for key, twins in TWIN_MODELS_MAP.items():
        if key == m_clean or m_clean in key:
            return twins
    return []


# Konfigurasi parameter nyata per domain
MARKETPLACE_CATALOG_CONFIGS: dict[str, MarketplaceDomainConfig] = {
    "carmudi.co.id": MarketplaceDomainConfig(
        name="Carmudi Indonesia",
        base_url="https://www.carmudi.co.id",
        catalog_template="https://www.carmudi.co.id/mobil-bekas-dijual/{brand_slug}/{model_slug}/indonesia",
        detail_url_pattern=re.compile(r"https?://(?:www\.)?carmudi\.co\.id/dijual/[^/]+/\d+", re.I),
        pagination_param="page_number",
        max_pages=5,
        items_per_page=25,
        supports_transmission=True,
        supports_year_filter=True,
        notes="Mendukung min_year, max_year, transmission=automatic|manual, page_number.",
    ),
    "carsome.id": MarketplaceDomainConfig(
        name="Carsome Indonesia",
        base_url="https://www.carsome.id",
        catalog_template="https://www.carsome.id/beli-mobil-bekas/{brand_slug}/{model_slug}",
        detail_url_pattern=re.compile(r"/beli-mobil-bekas/[^/]+/[^/]+/(?:\d{4}-)?[^/]+/[a-z0-9]+", re.I),
        pagination_param="pageNo",
        max_pages=5,
        items_per_page=20,
        supports_transmission=True,
        supports_year_filter=True,
        notes="Mendukung year, transmission=automatic|manual, pageNo. Tahun terlihat di slug.",
    ),
    "oto.com": MarketplaceDomainConfig(
        name="Oto.com",
        base_url="https://www.oto.com",
        catalog_template="https://www.oto.com/mobil-bekas/{brand_slug}-{model_slug}-{year}",
        detail_url_pattern=re.compile(r"https?://(?:www\.)?oto\.com/mobil-bekas/[^/]+/\d+", re.I),
        pagination_param="page",
        max_pages=3,
        items_per_page=20,
        supports_transmission=False,
        supports_year_filter=True,
        notes="Client-side JS rendering. Dicoba, bila tidak ada detail lolos dicatat di data dibuang.",
    ),
    "mobil123.com": MarketplaceDomainConfig(
        name="Mobil123",
        base_url="https://www.mobil123.com",
        catalog_template="https://www.mobil123.com/mobil-dijual/{brand_slug}/{model_slug}/indonesia",
        detail_url_pattern=re.compile(r"https?://(?:www\.)?mobil123\.com/dijual/[^/]+/\d+", re.I),
        pagination_param="page_number",
        max_pages=3,
        items_per_page=25,
        supports_transmission=True,
        supports_year_filter=True,
        notes="Cloudflare anti-scraping (HTTP 403). Dilewati dan dicatat di data dibuang.",
    ),
    "olx.co.id": MarketplaceDomainConfig(
        name="OLX Indonesia",
        base_url="https://www.olx.co.id",
        catalog_template="https://www.olx.co.id/mobil-bekas_c198/q-{brand_slug}-{model_slug}-{year}",
        detail_url_pattern=re.compile(r"https?://(?:www\.)?olx\.co\.id/item/[^/]+-iid-\d+", re.I),
        pagination_param="page",
        max_pages=2,
        items_per_page=20,
        supports_transmission=False,
        supports_year_filter=True,
        notes="Anti-bot connection limits. Dicoba dengan timeout pendek, dicatat jika gagal.",
    ),
    "mobilbekas.com": MarketplaceDomainConfig(
        name="Mobilbekas.com",
        base_url="https://mobilbekas.com",
        catalog_template="https://mobilbekas.com/search?keyword={brand_slug}+{model_slug}+{year}",
        detail_url_pattern=re.compile(r"https?://mobilbekas\.com/(?:iklan|detail)/[^/]+", re.I),
        pagination_param="page",
        max_pages=2,
        items_per_page=20,
        supports_transmission=False,
        supports_year_filter=True,
        notes="Katalog direct search. Dicoba, dilewati bila format berubah.",
    ),
}


def build_catalog_url(
    domain: str,
    brand: str,
    model: str,
    *,
    year: int | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    transmission: str = "",
    page: int = 1,
) -> str | None:
    """Build a filtered catalog URL for a specific domain based on real verified parameters."""
    cfg = MARKETPLACE_CATALOG_CONFIGS.get(domain)
    if not cfg:
        return None

    brand_slug = brand.strip().lower().replace(" ", "-")
    model_slug = model.strip().lower().replace(" ", "-")
    year_val = str(year) if year else ""

    url = cfg.catalog_template.format(
        brand_slug=brand_slug,
        model_slug=model_slug,
        year=year_val,
    )

    params: list[str] = []

    # Transmission mapping
    trans_upper = transmission.strip().upper()
    if cfg.supports_transmission:
        if trans_upper in ("AT", "AUTOMATIC", "OTOMATIS"):
            params.append("transmission=automatic")
        elif trans_upper in ("MT", "MANUAL"):
            params.append("transmission=manual")

    # Year filter mapping per domain
    if cfg.supports_year_filter:
        if domain == "carmudi.co.id":
            # Carmudi uses min_year and max_year
            actual_min = min_year if min_year is not None else (year - 1 if year else None)
            actual_max = max_year if max_year is not None else (year + 1 if year else None)
            if actual_min is not None:
                params.append(f"min_year={actual_min}")
            if actual_max is not None:
                params.append(f"max_year={actual_max}")
        elif domain == "carsome.id":
            # Carsome supports ?year=2020
            if year:
                params.append(f"year={year}")

    # Pagination
    if page > 1 and cfg.pagination_param:
        params.append(f"{cfg.pagination_param}={page}")

    if params:
        separator = "&" if "?" in url else "?"
        url = url + separator + "&".join(params)

    return url


def extract_year_from_slug(url: str) -> int | None:
    """Extract 4-digit vehicle year from URL path or slug if present."""
    match = re.search(r"(?:/|[_\-])((?:19|20)\d{2})(?:[_\-/]|$)", url)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def build_manual_search_urls(
    brand: str,
    model: str,
    year: int,
    transmission: str = "",
) -> dict[str, str]:
    """Generate direct manual search links for users to open when results are 0 or low."""
    brand_slug = brand.strip().lower().replace(" ", "-")
    model_slug = model.strip().lower().replace(" ", "-")
    trans_upper = transmission.strip().upper()
    trans_slug = "automatic" if trans_upper in ("AT", "AUTOMATIC", "OTOMATIS") else ("manual" if trans_upper in ("MT", "MANUAL") else "")

    links: dict[str, str] = {}

    # Carmudi
    carmudi_params = [f"min_year={year - 1}", f"max_year={year + 1}"]
    if trans_slug:
        carmudi_params.append(f"transmission={trans_slug}")
    links["Carmudi"] = f"https://www.carmudi.co.id/mobil-bekas-dijual/{brand_slug}/{model_slug}/indonesia?" + "&".join(carmudi_params)

    # Carsome
    carsome_params = [f"year={year}"]
    if trans_slug:
        carsome_params.append(f"transmission={trans_slug}")
    links["Carsome"] = f"https://www.carsome.id/beli-mobil-bekas/{brand_slug}/{model_slug}?" + "&".join(carsome_params)

    # Oto.com
    links["Oto.com"] = f"https://www.oto.com/mobil-bekas/{brand_slug}-{model_slug}-{year}"

    # OLX
    links["OLX"] = f"https://www.olx.co.id/mobil-bekas_c198/q-{brand_slug}-{model_slug}-{year}"

    # Mobil123
    links["Mobil123"] = f"https://www.mobil123.com/mobil-dijual/{brand_slug}/{model_slug}/indonesia"

    # Mobilbekas.com
    links["Mobilbekas"] = f"https://mobilbekas.com/search?keyword={brand_slug}+{model_slug}+{year}"

    return links

