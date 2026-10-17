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
    is_accessible_from_server: bool = True
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
        if key == m_clean or m_clean in key or key in m_clean:
            return twins
    # Also check with marketing prefix stripped
    m_sub = re.sub(r"^(?:all\s+new|the\s+new)\s+", "", m_clean).strip()
    for key, twins in TWIN_MODELS_MAP.items():
        if key == m_sub or m_sub in key or key in m_sub:
            return twins
    return []


# Aturan slug model per domain dan fallback global
DOMAIN_SLUG_RULES: dict[str, dict[str, str]] = {
    "carsome.id": {
        "grand livina": "livina",
        "all new avanza": "avanza",
        "brio satya": "brio",
        "innova reborn": "innova",
        "kijang innova": "innova",
        "all new xenia": "xenia",
    },
    "carmudi.co.id": {
        "grand livina": "grand-livina",
        "all new avanza": "avanza",
        "brio satya": "brio",
        "innova reborn": "innova",
        "kijang innova": "innova",
        "all new xenia": "xenia",
    },
    "default": {
        "all new avanza": "avanza",
        "all new xenia": "xenia",
        "all new ertiga": "ertiga",
        "all new livina": "livina",
        "grand livina": "grand-livina",
        "brio satya": "brio",
        "innova reborn": "innova",
        "kijang innova": "innova",
    },
}

MODEL_SLUG_VARIANTS: dict[str, list[str]] = {
    "grand livina": ["grand-livina", "livina", "grandlivina", "grand_livina"],
    "livina": ["livina", "grand-livina", "grandlivina"],
    "all new avanza": ["avanza", "all-new-avanza", "allnewavanza"],
    "brio satya": ["brio", "brio-satya", "briosatya"],
    "innova reborn": ["innova", "kijang-innova", "innova-reborn"],
    "kijang innova": ["innova", "kijang-innova"],
}


def slugify_model(domain: str, brand: str, model: str) -> str:
    """Return the normalized model slug for a given domain and model name."""
    m_clean = model.strip().lower()
    # Check domain specific mapping
    domain_rules = DOMAIN_SLUG_RULES.get(domain, {})
    if m_clean in domain_rules:
        return domain_rules[m_clean]

    # Check default rules
    default_rules = DOMAIN_SLUG_RULES.get("default", {})
    if m_clean in default_rules:
        return default_rules[m_clean]

    # Remove generic marketing prefixes like "all new" or "the new"
    m_sub = re.sub(r"^(?:all\s+new|the\s+new)\s+", "", m_clean).strip()
    if m_sub in default_rules:
        return default_rules[m_sub]

    return re.sub(r"[\s_]+", "-", m_sub).strip("-")


def get_model_slug_variants(domain: str, brand: str, model: str) -> list[str]:
    """Return ordered list of slug candidates to try if primary slug fails or returns 0."""
    primary = slugify_model(domain, brand, model)
    m_clean = model.strip().lower()
    candidates = [primary]

    known_variants = MODEL_SLUG_VARIANTS.get(m_clean, [])
    for v in known_variants:
        if v not in candidates:
            candidates.append(v)

    # Standard morphological variations
    hyphen_slug = re.sub(r"[\s_]+", "-", m_clean)
    no_space_slug = re.sub(r"[\s_-]+", "", m_clean)
    underscore_slug = re.sub(r"[\s-]+", "_", m_clean)
    for s in (hyphen_slug, no_space_slug, underscore_slug):
        if s and s not in candidates:
            candidates.append(s)

    return candidates


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
        is_accessible_from_server=True,
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
        is_accessible_from_server=True,
        notes="Mendukung year, transmission=automatic|manual, pageNo. Tahun terlihat di slug.",
    ),
    "oto.com": MarketplaceDomainConfig(
        name="Oto.com",
        base_url="https://www.oto.com",
        catalog_template="https://www.oto.com/mobil-bekas/{brand_slug}-{model_slug}-{year}",
        detail_url_pattern=re.compile(r"https?://(?:www\.)?oto\.com/detail-mobil-bekas/[^/]+", re.I),
        pagination_param="page",
        max_pages=3,
        items_per_page=20,
        supports_transmission=False,
        supports_year_filter=True,
        is_accessible_from_server=False,
        notes="tidak dapat diakses dari server (halaman butuh JavaScript)",
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
        is_accessible_from_server=False,
        notes="tidak dapat diakses dari server (diblokir 403 / Cloudflare)",
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
        is_accessible_from_server=False,
        notes="tidak dapat diakses dari server (koneksi timeout / anti-bot)",
    ),
    "mobilbekas.com": MarketplaceDomainConfig(
        name="Mobilbekas.com",
        base_url="https://mobilbekas.com",
        catalog_template="https://mobilbekas.com/search?keyword={brand_slug}+{model_slug}+{year}",
        detail_url_pattern=re.compile(r"https?://mobilbekas\.com/(?:(?:iklan|detail)/[^/]+|[^/]+/[^/]+/\d+-[^/]+\.html)", re.I),
        pagination_param="page",
        max_pages=2,
        items_per_page=20,
        supports_transmission=False,
        supports_year_filter=True,
        is_accessible_from_server=False,
        notes="parser belum terverifikasi",
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
    model_slug_override: str | None = None,
    include_year_filter: bool = True,
    include_transmission_filter: bool = True,
) -> str | None:
    """Build a filtered catalog URL for a specific domain based on real verified parameters."""
    cfg = MARKETPLACE_CATALOG_CONFIGS.get(domain)
    if not cfg:
        return None

    brand_slug = brand.strip().lower().replace(" ", "-")
    model_slug = model_slug_override or slugify_model(domain, brand, model)
    year_val = str(year) if year else ""

    url = cfg.catalog_template.format(
        brand_slug=brand_slug,
        model_slug=model_slug,
        year=year_val,
    )

    params: list[str] = []

    # Transmission mapping
    trans_upper = transmission.strip().upper()
    if cfg.supports_transmission and include_transmission_filter:
        if trans_upper in ("AT", "AUTOMATIC", "OTOMATIS"):
            params.append("transmission=automatic")
        elif trans_upper in ("MT", "MANUAL"):
            params.append("transmission=manual")

    # Domain specific location filters
    if domain == "carmudi.co.id":
        # Batasi listing hanya ke Jabodetabek agar Semarang, Surabaya, Malang tidak ikut terambil
        params.append("state=jabodetabek")

    # Year filter mapping per domain
    if cfg.supports_year_filter and include_year_filter:
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
    trans_upper = transmission.strip().upper()
    trans_slug = "automatic" if trans_upper in ("AT", "AUTOMATIC", "OTOMATIS") else ("manual" if trans_upper in ("MT", "MANUAL") else "")

    links: dict[str, str] = {}

    # Carmudi
    carmudi_slug = slugify_model("carmudi.co.id", brand, model)
    carmudi_params = [f"min_year={year - 1}", f"max_year={year + 1}"]
    if trans_slug:
        carmudi_params.append(f"transmission={trans_slug}")
    links["Carmudi"] = f"https://www.carmudi.co.id/mobil-bekas-dijual/{brand_slug}/{carmudi_slug}/indonesia?" + "&".join(carmudi_params)

    # Carsome
    carsome_slug = slugify_model("carsome.id", brand, model)
    carsome_params = [f"year={year}"]
    if trans_slug:
        carsome_params.append(f"transmission={trans_slug}")
    links["Carsome"] = f"https://www.carsome.id/beli-mobil-bekas/{brand_slug}/{carsome_slug}?" + "&".join(carsome_params)

    # Oto.com
    oto_slug = slugify_model("oto.com", brand, model)
    links["Oto.com"] = f"https://www.oto.com/mobil-bekas/{brand_slug}-{oto_slug}-{year}"

    # OLX
    olx_slug = slugify_model("olx.co.id", brand, model)
    links["OLX"] = f"https://www.olx.co.id/mobil-bekas_c198/q-{brand_slug}-{olx_slug}-{year}"

    # Mobil123
    m123_slug = slugify_model("mobil123.com", brand, model)
    links["Mobil123"] = f"https://www.mobil123.com/mobil-dijual/{brand_slug}/{m123_slug}/indonesia"

    # Mobilbekas.com
    mb_slug = slugify_model("mobilbekas.com", brand, model)
    links["Mobilbekas"] = f"https://mobilbekas.com/search?keyword={brand_slug}+{mb_slug}+{year}"

    return links

