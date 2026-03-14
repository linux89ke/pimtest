"""
jumia_scraper.py
----------------
Scrapes Jumia product detail pages to fill in columns that the post-QC
export file doesn't include:
  - Color
  - Product Warranty / Warranty Duration
  - Count Variations
  - Main Image URL  (if missing)

Only fetches the fields that are actually absent from the uploaded file,
so it's as fast as possible.

Usage (called from streamlit_app.py):
    from jumia_scraper import enrich_post_qc_df
    df = enrich_post_qc_df(df, country_code="KE", progress_callback=fn)
"""

from __future__ import annotations

import re
import time
import logging
import concurrent.futures
from typing import Dict, Optional, Callable

import requests
from bs4 import BeautifulSoup
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Country → base URL
# ---------------------------------------------------------------------------
COUNTRY_BASE_URLS: Dict[str, str] = {
    "KE": "https://www.jumia.co.ke",
    "UG": "https://www.jumia.ug",
    "NG": "https://www.jumia.com.ng",
    "GH": "https://www.jumia.com.gh",
    "MA": "https://www.jumia.ma",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ---------------------------------------------------------------------------
# Columns that might be absent from the post-QC export
# ---------------------------------------------------------------------------
SCRAPEABLE_COLS = {
    "Color":              "COLOR",
    "Product Warranty":   "PRODUCT_WARRANTY",
    "Warranty Duration":  "WARRANTY_DURATION",
    "Count Variations":   "COUNT_VARIATIONS",
    "Main Image":         "MAIN_IMAGE",
}


def _missing_cols(df: pd.DataFrame) -> list[str]:
    """Return the internal (UPPER) column names that are absent or all-blank."""
    missing = []
    for _, internal in SCRAPEABLE_COLS.items():
        if internal not in df.columns:
            missing.append(internal)
        elif df[internal].astype(str).str.strip().replace("nan", "").eq("").all():
            missing.append(internal)
    return missing


# ---------------------------------------------------------------------------
# URL builder  — search by SKU then follow first result
# ---------------------------------------------------------------------------

def _search_url(sku: str, base: str) -> str:
    return f"{base}/catalog/?q={sku}"


def _get_product_url(sku: str, base: str, session: requests.Session) -> Optional[str]:
    """Find the product detail URL from a catalog search page."""
    try:
        r = session.get(_search_url(sku, base), timeout=12, headers=_HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Primary: structured article links
        for sel in ["article.prd a.core", "a.core[href]", "a[href*='.html']"]:
            links = soup.select(sel)
            if links:
                href = links[0].get("href", "")
                if href.startswith("/"):
                    href = base + href
                if ".html" in href:
                    return href

        # Fallback: og:url
        og = soup.find("meta", property="og:url")
        if og and og.get("content"):
            return og["content"]

    except Exception as e:
        logger.debug(f"_get_product_url({sku}): {e}")
    return None


# ---------------------------------------------------------------------------
# Core page scraper
# ---------------------------------------------------------------------------

def _scrape_product_page(url: str, session: requests.Session) -> dict:
    """
    Scrape a single Jumia product page and return a dict with any of:
      COLOR, PRODUCT_WARRANTY, WARRANTY_DURATION, COUNT_VARIATIONS, MAIN_IMAGE
    Keys are only present when a value was actually found.
    """
    result: dict = {}
    try:
        r = session.get(url, timeout=15, headers=_HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # ── Main image ──────────────────────────────────────────────────────
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            result["MAIN_IMAGE"] = og_img["content"].strip()

        # ── Specifications table / dl list ──────────────────────────────────
        # Jumia uses two layouts: a <table> and a <dl> list
        specs: Dict[str, str] = {}

        # Layout A: <table class="sku-specs">
        for table in soup.select("table.sku-specs, table.-specs, table"):
            for row in table.select("tr"):
                cells = row.select("th, td")
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).lower()
                    val = cells[1].get_text(strip=True)
                    if key and val:
                        specs[key] = val

        # Layout B: <div class="sku-specs"> with <dl><dt><dd>
        for dl in soup.select("dl, .sku-specs"):
            dts = dl.select("dt")
            dds = dl.select("dd")
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True).lower()
                val = dd.get_text(strip=True)
                if key and val:
                    specs[key] = val

        # Layout C: Generic -content rows (newer Jumia layout)
        for item in soup.select(".-content .-row, .sku-attr-item, [class*='spec']"):
            spans = item.select("span, b, strong")
            if len(spans) >= 2:
                key = spans[0].get_text(strip=True).lower()
                val = spans[1].get_text(strip=True)
                if key and val:
                    specs[key] = val

        # ── Extract specific fields from specs ──────────────────────────────

        # COLOR
        for key in ("color", "colour", "color/finish", "colour/finish"):
            if key in specs:
                result["COLOR"] = specs[key]
                break
        # Fallback: look for color in title / breadcrumb
        if "COLOR" not in result:
            title_tag = soup.select_one("h1[class*='title'], h1")
            if title_tag:
                title_text = title_tag.get_text()
                # common color words
                color_pat = re.compile(
                    r'\b(black|white|silver|gold|blue|red|green|grey|gray|'
                    r'pink|purple|orange|yellow|brown|beige|navy|rose|'
                    r'champagne|graphite|midnight|titanium|cosmic)\b',
                    re.IGNORECASE
                )
                m = color_pat.search(title_text)
                if m:
                    result["COLOR"] = m.group(0).title()

        # WARRANTY
        for key in specs:
            if "warranty" in key:
                val = specs[key]
                # Try to split "1 Year" → duration
                dur_match = re.search(r'(\d+\s*(?:year|month|day|yr)s?)', val, re.IGNORECASE)
                if dur_match:
                    result["WARRANTY_DURATION"] = dur_match.group(1)
                result["PRODUCT_WARRANTY"] = val
                break

        # COUNT VARIATIONS (number of size/color options on the page)
        variation_count = 0

        # Jumia renders variations as swatches or radio buttons
        for sel in [
            ".-variations .-item",
            ".variation-list li",
            "[class*='variation'] [class*='item']",
            ".-attrs .-item",
        ]:
            items = soup.select(sel)
            if items:
                variation_count = len(items)
                break

        # Fallback: count <option> tags inside a select for size/color
        if variation_count == 0:
            for sel_tag in soup.select("select"):
                opts = [o for o in sel_tag.select("option")
                        if o.get_text(strip=True).lower() not in ("", "select", "choose")]
                if len(opts) > variation_count:
                    variation_count = len(opts)

        if variation_count > 0:
            result["COUNT_VARIATIONS"] = str(variation_count)

    except Exception as e:
        logger.debug(f"_scrape_product_page({url}): {e}")

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_post_qc_df(
    df: pd.DataFrame,
    country_code: str = "KE",
    max_workers: int = 6,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    delay_between_requests: float = 0.3,
) -> pd.DataFrame:
    """
    Enrich a post-QC dataframe with data scraped from Jumia product pages.

    Only scrapes columns that are genuinely absent / blank in the dataframe.
    If nothing is missing, returns the dataframe untouched immediately.

    Parameters
    ----------
    df : pd.DataFrame
        The normalised post-QC dataframe (after normalize_post_qc has run).
        Must contain PRODUCT_SET_SID and CATEGORY columns.
    country_code : str
        Two-letter Jumia country code: KE, UG, NG, GH, MA.
    max_workers : int
        Thread pool size for concurrent scraping.
    progress_callback : callable(done, total, current_sku)
        Optional — called after each SKU is processed.
    delay_between_requests : float
        Seconds to wait between requests per thread (politeness).

    Returns
    -------
    pd.DataFrame
        The input dataframe with missing columns filled in where data was found.
    """
    missing = _missing_cols(df)
    if not missing:
        logger.info("enrich_post_qc_df: nothing to scrape, all columns present")
        return df

    logger.info(f"enrich_post_qc_df: will scrape {missing} for {len(df)} rows")

    base_url = COUNTRY_BASE_URLS.get(country_code.upper(), COUNTRY_BASE_URLS["KE"])
    df = df.copy()

    # Ensure target columns exist
    for col in missing:
        if col not in df.columns:
            df[col] = ""

    skus = df["PRODUCT_SET_SID"].astype(str).str.strip().tolist()
    total = len(skus)

    # Shared session (connection pooling)
    session = requests.Session()
    session.headers.update(_HEADERS)

    def _process_one(idx_sku):
        idx, sku = idx_sku
        enriched = {}
        try:
            product_url = _get_product_url(sku, base_url, session)
            if product_url:
                time.sleep(delay_between_requests)
                enriched = _scrape_product_page(product_url, session)
        except Exception as e:
            logger.debug(f"enrich SKU {sku}: {e}")
        return idx, sku, enriched

    results: Dict[int, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, (i, s)): i for i, s in enumerate(skus)}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            idx, sku, enriched = future.result()
            results[idx] = enriched
            done_count += 1
            if progress_callback:
                progress_callback(done_count, total, sku)

    # Write results back into the dataframe
    for idx, enriched in results.items():
        for col, val in enriched.items():
            if col not in missing:
                continue
            if not val:
                continue
            current = str(df.at[idx, col]).strip()
            if current in ("", "nan", "None"):
                df.at[idx, col] = val

    session.close()
    logger.info("enrich_post_qc_df: scraping complete")
    return df


def build_product_url_from_sku(sku: str, country_code: str = "KE") -> str:
    """Convenience: return the search URL for a single SKU."""
    base = COUNTRY_BASE_URLS.get(country_code.upper(), COUNTRY_BASE_URLS["KE"])
    return _search_url(sku, base)
