"""
jumia_scraper.py
────────────────
Scrapes missing product data from Jumia for post-QC enrichment.

Supported countries : KE · UG · NG · GH · MA
Scraped fields      : COLOR, COUNT_VARIATIONS, PRODUCT_WARRANTY,
                      WARRANTY_DURATION, MAIN_IMAGE, PRICE, DISCOUNT,
                      RATING, REVIEW_COUNT, STOCK_STATUS
"""

from __future__ import annotations

import logging
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Country base URLs ──────────────────────────────────────────────────────────
COUNTRY_URLS: dict[str, str] = {
    "KE": "https://www.jumia.co.ke",
    "UG": "https://www.jumia.ug",
    "NG": "https://www.jumia.com.ng",
    "GH": "https://www.jumia.com.gh",
    "MA": "https://www.jumia.ma",
}

# ── All fields this module can fill ──────────────────────────────────────────
SCRAPABLE_FIELDS: list[str] = [
    "COLOR",
    "COUNT_VARIATIONS",
    "PRODUCT_WARRANTY",
    "WARRANTY_DURATION",
    "MAIN_IMAGE",
    "PRICE",
    "DISCOUNT",
    "RATING",
    "REVIEW_COUNT",
    "STOCK_STATUS",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)

# ── Possible SKU column names (checked in order) ──────────────────────────────
_SKU_CANDIDATES = [
    "PRODUCT_SET_SID", "SellerSku", "SELLER_SKU", "seller_sku",
    "SKU", "Sku", "sku",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_sku(sku: str) -> str:
    """Strip the Jumia suffix that follows a hyphen (e.g. -269939913)."""
    return sku.split("-")[0].strip()


def _find_sku_column(df: pd.DataFrame) -> str | None:
    for col in _SKU_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _needs_fill(df: pd.DataFrame, col: str, threshold: float = 0.5) -> bool:
    """
    Return True when a column is absent or has fewer than `threshold` of rows
    filled with non-empty, non-NaN values.
    """
    if col not in df.columns:
        return True
    filled = (
        df[col].astype(str).str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
        .ne("")
        .sum()
    )
    return (filled / max(len(df), 1)) < threshold


def _row_is_empty(val) -> bool:
    return not str(val).strip() or str(val).strip().lower() in ("nan", "none", "")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – find product URL via search
# ─────────────────────────────────────────────────────────────────────────────

def _search_product_url(sku: str, base_url: str) -> str | None:
    """Search Jumia for `sku` and return the first product page URL, or None."""
    search_url = f"{base_url}/catalog/?q={_base_sku(sku)}"
    try:
        resp = _SESSION.get(search_url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Search request failed for %s: %s", sku, exc)
        return None

    text = resp.text
    if any(p in text.lower() for p in ("no results", "there are no results", "aucun résultat")):
        return None

    soup = BeautifulSoup(text, "html.parser")

    # Primary selector used on most Jumia sites
    card = soup.select_one("article.prd a.core")
    if card:
        href = card.get("href", "")
        return (base_url + href) if href.startswith("/") else href or None

    # Fallback: any .html link that is NOT a category/catalog link
    for a in soup.select("a[href*='.html']"):
        href = a.get("href", "")
        if href and "/catalog/" not in href and "/c/" not in href:
            return (base_url + href) if href.startswith("/") else href

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – scrape data from the product page
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_product_page(url: str, base_url: str) -> dict[str, str]:
    """
    Download a Jumia product page and return a dict of all fields
    we could extract.  Missing fields are simply absent from the dict.
    """
    result: dict[str, str] = {}
    try:
        resp = _SESSION.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Product page request failed for %s: %s", url, exc)
        return result

    soup = BeautifulSoup(resp.text, "html.parser")
    full_text = soup.get_text(" ", strip=True)

    # ── Main Image ────────────────────────────────────────────────────────────
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        result["MAIN_IMAGE"] = og["content"]
    else:
        for img in soup.select("img.-fw.-fh, img[data-src*='jumia']"):
            src = img.get("data-src") or img.get("src", "")
            if src and ("product" in src or "unsafe" in src or "jumia.is" in src):
                result["MAIN_IMAGE"] = src if src.startswith("http") else base_url + src
                break

    # ── Price ─────────────────────────────────────────────────────────────────
    for sel in [
        "span.-b.-ltr.-tal.-fs24",
        "span.prc",
        "[class*='price'] span",
        "span[data-price]",
    ]:
        el = soup.select_one(sel)
        if el:
            raw = re.sub(r"[^\d.,]", "", el.get_text())
            if raw:
                result["PRICE"] = raw.replace(",", "")
                break

    # ── Discount ──────────────────────────────────────────────────────────────
    for sel in ["span.bdg._dsct._sm", "span._dsct", "[class*='discount']"]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(strip=True)
            if "%" in txt:
                result["DISCOUNT"] = txt
                break

    # ── Rating ────────────────────────────────────────────────────────────────
    for sel in ["div.stars._s", "[class*='stars'] span", "div[data-rating]"]:
        el = soup.select_one(sel)
        if el:
            rating_text = el.get("data-rating") or el.get_text(strip=True)
            m = re.search(r"(\d+\.?\d*)", rating_text)
            if m:
                result["RATING"] = m.group(1)
                break

    # ── Review count ──────────────────────────────────────────────────────────
    for sel in ["a[href*='#reviews']", "span[class*='revw']", "[class*='review'] span"]:
        el = soup.select_one(sel)
        if el:
            m = re.search(r"(\d+)", el.get_text())
            if m:
                result["REVIEW_COUNT"] = m.group(1)
                break

    # ── Stock status ──────────────────────────────────────────────────────────
    out_indicators = [
        "div.out-of-stock", "[class*='sold-out']",
        "[class*='out-of-stock']", "button[disabled]",
    ]
    result["STOCK_STATUS"] = "In Stock"
    for sel in out_indicators:
        if soup.select_one(sel):
            result["STOCK_STATUS"] = "Out of Stock"
            break

    # ── Colors / Variations ───────────────────────────────────────────────────
    colors: list[str] = []

    # Method A: variation list items (common Jumia pattern)
    for li in soup.select("ul.-pvs.-mvs li"):
        inp = li.select_one("input")
        if inp:
            val = inp.get("value") or inp.get("data-value", "")
            if val:
                colors.append(val.strip())

    # Method B: colour swatches / selectors
    if not colors:
        for el in soup.select("[data-type='color'] span, span.color-selector"):
            txt = el.get_text(strip=True)
            if txt:
                colors.append(txt)

    # Method C: size options (count as variations even if not colours)
    variation_count: int = 0
    if not colors:
        opts = soup.select("ul.-pvs li, select option:not([value=''])")
        variation_count = len(opts)

    if colors:
        unique_colors = list(dict.fromkeys(colors))  # deduplicate, preserve order
        result["COLOR"] = ", ".join(unique_colors)
        result["COUNT_VARIATIONS"] = str(len(unique_colors))
    elif variation_count:
        result["COUNT_VARIATIONS"] = str(variation_count)

    # ── Warranty ──────────────────────────────────────────────────────────────
    # Look for explicit warranty text in the description / spec table
    warranty_pattern = re.compile(
        r"(\d+[\s\-]?(?:year|month|yr|mo)[s]?\s+(?:warranty|guarantee))",
        re.IGNORECASE,
    )
    m = warranty_pattern.search(full_text)
    if m:
        result["PRODUCT_WARRANTY"] = "Yes"
        result["WARRANTY_DURATION"] = m.group(1).strip()
    else:
        # Check spec rows for the word "warranty"
        for row in soup.select("ul li, table tr, div.-fs14"):
            txt = row.get_text(" ", strip=True)
            if re.search(r"warranty", txt, re.IGNORECASE):
                val = re.sub(r"[Ww]arranty\s*[:\-]?\s*", "", txt).strip()
                if val and len(val) < 80:
                    result["PRODUCT_WARRANTY"] = "Yes"
                    result.setdefault("WARRANTY_DURATION", val)
                    break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def scrape_single_sku(
    sku: str,
    country_code: str = "KE",
) -> dict[str, str]:
    """
    Scrape all available fields for a single SKU.
    Returns a dict of field_name → value (only filled fields included).
    """
    base_url = COUNTRY_URLS.get(country_code, COUNTRY_URLS["KE"])
    url = _search_product_url(sku, base_url)
    if not url:
        return {}
    return _scrape_product_page(url, base_url)


def enrich_post_qc_df(
    df: pd.DataFrame,
    country_code: str = "KE",
    progress_callback=None,
    delay: float = 0.6,
) -> pd.DataFrame:
    """
    Enrich a post-QC dataframe with data scraped from Jumia.

    Parameters
    ----------
    df               : The normalised post-QC dataframe.
    country_code     : Two-letter country code — KE / UG / NG / GH / MA.
    progress_callback: Optional callable(done: int, total: int, sku: str).
    delay            : Seconds to wait between requests (politeness).

    Returns
    -------
    A new dataframe with missing fields filled where Jumia data was found.
    """
    base_url = COUNTRY_URLS.get(country_code, COUNTRY_URLS["KE"])

    # Which columns actually need scraping?
    cols_needed = [c for c in SCRAPABLE_FIELDS if _needs_fill(df, c)]
    if not cols_needed:
        logger.info("All scrapable columns already populated — skipping enrichment.")
        return df

    sku_col = _find_sku_column(df)
    if sku_col is None:
        logger.warning("No SKU column found in dataframe — cannot enrich.")
        return df

    df = df.copy()

    # Ensure every target column exists
    for col in cols_needed:
        if col not in df.columns:
            df[col] = ""

    total = len(df)
    logger.info(
        "Enriching %d rows | country=%s | fields=%s",
        total, country_code, cols_needed,
    )

    for seq, (row_idx, row) in enumerate(df.iterrows()):
        sku = str(row.get(sku_col, "")).strip()

        # Skip rows with no SKU
        if _row_is_empty(sku):
            if progress_callback:
                progress_callback(seq + 1, total, "—")
            continue

        # Skip rows that are fully populated for all needed columns
        row_needs = any(_row_is_empty(row.get(col, "")) for col in cols_needed)
        if not row_needs:
            if progress_callback:
                progress_callback(seq + 1, total, sku)
            continue

        try:
            product_url = _search_product_url(sku, base_url)
            if product_url:
                scraped = _scrape_product_page(product_url, base_url)
                for col, val in scraped.items():
                    if col in cols_needed and _row_is_empty(df.at[row_idx, col]):
                        df.at[row_idx, col] = val
            time.sleep(delay)
        except Exception as exc:
            logger.warning("Enrichment failed for SKU %s: %s", sku, exc)

        if progress_callback:
            progress_callback(seq + 1, total, sku)

    return df
