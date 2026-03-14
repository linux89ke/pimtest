"""
jumia_scraper.py
────────────────
Scrapes missing product data from Jumia for post-QC enrichment.

Supported countries : KE · UG · NG · GH · MA
Scraped fields      : COLOR, COUNT_VARIATIONS, PRODUCT_WARRANTY,
                      WARRANTY_DURATION, MAIN_IMAGE, PRICE, DISCOUNT,
                      RATING, REVIEW_COUNT, STOCK_STATUS,
                      DESCRIPTION, KEY_FEATURES, WHATS_IN_BOX,
                      MODEL, GTIN, WEIGHT, CATEGORY_PATH, IS_OFFICIAL_STORE
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
    # Pricing & availability
    "PRICE",
    "DISCOUNT",
    "STOCK_STATUS",
    # Identity & classification
    "MODEL",
    "GTIN",
    "BRAND",
    "CATEGORY_PATH",
    "IS_OFFICIAL_STORE",
    # Media
    "MAIN_IMAGE",
    # Physical
    "WEIGHT",
    # Variations
    "COLOR",
    "COLOR_IN_TITLE",
    "COUNT_VARIATIONS",
    "SIZES_AVAILABLE",
    # Warranty
    "PRODUCT_WARRANTY",
    "WARRANTY_DURATION",
    # Ratings
    "RATING",
    "REVIEW_COUNT",
    # Content
    "DESCRIPTION",
    "KEY_FEATURES",
    "KEY_SPECS",
    "SPECIFICATIONS",
    "WHATS_IN_BOX",
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

    Extraction order:
      1. JSON-LD <script type="application/ld+json"> — fastest, most reliable
      2. Embedded JSON in <script> window variables (__STORES__, dataLayer, etc.)
      3. HTML / CSS selector parsing — fallback for anything not in JSON
    """
    import json as _json

    result: dict[str, str] = {}
    try:
        resp = _SESSION.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Product page request failed for %s: %s", url, exc)
        return result

    soup = BeautifulSoup(resp.text, "html.parser")
    full_text = soup.get_text(" ", strip=True)

    # =========================================================================
    # STRATEGY 1 — JSON-LD
    # Jumia injects a Product schema block with price, image, name, description,
    # brand, and sometimes offers/availability.
    # =========================================================================
    ld_data: dict = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            blob = _json.loads(script.string or "")
            # Could be a list of schemas
            if isinstance(blob, list):
                for item in blob:
                    if isinstance(item, dict) and item.get("@type") == "Product":
                        ld_data = item
                        break
            elif isinstance(blob, dict) and blob.get("@type") == "Product":
                ld_data = blob
            if ld_data:
                break
        except Exception:
            continue

    if ld_data:
        # Image
        img = ld_data.get("image")
        if isinstance(img, list):
            img = img[0]
        if img:
            result["MAIN_IMAGE"] = str(img)

        # Description — JSON-LD often has a clean plain-text description
        desc = ld_data.get("description", "")
        if desc and len(desc) > 20:
            result["DESCRIPTION"] = desc.strip()

        # Brand
        brand = ld_data.get("brand", {})
        if isinstance(brand, dict):
            brand = brand.get("name", "")
        if brand:
            result["BRAND"] = str(brand).strip()

        # Price / offers
        offers = ld_data.get("offers", {})
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        if isinstance(offers, dict):
            # JSON-LD price is a raw number (often USD/base currency), NOT the
            # localised display price. Store tentatively; HTML selector wins.
            price = offers.get("price") or offers.get("lowPrice", "")
            if price:
                result["_ld_price_raw"] = str(price).replace(",", "")
            avail = offers.get("availability", "")
            if "InStock" in avail:
                result["_ld_stock"] = "In Stock"
            elif "OutOfStock" in avail:
                result["_ld_stock"] = "Out of Stock"

        # Rating
        agg = ld_data.get("aggregateRating", {})
        if isinstance(agg, dict):
            rv = agg.get("ratingValue", "")
            rc = agg.get("reviewCount", "") or agg.get("ratingCount", "")
            if rv:
                result["RATING"] = str(rv)
            if rc:
                result["REVIEW_COUNT"] = str(rc)

    # =========================================================================
    # STRATEGY 2 — embedded window JSON (dataLayer / __STORES__ / APP_DATA)
    # Jumia injects product attributes as a JS object that we can regex-parse.
    # =========================================================================
    for script in soup.find_all("script"):
        raw = script.string or ""
        if not raw or len(raw) < 50:
            continue

        # dataLayer push — common on Jumia
        if "dataLayer" in raw and "price" in raw.lower():
            m = re.search(r"dataLayer\s*=\s*(\[.*?\])\s*;", raw, re.DOTALL)
            if not m:
                m = re.search(r"dataLayer\.push\s*\((\{.*?\})\s*\)", raw, re.DOTALL)
            if m:
                try:
                    dl = _json.loads(m.group(1))
                    if isinstance(dl, list):
                        dl = dl[0] if dl else {}
                    # Flatten one level of ecommerce wrapping
                    dl = dl.get("ecommerce", dl)
                    dl = dl.get("detail", dl)
                    products = dl.get("products", [dl])
                    p = products[0] if products else dl
                    if not result.get("PRICE") and p.get("price"):
                        result["PRICE"] = str(p["price"]).replace(",", "")
                    if not result.get("BRAND") and p.get("brand"):
                        result["BRAND"] = str(p["brand"])
                    if not result.get("CATEGORY_PATH") and p.get("category"):
                        result["CATEGORY_PATH"] = str(p["category"])
                except Exception:
                    pass

        # __STORES__ — Jumia SPA state blob (older layout)
        if "__STORES__" in raw or "window.store" in raw.lower():
            m = re.search(r"__STORES__\s*=\s*(\{.+?\})\s*;", raw, re.DOTALL)
            if m:
                try:
                    store = _json.loads(m.group(1))
                    pdp = store.get("pdp", store)
                    prod = pdp.get("product", pdp)
                    if not result.get("PRICE") and prod.get("price"):
                        result["PRICE"] = str(prod["price"]).replace(",", "")
                    if not result.get("MAIN_IMAGE") and prod.get("image"):
                        result["MAIN_IMAGE"] = str(prod["image"])
                except Exception:
                    pass

    # =========================================================================
    # STRATEGY 3 — HTML / CSS parsing
    # Used for anything not captured above, and for KEY_FEATURES / WHATS_IN_BOX
    # which are always rendered as HTML lists.
    # =========================================================================

    # ── Main Image (HTML fallback) ────────────────────────────────────────────
    if not result.get("MAIN_IMAGE"):
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            result["MAIN_IMAGE"] = og["content"]
        else:
            for img in soup.select("img.-fw.-fh, img[data-src*='jumia']"):
                src = img.get("data-src") or img.get("src", "")
                if src and ("product" in src or "unsafe" in src or "jumia.is" in src):
                    result["MAIN_IMAGE"] = src if src.startswith("http") else base_url + src
                    break

    # ── Price (HTML — primary source for local-currency display price) ─────────
    # Always run; overwrites any tentative JSON-LD raw number.
    _html_price_found = False
    for sel in [
        "span.-b.-ltr.-tal.-fs24",   # Jumia current price span
        "span.prc",                   # alternate class
        "span[data-price]",           # data attribute variant
        "span.-b.-ltr.-tal",          # another Jumia variant
        "div.-prc-w span",            # price wrapper div
    ]:
        el = soup.select_one(sel)
        if el:
            raw_p = re.sub(r"[^\d,.]", "", el.get_text())
            if raw_p:
                try:
                    price_val = float(raw_p.replace(",", ""))
                    # Must be at least 100 to avoid matching ratings/fees
                    if price_val >= 100:
                        result["PRICE"] = f"KSh {int(price_val):,}" if price_val == int(price_val) else f"KSh {price_val:,.2f}"
                        _html_price_found = True
                        break
                except ValueError:
                    pass
    # Fallback to JSON-LD raw price only if HTML gave nothing
    if not _html_price_found and result.pop("_ld_price_raw", None):
        result["PRICE"] = result.get("PRICE", "")  # already cleared above

    # ── Discount ──────────────────────────────────────────────────────────────
    for sel in ["span.bdg._dsct._sm", "span._dsct", "[class*='discount']"]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(strip=True)
            if "%" in txt:
                result["DISCOUNT"] = txt
                break

    # ── Rating (HTML fallback) ────────────────────────────────────────────────
    if not result.get("RATING"):
        for sel in ["div.stars._s", "[class*='stars'] span", "div[data-rating]"]:
            el = soup.select_one(sel)
            if el:
                rating_text = el.get("data-rating") or el.get_text(strip=True)
                m = re.search(r"(\d+\.?\d*)", rating_text)
                if m:
                    result["RATING"] = m.group(1)
                    break
    # Explicit "no ratings" state — set 0 so the field is populated, not blank
    if not result.get("RATING"):
        if re.search(r"no\s+ratings?\s+(?:yet|available)", full_text, re.IGNORECASE):
            result["RATING"] = "0"

    # ── Review count (HTML fallback) ──────────────────────────────────────────
    if not result.get("REVIEW_COUNT"):
        for sel in ["a[href*='#reviews']", "span[class*='revw']", "[class*='review'] span"]:
            el = soup.select_one(sel)
            if el:
                m = re.search(r"(\d+)", el.get_text())
                if m:
                    result["REVIEW_COUNT"] = m.group(1)
                    break
        # Also check text like "13 verified ratings"
        if not result.get("REVIEW_COUNT"):
            m = re.search(r"(\d+)\s+verified\s+ratings?", full_text, re.IGNORECASE)
            if m:
                result["REVIEW_COUNT"] = m.group(1)
    # Explicit 0 when page says no ratings
    if not result.get("REVIEW_COUNT"):
        if re.search(r"no\s+ratings?\s+(?:yet|available)", full_text, re.IGNORECASE):
            result["REVIEW_COUNT"] = "0"

    # ── Stock status ──────────────────────────────────────────────────────────
    # Always scrape actual displayed text first (captures nuanced messages like
    # "Some variations with low stock"), then fall back to structural indicators.
    _stock_text = ""
    for sel in [
        "span[class*='stock']", "div[class*='stock']",
        "span[class*='avail']", "div[class*='avail']",
        "p[class*='stock']",
    ]:
        el = soup.select_one(sel)
        if el:
            _t = el.get_text(strip=True)
            if _t and len(_t) < 80:
                _stock_text = _t
                break
    # Also check page text for Jumia's specific low-stock message
    if not _stock_text:
        _low_m = re.search(
            r"((?:some\s+)?variations?\s+with\s+low\s+stock"
            r"|low\s+stock|only\s+\d+\s+left"
            r"|hurry[,!]?\s+(?:only\s+)?\d+"
            r"|in\s+stock|out\s+of\s+stock|sold\s+out)",
            full_text, re.IGNORECASE,
        )
        if _low_m:
            _stock_text = _low_m.group(1).strip()
    if _stock_text:
        result["STOCK_STATUS"] = _stock_text.capitalize()
    else:
        # Structural fallback
        out_indicators = [
            "div.out-of-stock", "[class*='sold-out']",
            "[class*='out-of-stock']", "button[disabled]",
        ]
        result["STOCK_STATUS"] = result.pop("_ld_stock", "In Stock")
        for sel in out_indicators:
            if soup.select_one(sel):
                result["STOCK_STATUS"] = "Out of Stock"
                break
    # Clear the tentative LD stock flag if still present
    result.pop("_ld_stock", None)

    # ── Colors / Sizes / Variations ──────────────────────────────────────────
    # Jumia uses the same variation pill UI for both colours (electronics) and
    # sizes (fashion).  We detect which type we're dealing with and populate
    # the right fields:
    #   COLOR + COUNT_VARIATIONS  — when pills are colour names
    #   SIZES_AVAILABLE + COUNT_VARIATIONS — when pills are sizes (EU/UK/US/numeric)
    #   COUNT_VARIATIONS alone    — when we can count but not label the type

    colors: list[str] = []
    sizes:  list[str] = []

    # ── Helper: classify a value as a size or a colour ────────────────────────
    _SIZE_RE = re.compile(
        r"""
        ^(
            (EU|UK|US|EU/UK)\s*\d{1,3}(\.\d)?   # EU 40, UK 9, US 10.5
          | \d{1,3}(\.\d)?                        # bare number: 40, 42, 9
          | (XS|S|M|L|XL|XXL|XXXL|3XL|4XL|5XL)  # letter sizes
          | (ONE\s*SIZE|FREE\s*SIZE)               # universal sizes
          | \d{1,3}\s*(cm|mm|in|inch|inches)      # measurement
        )$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def _is_size(val: str) -> bool:
        return bool(_SIZE_RE.match(val.strip()))

    def _collect_pills(selector: str) -> list[str]:
        """Return text values of all non-empty pill/option elements.
        Rejects spec-table rows — they contain ': ' like 'SKU: XYZ123'.
        """
        _SPEC_ROW = re.compile(
            r"^(sku|weight|model|gtin|size|material|dimension|barcode"
            r"|main\s+material|color\s*:|colour\s*:)\s*[:\(]",
            re.IGNORECASE,
        )
        vals: list[str] = []
        for el in soup.select(selector):
            # Prefer hidden input value (radio buttons inside li)
            inp = el.select_one("input")
            if inp:
                v = inp.get("value") or inp.get("data-value", "")
            else:
                v = el.get_text(strip=True)
            v = v.strip()
            if not v or len(v) >= 50:
                continue
            # Reject anything that looks like a spec row "Label: Value"
            if ": " in v or _SPEC_ROW.match(v):
                continue
            vals.append(v)
        return vals

    # Strategy A: variation list <li> items (most common Jumia pattern)
    #   Covers both  ul.-pvs.-mvs li  (electronics colours)
    #   and          ul.-pvs li       (fashion sizes shown as pills)
    pill_vals = _collect_pills(
        "ul.-pvs.-mvs li, ul.-pvs li, ul[class*='var'] li"
    )

    # Strategy B: explicit colour/size selectors
    if not pill_vals:
        pill_vals = _collect_pills(
            "[data-type='color'] span, span.color-selector, "
            "li[class*='color'] span, label[class*='color'], "
            "[data-type='size'] span, li[class*='size'] span"
        )

    # Strategy C: <select> dropdown options (some Jumia sellers use a dropdown)
    # IMPORTANT: Jumia pages also contain location/delivery area dropdowns with
    # 100+ options. We must scope to the product form only and cap results to
    # avoid counting those as product variations.
    if not pill_vals:
        # Only look inside the product add-to-cart form
        product_form = (
            soup.select_one("form#product_addtocart_form")
            or soup.select_one("form[action*='checkout']")
            or soup.select_one("section.-pvs")  # Jumia variation section
        )
        if product_form:
            raw_opts = _collect_pills(
                "select option:not([value='']):not([disabled])"
            )
            # Apply within-form filter manually
            scoped_opts: list[str] = []
            for sel_el in product_form.select(
                "select option:not([value='']):not([disabled])"
            ):
                v = sel_el.get_text(strip=True)
                if v and len(v) < 50:
                    scoped_opts.append(v)
            # Sanity cap: location dropdowns have 50+ options; real variation
            # selects almost never exceed 30
            if 0 < len(scoped_opts) <= 30:
                pill_vals = scoped_opts

    # Strategy D: div/button variation pills (newer Jumia layout)
    if not pill_vals:
        pill_vals = _collect_pills(
            "div[class*='var'] button, div[class*='size'] button, "
            "div[class*='color'] button"
        )

    # Now classify whatever we found
    if pill_vals:
        unique_vals = list(dict.fromkeys(v for v in pill_vals if v))
        for v in unique_vals:
            if _is_size(v):
                sizes.append(v)
            else:
                colors.append(v)

    # Strategy E2: split concatenated colour strings like "PinkPurple" or
    # "BlackRedBlue" — Jumia sometimes renders colour pills without separators.
    # IMPORTANT: must run BEFORE the write block so COUNT_VARIATIONS is correct.
    _SPLIT_COLOR_RE = re.compile(
        r"(Black|White|Silver|Gold|Blue|Red|Green|Purple|Pink|Grey|Gray|"
        r"Titanium|Midnight|Starlight|Ivory|Champagne|Rose|Copper|Yellow|"
        r"Orange|Violet|Navy|Cream|Brown|Coral|Aqua|Cyan|Teal|Lilac|"
        r"Maroon|Beige|Olive|Turquoise)"
    )
    if colors and len(colors) == 1:
        split_parts = _SPLIT_COLOR_RE.findall(colors[0])
        if len(split_parts) >= 2:
            colors = list(dict.fromkeys(split_parts))

    # Strategy E: title-word colour extraction — only when pills gave nothing
    if not colors and not sizes:
        title_el = soup.select_one("h1, title")
        if title_el:
            COLOR_WORDS = re.compile(
                r"\b(black|white|silver|gold|blue|red|green|purple|pink|"
                r"grey|gray|titanium|midnight|starlight|ivory|champagne|"
                r"rose|copper|yellow|orange|violet|navy|cream|brown|coral|"
                r"aqua|cyan|teal|lilac|maroon|beige|olive|turquoise)\b",
                re.IGNORECASE,
            )
            found = COLOR_WORDS.findall(title_el.get_text())
            if found:
                colors = list(dict.fromkeys(c.title() for c in found))

    # ── Write results ─────────────────────────────────────────────────────────
    if sizes and colors:
        # Both present (rare — e.g. a product with both size and colour options)
        result["SIZES_AVAILABLE"]  = ", ".join(sizes)
        result["COLOR"]            = ", ".join(colors)
        result["COUNT_VARIATIONS"] = str(len(sizes) * len(colors))
    elif sizes:
        result["SIZES_AVAILABLE"]  = ", ".join(sizes)
        result["COUNT_VARIATIONS"] = str(len(sizes))
    elif colors:
        result["COLOR"]            = ", ".join(colors)
        result["COUNT_VARIATIONS"] = str(len(colors))

    # ── COLOR_IN_TITLE — always runs, independent of pill detection ───────────
    # Extracts colour words directly from the product h1 title.
    # Useful even when COLOR is already set from pills — confirms/complements it.
    # Compound phrases (e.g. "Titanium Silver", "Midnight Blue") are matched
    # before single words so they are captured as one token.
    _COLOR_WORDS = re.compile(
        r"\b("
        # ── Compound phrases first (order matters) ─────────────────────────
        r"titanium\s+silver|titanium\s+gold|midnight\s+black|midnight\s+blue|"
        r"space\s+gray|space\s+grey|rose\s+gold|sky\s+blue|navy\s+blue|"
        r"forest\s+green|hot\s+pink|light\s+blue|dark\s+blue|dark\s+green|"
        r"light\s+green|deep\s+purple|bright\s+red|pearl\s+white|"
        r"starlight\s+white|starlight\s+silver|"
        # ── Single colour words ────────────────────────────────────────────
        r"black|white|silver|gold|blue|red|green|purple|pink|"
        r"grey|gray|titanium|midnight|starlight|ivory|champagne|"
        r"rose|copper|yellow|orange|violet|navy|cream|brown|coral|"
        r"aqua|cyan|teal|lilac|maroon|beige|olive|turquoise"
        r")\b",
        re.IGNORECASE,
    )
    h1 = soup.select_one("h1")
    title_text = h1.get_text(" ", strip=True) if h1 else ""
    if not title_text:
        title_tag = soup.find("title")
        title_text = title_tag.get_text(" ", strip=True) if title_tag else ""

    if title_text:
        found_title_colors = _COLOR_WORDS.findall(title_text)
        if found_title_colors:
            unique_title_colors = list(dict.fromkeys(
                c.title() for c in found_title_colors
            ))
            result["COLOR_IN_TITLE"] = ", ".join(unique_title_colors)
        else:
            # Explicitly mark as None so field is populated, not blank
            result["COLOR_IN_TITLE"] = "None"
    else:
        result["COLOR_IN_TITLE"] = "None"


    # ── Specifications table helper ───────────────────────────────────────────
    # Jumia renders specs as a flat list: "Label Value" lines inside a section.
    # Build a dict once and reuse it for MODEL, GTIN, WEIGHT, etc.
    spec_map: dict[str, str] = {}
    for section in soup.select("div[class*='spec'], section[class*='spec'], ul[class*='spec']"):
        items = section.find_all("li") or section.find_all("div", recursive=False)
        for item in items:
            txt = item.get_text(" ", strip=True)
            # "Label : Value" or "Label Value" patterns
            if ":" in txt:
                parts = txt.split(":", 1)
            else:
                # Try splitting on two-or-more whitespace
                parts = re.split(r"\s{2,}", txt, 1)
            if len(parts) == 2:
                k = parts[0].strip().lower().replace(" ", "_")
                v = parts[1].strip()
                if k and v:
                    spec_map[k] = v

    # Also scrape from the plain-text "Specifications" block as a fallback —
    # Jumia often renders specs as "Key Value\nKey Value" lines.
    if not spec_map:
        spec_section_m = re.search(
            r"Specifications?\s*\n(.*?)(?:Customer Feedback|Related results|\Z)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if spec_section_m:
            for line in spec_section_m.group(1).splitlines():
                line = line.strip()
                if ":" in line:
                    k, _, v = line.partition(":")
                    k = k.strip().lower().replace(" ", "_")
                    v = v.strip()
                    if k and v:
                        spec_map[k] = v

    # ── MODEL ─────────────────────────────────────────────────────────────────
    # Priority 1: JSON-LD "model" field (already in ld_data if parsed)
    if not result.get("MODEL"):
        model_val = ld_data.get("model", "") if ld_data else ""
        if model_val:
            result["MODEL"] = str(model_val).strip()

    # Priority 2: Specifications table
    if not result.get("MODEL"):
        for key in ("model", "model_name", "model_number"):
            if spec_map.get(key):
                result["MODEL"] = spec_map[key]
                break

    # Priority 3: regex on full text (Jumia lists "Model: Smart 10" in specs)
    if not result.get("MODEL"):
        m_mod = re.search(r"\bModel\s*[:\-]\s*([^\n,]+)", full_text, re.IGNORECASE)
        if m_mod:
            result["MODEL"] = m_mod.group(1).strip()

    # ── GTIN ──────────────────────────────────────────────────────────────────
    # Appears as "GTIN Barcode: 04894947084454" in the Specifications section.
    if not result.get("GTIN"):
        # JSON-LD gtin fields
        for gtin_key in ("gtin13", "gtin12", "gtin8", "gtin", "barcode"):
            val = ld_data.get(gtin_key, "") if ld_data else ""
            if val:
                result["GTIN"] = str(val).strip()
                break

    if not result.get("GTIN"):
        for key in ("gtin_barcode", "gtin", "barcode", "ean", "upc"):
            if spec_map.get(key):
                result["GTIN"] = spec_map[key]
                break

    if not result.get("GTIN"):
        m_gtin = re.search(
            r"GTIN\s*(?:Barcode)?\s*[:\-]\s*(\d{8,14})",
            full_text, re.IGNORECASE,
        )
        if m_gtin:
            result["GTIN"] = m_gtin.group(1).strip()

    # ── WEIGHT ────────────────────────────────────────────────────────────────
    # Jumia shows "Weight (kg): 0.187" in the spec table.
    if not result.get("WEIGHT"):
        for key in ("weight_(kg)", "weight_kg", "weight_(g)", "weight"):
            if spec_map.get(key):
                result["WEIGHT"] = spec_map[key]
                break

    if not result.get("WEIGHT"):
        # JSON-LD weight object
        wt = ld_data.get("weight", {}) if ld_data else {}
        if isinstance(wt, dict) and wt.get("value"):
            unit = wt.get("unitCode", "kg")
            result["WEIGHT"] = f"{wt['value']} {unit}"
        elif isinstance(wt, str) and wt:
            result["WEIGHT"] = wt

    if not result.get("WEIGHT"):
        m_wt = re.search(
            r"Weight\s*(?:\(kg\))?\s*[:\-]\s*([\d\.]+\s*(?:kg|g)?)",
            full_text, re.IGNORECASE,
        )
        if m_wt:
            result["WEIGHT"] = m_wt.group(1).strip()

    # ── CATEGORY_PATH ─────────────────────────────────────────────────────────
    # The breadcrumb trail is the most reliable source:
    # Home › Phones & Tablets › Mobile Phones › Smartphones › Android Phones
    if not result.get("CATEGORY_PATH"):
        crumbs: list[str] = []

        # CSS: Jumia breadcrumb links are inside .-bre nav or ol.breadcrumb
        for sel in [
            "ol.breadcrumb a", "nav[aria-label*='breadcrumb'] a",
            "div.-bre a", "a[class*='crumb']",
            "div[class*='breadcrumb'] a",
        ]:
            links = soup.select(sel)
            if links:
                crumbs = [a.get_text(strip=True) for a in links if a.get_text(strip=True)]
                break

        # Fallback: scan for a sequence of short links that look like categories
        if not crumbs:
            for nav in soup.find_all(["nav", "ol", "ul"]):
                links = nav.find_all("a")
                texts = [a.get_text(strip=True) for a in links if a.get_text(strip=True)]
                # A breadcrumb has 3–7 items, each short, no duplicates
                if 3 <= len(texts) <= 7 and len(set(texts)) == len(texts):
                    if all(len(t) < 50 for t in texts):
                        crumbs = texts
                        break

        if crumbs:
            # Drop "Home" prefix and the current product name (last item if long)
            if crumbs[0].lower() in ("home", "jumia"):
                crumbs = crumbs[1:]
            if crumbs and len(crumbs[-1]) > 60:
                crumbs = crumbs[:-1]
            if crumbs:
                result["CATEGORY_PATH"] = " > ".join(crumbs)

    # ── IS_OFFICIAL_STORE ─────────────────────────────────────────────────────
    # Jumia marks official stores with a "JMALL" tag badge and/or an
    # "Official Store" (singular) label near the product title.
    # IMPORTANT: every Jumia page has an "Official Stores" (plural) nav link —
    # we must NOT match that. Use \b word-boundary so "store\b" never matches
    # inside "stores", and scope the text search to exclude the <header>/<nav>.
    if not result.get("IS_OFFICIAL_STORE"):
        # Word-boundary regex: matches "Official Store" but NOT "Official Stores"
        _official_re = re.compile(r"\bofficial\s+store\b", re.IGNORECASE)

        # Remove header/nav from search scope to avoid the nav-bar link
        body_copy = soup.find("body")
        for nav_el in (body_copy.find_all(["header", "nav"]) if body_copy else []):
            nav_el.decompose()

        jmall_badge   = soup.select_one("a[href*='JMALL'], a[tag*='JMALL']")
        official_text = body_copy.find(string=_official_re) if body_copy else None
        official_img  = soup.find("img", alt=_official_re)

        result["IS_OFFICIAL_STORE"] = (
            "Yes" if any([jmall_badge, official_text, official_img]) else "No"
        )


    warranty_pattern = re.compile(
        r"(\d+[\s\-]?(?:year|month|yr|mo)[s]?\s+(?:warranty|guarantee))",
        re.IGNORECASE,
    )
    m = warranty_pattern.search(full_text)
    if m:
        result["PRODUCT_WARRANTY"] = "Yes"
        result["WARRANTY_DURATION"] = m.group(1).strip()
    else:
        for row in soup.select("ul li, table tr, div.-fs14"):
            txt = row.get_text(" ", strip=True)
            if re.search(r"warranty", txt, re.IGNORECASE):
                val = re.sub(r"[Ww]arranty\s*[:\-]?\s*", "", txt).strip()
                if val and len(val) < 80:
                    result["PRODUCT_WARRANTY"] = "Yes"
                    result.setdefault("WARRANTY_DURATION", val)
                    break
        # Also parse "Warranty Address" spec row on Jumia
        m2 = re.search(r"Warranty\s+Address\s*[:\-]?\s*(.+?)(?:\n|$)", full_text, re.IGNORECASE)
        if m2:
            result["PRODUCT_WARRANTY"] = "Yes"
            result.setdefault("WARRANTY_DURATION", m2.group(1).strip())

    # ── SPECIFICATIONS — flat key/value dump from the specs table ───────────────
    # Captures the Specifications section as a readable "Key: Value" string.
    # Jumia shows rows like: SKU, Size (L x W x H cm), Weight (kg), Main Material
    if not result.get("SPECIFICATIONS"):
        spec_lines: list[str] = []
        # CSS: Jumia spec items inside .-sku, .-size, .-wgt etc or generic -atr
        for section in soup.select(
            "div[class*='spec'], section[class*='spec'], "
            "ul[class*='spec'], div.-atr, section.-atr"
        ):
            for item in (section.find_all("li") or section.find_all("div", recursive=False)):
                txt = item.get_text(" ", strip=True)
                if ":" in txt and len(txt) < 200:
                    spec_lines.append(txt.strip())
        # Plain-text fallback — Jumia renders specs in "Specifications" section
        if not spec_lines:
            m_spec = re.search(
                r"Specifications?\s*\n(.*?)(?:Customer Feedback|Related results|\Z)",
                full_text, re.IGNORECASE | re.DOTALL,
            )
            if m_spec:
                for line in m_spec.group(1).splitlines():
                    line = line.strip()
                    if line and len(line) < 200:
                        spec_lines.append(line)
        # Also always add known fields we already scraped
        known = [
            ("SKU",             result.get("MODEL") or ""),
            ("Weight (kg)",     result.get("WEIGHT") or ""),
            ("Main Material",   spec_map.get("main_material", "")),
            ("Size (L x W H)", spec_map.get("size_(l_x_w_x_h_cm)", "")),
        ]
        for label, val in known:
            if val and not any(label.lower() in l.lower() for l in spec_lines):
                spec_lines.append(f"{label}: {val}")
        if spec_lines:
            result["SPECIFICATIONS"] = " | ".join(spec_lines)

    # ── KEY_SPECS — formatted summary of the most important spec fields ─────────
    # A compact, human-readable string combining the key attributes visible in
    # the Specifications panel (SKU, Size, Weight, Material).
    if not result.get("KEY_SPECS"):
        ks_parts: list[str] = []
        # Pull from spec_map (already built above)
        _ks_fields = [
            ("SKU",      "sku"),
            ("Size",     "size_(l_x_w_x_h_cm)"),
            ("Weight",   "weight_(kg)"),
            ("Material", "main_material"),
            ("GTIN",     "gtin_barcode"),
        ]
        for label, key in _ks_fields:
            val = spec_map.get(key, "")
            if val:
                ks_parts.append(f"{label}: {val}")
        # Also use already-scraped fields as fallback
        if not any("SKU" in p for p in ks_parts) and result.get("MODEL"):
            ks_parts.insert(0, f"SKU: {result['MODEL']}")
        if not any("Weight" in p for p in ks_parts) and result.get("WEIGHT"):
            ks_parts.append(f"Weight: {result['WEIGHT']} kg")
        if ks_parts:
            result["KEY_SPECS"] = " | ".join(ks_parts)

    # ── Description (HTML — preserves infographic <img> tags) ────────────────
    # Jumia's description section sits between the "Product details" heading
    # and the "Specifications" heading.  We capture it as *HTML*, not plain
    # text, so that infographic images are preserved for downstream rendering.
    #
    # Image handling
    # --------------
    # Jumia lazy-loads images: the real URL lives in data-src / data-original /
    # data-lazy while src holds a tiny SVG placeholder.  We resolve all img
    # src attributes to the real URL before serialising, and make relative
    # URLs absolute.  We also strip empty placeholder <img> tags entirely.
    if not result.get("DESCRIPTION"):

        STOP_HEADS = re.compile(
            r"^(specifications?|customer\s+feedback|related\s+results|seller\s+info|"
            r"promotions?|delivery|return\s+policy|seller\s+information)",
            re.IGNORECASE,
        )

        def _resolve_imgs(container) -> None:
            """Replace lazy-src placeholders with real URLs in-place."""
            for img in container.find_all("img"):
                real_src = (
                    img.get("data-src")
                    or img.get("data-original")
                    or img.get("data-lazy")
                    or img.get("src", "")
                )
                # Skip SVG placeholders and data URIs
                if not real_src or real_src.startswith("data:"):
                    img.decompose()
                    continue
                # Make relative URLs absolute
                if real_src.startswith("//"):
                    real_src = "https:" + real_src
                elif real_src.startswith("/"):
                    real_src = base_url + real_src
                img["src"] = real_src
                # Clean up lazy-load attributes so output HTML is tidy
                for attr in ("data-src", "data-original", "data-lazy",
                             "data-srcset", "srcset", "loading"):
                    if img.has_attr(attr):
                        del img[attr]
                # Ensure alt is present
                if not img.get("alt"):
                    img["alt"] = ""

        def _html(tag) -> str:
            """Return the outer HTML of a tag as a clean string."""
            return str(tag)

        desc_html_parts: list[str] = []

        # ── Strategy A: collect nodes between "Product details" and the
        #    next boundary heading, walking all siblings in document order.
        #    This captures both text blocks AND infographic images.
        collecting = False
        for tag in soup.find_all(
            ["h2", "h3", "h4", "p", "ul", "ol", "div", "img", "figure"],
            limit=500,
        ):
            txt = tag.get_text(" ", strip=True)

            # Trigger: "Product details" heading
            if tag.name in ("h2", "h3", "h4") and re.search(
                r"product\s+details?", txt, re.IGNORECASE
            ):
                collecting = True
                continue

            if not collecting:
                continue

            # Stop trigger: boundary headings
            if tag.name in ("h2", "h3", "h4") and STOP_HEADS.search(txt):
                break

            # Skip empty or navigation/boilerplate divs
            if tag.name == "div":
                cls = " ".join(tag.get("class", []))
                if any(
                    skip in cls.lower()
                    for skip in ("nav", "bread", "cart", "seller",
                                 "rate", "review", "footer", "header",
                                 "sidebar", "related", "promo", "banner")
                ):
                    continue
                # Only include divs that contain either an img OR substantial text
                has_img = bool(tag.find("img"))
                has_text = len(txt) > 30
                if not has_img and not has_text:
                    continue

            # Resolve lazy images before serialising
            _resolve_imgs(tag)

            serialised = _html(tag).strip()
            if serialised and serialised not in desc_html_parts:
                desc_html_parts.append(serialised)

        # ── Strategy B: look for a dedicated description container div
        if not desc_html_parts:
            for sel in [
                "div.-prd-desc",
                "div[class*='description']",
                "section[class*='description']",
                "div[class*='product-desc']",
                "div.-fs14",
            ]:
                container = soup.select_one(sel)
                if container:
                    _resolve_imgs(container)
                    desc_html_parts = [_html(container)]
                    break

        # ── Strategy C: collect standalone infographic images (data-src on
        #    product CDN domains) even when no description container is found.
        if not desc_html_parts:
            for img in soup.find_all("img"):
                src = (
                    img.get("data-src") or img.get("data-original")
                    or img.get("data-lazy") or img.get("src", "")
                )
                if src and not src.startswith("data:") and (
                    "product" in src or "unsafe" in src or "jumia.is" in src
                ):
                    if src.startswith("//"):
                        src = "https:" + src
                    elif src.startswith("/"):
                        src = base_url + src
                    desc_html_parts.append(
                        f'<img src="{src}" alt="" '
                        f'style="max-width:100%;height:auto;display:block;margin:8px 0;">'
                    )

        # ── Strategy D: JSON-LD plain text (last resort — no images)
        if not desc_html_parts and result.get("DESCRIPTION"):
            # Already set by JSON-LD in Strategy 1 — wrap in <p> tags
            plain = result["DESCRIPTION"]
            desc_html_parts = [
                f"<p>{para.strip()}</p>"
                for para in plain.split("\n\n")
                if para.strip()
            ]

        if desc_html_parts:
            # Deduplicate while preserving order
            seen_keys: set[str] = set()
            unique_parts: list[str] = []
            for part in desc_html_parts:
                key = re.sub(r"\s+", " ", part)[:120]
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_parts.append(part)

            # FIX: if ALL parts are <img> tags with no text content, the
            # description is image-only (Jumia infographic layout). In this case
            # store only the image URLs as a newline-separated list, not a
            # wall of HTML that clutters the table/Excel cell.
            text_parts = [p for p in unique_parts if not re.match(r"^<img", p.strip())]
            if not text_parts:
                # Extract unique image URLs and store as plain URL list
                img_urls = []
                for part in unique_parts:
                    m_src = re.search(r'src="([^"]+)"', part)
                    if m_src:
                        url = m_src.group(1)
                        if url not in img_urls:
                            img_urls.append(url)
                result["DESCRIPTION"] = "\n".join(img_urls) if img_urls else ""
            else:
                # Wrap in a container div with basic responsive styles
                result["DESCRIPTION"] = (
                    '<div class="jm-desc" style="font-family:sans-serif;'
                    'line-height:1.6;color:#313133;">\n'
                    + "\n".join(text_parts)
                    + "\n</div>"
                )

    # ── Key Features ──────────────────────────────────────────────────────────
    # Always a <ul> immediately following an h2/h3 "Key Features" heading.
    key_features: list[str] = []
    kf_pattern = re.compile(r"key\s+features?", re.IGNORECASE)

    for heading in soup.find_all(["h2", "h3", "h4", "b", "strong"]):
        if not kf_pattern.search(heading.get_text()):
            continue
        # Scan forward siblings for the first <ul>
        node = heading.find_next_sibling()
        while node:
            if node.name in ("h2", "h3", "h4"):
                break
            if node.name == "ul":
                for li in node.find_all("li", recursive=False):
                    txt = li.get_text(" ", strip=True)
                    if txt:
                        key_features.append(txt)
                break
            # Some Jumia pages wrap the list in a div first
            if node.name == "div":
                inner_ul = node.find("ul")
                if inner_ul:
                    for li in inner_ul.find_all("li", recursive=False):
                        txt = li.get_text(" ", strip=True)
                        if txt:
                            key_features.append(txt)
                    if key_features:
                        break
            node = node.find_next_sibling()
        if key_features:
            break

    # Fallback: any <ul> whose nearest preceding heading matches
    if not key_features:
        for ul in soup.find_all("ul"):
            prev_h = ul.find_previous(["h2", "h3", "h4", "b", "strong"])
            if prev_h and kf_pattern.search(prev_h.get_text()):
                for li in ul.find_all("li"):
                    txt = li.get_text(" ", strip=True)
                    if txt:
                        key_features.append(txt)
                break

    if key_features:
        result["KEY_FEATURES"] = " | ".join(key_features)

    # ── What's in the box ─────────────────────────────────────────────────────
    in_box: list[str] = []
    box_pattern = re.compile(r"what.{0,5}s?\s+in\s+the\s+box", re.IGNORECASE)

    for heading in soup.find_all(["h2", "h3", "h4", "b", "strong"]):
        if not box_pattern.search(heading.get_text()):
            continue
        node = heading.find_next_sibling()
        while node:
            if node.name in ("h2", "h3", "h4"):
                break
            if node.name == "ul":
                for li in node.find_all("li", recursive=False):
                    txt = li.get_text(" ", strip=True)
                    if txt:
                        in_box.append(txt)
                break
            if node.name == "div":
                inner_ul = node.find("ul")
                if inner_ul:
                    for li in inner_ul.find_all("li", recursive=False):
                        txt = li.get_text(" ", strip=True)
                        if txt:
                            in_box.append(txt)
                    if in_box:
                        break
            node = node.find_next_sibling()
        if in_box:
            break

    # Fallback: nearest preceding heading approach
    if not in_box:
        for ul in soup.find_all("ul"):
            prev_h = ul.find_previous(["h2", "h3", "h4", "b", "strong"])
            if prev_h and box_pattern.search(prev_h.get_text()):
                for li in ul.find_all("li"):
                    txt = li.get_text(" ", strip=True)
                    if txt:
                        in_box.append(txt)
                break

    # Regex fallback on plain text — stop at the next heading keyword
    if not in_box:
        m_box = re.search(
            r"what.{0,5}s?\s+in\s+the\s+box[:\s]+"
            r"(.*?)"
            r"(?=\n\s*\n|##|\bSpecifications?\b|\bCustomer\b|\bRelated\b|\Z)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if m_box:
            raw_box = m_box.group(1).strip()
            # Sanity cap — box content shouldn't be a whole paragraph
            if len(raw_box) < 300:
                items = re.split(r"[•\n,;]+", raw_box)
                in_box = [i.strip() for i in items if i.strip() and len(i.strip()) > 2]

    if in_box:
        result["WHATS_IN_BOX"] = " | ".join(in_box)

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
