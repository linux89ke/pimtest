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
    "COUNT_VARIATIONS",
    # Warranty
    "PRODUCT_WARRANTY",
    "WARRANTY_DURATION",
    # Ratings
    "RATING",
    "REVIEW_COUNT",
    # Content
    "DESCRIPTION",
    "KEY_FEATURES",
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
            price = offers.get("price") or offers.get("lowPrice", "")
            if price:
                result["PRICE"] = str(price).replace(",", "")
            avail = offers.get("availability", "")
            if "InStock" in avail:
                result["STOCK_STATUS"] = "In Stock"
            elif "OutOfStock" in avail:
                result["STOCK_STATUS"] = "Out of Stock"

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

    # ── Price (HTML fallback) ──────────────────────────────────────────────────
    if not result.get("PRICE"):
        for sel in [
            "span.-b.-ltr.-tal.-fs24",
            "span.prc",
            "[class*='price'] span",
            "span[data-price]",
        ]:
            el = soup.select_one(sel)
            if el:
                raw_p = re.sub(r"[^\d.,]", "", el.get_text())
                if raw_p:
                    result["PRICE"] = raw_p.replace(",", "")
                    break

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

    # ── Stock status ──────────────────────────────────────────────────────────
    if not result.get("STOCK_STATUS"):
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
    # Strategy A: hidden inputs inside the variation list (most reliable)
    colors: list[str] = []
    for li in soup.select("ul.-pvs.-mvs li, ul[class*='var'] li"):
        inp = li.select_one("input")
        if inp:
            val = inp.get("value") or inp.get("data-value", "")
            if val:
                colors.append(val.strip())

    # Strategy B: visible colour swatch labels / selectors
    if not colors:
        for el in soup.select(
            "[data-type='color'] span, span.color-selector, "
            "li[class*='color'] span, label[class*='color']"
        ):
            txt = el.get_text(strip=True)
            if txt and len(txt) < 40:
                colors.append(txt)

    # Strategy C: variation pills / size options — count only, no names
    variation_count: int = 0
    if not colors:
        opts = soup.select(
            "ul.-pvs li, select option:not([value='']):not([disabled]), "
            "ul[class*='var'] li, div[class*='var'] button"
        )
        variation_count = len(opts)

    # Strategy D: parse from title — e.g. "Titanium Silver" already in name
    #   Only fire when we truly found nothing above AND the title contains a
    #   known colour word, to avoid false positives.
    if not colors and not variation_count:
        title_el = soup.select_one("h1, title")
        if title_el:
            COLOR_WORDS = re.compile(
                r"\b(black|white|silver|gold|blue|red|green|purple|pink|grey|gray|"
                r"titanium|midnight|starlight|ivory|champagne|rose|copper|yellow|"
                r"orange|violet|navy|cream|brown|coral|aqua|cyan|teal|lilac)\b",
                re.IGNORECASE,
            )
            found = COLOR_WORDS.findall(title_el.get_text())
            if found:
                colors = list(dict.fromkeys(c.title() for c in found))

    if colors:
        unique_colors = list(dict.fromkeys(c for c in colors if c))
        result["COLOR"] = ", ".join(unique_colors)
        result["COUNT_VARIATIONS"] = str(len(unique_colors))
    elif variation_count:
        result["COUNT_VARIATIONS"] = str(variation_count)

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
    # "Official Store" label near the seller info.
    if not result.get("IS_OFFICIAL_STORE"):
        official_signals = [
            # Tag badge with JMALL href
            soup.select_one("a[href*='JMALL'], a[tag*='JMALL']"),
            # Text badge
            soup.find(string=re.compile(r"official\s+store", re.IGNORECASE)),
            # Image alt text
            soup.find("img", alt=re.compile(r"official\s+store", re.IGNORECASE)),
        ]
        result["IS_OFFICIAL_STORE"] = "Yes" if any(official_signals) else "No"


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

    # ── Description (HTML) ────────────────────────────────────────────────────
    # Jumia wraps the long product description in a section that sits between
    # the "Product details" heading and the "Specifications" heading.
    # The section contains alternating h2/h3 headings and paragraphs.
    if not result.get("DESCRIPTION"):
        desc_parts: list[str] = []

        # Walk every h2/h3 and the block of text that follows it,
        # stopping when we hit "Specifications" or "Customer Feedback".
        STOP_HEADS = re.compile(
            r"^(specifications?|customer\s+feedback|related\s+results|seller\s+info)",
            re.IGNORECASE,
        )
        collecting = False
        for tag in soup.find_all(
            ["h1", "h2", "h3", "h4", "p", "li"], limit=300
        ):
            txt = tag.get_text(" ", strip=True)
            if not txt:
                continue
            # Start collecting after "Product details" heading
            if tag.name in ("h2", "h3") and re.search(r"product\s+details?", txt, re.IGNORECASE):
                collecting = True
                continue
            # Stop at known boundary headings
            if collecting and tag.name in ("h2", "h3") and STOP_HEADS.search(txt):
                break
            if collecting and len(txt) > 15:
                desc_parts.append(txt)

        # If "Product details" heading wasn't found, grab all substantial
        # paragraphs that are NOT inside the spec table or rating block.
        if not desc_parts:
            for p in soup.find_all("p"):
                txt = p.get_text(" ", strip=True)
                if len(txt) > 40 and not re.search(
                    r"cookie|privacy|subscribe|newsletter|download", txt, re.IGNORECASE
                ):
                    desc_parts.append(txt)

        if desc_parts:
            seen: set[str] = set()
            unique: list[str] = []
            for part in desc_parts:
                key = part[:80]
                if key not in seen:
                    seen.add(key)
                    unique.append(part)
            result["DESCRIPTION"] = "\n\n".join(unique)

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

    # Regex fallback on plain text
    if not in_box:
        m_box = re.search(
            r"what.{0,5}s?\s+in\s+the\s+box[:\s]+(.*?)(?:\n\n|\Z)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if m_box:
            raw_box = m_box.group(1).strip()
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
