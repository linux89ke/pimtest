import pandas as pd
import re
import os
import functools
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LEGACY FALLBACK: level-1 USD caps (kept for backwards compatibility and as
# a final safety net when the category map can't be loaded)
# ---------------------------------------------------------------------------
CATEGORY_MAX_PRICES_USD = {
    "Automobile": 2000.0,
    "Computing": 5000.0,
    "Electronics": 5000.0,
    "Home & Office": 3000.0,
    "Industrial & Scientific": 2500.0,
    "Garden & Outdoors": 2000.0,
    "Musical Instruments": 2000.0,
    "Sporting Goods": 2000.0,
    "Gaming": 1200.0,
    "Baby Products": 600.0,
    "Toys & Games": 500.0,
    "Fashion": 500.0,
    "Health & Beauty": 500.0,
    "Grocery": 300.0,
    "Books, Movies & Music": 250.0,
    "Pet Supplies": 200.0,
}

# ---------------------------------------------------------------------------
# Country → column index mapping for the category_map xlsx
# col 0 = category_name  col 1 = category_code  col 2 = Category Path
# col 3 = NG  col 4 = EG  col 5 = IC  col 6 = MA
# col 7 = KE  col 8 = GH  col 9 = SN  col 10 = UG
# ---------------------------------------------------------------------------
_COUNTRY_COL = {
    "NG": 3,
    "EG": 4,
    "IC": 5,
    "MA": 6,
    "KE": 7,
    "GH": 8,
    "SN": 9,
    "UG": 10,
}

# Cache so the file is only parsed once per process
@functools.lru_cache(maxsize=1)
def _load_category_price_map(xlsx_path: str) -> dict:
    """
    Returns a nested dict:
        { country_code: { category_code_str: max_price_local_currency } }

    Also stores a per-country "level-1 fallback" so that if an exact code is
    missing we can still cap by the root category.

    Looks for the xlsx file in:
      1. The path passed in (absolute or relative)
      2. Same directory as this script
      3. CWD
    """
    candidates = [
        xlsx_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), xlsx_path),
        os.path.join(os.getcwd(), xlsx_path),
    ]

    found_path = None
    for p in candidates:
        if os.path.exists(p):
            found_path = p
            break

    if found_path is None:
        logger.warning(
            "category_map xlsx not found at any of: %s — falling back to USD caps",
            candidates,
        )
        return {}

    try:
        import openpyxl
        wb = openpyxl.load_workbook(found_path, read_only=True, data_only=True)
        ws = wb.active

        result = {cc: {} for cc in _COUNTRY_COL}

        for row in ws.iter_rows(values_only=True):
            cat_name, cat_code, cat_path = row[0], row[1], row[2]

            # Skip header or malformed rows
            if cat_code is None or not str(cat_code).strip().isdigit():
                continue

            code_str = str(int(cat_code))

            for cc, col_idx in _COUNTRY_COL.items():
                try:
                    price_val = row[col_idx]
                    if price_val is not None:
                        price = float(price_val)
                        if price > 0:
                            result[cc][code_str] = price
                except (IndexError, ValueError, TypeError):
                    pass

        total = sum(len(v) for v in result.values())
        logger.info("Loaded %d category price entries from %s", total, found_path)
        return result

    except Exception as e:
        logger.warning("Failed to load category map for pricing: %s", e)
        return {}


def _resolve_price_cap(
    cat_code: str,
    country_code: str,
    price_map: dict,
    code_to_path: dict,
    usd_fallback_map: dict,
) -> tuple[float, str]:
    """
    Returns (cap, source_description).

    Resolution order:
      1. Exact category code → per-country local-currency cap from xlsx
      2. Walk up the category path (parent → grandparent …) and use the
         first ancestor that has a cap
      3. Level-1 USD fallback from CATEGORY_MAX_PRICES_USD
      4. Hard default: 999_999_999 (never flag) if nothing matches
    """
    country_caps = price_map.get(country_code, {})
    clean_code = str(cat_code).strip().split(".")[0]

    # 1. Exact match
    if clean_code in country_caps:
        return country_caps[clean_code], f"exact code {clean_code}"

    # 2. Walk up the path hierarchy
    full_path = code_to_path.get(clean_code, "")
    if full_path and ">" in full_path:
        parts = [p.strip() for p in full_path.split(">")]
        # Reverse-walk: immediate parent first
        for depth in range(len(parts) - 1, 0, -1):
            ancestor_path = " > ".join(parts[:depth])
            # Find a code whose path matches the ancestor path
            for code, path in code_to_path.items():
                if path.strip() == ancestor_path and code in country_caps:
                    return country_caps[code], f"ancestor '{ancestor_path}'"

    # 3. USD fallback — use level-1 root from path or CATEGORY column
    level_1 = (
        full_path.split(">")[0].strip()
        if ">" in full_path
        else full_path.strip()
    )
    if level_1 in usd_fallback_map:
        return usd_fallback_map[level_1], f"USD fallback for '{level_1}'"

    # 4. No cap found
    return 999_999_999.0, "no cap"


# ---------------------------------------------------------------------------
# CATEGORY MAP FILE NAME — update this if your file is in a different location
# ---------------------------------------------------------------------------
CATEGORY_MAP_XLSX = "category_map.xlsx"


# ---------------------------------------------------------------------------
# VALIDATION 1 — Wrong Price
# ---------------------------------------------------------------------------
def check_wrong_price(data: pd.DataFrame) -> pd.DataFrame:
    """
    Flags products with missing/zero prices or extreme, unrealistic discounts (>95%).
    """
    if not {"GLOBAL_PRICE", "GLOBAL_SALE_PRICE"}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    d = data.copy()
    d["price"]      = pd.to_numeric(d["GLOBAL_PRICE"],      errors="coerce")
    d["sale_price"] = pd.to_numeric(d["GLOBAL_SALE_PRICE"], errors="coerce")

    invalid_base = d["price"].notna()     & (d["price"]      <= 0)
    invalid_sale = d["sale_price"].notna() & (d["sale_price"] <= 0)
    invalid_price = invalid_base | invalid_sale

    valid_prices    = (d["price"] > 0) & d["sale_price"].notna() & (d["sale_price"] > 0)
    discount_pct    = 1 - (d["sale_price"] / d["price"])
    extreme_discount = valid_prices & (discount_pct > 0.95)

    flagged = d[invalid_price | extreme_discount].copy()

    if not flagged.empty:
        def build_comment(row):
            p, sp = row["price"], row["sale_price"]
            if pd.isna(p) or p <= 0 or (pd.notna(sp) and sp <= 0):
                return f"Invalid price detected (Price: {p}, Sale: {sp})"
            return f"Extreme discount > 95% (Price: {p}, Sale: {sp})"

        flagged["Comment_Detail"] = flagged.apply(build_comment, axis=1)

    return (
        flagged
        .drop(columns=["price", "sale_price"], errors="ignore")
        .drop_duplicates(subset=["PRODUCT_SET_SID"])
    )


# ---------------------------------------------------------------------------
# VALIDATION 2 — Category Max Price Exceeded
# Reads per-category per-country local-currency caps from category_map.xlsx.
# ---------------------------------------------------------------------------
def check_category_max_price(
    data: pd.DataFrame,
    max_price_map: dict,          # kept for signature compatibility (ignored)
    code_to_path: dict = None,
    country_code: str = "KE",
    xlsx_path: str = None,
) -> pd.DataFrame:
    """
    Flags products whose price exceeds the per-category maximum for the
    given country (local currency), sourced from category_map.xlsx.

    Parameters
    ----------
    data          : product DataFrame
    max_price_map : legacy USD dict — kept so callers don't need to change;
                    used only as fallback when the xlsx is unavailable
    code_to_path  : dict mapping category_code str → full category path str
    country_code  : two-letter country code, e.g. "KE", "NG", "GH"
    xlsx_path     : override path to category_map.xlsx (default: auto-resolve)
    """
    required = {"CATEGORY_CODE", "GLOBAL_PRICE", "GLOBAL_SALE_PRICE"}
    if not required.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    if code_to_path is None:
        code_to_path = {}

    _xlsx = xlsx_path or CATEGORY_MAP_XLSX
    price_map = _load_category_price_map(_xlsx)

    d = data.copy()
    d["price"]      = pd.to_numeric(d["GLOBAL_PRICE"],      errors="coerce").fillna(0)
    d["sale_price"] = pd.to_numeric(d["GLOBAL_SALE_PRICE"], errors="coerce").fillna(0)
    d["max_listed_price"] = d[["price", "sale_price"]].max(axis=1)

    flagged_indices = []
    comment_map     = {}

    for idx, row in d.iterrows():
        listed_price = row["max_listed_price"]
        if listed_price <= 0:
            continue

        cat_code = str(row.get("CATEGORY_CODE", "")).strip().split(".")[0]
        cap, cap_source = _resolve_price_cap(
            cat_code, country_code, price_map, code_to_path, max_price_map
        )

        if listed_price > cap:
            flagged_indices.append(idx)
            # Determine local currency symbol for readable comment
            from constants import COUNTRY_CURRENCY
            _cc_map = {v["code"][:2]: v["symbol"] for k, v in COUNTRY_CURRENCY.items()}
            # match by country code via COUNTRY_CONFIG equivalent
            _country_name = {
                "KE": "Kenya", "UG": "Uganda", "NG": "Nigeria",
                "GH": "Ghana",  "MA": "Morocco",
            }.get(country_code, country_code)
            from constants import COUNTRY_CURRENCY as _CCY
            _sym = _CCY.get(_country_name, {}).get("symbol", "")
            comment_map[idx] = (
                f"Price ({_sym}{listed_price:,.0f}) exceeds max for this category "
                f"({_sym}{cap:,.0f}) [{cap_source}]"
            )

    if not flagged_indices:
        return pd.DataFrame(columns=data.columns)

    result = d.loc[flagged_indices].copy()
    result["Comment_Detail"] = result.index.map(comment_map)
    return (
        result
        .drop(columns=["price", "sale_price", "max_listed_price"], errors="ignore")
        .drop_duplicates(subset=["PRODUCT_SET_SID"])
    )


# ---------------------------------------------------------------------------
# VALIDATION 3 — Suspicious Discount (>50 %)
# ---------------------------------------------------------------------------
def check_suspicious_discount(data: pd.DataFrame) -> pd.DataFrame:
    """
    Flags products where the sale price is more than 50 % below the regular
    price.  Both GLOBAL_PRICE and GLOBAL_SALE_PRICE must be present and
    positive for the check to apply.

    A discount of exactly 50 % is acceptable; only strictly > 50 % is flagged.
    Products with discounts > 95 % are already handled by check_wrong_price,
    but we flag them here too so the seller gets a clearer message.
    """
    if not {"GLOBAL_PRICE", "GLOBAL_SALE_PRICE"}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    d = data.copy()
    d["price"]      = pd.to_numeric(d["GLOBAL_PRICE"],      errors="coerce")
    d["sale_price"] = pd.to_numeric(d["GLOBAL_SALE_PRICE"], errors="coerce")

    # Only compare rows where both prices are valid and positive
    valid = (
        d["price"].notna()      & (d["price"]      > 0) &
        d["sale_price"].notna() & (d["sale_price"] > 0) &
        (d["sale_price"] < d["price"])   # sale must actually be lower
    )

    discount_pct = 1 - (d["sale_price"] / d["price"])
    flagged_mask = valid & (discount_pct > 0.50)

    flagged = d[flagged_mask].copy()

    if not flagged.empty:
        def build_comment(row):
            p   = row["price"]
            sp  = row["sale_price"]
            pct = (1 - sp / p) * 100
            return (
                f"Suspicious discount of {pct:.1f}% "
                f"(Regular: {p:,.2f} → Sale: {sp:,.2f})"
            )

        flagged["Comment_Detail"] = flagged.apply(build_comment, axis=1)

    return (
        flagged
        .drop(columns=["price", "sale_price"], errors="ignore")
        .drop_duplicates(subset=["PRODUCT_SET_SID"])
    )
