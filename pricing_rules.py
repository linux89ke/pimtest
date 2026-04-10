import pandas as pd
import re
import os
import functools
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LEGACY FALLBACK: level-1 USD caps (kept as final safety net when the
# category map can't be loaded).  These are in USD and will be converted
# to local currency via the exchange-rate helper before comparison.
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

# ---------------------------------------------------------------------------
# Country metadata: name, currency symbol, and approximate USD exchange rate.
# Rates are used to convert USD-denominated input prices to local currency
# for comparison against the local-currency caps in category_map.xlsx.
# Update these rates periodically or wire in a live rates API if needed.
# ---------------------------------------------------------------------------
_COUNTRY_META = {
    "KE": {"name": "Kenya",   "symbol": "KSh", "usd_rate": 129.0},
    "UG": {"name": "Uganda",  "symbol": "USh", "usd_rate": 3700.0},
    "NG": {"name": "Nigeria", "symbol": "₦",   "usd_rate": 1550.0},
    "GH": {"name": "Ghana",   "symbol": "GH₵", "usd_rate": 15.5},
    "MA": {"name": "Morocco", "symbol": "MAD", "usd_rate": 10.0},
    "EG": {"name": "Egypt",   "symbol": "EGP", "usd_rate": 48.5},
    "SN": {"name": "Senegal", "symbol": "XOF", "usd_rate": 615.0},
    "IC": {"name": "Ivory Coast", "symbol": "XOF", "usd_rate": 615.0},
}


def _get_usd_rate(country_code: str) -> float:
    """Return the USD → local currency conversion rate for a country code."""
    return _COUNTRY_META.get(country_code, {}).get("usd_rate", 1.0)


def _get_symbol(country_code: str) -> str:
    """Return the local currency symbol for a country code."""
    return _COUNTRY_META.get(country_code, {}).get("symbol", "$")


def usd_to_local(amount_usd: float, country_code: str) -> float:
    """Convert a USD amount to the local currency for the given country."""
    return amount_usd * _get_usd_rate(country_code)


# ---------------------------------------------------------------------------
# Cache so the xlsx is only parsed once per process lifetime.
# Call _load_category_price_map.cache_clear() if the file changes at runtime.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _load_category_price_map(xlsx_path: str) -> dict:
    """
    Returns a nested dict:
        { country_code: { category_code_str: max_price_local_currency } }

    The caps stored in the xlsx are already in local currency for each country
    column — no conversion is needed for the xlsx path.

    Looks for the xlsx in:
      1. The path passed in (absolute or relative to CWD)
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
            "category_map xlsx not found at any of: %s — "
            "falling back to USD caps (will be converted to local currency)",
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
    Returns (cap_in_local_currency, source_description).

    Resolution order:
      1. Exact category code  → per-country local-currency cap from xlsx
      2. Walk up category path (parent → grandparent …) → first ancestor cap
      3. Level-1 USD fallback from CATEGORY_MAX_PRICES_USD, converted to local
      4. Hard default 999_999_999 (never flag) if nothing matches
    """
    country_caps = price_map.get(country_code, {})
    clean_code = str(cat_code).strip().split(".")[0]

    # 1. Exact match (already in local currency)
    if clean_code in country_caps:
        return country_caps[clean_code], f"exact code {clean_code}"

    # 2. Walk up the path hierarchy (already in local currency)
    full_path = code_to_path.get(clean_code, "")
    if full_path and ">" in full_path:
        parts = [p.strip() for p in full_path.split(">")]
        for depth in range(len(parts) - 1, 0, -1):
            ancestor_path = " > ".join(parts[:depth])
            for code, path in code_to_path.items():
                if path.strip() == ancestor_path and code in country_caps:
                    return country_caps[code], f"ancestor '{ancestor_path}'"

    # 3. USD fallback — convert to local currency before returning
    level_1 = (
        full_path.split(">")[0].strip()
        if ">" in full_path
        else full_path.strip()
    )
    if level_1 in usd_fallback_map:
        cap_usd = usd_fallback_map[level_1]
        cap_local = usd_to_local(cap_usd, country_code)
        rate = _get_usd_rate(country_code)
        return cap_local, f"USD fallback for '{level_1}' (${cap_usd:,.0f} × {rate} rate)"

    # 4. No cap found
    return 999_999_999.0, "no cap"


# ---------------------------------------------------------------------------
# CATEGORY MAP FILE NAME — update if your file lives elsewhere
# ---------------------------------------------------------------------------
CATEGORY_MAP_XLSX = "category_map.xlsx"


# ---------------------------------------------------------------------------
# VALIDATION 1 — Discount too high
# Flags products where discount exceeds 51% (when both prices are present
# and valid). If either price is blank/zero/missing, the check is skipped.
# ---------------------------------------------------------------------------
def check_wrong_price(data: pd.DataFrame, country_code: str = "KE") -> pd.DataFrame:
    """
    Flags products with a discount > 51% (sale price vs regular price).

    Rules:
      • If GLOBAL_PRICE is blank/zero/missing → skip (no discount to check)
      • If GLOBAL_SALE_PRICE is blank/zero/missing → skip (no discount applied)
      • Only flag when both prices are valid positive numbers AND
        discount = (1 - sale/price) > 0.51

    Input prices are assumed to be in USD and are converted to local currency
    for the comment string only — the ratio check is currency-neutral.
    """
    if not {"GLOBAL_PRICE", "GLOBAL_SALE_PRICE"}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    rate = _get_usd_rate(country_code)
    sym  = _get_symbol(country_code)

    d = data.copy()
    d["price"]      = pd.to_numeric(d["GLOBAL_PRICE"],      errors="coerce")
    d["sale_price"] = pd.to_numeric(d["GLOBAL_SALE_PRICE"], errors="coerce")

    # Both prices must be present and positive — skip if either is blank/zero
    valid_both = (
        d["price"].notna()      & (d["price"]      > 0) &
        d["sale_price"].notna() & (d["sale_price"] > 0)
    )

    discount_pct  = 1 - (d["sale_price"] / d["price"])
    high_discount = valid_both & (discount_pct > 0.51)

    flagged = d[high_discount].copy()

    if not flagged.empty:
        def build_comment(row):
            p_usd  = row["price"]
            sp_usd = row["sale_price"]
            pct    = (1 - sp_usd / p_usd) * 100
            p_loc  = p_usd  * rate
            sp_loc = sp_usd * rate
            return (
                f"Discount too high: {pct:.1f}% "
                f"(Price: USD {p_usd:,.2f} / {sym}{p_loc:,.0f} → "
                f"Sale: USD {sp_usd:,.2f} / {sym}{sp_loc:,.0f})"
            )

        flagged["Comment_Detail"] = flagged.apply(build_comment, axis=1)

    return (
        flagged
        .drop(columns=["price", "sale_price"], errors="ignore")
        .drop_duplicates(subset=["PRODUCT_SET_SID"])
    )


# ---------------------------------------------------------------------------
# VALIDATION 2 — Category Max Price Exceeded
# Upload prices are in USD → convert to local currency before comparing
# against the local-currency caps stored in category_map.xlsx.
# ---------------------------------------------------------------------------
def check_category_max_price(
    data: pd.DataFrame,
    max_price_map: dict,          # legacy USD dict — still used as fallback
    code_to_path: dict = None,
    country_code: str = "KE",
    xlsx_path: str = None,
) -> pd.DataFrame:
    """
    Flags products whose price (converted from USD to local currency) exceeds
    the per-category cap for the given country.

    Parameters
    ----------
    data          : product DataFrame (prices in USD)
    max_price_map : legacy USD fallback caps (converted to local at runtime)
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

    # --- Hoist all per-country lookups outside the row loop ---
    rate = _get_usd_rate(country_code)
    sym  = _get_symbol(country_code)

    d = data.copy()
    # Parse USD prices, fill NaN with 0 so max() works cleanly
    d["price_usd"]      = pd.to_numeric(d["GLOBAL_PRICE"],      errors="coerce").fillna(0)
    d["sale_price_usd"] = pd.to_numeric(d["GLOBAL_SALE_PRICE"], errors="coerce").fillna(0)

    # Convert to local currency for comparison against local-currency caps
    d["price_local"]      = d["price_usd"]      * rate
    d["sale_price_local"] = d["sale_price_usd"] * rate

    # Use the higher of the two local prices as the "listed price" to check
    d["max_listed_local"] = d[["price_local", "sale_price_local"]].max(axis=1)

    flagged_indices = []
    comment_map     = {}

    for idx, row in d.iterrows():
        listed_local = row["max_listed_local"]
        if listed_local <= 0:
            continue

        cat_code = str(row.get("CATEGORY_CODE", "")).strip().split(".")[0]
        cap_local, cap_source = _resolve_price_cap(
            cat_code, country_code, price_map, code_to_path, max_price_map
        )

        if listed_local > cap_local:
            flagged_indices.append(idx)
            # Original USD price for context
            usd_price = row["price_usd"] if row["price_usd"] >= row["sale_price_usd"] else row["sale_price_usd"]
            comment_map[idx] = (
                f"Price (USD {usd_price:,.2f} → {sym}{listed_local:,.0f}) "
                f"exceeds category max ({sym}{cap_local:,.0f}) [{cap_source}]"
            )

    if not flagged_indices:
        return pd.DataFrame(columns=data.columns)

    result = d.loc[flagged_indices].copy()
    result["Comment_Detail"] = result.index.map(comment_map)
    return (
        result
        .drop(columns=["price_usd", "sale_price_usd", "price_local",
                        "sale_price_local", "max_listed_local"], errors="ignore")
        .drop_duplicates(subset=["PRODUCT_SET_SID"])
    )


# ---------------------------------------------------------------------------
# VALIDATION 3 — Suspicious Discount (> 50 %, up to 95 %)
# Discounts > 95 % are handled exclusively by check_wrong_price.
# ---------------------------------------------------------------------------
def check_suspicious_discount(data: pd.DataFrame, country_code: str = "KE") -> pd.DataFrame:
    """
    Flags products where the sale price is more than 50 % below the regular
    price.  The 50–95 % window is checked here; > 95 % belongs to Wrong Price.

    Input prices are in USD; they are converted to local currency for the
    comment string so QC agents see familiar figures.
    """
    if not {"GLOBAL_PRICE", "GLOBAL_SALE_PRICE"}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    rate = _get_usd_rate(country_code)
    sym  = _get_symbol(country_code)

    d = data.copy()
    d["price"]      = pd.to_numeric(d["GLOBAL_PRICE"],      errors="coerce")
    d["sale_price"] = pd.to_numeric(d["GLOBAL_SALE_PRICE"], errors="coerce")

    # Both prices must be valid, positive, and sale must actually be lower
    valid = (
        d["price"].notna()      & (d["price"]      > 0) &
        d["sale_price"].notna() & (d["sale_price"] > 0) &
        (d["sale_price"] < d["price"])
    )

    discount_pct = 1 - (d["sale_price"] / d["price"])

    # Only the 51–95 % window is now handled by check_wrong_price (> 51 %).
    # Suspicious Discount covers the same range for legacy/separate reporting.
    flagged_mask = valid & (discount_pct > 0.50) & (discount_pct <= 0.95)

    flagged = d[flagged_mask].copy()

    if not flagged.empty:
        def build_comment(row):
            p_usd  = row["price"]
            sp_usd = row["sale_price"]
            pct    = (1 - sp_usd / p_usd) * 100
            p_loc  = p_usd  * rate
            sp_loc = sp_usd * rate
            return (
                f"Suspicious discount of {pct:.1f}% "
                f"(Regular: USD {p_usd:,.2f} / {sym}{p_loc:,.0f} → "
                f"Sale: USD {sp_usd:,.2f} / {sym}{sp_loc:,.0f})"
            )

        flagged["Comment_Detail"] = flagged.apply(build_comment, axis=1)

    return (
        flagged
        .drop(columns=["price", "sale_price"], errors="ignore")
        .drop_duplicates(subset=["PRODUCT_SET_SID"])
    )
