import sys
import os
import re
import hashlib
import traceback
import logging
import json
import concurrent.futures
from io import BytesIO

# ------------------------------------------------------------------
# PATH FIX — must be first, before any local imports.
# ------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
try:
    import requests as _requests
    from PIL import Image as _Image
    _IMAGE_CHECK_OK = True
except ImportError:
    _IMAGE_CHECK_OK = False

# ------------------------------------------------------------------
# LOCAL IMPORTS
# ------------------------------------------------------------------
try:
    from translations import LANGUAGES, get_translation
    _TRANSLATIONS_OK = True
except ImportError:
    _TRANSLATIONS_OK = False
    def get_translation(lang, key): return key

try:
    from postqc import (
        detect_file_type,
        normalize_post_qc,
        load_category_map,
    )
    _POSTQC_OK = True
except Exception as _postqc_err:
    _POSTQC_OK = False
    _postqc_err_msg = str(_postqc_err)

try:
    import _preqc_registry  # noqa: F401
except ImportError:
    pass

try:
    import importlib as _importlib
    _main_mod = (
        sys.modules.get("streamlit_app")
        or _importlib.import_module("streamlit_app")
    )
    validate_products      = _main_mod.validate_products
    load_all_support_files = _main_mod.load_all_support_files
    render_flag_expander   = _main_mod.render_flag_expander
    CountryValidator       = _main_mod.CountryValidator
    generate_smart_export  = _main_mod.generate_smart_export
    PRODUCTSETS_COLS       = _main_mod.PRODUCTSETS_COLS
    REJECTION_REASONS_COLS = _main_mod.REJECTION_REASONS_COLS
    clean_category_code    = _main_mod.clean_category_code
    # Image grid helpers
    build_fast_grid_html            = _main_mod.build_fast_grid_html
    analyze_image_quality_cached    = _main_mod.analyze_image_quality_cached
    format_local_price              = _main_mod.format_local_price
    GRID_COLS                       = _main_mod.GRID_COLS
    REASON_MAP                      = _main_mod.REASON_MAP
    _FULL_VALIDATION_OK    = True
except Exception as _fv_err:
    _FULL_VALIDATION_OK = False
    _fv_err_msg = str(_fv_err)
    REASON_MAP = {
        "REJECT_POOR_IMAGE": "Poor images",
        "REJECT_WRONG_CAT":  "Wrong Category",
        "REJECT_FAKE":       "Suspected Fake product",
        "REJECT_BRAND":      "Restricted brands",
        "REJECT_PROHIBITED": "Prohibited products",
        "REJECT_COLOR":      "Missing COLOR",
        "REJECT_WRONG_BRAND":"Generic branded products with genuine brands",
        "OTHER_CUSTOM":      "Other Reason (Custom)",
    }
    GRID_COLS = ['PRODUCT_SET_SID','NAME','BRAND','CATEGORY','SELLER_NAME',
                 'MAIN_IMAGE','GLOBAL_SALE_PRICE','GLOBAL_PRICE','COLOR']

# ── Post-QC aware brand-check overrides ───────────────────────────────────────
if _FULL_VALIDATION_OK:
    import pandas as _pd_patch

    def _pq_check_fashion_brand_issues(data: "pd.DataFrame",
                                        valid_category_codes_fas: list) -> "pd.DataFrame":
        if not {"CATEGORY_CODE", "BRAND"}.issubset(data.columns):
            return _pd_patch.DataFrame(columns=data.columns)
        fas_set = set(clean_category_code(c) for c in valid_category_codes_fas)
        brand_is_fashion = data["BRAND"].astype(str).str.strip().str.lower() == "fashion"
        code_not_fas     = ~data["CATEGORY_CODE"].apply(clean_category_code).isin(fas_set)
        if "CATEGORY" in data.columns:
            path_is_fashion = data["CATEGORY"].astype(str).str.contains(
                r"\bfashion\b", case=False, na=False, regex=True
            )
        else:
            path_is_fashion = _pd_patch.Series(False, index=data.index)
        flagged = data[brand_is_fashion & code_not_fas & ~path_is_fashion].copy()
        return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

    def _pq_check_generic_brand_issues(data: "pd.DataFrame",
                                        valid_category_codes_fas: list) -> "pd.DataFrame":
        if not {"CATEGORY_CODE", "BRAND"}.issubset(data.columns):
            return _pd_patch.DataFrame(columns=data.columns)
        fas_set = set(clean_category_code(c) for c in valid_category_codes_fas)
        brand_is_generic = data["BRAND"].astype(str).str.lower() == "generic"
        code_in_fas      = data["CATEGORY_CODE"].apply(clean_category_code).isin(fas_set)
        if "CATEGORY" in data.columns:
            path_is_fashion = data["CATEGORY"].astype(str).str.contains(
                r"\bfashion\b", case=False, na=False, regex=True
            )
        else:
            path_is_fashion = _pd_patch.Series(False, index=data.index)
        flagged = data[brand_is_generic & (code_in_fas | path_is_fashion)].copy()
        return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

    _main_mod.check_fashion_brand_issues  = _pq_check_fashion_brand_issues
    _main_mod.check_generic_brand_issues  = _pq_check_generic_brand_issues

try:
    from jumia_scraper import (
        enrich_post_qc_df,
        SCRAPABLE_FIELDS,
        scrape_single_sku,
        COUNTRY_URLS,
    )
    _SCRAPER_OK = True
except ImportError:
    _SCRAPER_OK = False

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Post-QC Validation",
        page_icon="🔍",
        layout=st.session_state.get("layout_mode", "wide"),
    )
except Exception:
    pass

# ------------------------------------------------------------------
# THEME
# ------------------------------------------------------------------
ORANGE  = "#F68B1E"
ORANGE2 = "#FF9933"
RED     = "#E73C17"
DARK    = "#313133"
MED     = "#5A5A5C"
LIGHT   = "#F5F5F5"
BORDER  = "#E0E0E0"
GREEN   = "#4CAF50"
BLUE    = "#1976D2"

st.markdown(f"""
<style>
.stButton > button {{ border-radius: 4px; font-weight: 600; }}
.stButton > button[kind="primary"] {{
    background-color: {ORANGE} !important;
    border: none !important; color: white !important;
}}
.stButton > button[kind="primary"]:hover {{ background-color: {ORANGE2} !important; }}
div[data-testid="stMetric"] {{
    background: {LIGHT}; border-radius: 0 0 8px 8px;
    padding: 12px 16px 16px; text-align: center;
}}
div[data-testid="stMetricValue"] {{
    color: {DARK}; font-weight: 700; font-size: 26px !important;
}}
div[data-testid="stMetricLabel"] {{
    color: {MED}; font-size: 11px; text-transform: uppercase;
}}
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER}; border-radius: 8px;
}}
div[data-testid="stExpander"] summary {{
    background-color: {LIGHT}; padding: 12px; border-radius: 8px 8px 0 0;
}}
h1, h2, h3 {{ color: {DARK} !important; }}
.enrich-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    background: {GREEN};
    color: white;
    margin-left: 6px;
}}
.field-chip {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 14px;
    font-size: 12px;
    font-weight: 500;
    background: #E8F5E9;
    color: #2E7D32;
    border: 1px solid #A5D6A7;
    margin: 2px 3px 2px 0;
}}
.field-chip.missing {{
    background: #FFF3E0;
    color: #E65100;
    border-color: #FFCC80;
}}
@media (prefers-color-scheme: dark) {{
    div[data-testid="stMetricValue"] {{ color: #F5F5F5 !important; }}
    div[data-testid="stMetric"]      {{ background: #2a2a2e !important; }}
    h1, h2, h3                       {{ color: #F5F5F5 !important; }}
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ------------------------------------------------------------------
_SS_DEFAULTS = {
    "ui_lang":            "en",
    "selected_country":   "Kenya",
    "layout_mode":        "wide",
    "pq_country":         "Kenya",
    "pq_data":            pd.DataFrame(),
    "pq_last_sig":        None,
    "enriched_df":        None,
    "enrichment_summary": {},
    "enrichment_done":    False,
    "pq_val_report":      pd.DataFrame(),
    "pq_val_results":     {},
    "pq_val_exports":     {},
    "pq_flags_init":      False,
    "display_df_cache":   {},
    "exports_cache":      {},
    "final_report":       pd.DataFrame(),
    "main_toasts":        [],
    "flags_expanded_initialized": False,
    # image grid state
    "pq_grid_page":          0,
    "pq_grid_items_per_page": 50,
    "pq_bridge_counter":     0,
    "pq_do_scroll_top":      False,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ------------------------------------------------------------------
# CANONICAL FIELD LIST
# ------------------------------------------------------------------
_ALL_FIELDS = [
    "PRICE", "DISCOUNT", "STOCK_STATUS",
    "MODEL", "GTIN", "BRAND", "CATEGORY_PATH", "IS_OFFICIAL_STORE",
    "MAIN_IMAGE", "WEIGHT",
    "COLOR", "COLOR_IN_TITLE", "COUNT_VARIATIONS", "SIZES_AVAILABLE",
    "PRODUCT_WARRANTY", "WARRANTY_DURATION",
    "RATING", "REVIEW_COUNT",
    "DESCRIPTION", "KEY_FEATURES", "KEY_SPECS", "SPECIFICATIONS",
    "WHATS_IN_BOX",
]

# ------------------------------------------------------------------
# COLUMN MAP
# ------------------------------------------------------------------
_PQ_COL_MAP = {
    "sku":               "PRODUCT_SET_SID",
    "name":              "NAME",
    "brand":             "BRAND",
    "category":          "CATEGORY",
    "categories":        "CATEGORY",
    "price":             "GLOBAL_SALE_PRICE",
    "newprice":          "GLOBAL_SALE_PRICE",
    "old price":         "GLOBAL_PRICE",
    "oldprice":          "GLOBAL_PRICE",
    "seller":            "SELLER_NAME",
    "sellername":        "SELLER_NAME",
    "seller name":       "SELLER_NAME",
    "image url":         "MAIN_IMAGE",
    "image":             "MAIN_IMAGE",
    "main image":        "MAIN_IMAGE",
    "url":               "PRODUCT_URL",
    "product url":       "PRODUCT_URL",
    "stock":             "STOCK_STATUS",
    "stock status":      "STOCK_STATUS",
    "rating":            "RATING",
    "averagerating":     "RATING",
    "average rating":    "RATING",
    "total ratings":     "REVIEW_COUNT",
    "totalratings":      "REVIEW_COUNT",
    "review count":      "REVIEW_COUNT",
    "discount":          "DISCOUNT",
    "tags":              "TAGS",
    "jumia express":     "JUMIA_EXPRESS",
    "isexpress":         "JUMIA_EXPRESS",
    "shop global":       "SHOP_GLOBAL",
    "isglobal":          "SHOP_GLOBAL",
    "color":             "COLOR",
    "colour":            "COLOR",
    "product warranty":  "PRODUCT_WARRANTY",
    "warranty":          "PRODUCT_WARRANTY",
    "warranty duration": "WARRANTY_DURATION",
    "count variations":  "COUNT_VARIATIONS",
    "variations":        "COUNT_VARIATIONS",
    "seller sku":        "SELLER_SKU",
    "parentsku":         "PARENTSKU",
    "parent sku":        "PARENTSKU",
    "description":       "DESCRIPTION",
}

def _standardise_pq(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    rename = {}
    for col in df.columns:
        mapped = _PQ_COL_MAP.get(col.lower().strip())
        if mapped:
            rename[col] = mapped
    return df.rename(columns=rename)

COUNTRY_CODES = {
    "Kenya":   "KE",
    "Uganda":  "UG",
    "Nigeria": "NG",
    "Ghana":   "GH",
    "Morocco": "MA",
}
COUNTRIES = list(COUNTRY_CODES.keys())

def _t(key): return get_translation(st.session_state.ui_lang, key)
def _code(name: str) -> str: return COUNTRY_CODES.get(name, "KE")

def _load_support_files() -> dict:
    if "support_files" in st.session_state:
        return st.session_state.support_files
    if _FULL_VALIDATION_OK:
        try:
            sf = load_all_support_files()
            st.session_state.support_files = sf
            return sf
        except Exception:
            pass
    cat_map = load_category_map() if _POSTQC_OK else {}
    return {"category_map": cat_map}

def _reset_results():
    st.session_state.pq_data            = pd.DataFrame()
    st.session_state.pq_last_sig        = None
    st.session_state.enriched_df        = None
    st.session_state.enrichment_summary = {}
    st.session_state.enrichment_done    = False
    st.session_state.pq_val_report      = pd.DataFrame()
    st.session_state.pq_val_results     = {}
    st.session_state.pq_val_exports     = {}
    st.session_state.pq_flags_init      = False
    st.session_state.final_report       = pd.DataFrame()
    # image grid
    st.session_state.pq_grid_page       = 0
    st.session_state.pq_bridge_counter  = 0
    st.session_state.pq_do_scroll_top   = False
    # clear per-product rejection keys from previous run
    _keys_to_del = [k for k in st.session_state.keys()
                    if k.startswith("pq_qrej_")]
    for _k in _keys_to_del:
        del st.session_state[_k]

def _count_filled(series: pd.Series) -> int:
    return int(
        series.astype(str).str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
        .ne("")
        .sum()
    )

def _build_enrichment_summary(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    summary = {}
    for col in SCRAPABLE_FIELDS:
        b_count = _count_filled(before[col]) if col in before.columns else 0
        a_count = _count_filled(after[col])  if col in after.columns  else 0
        summary[col] = {
            "before": b_count,
            "after":  a_count,
            "filled": max(0, a_count - b_count),
        }
    return summary

def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched Data")
    return buf.getvalue()

def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def _clean_sku(raw: str) -> str:
    return raw.split("-")[0].strip()

def _extract_sku_from_url(url: str) -> str | None:
    slug = re.sub(r"\.html.*$", "", url.rstrip("/").split("/")[-1])
    parts = slug.split("-")
    for part in reversed(parts):
        if not (12 <= len(part) <= 30): continue
        if not re.search(r"[A-Za-z]", part) or not re.search(r"[0-9]", part): continue
        if re.fullmatch(r"[\dA-Z]*(ML|KG|GB|TB|CM|MM|GM|MG|PC|PCS|L)+[\dA-Z]*", part, re.IGNORECASE): continue
        if re.search(r"(NAFAMZ|GANAFAMZ|HANAFAMZ|FANAFAMZ)$", part, re.IGNORECASE):
            return part.upper()
    for part in reversed(parts):
        if not (12 <= len(part) <= 30): continue
        if not re.search(r"[A-Za-z]", part) or not re.search(r"[0-9]", part): continue
        if re.fullmatch(r"[\dA-Z]*(ML|KG|GB|TB|CM|MM|GM|MG|PC|PCS|L)+[\dA-Z]*", part, re.IGNORECASE): continue
        return part.upper()
    return None

def _auto_detect_country_from_url(url: str) -> str | None:
    _domain_map = {
        "jumia.co.ke": "KE", "jumia.ug": "UG", "jumia.com.ng": "NG",
        "jumia.com.gh": "GH", "jumia.ma": "MA",
    }
    for domain, code in _domain_map.items():
        if domain in url:
            return code
    return None

def _parse_lookup_inputs(raw_text: str) -> list[dict]:
    entries = []
    seen_skus: set[str] = set()
    for line in raw_text.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith("http"):
            sku = _extract_sku_from_url(line)
            if not sku:
                slug = re.sub(r"\.html.*$", "", line.rstrip("/").split("/")[-1])
                num_m = re.search(r"(\d{6,})$", slug)
                sku = num_m.group(1) if num_m else slug[:20]
            hint = _auto_detect_country_from_url(line)
            direct_url = line
        else:
            sku = _clean_sku(line)
            hint = None
            direct_url = None
        if sku and sku not in seen_skus:
            seen_skus.add(sku)
            entries.append({
                "input": line, "sku": sku, "url": direct_url, "country_hint": hint,
            })
    return entries

# ------------------------------------------------------------------
# NEW HELPERS: data-quality checks
# ------------------------------------------------------------------

def _derive_official_store_from_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive IS_OFFICIAL_STORE from the TAGS column when it is absent.
    Any product tagged JMALL is an official-store listing.
    Only writes to IS_OFFICIAL_STORE if the column is fully absent or all-blank.
    """
    if "TAGS" not in df.columns:
        return df
    col_present = (
        "IS_OFFICIAL_STORE" in df.columns
        and df["IS_OFFICIAL_STORE"].astype(str).str.strip()
            .replace({"nan": "", "None": ""}).ne("").any()
    )
    if col_present:
        return df
    df = df.copy()
    df["IS_OFFICIAL_STORE"] = df["TAGS"].str.contains(
        "JMALL", case=False, na=False
    ).map({True: "Yes", False: "No"})
    return df


def _calc_discount_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Data-Grab uploads (and any file missing DISCOUNT), calculate the
    discount percentage from GLOBAL_PRICE and GLOBAL_SALE_PRICE.
    Skips rows that already have a value.
    """
    if "DISCOUNT" not in df.columns:
        df = df.copy()
        df["DISCOUNT"] = ""
    else:
        df = df.copy()

    missing = df["DISCOUNT"].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("")
    if not missing.any():
        return df

    op = pd.to_numeric(
        df.get("GLOBAL_PRICE", pd.Series(dtype=float)), errors="coerce"
    )
    sp = pd.to_numeric(
        df.get("GLOBAL_SALE_PRICE", pd.Series(dtype=float)), errors="coerce"
    )
    mask = missing & (op > 0) & (sp < op)
    df.loc[mask, "DISCOUNT"] = (
        ((op - sp) / op * 100).round(0).astype("Int64").astype(str) + "%"
    )[mask]
    return df


def _flag_corrupted_old_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark OLD_PRICE_CORRUPTED = True when GLOBAL_PRICE is implausibly large
    relative to GLOBAL_SALE_PRICE (>500×). This catches concatenated-number
    artefacts like 82848717 when the real price is ~5000.
    The flag column is used by the discount-integrity check below.
    """
    df = df.copy()
    op = pd.to_numeric(df.get("GLOBAL_PRICE",  pd.Series(dtype=float)), errors="coerce")
    sp = pd.to_numeric(df.get("GLOBAL_SALE_PRICE", pd.Series(dtype=float)), errors="coerce")
    df["_OLD_PRICE_CORRUPTED"] = (op > 0) & (sp > 0) & (op > sp * 500)
    return df


def _flag_discount_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark _DISCOUNT_MISMATCH = True when the stated DISCOUNT diverges from
    the value calculated from GLOBAL_PRICE / GLOBAL_SALE_PRICE by more than
    5 percentage points. Rows with corrupted old prices are excluded.
    """
    df = df.copy()
    if "DISCOUNT" not in df.columns:
        df["_DISCOUNT_MISMATCH"] = False
        return df

    op  = pd.to_numeric(df.get("GLOBAL_PRICE",      pd.Series(dtype=float)), errors="coerce")
    sp  = pd.to_numeric(df.get("GLOBAL_SALE_PRICE",  pd.Series(dtype=float)), errors="coerce")
    stated = pd.to_numeric(
        df["DISCOUNT"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )
    calc = ((op - sp) / op * 100).where((op > 0) & (sp < op))
    corrupted = df.get("_OLD_PRICE_CORRUPTED", pd.Series(False, index=df.index))
    df["_DISCOUNT_MISMATCH"] = (
        stated.notna() & calc.notna() & ~corrupted & (abs(calc - stated) > 5)
    )
    return df


def _flag_brand_in_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark _BRAND_IN_TITLE = True when the brand name appears 2+ times in NAME.
    """
    df = df.copy()
    if "BRAND" not in df.columns or "NAME" not in df.columns:
        df["_BRAND_IN_TITLE"] = False
        return df

    def _check(row):
        brand = str(row.get("BRAND", "")).lower().strip()
        name  = str(row.get("NAME",  "")).lower()
        return bool(brand) and name.count(brand) >= 2

    df["_BRAND_IN_TITLE"] = df.apply(_check, axis=1)
    return df


def _flag_rating_no_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark _RATING_NO_REVIEWS = True when a product has a non-zero RATING but
    REVIEW_COUNT is 0 or missing.
    """
    df = df.copy()
    rating  = pd.to_numeric(df.get("RATING",       pd.Series(dtype=float)), errors="coerce")
    reviews = pd.to_numeric(df.get("REVIEW_COUNT",  pd.Series(dtype=float)), errors="coerce")
    df["_RATING_NO_REVIEWS"] = rating.gt(0) & (reviews.isna() | reviews.eq(0))
    return df


def _render_data_quality_flags(df: pd.DataFrame) -> None:
    """
    Render expandable sections for each internal data-quality flag.
    Called after pipeline processing, before the main validation block.
    """
    flag_cfg = [
        (
            "_OLD_PRICE_CORRUPTED",
            "⚠️ Corrupted old price",
            "Old price is 500× or more the current price — likely two values concatenated by the export system.",
            ["PRODUCT_SET_SID", "NAME", "GLOBAL_SALE_PRICE", "GLOBAL_PRICE", "DISCOUNT"],
        ),
        (
            "_DISCOUNT_MISMATCH",
            "⚠️ Discount doesn't match prices",
            "Stated discount differs from the calculated (old − new) / old by more than 5 percentage points.",
            ["PRODUCT_SET_SID", "NAME", "GLOBAL_SALE_PRICE", "GLOBAL_PRICE", "DISCOUNT"],
        ),
        (
            "_BRAND_IN_TITLE",
            "⚠️ Brand repeated in title",
            "The brand name appears two or more times in the product title.",
            ["PRODUCT_SET_SID", "NAME", "BRAND"],
        ),
        (
            "_RATING_NO_REVIEWS",
            "⚠️ Rating with no reviews",
            "Product has a star rating but zero (or missing) review count.",
            ["PRODUCT_SET_SID", "NAME", "RATING", "REVIEW_COUNT"],
        ),
    ]

    shown_any = False
    for flag_col, title, desc, show_cols in flag_cfg:
        if flag_col not in df.columns:
            continue
        flagged = df[df[flag_col] == True]  # noqa: E712
        if flagged.empty:
            continue
        shown_any = True
        display_cols = [c for c in show_cols if c in flagged.columns]
        with st.expander(f"{title} ({len(flagged)} product{'s' if len(flagged) != 1 else ''})", expanded=False):
            st.caption(desc)
            st.dataframe(
                flagged[display_cols].fillna("").replace("nan", ""),
                use_container_width=True,
                hide_index=True,
            )

    if shown_any:
        st.info(
            "💡 These data-quality issues were detected in the uploaded file. "
            "They don't affect the validation pass/fail status but should be "
            "reviewed before publishing.",
            icon="ℹ️",
        )


def _restore_pq_item(sid: str):
    """Mark a manually-rejected post-QC item as approved again."""
    st.session_state.final_report.loc[
        st.session_state.final_report["ProductSetSid"] == sid,
        ["Status", "Reason", "Comment", "FLAG"],
    ] = ["Approved", "", "", "Approved by User"]
    st.session_state.pop(f"pq_qrej_{sid}", None)
    st.session_state.pop(f"pq_qrej_reason_{sid}", None)
    st.session_state.exports_cache.clear()
    st.session_state.display_df_cache.clear()
    st.session_state.main_toasts.append("Restored item to approved.")


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🔍 Post-QC Settings")

    pq_country = st.selectbox(
        "Country", COUNTRIES,
        index=COUNTRIES.index(st.session_state.pq_country),
        key="pq_country_select",
    )
    if pq_country != st.session_state.pq_country:
        st.session_state.pq_country = pq_country
        _reset_results()

    country_code = _code(pq_country)
    st.session_state.pq_country = pq_country
    st.session_state._pq_country_code = country_code
    st.markdown("---")

    if _SCRAPER_OK:
        st.caption("🌐 Jumia enrichment active — missing fields auto-filled on upload.")
    else:
        st.caption("⚠️ `jumia_scraper.py` not found. Auto-enrichment disabled.")

    st.markdown("---")
    if st.button("🗑 Clear Results", use_container_width=True, type="secondary"):
        _reset_results()
        st.rerun()

# ------------------------------------------------------------------
# HEADER BANNER & MAIN UI
# ------------------------------------------------------------------
st.markdown(f"""
<div style="background:linear-gradient(135deg,{ORANGE},{ORANGE2});
padding:20px 24px;border-radius:10px;margin-bottom:20px;
box-shadow:0 4px 12px rgba(246,139,30,0.25);">
<h2 style="color:white;margin:0;font-size:26px;font-weight:700;">
🔍 Post-QC Validation</h2>
<p style="color:rgba(255,255,255,0.9);margin:6px 0 0;font-size:13px;">
Validate Jumia product data, auto-fill fields via Scraper, and generate detailed reports.</p>
</div>
""", unsafe_allow_html=True)

if not _POSTQC_OK:
    st.error(f"**postqc.py could not be imported.**\nError: `{_postqc_err_msg}`")
    st.stop()
if not _FULL_VALIDATION_OK:
    st.warning(f"⚠️ Full validation engine not available: `{_fv_err_msg}`. Validations skipped.")

st.header("📥 Input Source", anchor=False)
input_mode = st.radio(
    "Choose Input Method:",
    ["📁 Upload Post-QC File", "📊 Data Grab Upload", "🌐 Paste SKUs / URLs"],
    horizontal=True,
    key="input_mode",
)

# ------------------------------------------------------------------
# INPUT FETCHING
# ------------------------------------------------------------------
all_dfs = []
sig = None

if input_mode in ["📁 Upload Post-QC File", "📊 Data Grab Upload"]:
    uploaded = st.file_uploader(
        f"Drop your {input_mode.split(' ', 1)[1]} here",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key=f"pq_uploader_{input_mode}",
    )
    if uploaded:
        files_data = [{"name": f.name, "bytes": f.read()} for f in uploaded]
        sig = hashlib.md5(
            (
                str(sorted(f["name"] + str(len(f["bytes"])) for f in files_data))
                + country_code
                + input_mode
            ).encode()
        ).hexdigest()

        if st.session_state.pq_last_sig != sig:
            for uf in files_data:
                buf = BytesIO(uf["bytes"])
                if uf["name"].endswith(".xlsx"):
                    raw = pd.read_excel(buf, engine="openpyxl", dtype=str)
                else:
                    if input_mode == "📊 Data Grab Upload":
                        try:
                            raw = pd.read_csv(buf, sep="|", dtype=str)
                            if len(raw.columns) <= 1:
                                buf.seek(0)
                                raw = pd.read_csv(buf, sep=",", dtype=str)
                        except Exception:
                            buf.seek(0)
                            raw = pd.read_csv(buf, dtype=str)
                    else:
                        try:
                            raw = pd.read_csv(buf, dtype=str)
                            if len(raw.columns) <= 1:
                                buf.seek(0)
                                raw = pd.read_csv(buf, sep=";", encoding="ISO-8859-1", dtype=str)
                        except Exception:
                            buf.seek(0)
                            raw = pd.read_csv(buf, sep=";", encoding="ISO-8859-1", dtype=str)

                _is_pq = _POSTQC_OK and detect_file_type(raw) == "post_qc"
                if not _is_pq:
                    raw = _standardise_pq(raw)
                all_dfs.append(raw)
    else:
        if st.session_state.pq_last_sig is not None:
            _reset_results()

elif input_mode == "🌐 Paste SKUs / URLs":
    if not _SCRAPER_OK:
        st.error("⚠️ `jumia_scraper.py` is required for this functionality.")
        st.stop()

    lookup_input_raw = st.text_area(
        "SKUs or Jumia product URLs (one per line)",
        height=160,
        placeholder="GE840EA6C62GANAFAMZ\nhttps://www.jumia.co.ke/some-product.html"
    )

    if lookup_input_raw.strip():
        sig = hashlib.md5(
            (lookup_input_raw + country_code + input_mode).encode()
        ).hexdigest()

        if st.session_state.pq_last_sig != sig:
            do_process = st.button("🔍 Search & Process", type="primary")
            if do_process:
                entries = _parse_lookup_inputs(lookup_input_raw)
                if not entries:
                    st.warning("Please enter at least one valid SKU or URL.")
                else:
                    st.info(f"🔍 Looking up **{len(entries)}** item(s)…")
                    _sbar = st.progress(0, text="Starting…")
                    _stxt = st.empty()

                    rows = []
                    for idx, entry in enumerate(entries):
                        sku = entry["sku"]
                        direct_url = entry["url"]
                        eff_code = entry["country_hint"] or country_code

                        _sbar.progress(idx / len(entries), text=f"Fetching {idx + 1}/{len(entries)} — {sku}")
                        _stxt.caption(f"⏱ {'URL' if direct_url else 'SKU'}: **{entry['input'][:80]}**")

                        try:
                            if direct_url:
                                from jumia_scraper import _scrape_product_page
                                _base = (COUNTRY_URLS or {}).get(eff_code, "https://www.jumia.co.ke")
                                scraped = _scrape_product_page(direct_url, _base)
                            else:
                                scraped = scrape_single_sku(sku, country_code=eff_code)
                        except Exception as exc:
                            logger.warning("Lookup failed for %s: %s", sku, exc)
                            scraped = {}

                        row = {
                            "PRODUCT_SET_SID": sku,
                            "Input": entry["input"],
                            "Type": "URL" if direct_url else "SKU",
                            "Country": eff_code,
                            "Found": "Yes" if scraped else "No",
                        }
                        row.update(scraped)
                        rows.append(row)

                    _sbar.progress(1.0, text="Done!")
                    _stxt.empty()
                    _sbar.empty()

                    raw = pd.DataFrame(rows)
                    raw = _standardise_pq(raw)
                    all_dfs.append(raw)
        else:
            st.success("✅ Inputs processed successfully. Results are displayed below.")

# ------------------------------------------------------------------
# PIPELINE PROCESSING
# ------------------------------------------------------------------
if all_dfs and sig and st.session_state.pq_last_sig != sig:
    _reset_results()

    try:
        support_files = _load_support_files()
        cat_map = support_files.get("category_map", {})
        support_files_pq = dict(support_files)
        support_files_pq["country_code"] = country_code
        support_files_pq["country_name"] = pq_country

        import unicodedata as _ud
        def _nk(s: str) -> str:
            s = _ud.normalize("NFKD", str(s)).lower()
            return re.sub(r"[^a-z0-9]", "", s)
        def _norm_sep(s: str) -> str:
            return re.sub(r"\s*[>/|\\]\s*", " / ", str(s).strip()).lower()

        _cmap_by_path, _cmap_by_name, _cmap_by_seg = {}, {}, {}
        _cmap_xlsx_path = "category_map.xlsx"
        if os.path.exists(_cmap_xlsx_path):
            try:
                _cm_df = pd.read_excel(_cmap_xlsx_path, engine="openpyxl", dtype=str)
                _cm_df.columns = [c.strip().lower() for c in _cm_df.columns]
                _col_name = next((c for c in _cm_df.columns if "name" in c and "path" not in c), None)
                _col_code = next((c for c in _cm_df.columns if "code" in c), None)
                _col_path = next((c for c in _cm_df.columns if "path" in c), None)
                if _col_name and _col_code:
                    for _, _row in _cm_df.iterrows():
                        _cs = str(_row[_col_code]).split(".")[0].strip()
                        if not re.match(r"^\d+$", _cs): continue
                        _nm = str(_row[_col_name]).strip()
                        if _nm and _nm.lower() not in ("nan", "none", ""):
                            _cmap_by_name[_nm.lower()] = _cs
                            _cmap_by_seg.setdefault(_nm.lower(), _cs)
                        if _col_path:
                            _pt = str(_row[_col_path]).strip()
                            if _pt and _pt.lower() not in ("nan", "none", ""):
                                _cmap_by_path[_norm_sep(_pt)] = _cs
                                for _seg in _pt.split(" / "):
                                    _seg = _seg.strip()
                                    if _seg and _seg.lower() not in ("nan","none",""):
                                        _cmap_by_seg.setdefault(_seg.lower(), _cs)
            except Exception as _cm_err:
                logger.warning(f"category_map.xlsx load failed: {_cm_err}")
        else:
            for _n, _c in cat_map.items():
                _cs = str(_c).split(".")[0].strip()
                if re.match(r"^\d+$", _cs):
                    _cmap_by_name[_n.lower()] = _cs
                    _cmap_by_seg[_n.lower()]   = _cs

        _cmap_seg_sorted = sorted(_cmap_by_seg.items(), key=lambda x: len(x[0]), reverse=True)

        def _resolve_cat_code(category: str, existing: str = "") -> str:
            _existing = str(existing).strip()
            raw = str(category).strip()
            if not raw or raw.lower() in ("nan", "none", ""): return _existing
            def _valid(v: str) -> str:
                v = str(v).split(".")[0].strip()
                return v if re.match(r"^\d+$", v) else ""

            c = _valid(_cmap_by_path.get(_norm_sep(raw), ""))
            if c: return c

            segs = [s.strip() for s in re.split(r"\s*[>/|\\]\s*", raw) if s.strip()]
            for seg in reversed(segs):
                c = _valid(_cmap_by_name.get(seg.lower(), ""))
                if c: return c

            for seg in reversed(segs):
                sn = _nk(seg)
                if not sn: continue
                best_c, best_len = "", 0
                for k, v in _cmap_seg_sorted:
                    kn = _nk(k)
                    if not kn: continue
                    if len(kn) < 3: continue
                    if kn == sn or kn in sn or sn in kn:
                        c = _valid(v)
                        if c and len(kn) > best_len:
                            best_c, best_len = c, len(kn)
                if best_c: return best_c

            if re.match(r"^\d+$", _existing): return _existing
            return _existing

        # ── Normalise all loaded DataFrames ────────────────────
        norm_dfs = []
        for df in all_dfs:
            if "PRODUCT_SET_SID" in df.columns:
                ndf = df.copy()
                if "CATEGORY_CODE" not in ndf.columns:
                    ndf["CATEGORY_CODE"] = ""
            else:
                ndf = normalize_post_qc(df, category_map=cat_map)
                if "CATEGORY_CODE" not in ndf.columns:
                    ndf["CATEGORY_CODE"] = ""
            if "CATEGORY" in ndf.columns:
                ndf["CATEGORY_CODE"] = ndf.apply(
                    lambda r: _resolve_cat_code(r.get("CATEGORY", ""), r.get("CATEGORY_CODE", "")), axis=1
                )
            norm_dfs.append(ndf)

        merged = pd.concat(norm_dfs, ignore_index=True)
        # ── Dedup on SKU (keep first occurrence) ───────────────
        # Duplicate-SKU validation is intentionally removed from this page.
        # The dedup still runs so the rest of the pipeline works correctly,
        # but no flag or warning is surfaced to the user.
        merged_dedup = merged.drop_duplicates(subset=["PRODUCT_SET_SID"], keep="first")

        # ── Scraper enrichment ─────────────────────────────────
        if _SCRAPER_OK and input_mode != "🌐 Paste SKUs / URLs":
            _missing_cols = [
                c for c in SCRAPABLE_FIELDS
                if c not in merged_dedup.columns
                or merged_dedup[c].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("").mean() > 0.5
            ]
            if _missing_cols:
                st.info(f"🌐 Enriching **{len(merged_dedup)}** products from Jumia ({', '.join(_missing_cols)})…")
                _bar = st.progress(0, text="Starting enrichment…")
                _txt = st.empty()
                _before_df = merged_dedup.copy()

                def _cb(done, total, sku, bar=_bar, txt=_txt):
                    bar.progress(done / max(total, 1), text=f"Scraped {done}/{total} — {sku}")
                    txt.caption(f"⏱ Last scraped: **{sku}**")

                merged_dedup = enrich_post_qc_df(
                    merged_dedup, country_code=country_code, progress_callback=_cb
                )
                _bar.empty()
                _txt.empty()

                enrich_summary = _build_enrichment_summary(_before_df, merged_dedup)
                total_filled = sum(v["filled"] for v in enrich_summary.values())

                st.session_state.enriched_df        = merged_dedup.copy()
                st.session_state.enrichment_summary = enrich_summary
                st.session_state.enrichment_done    = True

                if total_filled:
                    st.toast(f"✅ {total_filled} cell(s) filled from Jumia", icon="🌐")
                else:
                    st.toast("No new data found on Jumia.", icon="ℹ️")
            else:
                st.toast("All enrichable columns already populated — no scraping needed.", icon="ℹ️")

        elif input_mode == "🌐 Paste SKUs / URLs":
            st.session_state.enriched_df = merged_dedup.copy()
            st.session_state.enrichment_done = True

        # ── Prepare data for full validation ───────────────────
        ready = merged_dedup.copy()

        if "CATEGORY_PATH" in ready.columns:
            if "CATEGORY" not in ready.columns:
                ready["CATEGORY"] = ready["CATEGORY_PATH"]
            else:
                ready["CATEGORY"] = ready["CATEGORY"].where(ready["CATEGORY"].ne(""), ready["CATEGORY_PATH"])

        ready["ACTIVE_STATUS_COUNTRY"] = country_code

        if "CATEGORY" in ready.columns:
            if "CATEGORY_CODE" not in ready.columns:
                ready["CATEGORY_CODE"] = ""
            ready["CATEGORY_CODE"] = ready.apply(
                lambda r: _resolve_cat_code(r.get("CATEGORY", ""), r.get("CATEGORY_CODE", "")), axis=1
            )
            _unresolved = ready[~ready["CATEGORY_CODE"].astype(str).str.match(r"^\d+$")]["CATEGORY"].dropna().unique()
            if len(_unresolved):
                st.warning(
                    f"⚠️ **{len(_unresolved)}** category path(s) could not be resolved to a numeric code.\n\n"
                    + "\n".join(f"- `{c}`" for c in _unresolved[:10])
                    + ("\n- *(and more…)*" if len(_unresolved) > 10 else ""),
                    icon="⚠️",
                )

        if "LIST_VARIATIONS" not in ready.columns or (
            ready["LIST_VARIATIONS"].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("").all()
        ):
            _var_parts = []
            for _vcol in ["COLOR", "SIZES_AVAILABLE"]:
                if _vcol in ready.columns:
                    _var_parts.append(ready[_vcol].astype(str).str.strip().replace({"nan": "", "None": ""}))
            if _var_parts:
                import functools as _ft
                ready["LIST_VARIATIONS"] = _ft.reduce(
                    lambda a, b: a.where(b.eq(""), a + " | " + b).where(a.ne(""), b), _var_parts
                ).replace({"": pd.NA})
            else:
                ready["LIST_VARIATIONS"] = ""

        def _parse_price_str(s) -> str:
            try:
                clean = re.sub(r"[^\d.]", "", str(s))
                val = float(clean) if clean else None
                if val and val > 0: return str(val)
            except (ValueError, TypeError): pass
            return ""

        _file_price_candidates = [c for c in ["GLOBAL_SALE_PRICE", "GLOBAL_PRICE", "SALE_PRICE"] if c in ready.columns]
        _file_price_col = _file_price_candidates[0] if _file_price_candidates else None

        if _file_price_col and _file_price_col != "GLOBAL_SALE_PRICE":
            ready["GLOBAL_SALE_PRICE"] = ready[_file_price_col].astype(str).apply(_parse_price_str)
        elif "GLOBAL_SALE_PRICE" not in ready.columns:
            ready["GLOBAL_SALE_PRICE"] = ""

        _sp_empty = ready["GLOBAL_SALE_PRICE"].astype(str).str.strip().eq("")
        if "PRICE" in ready.columns and _sp_empty.any():
            ready.loc[_sp_empty, "GLOBAL_SALE_PRICE"] = (
                ready.loc[_sp_empty, "PRICE"].astype(str).apply(_parse_price_str)
            )

        if "GLOBAL_PRICE" not in ready.columns or (
            ready["GLOBAL_PRICE"].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("").all()
        ):
            ready["GLOBAL_PRICE"] = ready["GLOBAL_SALE_PRICE"]

        if "PARENTSKU" not in ready.columns: ready["PARENTSKU"] = ""

        for _col_needed in [
            "NAME", "BRAND", "COLOR", "SELLER_NAME",
            "PRODUCT_WARRANTY", "WARRANTY_DURATION", "COUNT_VARIATIONS", "COLOR_FAMILY",
        ]:
            if _col_needed not in ready.columns: ready[_col_needed] = ""
        for _col_str in ["NAME", "BRAND", "COLOR", "SELLER_NAME"]:
            if _col_str in ready.columns:
                ready[_col_str] = ready[_col_str].astype(str).fillna("")

        # ── NEW: derive IS_OFFICIAL_STORE from TAGS ────────────
        ready = _derive_official_store_from_tags(ready)

        # ── NEW: calculate DISCOUNT when missing ───────────────
        ready = _calc_discount_if_missing(ready)

        # ── NEW: internal quality flags (not pass/fail — advisory) ─
        ready = _flag_corrupted_old_price(ready)
        ready = _flag_discount_mismatch(ready)
        ready = _flag_brand_in_title(ready)
        ready = _flag_rating_no_reviews(ready)

        # ── Run full validation pipeline ───────────────────────
        if _FULL_VALIDATION_OK:
            # Strip internal flag columns before passing to validator
            # (validator doesn't know about them and may choke on extras)
            _internal_flags = [c for c in ready.columns if c.startswith("_")]
            _ready_for_val  = ready.drop(columns=_internal_flags, errors="ignore")

            with st.spinner("Running validations…"):
                _cv = CountryValidator(pq_country)
                try:
                    _sf_full = load_all_support_files()
                except Exception:
                    _sf_full = support_files_pq

                data_has_warranty = all(
                    c in _ready_for_val.columns
                    for c in ["PRODUCT_WARRANTY", "WARRANTY_DURATION"]
                )
                _val_report, _val_results = validate_products(
                    _ready_for_val, _sf_full, _cv, data_has_warranty
                )

            st.session_state.pq_val_report  = _val_report
            st.session_state.pq_val_results = _val_results
            st.session_state.pq_flags_init  = False

        st.session_state.pq_data     = ready   # keep flags for advisory display
        st.session_state.pq_last_sig = sig
        if not st.session_state.enrichment_done:
            st.session_state.enriched_df = ready.copy()

    except Exception as exc:
        st.error(f"Processing error: {exc}")
        st.code(traceback.format_exc())

# ------------------------------------------------------------------
# ENRICHMENT PANEL
# ------------------------------------------------------------------
if st.session_state.enrichment_done and st.session_state.enriched_df is not None:
    edf      = st.session_state.enriched_df
    esummary = st.session_state.enrichment_summary

    st.markdown("---")

    if esummary:
        total_cells_filled = sum(v["filled"] for v in esummary.values())
        cols_filled        = [c for c, v in esummary.items() if v["filled"] > 0]

        st.markdown(
            f"### 🌐 Jumia Enrichment Results "
            f'<span class="enrich-badge">+{total_cells_filled} cells filled</span>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Products processed", len(edf))
        m2.metric("Fields attempted",   len(SCRAPABLE_FIELDS))
        m3.metric("Fields enriched",    len(cols_filled))
        m4.metric("Total cells filled", total_cells_filled)

        with st.expander("📊 Field-by-field breakdown", expanded=True):
            chip_html = ""
            for col in SCRAPABLE_FIELDS:
                info  = esummary.get(col, {"before": 0, "after": 0, "filled": 0})
                label = col.replace("_", " ").title()
                if info["filled"] > 0:
                    chip_html += f'<span class="field-chip">✅ {label} (+{info["filled"]})</span>'
                else:
                    status = "—" if info["after"] == 0 else f'{info["after"]} rows'
                    chip_html += f'<span class="field-chip missing">⬜ {label} ({status})</span>'
            st.markdown(chip_html, unsafe_allow_html=True)

    else:
        st.markdown(
            f"### 🌐 Data Generated "
            f'<span class="enrich-badge">{len(edf)} Products Loaded</span>',
            unsafe_allow_html=True,
        )

    with st.expander(f"📋 Inline preview — enriched data ({len(edf)} rows)", expanded=False):
        _preview_cols = [
            c for c in edf.columns
            if not c.startswith("_")
            and edf[c].astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").any()
        ]
        st.dataframe(
            edf[_preview_cols].fillna("").replace("nan", ""),
            use_container_width=True, height=400,
        )

    st.markdown("#### ⬇️ Download Enriched Dataset")
    dl_col1, dl_col2 = st.columns(2)
    # Strip internal flag cols from download
    _edf_clean = edf[[c for c in edf.columns if not c.startswith("_")]]
    with dl_col1:
        st.download_button(
            label="📥 Download as Excel (.xlsx)",
            data=_to_excel_bytes(_edf_clean),
            file_name=f"enriched_data_{pq_country.lower()}_{country_code}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, type="primary",
        )
    with dl_col2:
        st.download_button(
            label="📄 Download as CSV (.csv)",
            data=_to_csv_bytes(_edf_clean),
            file_name=f"enriched_data_{pq_country.lower()}_{country_code}.csv",
            mime="text/csv", use_container_width=True,
        )

# ------------------------------------------------------------------
# DATA QUALITY FLAGS (advisory — shown before validation results)
# ------------------------------------------------------------------
_pq_data = st.session_state.pq_data
if not _pq_data.empty:
    _has_any_flag = any(
        c in _pq_data.columns and _pq_data[c].any()
        for c in ["_OLD_PRICE_CORRUPTED", "_DISCOUNT_MISMATCH",
                  "_BRAND_IN_TITLE", "_RATING_NO_REVIEWS"]
    )
    if _has_any_flag:
        st.markdown("---")
        st.subheader("🔎 Data Quality Notices", anchor=False)
        _render_data_quality_flags(_pq_data)

# ------------------------------------------------------------------
# FULL VALIDATION RESULTS
# ------------------------------------------------------------------
_val_report = st.session_state.get("pq_val_report", pd.DataFrame())

if _FULL_VALIDATION_OK and not _val_report.empty and not _pq_data.empty:
    _app_df = _val_report[_val_report["Status"] == "Approved"]
    _rej_df = _val_report[_val_report["Status"] == "Rejected"]
    _sf     = _load_support_files()
    _cv     = CountryValidator(pq_country)

    st.markdown("---")
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,{ORANGE},{ORANGE2});
        padding:18px 24px;border-radius:10px;margin-bottom:16px;
        box-shadow:0 4px 12px rgba(246,139,30,0.25);'>
        <h3 style='color:white;margin:0;font-size:20px;font-weight:700;'>
        🛡️ Validation Results</h3></div>""",
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    # Use len of pq_data without internal flag cols as "Total Products"
    _total = len(_pq_data[[c for c in _pq_data.columns if not c.startswith("_")]])
    for _col, _lbl, _val, _color in [
        (mc1, "Total Products", _total,         DARK),
        (mc2, "Approved",       len(_app_df),   GREEN),
        (mc3, "Rejected",       len(_rej_df),   RED),
        (mc4, "Rejection Rate", f"{(len(_rej_df)/_total*100) if _total > 0 else 0:.1f}%", ORANGE),
    ]:
        with _col:
            st.markdown(f"<div style='height:4px;background:{_color};border-radius:4px 4px 0 0;'></div>", unsafe_allow_html=True)
            st.metric(_lbl, _val)

    # Always sync final_report so the image grid has data regardless of
    # whether there are rejections or all products are approved.
    if st.session_state.final_report.empty:
        st.session_state.final_report = st.session_state.pq_val_report.copy()

    if not _rej_df.empty:
        st.subheader("🚩 Flags Breakdown", anchor=False)

        if not st.session_state.pq_flags_init:
            _top = _rej_df["FLAG"].value_counts().index[0]
            st.session_state[f"pqexp_{_top}"] = True
            st.session_state.pq_flags_init = True

        st.session_state.final_report = st.session_state.pq_val_report.copy()
        _data_has_w = all(c in _pq_data.columns for c in ["PRODUCT_WARRANTY", "WARRANTY_DURATION"])

        for _flag_title in _rej_df["FLAG"].unique():
            _flag_df = _rej_df[_rej_df["FLAG"] == _flag_title]
            with st.expander(f"{_flag_title} ({len(_flag_df)})", key=f"pqexp_{_flag_title}"):
                render_flag_expander(_flag_title, _flag_df, _pq_data, _data_has_w, _sf, _cv)

        st.session_state.pq_val_report = st.session_state.final_report.copy()
    else:
        st.success("✅ All products passed validation — no rejections found.")

    st.markdown("---")
    st.markdown("#### ⬇️ Download Validation Reports")
    _date_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    _fname_base = f"Validation_Report_{pq_country}_{country_code}_{_date_str}"
    _reasons_df = _sf.get("reasons", pd.DataFrame())

    _export_cfg = [
        ("PIM Export",    _val_report, "All products with status"),
        ("Rejected Only", _rej_df,     "Only rejected products"),
        ("Approved Only", _app_df,     "Only approved products"),
    ]

    _ecols = st.columns(3)
    for _ei, (_etitle, _edf, _edesc) in enumerate(_export_cfg):
        with _ecols[_ei]:
            _ekey = f"pq_exp_{_etitle}"
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style='font-weight:700;font-size:15px;'>{_etitle}</div>"
                    f"<div style='font-size:11px;opacity:.7;margin-top:4px;'>{_edesc}</div>"
                    f"<div style='background:{LIGHT};color:{ORANGE};padding:6px;"
                    f"border-radius:6px;margin-top:10px;font-weight:600;'>"
                    f"{len(_edf):,} rows</div></div>",
                    unsafe_allow_html=True,
                )
                if _ekey not in st.session_state.pq_val_exports:
                    if st.button("Generate", key=f"gen_{_ekey}", type="primary", use_container_width=True):
                        _res, _fn, _mime = generate_smart_export(
                            _edf, f"{_fname_base}_{_etitle.replace(' ','_')}", "simple", _reasons_df,
                        )
                        st.session_state.pq_val_exports[_ekey] = {
                            "data": _res.getvalue(), "fname": _fn, "mime": _mime
                        }
                        st.rerun()
                else:
                    _ec = st.session_state.pq_val_exports[_ekey]
                    st.download_button(
                        "📥 Download", data=_ec["data"], file_name=_ec["fname"],
                        mime=_ec["mime"], use_container_width=True, type="primary",
                        key=f"dl_{_ekey}",
                    )

# ------------------------------------------------------------------
# JTBRIDGE — receives reject / undo messages from the iframe
# Uses pq_qrej_ prefixed keys to avoid colliding with main app state.
# ------------------------------------------------------------------
_pq_bridge_val = st.text_input(
    "jtbridge_pq", value="",
    placeholder="JTBRIDGE_UNIQUE_DO_NOT_USE",
    key=f"pq_bridge_{st.session_state.pq_bridge_counter}",
    label_visibility="collapsed",
)

if _pq_bridge_val:
    try:
        _pq_msg = json.loads(_pq_bridge_val)

        if _pq_msg.get("action") == "reject":
            _pq_payload = _pq_msg.get("payload", {})
            if isinstance(_pq_payload, dict) and _pq_payload:
                _pq_sf = _load_support_files()
                _pq_rgroups: dict = {}
                for _sid, _rkey in _pq_payload.items():
                    _pq_rgroups.setdefault(_rkey, []).append(_sid)
                _pq_total = 0
                for _rkey, _sids in _pq_rgroups.items():
                    _flag    = REASON_MAP.get(_rkey, "Poor images")
                    _rinfo   = _pq_sf.get("flags_mapping", {}).get(
                        _flag,
                        {"reason": "1000042 - Kindly follow our product image upload guideline.",
                         "en": "Poor Image Quality"},
                    )
                    _rcode = _rinfo["reason"]
                    _rcmt  = _rinfo.get("en", "Manual image rejection")
                    st.session_state.final_report.loc[
                        st.session_state.final_report["ProductSetSid"].isin(_sids),
                        ["Status", "Reason", "Comment", "FLAG"],
                    ] = ["Rejected", _rcode, _rcmt, _flag]
                    for _s in _sids:
                        st.session_state[f"pq_qrej_{_s}"]        = True
                        st.session_state[f"pq_qrej_reason_{_s}"] = _flag
                    _pq_total += len(_sids)
                st.session_state.exports_cache.clear()
                st.session_state.display_df_cache.clear()
                st.session_state.pq_val_exports.clear()
                # Sync final_report back into pq_val_report so the flags
                # breakdown above updates on re-render
                st.session_state.pq_val_report = st.session_state.final_report.copy()
                st.session_state.main_toasts.append(
                    (f"Rejected {_pq_total} product(s) via image review", "🖼️")
                )
                st.session_state.pq_bridge_counter += 1
                st.session_state.pq_do_scroll_top = False
                st.rerun()

        elif _pq_msg.get("action") == "undo":
            _pq_payload = _pq_msg.get("payload", {})
            if isinstance(_pq_payload, dict):
                for _sid in _pq_payload.keys():
                    _restore_pq_item(_sid)
                st.session_state.pq_val_report = st.session_state.final_report.copy()
                st.session_state.pq_bridge_counter += 1
                st.session_state.pq_do_scroll_top = False
                st.rerun()

    except Exception as _pq_bridge_err:
        logger.error(f"PQ bridge parse error: {_pq_bridge_err}")


# ------------------------------------------------------------------
# IMAGE REVIEW GRID
# Shown after validation results when data is available.
# ------------------------------------------------------------------
@st.fragment
def _render_pq_image_grid():
    _fr   = st.session_state.get("final_report", pd.DataFrame())
    _data = st.session_state.pq_data

    if _fr.empty or _data.empty:
        return

    # Read country from session state — pq_country is a local var in the outer
    # script and is NOT available during fragment-scoped reruns.
    _pq_country      = st.session_state.get("pq_country", "Kenya")

    _sf = _load_support_files()

    st.markdown("---")
    st.header("\U0001f5bc\ufe0f Manual Image Review", anchor=False)
    st.caption(
        "Inspect product images on this page. Use **Poor Image** or the "
        "dropdown to reject individual cards, then **Batch Reject** "
        "to commit. Rejections sync back to the validation report above."
    )

    # Show approved + already-manually-rejected (so undo overlay works)
    _committed_sids = {
        k.replace("pq_qrej_", "")
        for k in st.session_state.keys()
        if k.startswith("pq_qrej_") and "reason" not in k
    }
    _approved_mask = (
        (_fr["Status"] == "Approved") |
        (_fr["ProductSetSid"].isin(_committed_sids))
    )
    _grid_fr = _fr[_approved_mask]

    # ── filters ──────────────────────────────────────────────────────
    _gc1, _gc2, _gc3 = st.columns([1.5, 1.5, 2])
    with _gc1:
        _gsearch_n = st.text_input("Search by Name", placeholder="Product name\u2026",
                                   key="pq_grid_search_n")
    with _gc2:
        _gsearch_sc = st.text_input("Search by Seller/Category",
                                    placeholder="Seller or Category\u2026",
                                    key="pq_grid_search_sc")
    with _gc3:
        st.session_state.pq_grid_items_per_page = st.select_slider(
            "Items per page", options=[20, 50, 100, 200],
            value=st.session_state.pq_grid_items_per_page,
            key="pq_grid_ipp",
        )

    # ── merge product data ────────────────────────────────────────────
    _avail_cols  = [c for c in GRID_COLS if c in _data.columns]
    _review_data = pd.merge(
        _grid_fr[["ProductSetSid"]],
        _data[_avail_cols],
        left_on="ProductSetSid", right_on="PRODUCT_SET_SID", how="left",
    )
    if "MAIN_IMAGE" not in _review_data.columns:
        _review_data["MAIN_IMAGE"] = ""

    if _gsearch_n:
        _review_data = _review_data[
            _review_data["NAME"].astype(str).str.contains(_gsearch_n, case=False, na=False)
        ]
    if _gsearch_sc:
        _mc = (
            _review_data["CATEGORY"].astype(str).str.contains(_gsearch_sc, case=False, na=False)
            if "CATEGORY" in _review_data.columns
            else pd.Series(False, index=_review_data.index)
        )
        _ms = _review_data["SELLER_NAME"].astype(str).str.contains(
            _gsearch_sc, case=False, na=False
        )
        _review_data = _review_data[_mc | _ms]

    if _review_data.empty:
        st.info("No products to display in the image grid.")
        return

    # ── pagination ────────────────────────────────────────────────────
    _ipp         = st.session_state.pq_grid_items_per_page
    _total_pages = max(1, (len(_review_data) + _ipp - 1) // _ipp)
    if st.session_state.pq_grid_page >= _total_pages:
        st.session_state.pq_grid_page = 0

    _pg_c1, _pg_c2, _pg_c3 = st.columns([1, 2, 1], vertical_alignment="center")
    with _pg_c1:
        if st.button("\u25c4 Prev", use_container_width=True,
                     disabled=st.session_state.pq_grid_page == 0,
                     key="pq_grid_prev"):
            st.session_state.pq_grid_page = max(0, st.session_state.pq_grid_page - 1)
            st.session_state.pq_do_scroll_top = True
            st.rerun(scope="fragment")
    with _pg_c2:
        _new_page = st.number_input(
            f"Page (of {_total_pages} \u00b7 {len(_review_data)} items)",
            min_value=1, max_value=max(1, _total_pages),
            value=st.session_state.pq_grid_page + 1, step=1,
            key="pq_grid_page_input",
        )
        if _new_page - 1 != st.session_state.pq_grid_page:
            st.session_state.pq_grid_page = _new_page - 1
            st.session_state.pq_do_scroll_top = True
            st.rerun(scope="fragment")
    with _pg_c3:
        if st.button("Next \u25ba", use_container_width=True,
                     disabled=st.session_state.pq_grid_page >= _total_pages - 1,
                     key="pq_grid_next"):
            st.session_state.pq_grid_page += 1
            st.session_state.pq_do_scroll_top = True
            st.rerun(scope="fragment")

    _page_start = st.session_state.pq_grid_page * _ipp
    _page_data  = _review_data.iloc[_page_start : _page_start + _ipp]

    # ── image quality warnings (async) ───────────────────────────────
    _page_warnings: dict = {}
    if _IMAGE_CHECK_OK:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as _ex:
            _fut_map = {
                _ex.submit(
                    _analyze_image_quality,
                    str(_r.get("MAIN_IMAGE", "")).strip()
                ): str(_r["PRODUCT_SET_SID"])
                for _, _r in _page_data.iterrows()
            }
            for _fut in concurrent.futures.as_completed(_fut_map):
                _warns = _fut.result()
                if _warns:
                    _page_warnings[_fut_map[_fut]] = _warns

    # ── committed rejections for this page ───────────────────────────
    _rejected_state = {
        _sid: st.session_state[f"pq_qrej_reason_{_sid}"]
        for _sid in _page_data["PRODUCT_SET_SID"].astype(str)
        if st.session_state.get(f"pq_qrej_{_sid}")
    }

    _cols_per_row = 3 if st.session_state.get("layout_mode") == "centered" else 4

    # ── build and render the grid iframe ─────────────────────────────
    # Try the main app grid builder first; fall back to self-contained version.
    # Use _pq_country (from session state) NOT the outer-scope pq_country.
    _grid_html = None
    try:
        _grid_html = build_fast_grid_html(
            _page_data, _sf.get("flags_mapping", {}),
            _pq_country, _page_warnings, _rejected_state, _cols_per_row,
        )
    except Exception as _grid_err:
        logger.warning("build_fast_grid_html failed (%s), using fallback", _grid_err)

    if _grid_html is None:
        _grid_html = _build_pq_grid_html(
            _page_data, _page_warnings, _rejected_state, _cols_per_row,
        )

    components.html(_grid_html, height=800, scrolling=True)

    if st.session_state.get("pq_do_scroll_top", False):
        components.html(
            "<script>"
            "window.parent.document.querySelector('.main')"
            ".scrollTo({top:0,behavior:'smooth'});"
            "</script>",
            height=0,
        )
        st.session_state.pq_do_scroll_top = False


# ------------------------------------------------------------------
# Lightweight image quality checker (no dependency on main app cache)
# ------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def _analyze_image_quality(url: str):
    if not url or not url.startswith("http"):
        return []
    if not _IMAGE_CHECK_OK:
        return []
    warnings = []
    try:
        resp = _requests.get(url, timeout=1, stream=True)
        if resp.status_code == 200:
            img = _Image.open(resp.raw)
            w, h = img.size
            if w < 300 or h < 300:
                warnings.append("Low Resolution")
            ratio = h / w if w > 0 else 1
            if ratio > 1.5:
                warnings.append("Tall (Screenshot?)")
            elif ratio < 0.6:
                warnings.append("Wide Aspect")
    except Exception:
        pass
    return warnings


# ------------------------------------------------------------------
# Fallback grid builder (used if build_fast_grid_html isn't importable)
# Mirrors the main app's grid but scoped to post-QC page keys.
# ------------------------------------------------------------------
def _build_pq_grid_html(page_data, page_warnings, rejected_state, cols_per_row):
    import json as _json

    O = ORANGE
    R = RED
    G = GREEN

    committed_json = _json.dumps(rejected_state)

    cards_data = []
    for _, row in page_data.iterrows():
        sid     = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip()
        if img_url.startswith("http://"):
            img_url = img_url.replace("http://", "https://")
        if not img_url.startswith("http"):
            img_url = "https://via.placeholder.com/150?text=No+Image"
        try:
            sp   = row.get("GLOBAL_SALE_PRICE")
            rp   = row.get("GLOBAL_PRICE")
            uval = sp if pd.notna(sp) and str(sp).strip() not in ("", "nan") else rp
            price_str = f"KSh {float(str(uval).replace(',','')):.0f}" if pd.notna(uval) else ""
        except Exception:
            price_str = ""
        cards_data.append({
            "sid":      sid,
            "img":      img_url,
            "name":     str(row.get("NAME",  "")),
            "brand":    str(row.get("BRAND", "Unknown Brand")),
            "cat":      str(row.get("CATEGORY", "Unknown Category")),
            "seller":   str(row.get("SELLER_NAME", "Unknown Seller")),
            "warnings": page_warnings.get(sid, []),
            "price":    price_str,
        })
    cards_json = _json.dumps(cards_data)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:sans-serif;}}
body{{background:#f5f5f5;padding:8px;}}
.ctrl-bar{{
  position:sticky;top:0;z-index:999;
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  padding:8px 12px;background:rgba(255,255,255,0.96);
  border-bottom:2px solid {O};border-radius:4px;margin-bottom:12px;
  box-shadow:0 4px 16px rgba(0,0,0,0.12);
}}
.sel-count{{font-weight:700;color:{O};font-size:13px;min-width:80px;}}
.reason-sel{{flex:1;min-width:160px;padding:6px 10px;border:1px solid #ccc;border-radius:4px;font-size:12px;background:#fff;}}
.batch-btn{{padding:7px 14px;background:{O};color:#fff;border:none;border-radius:4px;font-weight:700;font-size:12px;cursor:pointer;}}
.batch-btn:hover{{opacity:.88;}}
.desel-btn{{padding:7px 12px;background:#fff;color:#555;border:1px solid #ccc;border-radius:4px;font-size:12px;cursor:pointer;}}
.grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}
.card{{border:2px solid #e0e0e0;border-radius:8px;padding:10px;background:#fff;position:relative;transition:border-color .15s;z-index:1;}}
.card.selected{{border-color:{G};box-shadow:0 0 0 3px rgba(76,175,80,.2);background:rgba(76,175,80,.04);}}
.card.staged-rej{{border-color:{R};box-shadow:0 0 0 3px rgba(231,60,23,.2);background:rgba(231,60,23,.04);}}
.card.committed-rej{{border-color:#bbb;opacity:.6;}}
.img-wrap{{position:relative;cursor:pointer;height:180px;display:flex;align-items:center;justify-content:center;background:#fff;border-radius:6px;}}
.card-img{{width:100%;height:180px;object-fit:contain;border-radius:6px;transition:transform .2s;}}
.card.committed-rej .card-img{{filter:grayscale(80%);}}
.card-img.zoomed{{transform:scale(2.3);box-shadow:0 15px 50px rgba(0,0,0,0.6);border:2px solid {O};z-index:9999;position:relative;border-radius:8px;}}
.zoom-btn{{position:absolute;bottom:6px;left:6px;width:28px;height:28px;background:rgba(255,255,255,.95);border-radius:50%;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,.3);z-index:100;font-size:14px;}}
.tick{{position:absolute;bottom:6px;right:6px;width:22px;height:22px;border-radius:50%;background:rgba(0,0,0,.18);display:flex;align-items:center;justify-content:center;color:transparent;font-size:13px;font-weight:900;pointer-events:none;z-index:10;}}
.card.selected .tick{{background:{G};color:#fff;}}
.card.staged-rej .tick{{background:{R};color:#fff;}}
.warn-wrap{{position:absolute;top:6px;right:6px;display:flex;flex-direction:column;gap:3px;z-index:5;pointer-events:none;}}
.warn-badge{{background:rgba(255,193,7,.95);color:#313133;font-size:9px;font-weight:800;padding:3px 7px;border-radius:10px;}}
.price-badge{{position:absolute;top:6px;left:6px;background:rgba(76,175,80,.95);color:#fff;font-size:10px;font-weight:800;padding:3px 7px;border-radius:10px;z-index:5;pointer-events:none;}}
.rej-overlay{{display:none;position:absolute;inset:0;background:rgba(255,255,255,.90);border-radius:6px;flex-direction:column;align-items:center;justify-content:center;z-index:20;gap:5px;padding:8px;text-align:center;}}
.card.committed-rej .rej-overlay{{display:flex;}}
.card.staged-rej .rej-overlay.staged{{display:flex;}}
.rej-badge{{background:{R};color:#fff;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:700;}}
.rej-badge.pending{{background:{O};}}
.rej-label{{font-size:10px;color:{R};font-weight:600;max-width:120px;}}
.undo-btn{{margin-top:8px;padding:6px 12px;background:#313133;color:#fff;border:none;border-radius:4px;font-size:11px;font-weight:bold;cursor:pointer;}}
.meta{{font-size:11px;margin-top:8px;line-height:1.4;}}
.meta .nm{{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.meta .br{{color:{O};font-weight:700;margin:2px 0;}}
.meta .ct{{color:#666;font-size:10px;}}
.meta .sl{{color:#999;font-size:9px;margin-top:4px;border-top:1px dashed #eee;padding-top:4px;}}
.acts{{display:flex;gap:4px;margin-top:8px;}}
.act-btn{{flex:1;padding:6px;font-size:11px;border:none;border-radius:4px;cursor:pointer;font-weight:700;color:#fff;background:{O};}}
.act-more{{flex:1;font-size:11px;border:1px solid #ccc;border-radius:4px;outline:none;cursor:pointer;background:#fff;}}
</style>
</head>
<body>
<div class="ctrl-bar">
  <span class="sel-count" id="sel-count">0 pending</span>
  <select class="reason-sel" id="batch-reason">
    <option value="REJECT_POOR_IMAGE">Poor Image</option>
    <option value="REJECT_WRONG_CAT">Wrong Category</option>
    <option value="REJECT_FAKE">Fake Product</option>
    <option value="REJECT_BRAND">Restricted Brand</option>
    <option value="REJECT_WRONG_BRAND">Wrong Brand</option>
    <option value="REJECT_PROHIBITED">Prohibited</option>
    <option value="REJECT_COLOR">Missing Color</option>
  </select>
  <button class="batch-btn" onclick="doBatchReject()">Batch Reject</button>
  <button class="desel-btn" onclick="doSelectAll()">Select All</button>
  <button class="desel-btn" onclick="doDeselAll()">Deselect All</button>
</div>
<div class="grid" id="card-grid"></div>
<script>
function esc(s){{return(s||'').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}}
var CARDS={cards_json};
var COMMITTED={committed_json};
window._pqSel=window._pqSel||{{}};
window._pqStaged=window._pqStaged||{{}};
var sel=window._pqSel,staged=window._pqStaged;
function sendMsg(type,payload){{
  try{{
    var par=window.parent;
    var inputs=par.document.querySelectorAll('input[type="text"]');
    var bridge=null;
    for(var i=0;i<inputs.length;i++){{
      if(inputs[i].placeholder==='JTBRIDGE_UNIQUE_DO_NOT_USE'){{bridge=inputs[i];break;}}
    }}
    if(!bridge)return;
    var msg=JSON.stringify({{action:type,payload:payload}});
    bridge.focus({{preventScroll:true}});
    Object.getOwnPropertyDescriptor(par.HTMLInputElement.prototype,'value').set.call(bridge,msg);
    bridge.dispatchEvent(new par.Event('input',{{bubbles:true}}));
    setTimeout(function(){{bridge.blur();bridge.dispatchEvent(new par.KeyboardEvent('keydown',{{bubbles:true,cancelable:true,key:'Enter',keyCode:13}}));}},150);
  }}catch(e){{console.error('bridge error:',e);}}
}}
function upd(){{
  var n=Object.keys(sel).length+Object.keys(staged).length;
  document.getElementById('sel-count').textContent=n+' pending';
}}
function renderCard(c){{
  var sid=c.sid,img=esc(c.img);
  var isC=sid in COMMITTED,isS=sid in staged,isSel=!isC&&!isS&&(sid in sel);
  var cls='card'+(isC?' committed-rej':isS?' staged-rej':isSel?' selected':'');
  var warns=(c.warnings||[]).map(w=>'<span class="warn-badge">'+esc(w)+'</span>').join('');
  var price=c.price?'<div class="price-badge">'+esc(c.price)+'</div>':'';
  var zoom='<div class="zoom-btn" onclick="event.stopPropagation();toggleZoom(\''+sid+'\')">&#128269;</div>';
  var overlay='',acts='';
  if(isC){{
    overlay='<div class="rej-overlay"><div class="rej-badge">REJECTED</div><div class="rej-label">'+esc((COMMITTED[sid]||'').replace(/_/g,' '))+'</div><button class="undo-btn" onclick="event.stopPropagation();undoR(\''+sid+'\')">Undo</button></div>';
  }}else if(isS){{
    overlay='<div class="rej-overlay staged"><div class="rej-badge pending">PENDING</div><div class="rej-label">'+esc((staged[sid]||'').replace(/_/g,' '))+'</div><button class="undo-btn" onclick="event.stopPropagation();clrS(\''+sid+'\')">Clear</button></div>';
  }}else{{
    acts='<div class="acts"><button class="act-btn" onclick="event.stopPropagation();stageR(\''+sid+'\',\'REJECT_POOR_IMAGE\')">Poor Image</button><select class="act-more" onchange="if(this.value){{event.stopPropagation();stageR(\''+sid+'\',this.value);this.value=\'\'}}"><option value="">More…</option><option value="REJECT_WRONG_CAT">Wrong Category</option><option value="REJECT_FAKE">Fake</option><option value="REJECT_BRAND">Restricted Brand</option><option value="REJECT_PROHIBITED">Prohibited</option><option value="REJECT_COLOR">Missing Color</option><option value="REJECT_WRONG_BRAND">Wrong Brand</option></select></div>';
  }}
  var shortName=c.name.length>38?esc(c.name.slice(0,38))+'…':esc(c.name);
  return '<div class="'+cls+'" id="card-'+sid+'"><div class="img-wrap" onclick="toggleSel(\''+sid+'\',event)">'+price+'<div class="warn-wrap">'+warns+'</div><img class="card-img" src="'+img+'" onerror="this.src=\'https://via.placeholder.com/150?text=No+Image\'">'+zoom+overlay+'<div class="tick">&#10003;</div></div><div class="meta"><div class="nm" title="'+esc(c.name)+'">'+shortName+'</div><div class="br">'+esc(c.brand)+'</div><div class="ct">'+esc(c.cat)+'</div><div class="sl">'+esc(c.seller)+'</div></div>'+acts+'</div>';
}}
function repl(sid){{var el=document.getElementById('card-'+sid);if(!el)return;var c=CARDS.find(x=>x.sid===sid);if(c){{var t=document.createElement('div');t.innerHTML=renderCard(c);el.replaceWith(t.firstElementChild);}}}}
function toggleZoom(sid){{var img=document.querySelector('#card-'+sid+' .card-img');if(!img)return;if(img.classList.contains('zoomed')){{img.classList.remove('zoomed');img.closest('.card').style.zIndex='1';}}else{{document.querySelectorAll('.zoomed').forEach(e=>{{e.classList.remove('zoomed');if(e.closest('.card'))e.closest('.card').style.zIndex='1';}});img.classList.add('zoomed');img.closest('.card').style.zIndex='999';}}}}
function toggleSel(sid,e){{var img=document.querySelector('#card-'+sid+' .card-img');if(img&&img.classList.contains('zoomed')){{img.classList.remove('zoomed');img.closest('.card').style.zIndex='1';return;}}if(sid in COMMITTED)return;if(sid in staged){{delete staged[sid];}}else if(sid in sel){{delete sel[sid];}}else{{sel[sid]=true;}}repl(sid);upd();}}
function stageR(sid,r){{if(sid in sel)delete sel[sid];staged[sid]=r;repl(sid);upd();}}
function clrS(sid){{delete staged[sid];repl(sid);upd();}}
function undoR(sid){{sendMsg('undo',{{[sid]:true}});delete COMMITTED[sid];repl(sid);upd();}}
function doBatchReject(){{
  var br=document.getElementById('batch-reason').value,payload={{}},count=0;
  for(var s in staged){{payload[s]=staged[s];count++;}}
  for(var s in sel){{payload[s]=br;count++;}}
  if(!count)return;
  for(var s in payload){{COMMITTED[s]=payload[s];delete sel[s];delete staged[s];}}
  sendMsg('reject',payload);
  renderAll();upd();
}}
function doSelectAll(){{CARDS.forEach(c=>{{if(!(c.sid in COMMITTED)&&!(c.sid in staged))sel[c.sid]=true;}});renderAll();upd();}}
function doDeselAll(){{for(var k in sel)delete sel[k];for(var k in staged)delete staged[k];renderAll();upd();}}
function renderAll(){{document.getElementById('card-grid').innerHTML=CARDS.map(renderCard).join('');upd();}}
renderAll();
</script>
</body>
</html>"""


# ------------------------------------------------------------------
# CALL THE IMAGE GRID
# Only when validation has run and produced data.
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# CALL THE IMAGE GRID
# Render whenever product data is available — works with or without
# validation results, and whether products are approved or rejected.
# ------------------------------------------------------------------
_grid_data_ready  = not st.session_state.pq_data.empty
_grid_report_ready = not st.session_state.pq_val_report.empty or not st.session_state.final_report.empty

if _grid_data_ready and _grid_report_ready:
    # Ensure final_report is populated even if validation block was skipped
    # (e.g. _FULL_VALIDATION_OK is False but enrichment ran)
    if st.session_state.final_report.empty and not st.session_state.pq_val_report.empty:
        st.session_state.final_report = st.session_state.pq_val_report.copy()
    _render_pq_image_grid()
