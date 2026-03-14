import sys
import os
import re
import hashlib
import traceback
import logging
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

# Import the full validation pipeline and support helpers from the main app.
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
    _FULL_VALIDATION_OK    = True
except Exception as _fv_err:
    _FULL_VALIDATION_OK = False
    _fv_err_msg = str(_fv_err)

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
    "scraper_enabled":    False,
    # enrichment state
    "enriched_df":        None,
    "enrichment_summary": {},
    "enrichment_done":    False,
    # full validation results
    "pq_val_report":      pd.DataFrame(),
    "pq_val_results":     {},
    "pq_val_exports":     {},
    "pq_flags_init":      False,
    "display_df_cache":   {},
    "exports_cache":      {},
    "final_report":       pd.DataFrame(),
    "main_toasts":        [],
    "flags_expanded_initialized": False,
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
# MAPPER & HELPERS
# ------------------------------------------------------------------
_PQ_COL_MAP = {
    # Post-QC standard
    "sku":               "PRODUCT_SET_SID",
    "name":              "NAME",
    "brand":             "BRAND",
    "category":          "CATEGORY",
    "categories":        "CATEGORY",      # Data Grab
    "price":             "GLOBAL_SALE_PRICE",
    "newprice":          "GLOBAL_SALE_PRICE", # Data Grab
    "old price":         "GLOBAL_PRICE",
    "oldprice":          "GLOBAL_PRICE",  # Data Grab
    "seller":            "SELLER_NAME",
    "sellername":        "SELLER_NAME",   # Data Grab
    "seller name":       "SELLER_NAME",
    "image url":         "MAIN_IMAGE",
    "image":             "MAIN_IMAGE",    # Data Grab
    "main image":        "MAIN_IMAGE",
    "url":               "PRODUCT_URL",   # Data Grab
    "product url":       "PRODUCT_URL",
    "stock":             "STOCK_STATUS",
    "stock status":      "STOCK_STATUS",
    "rating":            "RATING",
    "averagerating":     "RATING",        # Data Grab
    "average rating":    "RATING",
    "total ratings":     "REVIEW_COUNT",
    "totalratings":      "REVIEW_COUNT",  # Data Grab
    "review count":      "REVIEW_COUNT",
    "discount":          "DISCOUNT",
    "tags":              "TAGS",
    "jumia express":     "JUMIA_EXPRESS",
    "isexpress":         "JUMIA_EXPRESS", # Data Grab
    "shop global":       "SHOP_GLOBAL",
    "isglobal":          "SHOP_GLOBAL",   # Data Grab
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
    st.markdown("---")

    if _SCRAPER_OK:
        st.subheader("🌐 Jumia Enrichment")
        st.session_state.scraper_enabled = st.toggle(
            "Auto-fill missing fields from Jumia",
            value=st.session_state.scraper_enabled,
            help="Uses product SKUs to scrape additional data from Jumia. Runs once per file upload.",
        )
        if st.session_state.scraper_enabled:
            st.caption("⏱ ~1–3 s per product row. Runs once on upload.")
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
                + str(st.session_state.scraper_enabled)
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

                # Standardize columns unless it's perfectly native post-qc format
                _is_pq = _POSTQC_OK and detect_file_type(raw) == "post_qc"
                if not _is_pq:
                    raw = _standardise_pq(raw)
                all_dfs.append(raw)
    else:
        # File uploader cleared
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
            (lookup_input_raw + country_code + str(st.session_state.scraper_enabled) + input_mode).encode()
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

        # Load Category Map rules dynamically
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

        # Normalize across all loaded DataFrames
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
        merged_dedup = merged.drop_duplicates(subset=["PRODUCT_SET_SID"], keep="first")

        # ── Scraper enrichment ─────────────────────────────────
        if _SCRAPER_OK and st.session_state.scraper_enabled and input_mode != "🌐 Paste SKUs / URLs":
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

        # If scraper dumped info into CATEGORY_PATH but CATEGORY is missing, merge them up
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
                    + "\n".join(f"- `{c}`" for c in _unresolved[:10]) + ("\n- *(and more…)*" if len(_unresolved) > 10 else ""),
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
            ready.loc[_sp_empty, "GLOBAL_SALE_PRICE"] = ready.loc[_sp_empty, "PRICE"].astype(str).apply(_parse_price_str)

        if "GLOBAL_PRICE" not in ready.columns or ready["GLOBAL_PRICE"].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("").all():
            ready["GLOBAL_PRICE"] = ready["GLOBAL_SALE_PRICE"]

        if "PARENTSKU" not in ready.columns: ready["PARENTSKU"] = ""

        for _col_needed in ["NAME", "BRAND", "COLOR", "SELLER_NAME", "PRODUCT_WARRANTY", "WARRANTY_DURATION", "COUNT_VARIATIONS", "COLOR_FAMILY"]:
            if _col_needed not in ready.columns: ready[_col_needed] = ""
        for _col_str in ["NAME", "BRAND", "COLOR", "SELLER_NAME"]:
            if _col_str in ready.columns: ready[_col_str] = ready[_col_str].astype(str).fillna("")

        # ── Run full validation pipeline ───────────────────────
        if _FULL_VALIDATION_OK:
            with st.spinner("Running validations…"):
                _cv = CountryValidator(pq_country)
                try: _sf_full = load_all_support_files()
                except Exception: _sf_full = support_files_pq
                
                data_has_warranty = all(c in ready.columns for c in ["PRODUCT_WARRANTY", "WARRANTY_DURATION"])
                _val_report, _val_results = validate_products(ready, _sf_full, _cv, data_has_warranty)
                
            st.session_state.pq_val_report  = _val_report
            st.session_state.pq_val_results = _val_results
            st.session_state.pq_flags_init  = False

        st.session_state.pq_data     = ready
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
        cols_with_data     = [c for c, v in esummary.items() if v["after"] > 0]
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
        preview_cols = [c for c in edf.columns if edf[c].astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").any()]
        st.dataframe(edf[preview_cols].fillna("").replace("nan", ""), use_container_width=True, height=400)

    st.markdown("#### ⬇️ Download Enriched Dataset")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="📥 Download as Excel (.xlsx)",
            data=_to_excel_bytes(edf),
            file_name=f"enriched_data_{pq_country.lower()}_{country_code}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, type="primary",
        )
    with dl_col2:
        st.download_button(
            label="📄 Download as CSV (.csv)",
            data=_to_csv_bytes(edf),
            file_name=f"enriched_data_{pq_country.lower()}_{country_code}.csv",
            mime="text/csv", use_container_width=True,
        )

# ------------------------------------------------------------------
# FULL VALIDATION RESULTS
# ------------------------------------------------------------------
_val_report = st.session_state.get("pq_val_report", pd.DataFrame())
_pq_data    = st.session_state.pq_data

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
    for _col, _lbl, _val, _color in [
        (mc1, "Total Products", len(_pq_data),  DARK),
        (mc2, "Approved",       len(_app_df),   GREEN),
        (mc3, "Rejected",       len(_rej_df),   RED),
        (mc4, "Rejection Rate", f"{(len(_rej_df)/len(_pq_data)*100) if len(_pq_data)>0 else 0:.1f}%", ORANGE),
    ]:
        with _col:
            st.markdown(f"<div style='height:4px;background:{_color};border-radius:4px 4px 0 0;'></div>", unsafe_allow_html=True)
            st.metric(_lbl, _val)

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
                        st.session_state.pq_val_exports[_ekey] = {"data": _res.getvalue(), "fname": _fn, "mime": _mime}
                        st.rerun()
                else:
                    _ec = st.session_state.pq_val_exports[_ekey]
                    st.download_button("📥 Download", data=_ec["data"], file_name=_ec["fname"], mime=_ec["mime"], use_container_width=True, type="primary", key=f"dl_{_ekey}")
