import sys
import os
import hashlib
import traceback
import logging

# Make root-level modules (postqc, _preqc_registry, jumia_scraper, translations) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from translations import LANGUAGES, get_translation

try:
    from postqc import (
        detect_file_type, normalize_post_qc,
        run_checks as run_post_qc_checks,
        render_post_qc_section, load_category_map,
    )
    _POSTQC_AVAILABLE = True
except ImportError:
    _POSTQC_AVAILABLE = False

try:
    import _preqc_registry as _reg
except ImportError:
    _reg = None

try:
    from jumia_scraper import enrich_post_qc_df
    _SCRAPER_AVAILABLE = True
except ImportError:
    _SCRAPER_AVAILABLE = False

logger = logging.getLogger(__name__)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
try:
    st.set_page_config(
        page_title="Post-QC Validation",
        page_icon=":material/fact_check:",
        layout=st.session_state.get("layout_mode", "wide"),
    )
except Exception:
    pass

# -------------------------------------------------
# JUMIA THEME (keep consistent with main app)
# -------------------------------------------------
JUMIA_COLORS = {
    'primary_orange':  '#F68B1E',
    'secondary_orange':'#FF9933',
    'jumia_red':       '#E73C17',
    'dark_gray':       '#313133',
    'medium_gray':     '#5A5A5C',
    'light_gray':      '#F5F5F5',
    'border_gray':     '#E0E0E0',
    'success_green':   '#4CAF50',
    'warning_yellow':  '#FFC107',
    'white':           '#FFFFFF',
}

st.markdown(f"""
<style>
    .stButton > button {{ border-radius: 4px; font-weight: 600; }}
    .stButton > button[kind="primary"] {{
        background-color: {JUMIA_COLORS['primary_orange']} !important;
        border: none !important; color: white !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {JUMIA_COLORS['secondary_orange']} !important;
    }}
    div[data-testid="stMetric"] {{
        background: {JUMIA_COLORS['light_gray']};
        border-radius: 0 0 8px 8px;
        padding: 12px 16px 16px 16px;
        text-align: center;
    }}
    div[data-testid="stMetricValue"] {{ color: {JUMIA_COLORS['dark_gray']}; font-weight: 700; font-size: 26px !important; }}
    div[data-testid="stMetricLabel"] {{ color: {JUMIA_COLORS['medium_gray']}; font-size: 11px; text-transform: uppercase; }}
    div[data-testid="stExpander"] {{ border: 1px solid {JUMIA_COLORS['border_gray']}; border-radius: 8px; }}
    div[data-testid="stExpander"] summary {{ background-color: {JUMIA_COLORS['light_gray']}; padding: 12px; border-radius: 8px 8px 0 0; }}
    h1, h2, h3 {{ color: {JUMIA_COLORS['dark_gray']} !important; }}
    @media (prefers-color-scheme: dark) {{
        div[data-testid="stMetricValue"] {{ color: #F5F5F5 !important; }}
        div[data-testid="stMetric"] {{ background: #2a2a2e !important; }}
        h1, h2, h3 {{ color: #F5F5F5 !important; }}
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE DEFAULTS  (only set if not already
# initialised by the main page)
# -------------------------------------------------
if 'ui_lang'              not in st.session_state: st.session_state.ui_lang = "en"
if 'selected_country'     not in st.session_state: st.session_state.selected_country = "Kenya"
if 'layout_mode'          not in st.session_state: st.session_state.layout_mode = "wide"
if 'pq_summary'           not in st.session_state: st.session_state.pq_summary = pd.DataFrame()
if 'pq_results'           not in st.session_state: st.session_state.pq_results = {}
if 'pq_data'              not in st.session_state: st.session_state.pq_data = pd.DataFrame()
if 'pq_last_sig'          not in st.session_state: st.session_state.pq_last_sig = None
if 'pq_exports_cache'     not in st.session_state: st.session_state.pq_exports_cache = {}
if 'pq_cached_files'      not in st.session_state: st.session_state.pq_cached_files = []
if 'scraper_enabled'      not in st.session_state: st.session_state.scraper_enabled = False

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def _t(key):
    return get_translation(st.session_state.ui_lang, key)


def _load_support_files():
    """Re-use main app's cached support files if available, else load fresh."""
    # If the main page has already loaded them they're in st.session_state
    if "support_files" in st.session_state:
        return st.session_state.support_files
    # Otherwise load our own copy (same function, same cache key)
    try:
        # Import the loader from the main app without executing it as __main__
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "streamlit_app", os.path.join(root, "streamlit_app.py")
        )
        # We only need the function, not to run the full script —
        # use a lighter path: just call load_all_support_files directly
        # by importing via sys.modules if it's already there
        mod = sys.modules.get("streamlit_app")
        if mod and hasattr(mod, "load_all_support_files"):
            return mod.load_all_support_files()
    except Exception:
        pass

    # Absolute fallback — load category map only (minimum needed for post-QC)
    cat_map = load_category_map() if _POSTQC_AVAILABLE else {}
    return {"category_map": cat_map}


def _country_code(country_name: str) -> str:
    return {"Kenya": "KE", "Uganda": "UG", "Nigeria": "NG",
            "Ghana": "GH", "Morocco": "MA"}.get(country_name, "KE")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header(":material/fact_check: Post-QC Settings")

    # Country selector — independent from main page
    pq_country = st.selectbox(
        "Country",
        ["Kenya", "Uganda", "Nigeria", "Ghana", "Morocco"],
        index=["Kenya", "Uganda", "Nigeria", "Ghana", "Morocco"].index(
            st.session_state.get("selected_country", "Kenya")
        ),
        key="pq_country_select",
    )
    if pq_country != st.session_state.get("pq_country"):
        st.session_state.pq_country = pq_country
        st.session_state.pq_last_sig = None
        st.session_state.pq_summary  = pd.DataFrame()
        st.session_state.pq_results  = {}
        st.session_state.pq_data     = pd.DataFrame()
        st.session_state.pq_exports_cache = {}

    country_code = _country_code(st.session_state.get("pq_country", pq_country))

    st.markdown("---")

    # Scraper toggle
    if _SCRAPER_AVAILABLE:
        st.header(":material/travel_explore: Enrichment")
        st.session_state.scraper_enabled = st.toggle(
            "Auto-fill missing fields from Jumia",
            value=st.session_state.scraper_enabled,
            help=(
                "Scrapes Color, Warranty, Variation count from Jumia "
                "product pages for columns absent in your upload."
            ),
        )
        if st.session_state.scraper_enabled:
            st.caption(":material/info: ~1–3 s per product. Runs once on upload.")
    else:
        st.caption("Install `beautifulsoup4` + `requests` to enable auto-enrichment.")

    st.markdown("---")

    # Clear cache
    if st.button("Clear Results", use_container_width=True, type="secondary"):
        st.session_state.pq_summary       = pd.DataFrame()
        st.session_state.pq_results       = {}
        st.session_state.pq_data          = pd.DataFrame()
        st.session_state.pq_last_sig      = None
        st.session_state.pq_exports_cache = {}
        st.session_state.pq_cached_files  = []
        st.rerun()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(f"""
<div style='background: linear-gradient(135deg, {JUMIA_COLORS['primary_orange']},
{JUMIA_COLORS['secondary_orange']}); padding: 20px 24px; border-radius: 10px;
margin-bottom: 20px; box-shadow: 0 4px 12px rgba(246,139,30,0.25);'>
<h2 style='color:white;margin:0;font-size:26px;font-weight:700;'>
:material/fact_check: Post-QC Validation</h2>
<p style='color:rgba(255,255,255,0.9);margin:6px 0 0 0;font-size:13px;'>
Upload a Jumia post-QC export to run quality checks</p>
</div>
""", unsafe_allow_html=True)

if not _POSTQC_AVAILABLE:
    st.error("postqc.py could not be imported. Make sure it is in the same folder as streamlit_app.py.")
    st.stop()

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
st.header(":material/upload_file: Upload Post-QC File", anchor=False)
st.caption("Expected columns: SKU, Name, Brand, Category, Price, Seller — plus any extras your export includes.")

uploaded = st.file_uploader(
    "Drop your post-QC export here",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    key="pq_uploader",
)

if uploaded:
    st.session_state.pq_cached_files = [
        {"name": f.name, "bytes": f.read()} for f in uploaded
    ]
elif uploaded is not None and len(uploaded) == 0:
    st.session_state.pq_cached_files  = []
    st.session_state.pq_summary       = pd.DataFrame()
    st.session_state.pq_results       = {}
    st.session_state.pq_data          = pd.DataFrame()
    st.session_state.pq_last_sig      = None
    st.session_state.pq_exports_cache = {}

files = st.session_state.pq_cached_files

# -------------------------------------------------
# PROCESSING
# -------------------------------------------------
if files:
    sig = hashlib.md5(
        (str(sorted(f["name"] + str(len(f["bytes"])) for f in files)) + country_code).encode()
    ).hexdigest()

    if st.session_state.pq_last_sig != sig:
        # Reset
        st.session_state.pq_summary       = pd.DataFrame()
        st.session_state.pq_results       = {}
        st.session_state.pq_data          = pd.DataFrame()
        st.session_state.pq_exports_cache = {}

        try:
            from io import BytesIO as _BIO

            all_dfs = []
            for uf in files:
                buf = _BIO(uf["bytes"])
                if uf["name"].endswith(".xlsx"):
                    raw = pd.read_excel(buf, engine="openpyxl", dtype=str)
                else:
                    try:
                        raw = pd.read_csv(buf, dtype=str)
                        if len(raw.columns) <= 1:
                            buf.seek(0)
                            raw = pd.read_csv(buf, sep=";", encoding="ISO-8859-1", dtype=str)
                    except Exception:
                        buf.seek(0)
                        raw = pd.read_csv(buf, sep=";", encoding="ISO-8859-1", dtype=str)

                # Confirm it's a post-QC file
                if detect_file_type(raw) != "post_qc":
                    st.error(
                        f"**{uf['name']}** doesn't look like a post-QC export. "
                        "Expected columns: SKU, Name, Brand, Category, Price, Seller."
                    )
                    st.stop()

                all_dfs.append(raw)

            # Load support files (uses cached version from main page if available)
            support_files = _load_support_files()
            cat_map = support_files.get("category_map", {})

            support_files_pq = dict(support_files)
            support_files_pq["country_code"] = country_code
            support_files_pq["country_name"] = pq_country

            # Normalise
            try:
                norm_dfs = [normalize_post_qc(df, category_map=cat_map) for df in all_dfs]
            except TypeError:
                import re as _re
                norm_dfs = []
                for df in all_dfs:
                    ndf = normalize_post_qc(df)
                    if cat_map and "CATEGORY" in ndf.columns:
                        def _resolve(raw, cmap=cat_map):
                            if not raw or raw == "nan": return ""
                            segs = [s.strip() for s in _re.split(r"[>/]", str(raw)) if s.strip()]
                            for seg in reversed(segs):
                                code = cmap.get(seg.lower())
                                if code: return code
                            last = segs[-1] if segs else raw
                            return _re.sub(r"[^a-z0-9]", "_", last.lower())
                        ndf["CATEGORY_CODE"] = ndf["CATEGORY"].astype(str).apply(_resolve)
                    norm_dfs.append(ndf)

            merged = pd.concat(norm_dfs, ignore_index=True)
            merged_dedup = merged.drop_duplicates(subset=["PRODUCT_SET_SID"], keep="first")

            # ── Scraper enrichment ────────────────────────────────────────
            if _SCRAPER_AVAILABLE and st.session_state.scraper_enabled:
                _missing = [
                    c for c in ["COLOR", "PRODUCT_WARRANTY", "WARRANTY_DURATION",
                                "COUNT_VARIATIONS", "MAIN_IMAGE"]
                    if c not in merged_dedup.columns
                    or merged_dedup[c].astype(str).str.strip().replace("nan", "").eq("").all()
                ]
                if _missing:
                    _n = len(merged_dedup)
                    _bar  = st.progress(0, text=f"Enriching {_n} products from Jumia…")
                    _txt  = st.empty()

                    def _cb(done, total, sku):
                        _bar.progress(done / max(total, 1),
                                      text=f"Scraped {done}/{total} — {sku}")
                        _txt.caption(f"Last: {sku}")

                    merged_dedup = enrich_post_qc_df(
                        merged_dedup,
                        country_code=country_code,
                        progress_callback=_cb,
                    )
                    _bar.empty()
                    _txt.empty()

                    _filled = sum(
                        1 for c in _missing
                        if c in merged_dedup.columns
                        and not merged_dedup[c].astype(str).str.strip()
                                    .replace("nan", "").eq("").all()
                    )
                    if _filled:
                        st.toast(
                            f":material/check_circle: Enriched {_filled} column(s) from Jumia",
                            icon=":material/travel_explore:",
                        )
                else:
                    st.toast("All columns present — no scraping needed.",
                             icon=":material/info:")
            # ─────────────────────────────────────────────────────────────

            with st.spinner("Running Post-QC checks…"):
                summary_df, results = run_post_qc_checks(merged_dedup, support_files_pq)

            st.session_state.pq_summary  = summary_df
            st.session_state.pq_results  = results
            st.session_state.pq_data     = merged_dedup
            st.session_state.pq_last_sig = sig

        except Exception as e:
            st.error(f"Processing error: {e}")
            st.code(traceback.format_exc())

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
if not st.session_state.pq_summary.empty:
    # Temporarily swap session state keys so render_post_qc_section
    # (which reads post_qc_summary / post_qc_results / post_qc_data)
    # works unchanged from the original implementation.
    _orig_summary  = st.session_state.get("post_qc_summary",  pd.DataFrame())
    _orig_results  = st.session_state.get("post_qc_results",  {})
    _orig_data     = st.session_state.get("post_qc_data",     pd.DataFrame())
    _orig_ecache   = st.session_state.get("exports_cache",    {})

    st.session_state.post_qc_summary  = st.session_state.pq_summary
    st.session_state.post_qc_results  = st.session_state.pq_results
    st.session_state.post_qc_data     = st.session_state.pq_data
    st.session_state.exports_cache    = st.session_state.pq_exports_cache

    support_files = _load_support_files()
    render_post_qc_section(support_files)

    # Persist any export cache entries written by render_post_qc_section
    st.session_state.pq_exports_cache = st.session_state.exports_cache

    # Restore originals
    st.session_state.post_qc_summary = _orig_summary
    st.session_state.post_qc_results = _orig_results
    st.session_state.post_qc_data    = _orig_data
    st.session_state.exports_cache   = _orig_ecache

elif files:
    st.info("File uploaded — results will appear here once processing completes.")
else:
    st.info("Upload a post-QC export above to get started.")
