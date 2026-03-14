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
        run_checks as run_post_qc_checks,
        render_post_qc_section,
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
    "pq_summary":         pd.DataFrame(),
    "pq_results":         {},
    "pq_data":            pd.DataFrame(),
    "pq_last_sig":        None,
    "pq_exports_cache":   {},
    "pq_cached_files":    [],
    "scraper_enabled":    False,
    # enrichment state
    "enriched_df":        None,
    "enrichment_summary": {},   # col -> {before, after, filled}
    "enrichment_done":    False,
    # sku lookup state
    "sku_results_df":     pd.DataFrame(),
    "sku_search_done":    False,
    "sku_lookup_country": "Kenya",
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ------------------------------------------------------------------
# CANONICAL FIELD LIST  (module-level so it's always in scope)
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
# HELPERS
# ------------------------------------------------------------------
COUNTRY_CODES = {
    "Kenya":   "KE",
    "Uganda":  "UG",
    "Nigeria": "NG",
    "Ghana":   "GH",
    "Morocco": "MA",
}
COUNTRIES = list(COUNTRY_CODES.keys())


def _t(key):
    return get_translation(st.session_state.ui_lang, key)


def _code(name: str) -> str:
    return COUNTRY_CODES.get(name, "KE")


def _load_support_files() -> dict:
    if "support_files" in st.session_state:
        return st.session_state.support_files
    mod = sys.modules.get("streamlit_app") or sys.modules.get("__main__")
    if mod and hasattr(mod, "load_all_support_files"):
        try:
            sf = mod.load_all_support_files()
            st.session_state.support_files = sf
            return sf
        except Exception:
            pass
    cat_map = load_category_map() if _POSTQC_OK else {}
    return {"category_map": cat_map}


def _reset_results():
    st.session_state.pq_summary       = pd.DataFrame()
    st.session_state.pq_results       = {}
    st.session_state.pq_data          = pd.DataFrame()
    st.session_state.pq_last_sig      = None
    st.session_state.pq_exports_cache = {}
    st.session_state.enriched_df        = None
    st.session_state.enrichment_summary = {}
    st.session_state.enrichment_done    = False


def _count_filled(series: pd.Series) -> int:
    return int(
        series.astype(str).str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
        .ne("")
        .sum()
    )


def _build_enrichment_summary(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Compare before/after to record how many cells each field gained."""
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


def _reset_sku_search():
    st.session_state.sku_results_df  = pd.DataFrame()
    st.session_state.sku_search_done = False


def _clean_sku(raw: str) -> str:
    """Strip Jumia suffix (everything after the first hyphen)."""
    return raw.split("-")[0].strip()


def _render_sku_result_card(sku: str, data: dict):
    """Render a single SKU result as a styled card."""
    has_data = bool(data)
    border_color = GREEN if has_data else RED

    field_labels = {
        # Pricing & availability
        "PRICE":             "Price",
        "DISCOUNT":          "Discount",
        "STOCK_STATUS":      "Stock Status",
        # Identity & classification
        "MODEL":             "Model",
        "GTIN":              "GTIN / Barcode",
        "BRAND":             "Brand",
        "CATEGORY_PATH":     "Category Path",
        "IS_OFFICIAL_STORE": "Official Store",
        # Media
        "MAIN_IMAGE":        "Main Image URL",
        # Physical
        "WEIGHT":            "Weight",
        # Variations
        "COLOR":             "Color(s)",
        "COLOR_IN_TITLE":    "Color in Title",
        "COUNT_VARIATIONS":  "# Variations",
        "SIZES_AVAILABLE":   "Sizes Available",
        # Warranty
        "PRODUCT_WARRANTY":  "Warranty",
        "WARRANTY_DURATION": "Warranty Duration",
        # Ratings
        "RATING":            "Rating",
        "REVIEW_COUNT":      "Reviews",
        # Content
        "DESCRIPTION":       "Description",
        "KEY_FEATURES":      "Key Features",
        "KEY_SPECS":         "Key Specs",
        "SPECIFICATIONS":    "Specifications",
        "WHATS_IN_BOX":      "What's in the Box",
    }

    rows_html = ""
    for key, label in field_labels.items():
        val = data.get(key, "")
        if val:
            rows_html += (
                f'<tr>'
                f'<td style="color:{MED};font-size:12px;padding:3px 8px 3px 0;'
                f'white-space:nowrap;font-weight:500;">{label}</td>'
                f'<td style="font-size:13px;padding:3px 0;word-break:break-all;">{val}</td>'
                f'</tr>'
            )
        else:
            rows_html += (
                f'<tr>'
                f'<td style="color:{MED};font-size:12px;padding:3px 8px 3px 0;'
                f'white-space:nowrap;font-weight:500;">{label}</td>'
                f'<td style="font-size:13px;padding:3px 0;color:#BDBDBD;font-style:italic;">—</td>'
                f'</tr>'
            )

    body = (
        rows_html if rows_html
        else f'<p style="color:{RED};font-size:13px;margin:0;">No data found on Jumia</p>'
    )

    st.markdown(
        f"""
        <div style="border:1px solid {border_color};border-radius:8px;
                    padding:14px 16px;margin-bottom:12px;background:{LIGHT};">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="font-weight:700;font-size:15px;color:{DARK};">{sku}</span>
            <span style="font-size:11px;padding:2px 8px;border-radius:10px;
                         background:{'#E8F5E9' if has_data else '#FFEBEE'};
                         color:{'#2E7D32' if has_data else '#B71C1C'};
                         font-weight:600;">
              {'✅ Found' if has_data else '❌ Not found'}
            </span>
          </div>
          <table style="border-collapse:collapse;width:100%;">{body}</table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🔍 Post-QC Settings")

    pq_country = st.selectbox(
        "Country",
        COUNTRIES,
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
            help=(
                "Uses product SKUs to scrape Color, Warranty, Variations, "
                "Price, Discount, Rating, Image and Stock status from Jumia. "
                "Runs once per file upload."
            ),
        )
        if st.session_state.scraper_enabled:
            st.caption("⏱ ~1–3 s per product row. Runs once on upload.")
            jumia_site_label = {
                "KE": "jumia.co.ke",
                "UG": "jumia.ug",
                "NG": "jumia.com.ng",
                "GH": "jumia.com.gh",
                "MA": "jumia.ma",
            }.get(country_code, "jumia.co.ke")
            st.caption(f"🔗 Target: **{jumia_site_label}**")
    else:
        st.caption(
            "⚠️ `jumia_scraper.py` not found. Place it in the repo root to "
            "enable auto-enrichment."
        )

    st.markdown("---")

    if st.button("🗑 Clear Results", use_container_width=True, type="secondary"):
        _reset_results()
        st.session_state.pq_cached_files = []
        st.rerun()

# ------------------------------------------------------------------
# HEADER BANNER
# ------------------------------------------------------------------
st.markdown(f"""
<div style="background:linear-gradient(135deg,{ORANGE},{ORANGE2});
padding:20px 24px;border-radius:10px;margin-bottom:20px;
box-shadow:0 4px 12px rgba(246,139,30,0.25);">
<h2 style="color:white;margin:0;font-size:26px;font-weight:700;">
🔍 Post-QC Validation</h2>
<p style="color:rgba(255,255,255,0.9);margin:6px 0 0;font-size:13px;">
Upload a Jumia post-QC export · missing fields auto-filled from Jumia</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# HARD STOP if postqc.py failed to import
# ------------------------------------------------------------------
if not _POSTQC_OK:
    st.error(
        f"**postqc.py could not be imported.**\n\n"
        f"Error: `{_postqc_err_msg}`\n\n"
        "Make sure `postqc.py` is in the same folder as `streamlit_app.py`."
    )
    st.stop()

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_upload, tab_sku = st.tabs(["📁 Upload & Validate", "🔎 SKU Lookup"])

# ══════════════════════════════════════════════════════════════════
# TAB 2 — SKU LOOKUP  (defined first so helpers are in scope)
# ══════════════════════════════════════════════════════════════════
with tab_sku:
    if not _SCRAPER_OK:
        st.error(
            "⚠️ `jumia_scraper.py` not found — SKU Lookup requires it. "
            "Place the file in your repo root and restart."
        )
    else:
        st.markdown("### 🔎 SKU Lookup")
        st.caption(
            "Enter one or more SKUs to fetch their data directly from Jumia — "
            "no file upload needed."
        )

        # ── Controls ──────────────────────────────────────────────
        sku_col_a, sku_col_b = st.columns([2, 1])

        with sku_col_a:
            sku_input_raw = st.text_area(
                "SKUs to look up (one per line)",
                height=140,
                placeholder=(
                    "GE840EA6C62GANAFAMZ-269939913\n"
                    "AP456EA7D89HANAFAMZ\n"
                    "SA123BC4D56EGANAFAMZ"
                ),
                key="sku_lookup_input",
            )

        with sku_col_b:
            lookup_country = st.selectbox(
                "Country",
                COUNTRIES,
                index=COUNTRIES.index(st.session_state.sku_lookup_country),
                key="sku_lookup_country_select",
            )
            st.session_state.sku_lookup_country = lookup_country
            lookup_code = _code(lookup_country)

            jumia_domain = {
                "KE": "jumia.co.ke",
                "UG": "jumia.ug",
                "NG": "jumia.com.ng",
                "GH": "jumia.com.gh",
                "MA": "jumia.ma",
            }.get(lookup_code, "jumia.co.ke")
            st.caption(f"🔗 Will search **{jumia_domain}**")

            st.markdown("")
            do_search = st.button(
                "🔍 Search SKUs",
                use_container_width=True,
                type="primary",
                key="sku_search_btn",
            )
            if st.button("🗑 Clear", use_container_width=True, key="sku_clear_btn"):
                _reset_sku_search()
                st.rerun()

        # ── Run search ────────────────────────────────────────────
        if do_search and sku_input_raw and sku_input_raw.strip():
            raw_lines = [l.strip() for l in sku_input_raw.splitlines() if l.strip()]
            skus      = [_clean_sku(l) for l in raw_lines]
            skus      = list(dict.fromkeys(skus))   # deduplicate, preserve order

            if not skus:
                st.warning("Please enter at least one SKU.")
            else:
                _reset_sku_search()
                st.info(f"🔍 Searching **{len(skus)}** SKU(s) on {jumia_domain}…")
                _sbar = st.progress(0, text="Starting…")
                _stxt = st.empty()

                rows: list[dict] = []
                for idx, sku in enumerate(skus):
                    _sbar.progress(
                        idx / len(skus),
                        text=f"Looking up {idx + 1}/{len(skus)} — {sku}",
                    )
                    _stxt.caption(f"⏱ Searching: **{sku}**")

                    try:
                        data = scrape_single_sku(sku, country_code=lookup_code)
                    except Exception as exc:
                        logger.warning("SKU lookup failed for %s: %s", sku, exc)
                        data = {}

                    row = {"SKU": sku, "Found": "Yes" if data else "No"}
                    row.update(data)
                    rows.append(row)

                _sbar.progress(1.0, text="Done!")
                _stxt.empty()
                _sbar.empty()

                results_df = pd.DataFrame(rows)
                # Ensure every canonical field exists as a column (empty string
                # when not scraped). Merge with live SCRAPABLE_FIELDS in case
                # the installed jumia_scraper has additional fields.
                _all_cols = list(dict.fromkeys(
                    _ALL_FIELDS + (SCRAPABLE_FIELDS if _SCRAPER_OK else [])
                ))
                for col in _all_cols:
                    if col not in results_df.columns:
                        results_df[col] = ""
                results_df = results_df.fillna("").replace("nan", "")

                st.session_state.sku_results_df  = results_df
                st.session_state.sku_search_done = True

                found_n    = (results_df["Found"] == "Yes").sum()
                not_found  = len(results_df) - found_n
                st.toast(
                    f"✅ {found_n} found · ❌ {not_found} not found",
                    icon="🔎",
                )

        # ── Display results ───────────────────────────────────────
        if st.session_state.sku_search_done and not st.session_state.sku_results_df.empty:
            res_df   = st.session_state.sku_results_df
            found_n  = (res_df["Found"] == "Yes").sum()
            total_n  = len(res_df)

            st.markdown("---")
            ra, rb, rc = st.columns(3)
            ra.metric("SKUs searched", total_n)
            rb.metric("Found on Jumia", int(found_n))
            rc.metric("Not found", int(total_n - found_n))

            # ── Card view ─────────────────────────────────────────
            with st.expander("🃏 Card view — one card per SKU", expanded=True):
                for _, row in res_df.iterrows():
                    sku  = row["SKU"]
                    data = {
                        k: str(row.get(k, "")).strip()
                        for k in SCRAPABLE_FIELDS
                        if str(row.get(k, "")).strip() not in ("", "nan")
                    }
                    _render_sku_result_card(sku, data)

            # ── Table view ────────────────────────────────────────
            with st.expander("📋 Table view", expanded=False):
                # Always show SKU + Found + every known field (filled or empty)
                base_cols = ["SKU", "Found"]
                all_cols = base_cols + [c for c in _ALL_FIELDS if c not in base_cols]
                # Append any unexpected extra columns from the df
                extra_cols = [c for c in res_df.columns if c not in all_cols]
                show_cols = [c for c in all_cols + extra_cols if c in res_df.columns]
                st.dataframe(
                    res_df[show_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, 50 + 35 * len(res_df)),
                )

            # ── Downloads ─────────────────────────────────────────
            st.markdown("#### ⬇️ Download Results")
            d1, d2 = st.columns(2)
            _fname = f"sku_lookup_{lookup_country.lower()}_{lookup_code}"
            with d1:
                # Re-order res_df columns to canonical order before download
                _dl_cols = [c for c in (["SKU", "Found"] + _ALL_FIELDS)
                            if c in res_df.columns]
                _dl_cols += [c for c in res_df.columns if c not in _dl_cols]
                _res_df_ordered = res_df[_dl_cols]
                st.download_button(
                    label="📥 Download as Excel (.xlsx)",
                    data=_to_excel_bytes(_res_df_ordered),
                    file_name=f"{_fname}.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                    type="primary",
                )
            with d2:
                st.download_button(
                    label="📄 Download as CSV (.csv)",
                    data=_to_csv_bytes(_res_df_ordered),
                    file_name=f"{_fname}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.caption(
                f"{total_n} SKUs · {len(res_df.columns)} columns · "
                f"scraped from {jumia_domain}"
            )

# ══════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════
with tab_upload:

    st.header("📁 Upload Post-QC File", anchor=False)
    st.caption(
        "Expected columns: **SKU, Name, Brand, Category, Price, Seller** "
        "— plus any extras your export includes."
    )

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
        st.session_state.pq_cached_files = []
        _reset_results()

    files = st.session_state.pq_cached_files

    # ------------------------------------------------------------------
    # PROCESSING
    # ------------------------------------------------------------------
    if files:
        sig = hashlib.md5(
            (
                str(sorted(f["name"] + str(len(f["bytes"])) for f in files))
                + country_code
                + str(st.session_state.scraper_enabled)
            ).encode()
        ).hexdigest()

        if st.session_state.pq_last_sig != sig:
            _reset_results()

            try:
                all_dfs = []
                for uf in files:
                    buf = BytesIO(uf["bytes"])
                    if uf["name"].endswith(".xlsx"):
                        raw = pd.read_excel(buf, engine="openpyxl", dtype=str)
                    else:
                        try:
                            raw = pd.read_csv(buf, dtype=str)
                            if len(raw.columns) <= 1:
                                buf.seek(0)
                                raw = pd.read_csv(
                                    buf, sep=";", encoding="ISO-8859-1", dtype=str
                                )
                        except Exception:
                            buf.seek(0)
                            raw = pd.read_csv(
                                buf, sep=";", encoding="ISO-8859-1", dtype=str
                            )

                    if detect_file_type(raw) != "post_qc":
                        st.error(
                            f"**{uf['name']}** doesn't look like a post-QC export. "
                            "Expected columns: SKU, Name, Brand, Category, Price, Seller."
                        )
                        st.stop()

                    all_dfs.append(raw)

                # ── Support files ──────────────────────────────────────
                support_files    = _load_support_files()
                cat_map          = support_files.get("category_map", {})
                support_files_pq = dict(support_files)
                support_files_pq["country_code"] = country_code
                support_files_pq["country_name"] = pq_country

                # ── Normalise ──────────────────────────────────────────
                norm_dfs = []
                for df in all_dfs:
                    ndf = normalize_post_qc(df, category_map=cat_map)
                    if cat_map and "CATEGORY" in ndf.columns:
                        resolved = ndf["CATEGORY_CODE"].str.match(r"^\d+$").sum()
                        if resolved == 0:
                            def _resolve(raw, cmap=cat_map):
                                if not raw or raw == "nan":
                                    return ""
                                segs = [
                                    s.strip()
                                    for s in re.split(r"[>/]", str(raw))
                                    if s.strip()
                                ]
                                for seg in reversed(segs):
                                    code = cmap.get(seg.lower())
                                    if code:
                                        return code
                                last = segs[-1] if segs else raw
                                return re.sub(r"[^a-z0-9]", "_", last.lower())
                            ndf["CATEGORY_CODE"] = (
                                ndf["CATEGORY"].astype(str).apply(_resolve)
                            )
                    norm_dfs.append(ndf)

                merged = pd.concat(norm_dfs, ignore_index=True)
                merged_dedup = merged.drop_duplicates(
                    subset=["PRODUCT_SET_SID"], keep="first"
                )

                # ── Scraper enrichment ─────────────────────────────────
                if _SCRAPER_OK and st.session_state.scraper_enabled:
                    _missing_cols = [
                        c for c in SCRAPABLE_FIELDS
                        if c not in merged_dedup.columns
                        or merged_dedup[c]
                            .astype(str).str.strip()
                            .replace({"nan": "", "None": ""})
                            .eq("")
                            .mean() > 0.5
                    ]
                    if _missing_cols:
                        st.info(
                            f"🌐 Enriching **{len(merged_dedup)}** products from Jumia "
                            f"({', '.join(_missing_cols)})…"
                        )
                        _bar  = st.progress(0, text="Starting enrichment…")
                        _txt  = st.empty()
                        _before_df = merged_dedup.copy()

                        def _cb(done, total, sku, bar=_bar, txt=_txt):
                            bar.progress(
                                done / max(total, 1),
                                text=f"Scraped {done}/{total} — {sku}",
                            )
                            txt.caption(f"⏱ Last scraped: **{sku}**")

                        merged_dedup = enrich_post_qc_df(
                            merged_dedup,
                            country_code=country_code,
                            progress_callback=_cb,
                        )
                        _bar.empty()
                        _txt.empty()

                        enrich_summary = _build_enrichment_summary(
                            _before_df, merged_dedup
                        )
                        total_filled = sum(v["filled"] for v in enrich_summary.values())

                        st.session_state.enriched_df        = merged_dedup.copy()
                        st.session_state.enrichment_summary = enrich_summary
                        st.session_state.enrichment_done    = True

                        if total_filled:
                            st.toast(f"✅ {total_filled} cell(s) filled from Jumia", icon="🌐")
                        else:
                            st.toast("No new data found on Jumia.", icon="ℹ️")
                    else:
                        st.toast(
                            "All enrichable columns already populated — no scraping needed.",
                            icon="ℹ️",
                        )

                # ── Run QC checks ──────────────────────────────────────
                with st.spinner("Running Post-QC checks…"):
                    summary_df, results = run_post_qc_checks(
                        merged_dedup, support_files_pq
                    )

                st.session_state.pq_summary  = summary_df
                st.session_state.pq_results  = results
                st.session_state.pq_data     = merged_dedup
                st.session_state.pq_last_sig = sig

                if not st.session_state.enrichment_done:
                    st.session_state.enriched_df = merged_dedup.copy()

            except Exception as exc:
                st.error(f"Processing error: {exc}")
                st.code(traceback.format_exc())

    # ------------------------------------------------------------------
    # ENRICHMENT PANEL
    # ------------------------------------------------------------------
    if st.session_state.enrichment_done and st.session_state.enriched_df is not None:
        edf      = st.session_state.enriched_df
        esummary = st.session_state.enrichment_summary

        total_cells_filled = sum(v["filled"] for v in esummary.values())
        cols_with_data     = [c for c, v in esummary.items() if v["after"] > 0]
        cols_filled        = [c for c, v in esummary.items() if v["filled"] > 0]

        st.markdown("---")
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
                    chip_html += (
                        f'<span class="field-chip">✅ {label} (+{info["filled"]})</span>'
                    )
                else:
                    status = "—" if info["after"] == 0 else f'{info["after"]} rows'
                    chip_html += (
                        f'<span class="field-chip missing">⬜ {label} ({status})</span>'
                    )
            st.markdown(chip_html, unsafe_allow_html=True)

            detail_rows = []
            for col in SCRAPABLE_FIELDS:
                info = esummary.get(col, {"before": 0, "after": 0, "filled": 0})
                detail_rows.append({
                    "Field":           col,
                    "Before (filled)": info["before"],
                    "After (filled)":  info["after"],
                    "Newly filled":    info["filled"],
                    "Total rows":      len(edf),
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        with st.expander(
            f"📋 Inline preview — enriched data ({len(edf)} rows)", expanded=False
        ):
            preview_cols   = [
                c for c in edf.columns
                if edf[c].astype(str).str.strip()
                .replace({"nan": "", "None": ""}).ne("").any()
            ]
            enriched_first = cols_with_data + [
                c for c in preview_cols if c not in cols_with_data
            ]
            st.dataframe(
                edf[enriched_first].fillna("").replace("nan", ""),
                use_container_width=True,
                height=400,
            )

        st.markdown("#### ⬇️ Download Enriched Dataset")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📥 Download as Excel (.xlsx)",
                data=_to_excel_bytes(edf),
                file_name=f"enriched_postqc_{pq_country.lower()}_{country_code}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet"
                ),
                use_container_width=True,
                type="primary",
            )
        with dl_col2:
            st.download_button(
                label="📄 Download as CSV (.csv)",
                data=_to_csv_bytes(edf),
                file_name=f"enriched_postqc_{pq_country.lower()}_{country_code}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.caption(
            f"File contains all original columns plus enriched Jumia data · "
            f"{len(edf)} rows · {len(edf.columns)} columns"
        )
        st.markdown("---")

    # ------------------------------------------------------------------
    # QC RESULTS
    # ------------------------------------------------------------------
    if not st.session_state.pq_summary.empty:
        _save = {
            k: st.session_state.get(k)
            for k in ("post_qc_summary", "post_qc_results",
                      "post_qc_data", "exports_cache")
        }

        st.session_state.post_qc_summary = st.session_state.pq_summary
        st.session_state.post_qc_results = st.session_state.pq_results
        st.session_state.post_qc_data    = st.session_state.pq_data
        st.session_state.exports_cache   = st.session_state.pq_exports_cache

        support_files = _load_support_files()
        render_post_qc_section(support_files)

        st.session_state.pq_exports_cache = st.session_state.exports_cache

        for k, v in _save.items():
            if v is None:
                st.session_state.pop(k, None)
            else:
                st.session_state[k] = v

    elif files:
        st.info("⏳ File uploaded — results will appear here once processing completes.")
    else:
        st.info("👆 Upload a post-QC export above to get started.")
