"""
postqc.py
---------
Self-contained Post-QC module for the Jumia Product Validation Tool.

Handles detection, normalisation, validation checks, UI rendering and
export for post-QC (live listing / scrape) files.  Import and call
render_post_qc_section() from app.py — nothing else needs to be touched
in the main file.
"""

import re
import logging
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
POST_QC_SIGNATURE_COLS = {'sku', 'name', 'brand', 'category', 'price', 'seller'}

JUMIA_COLORS = {
    'primary_orange':  '#F68B1E',
    'secondary_orange':'#FF9933',
    'jumia_red':       '#E73C17',
    'dark_gray':       '#313133',
    'medium_gray':     '#5A5A5C',
    'light_gray':      '#F5F5F5',
    'success_green':   '#4CAF50',
    'warning_yellow':  '#FFC107',
}

# Order that check expanders are shown in the UI
CHECK_ORDER = [
    "Duplicate SKU",
    "Brand Repeated in Name",
    "Fashion Brand",
    "Fake Discount",
    "Low Rating (< 3.0)",
    "No Ratings",
    "Single-word Name",
    "Unnecessary Words in Name",
]

# -------------------------------------------------
# FILE DETECTION
# -------------------------------------------------

def detect_file_type(df: pd.DataFrame) -> str:
    """
    Returns 'post_qc' when the DataFrame looks like a Jumia live-listing
    / scrape export, otherwise returns 'pre_qc'.
    """
    cols_lower = set(df.columns.str.strip().str.lower())
    if POST_QC_SIGNATURE_COLS.issubset(cols_lower):
        return 'post_qc'
    return 'pre_qc'


# -------------------------------------------------
# NORMALISATION
# -------------------------------------------------

def normalize_post_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename post-QC columns to internal canonical names and add any
    synthetic columns needed by the check functions.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    col_map = {
        'SKU':           'PRODUCT_SET_SID',
        'Name':          'NAME',
        'Brand':         'BRAND',
        'Category':      'CATEGORY',
        'Price':         'GLOBAL_PRICE',
        'Old Price':     'OLD_PRICE',
        'Seller':        'SELLER_NAME',
        'Image URL':     'MAIN_IMAGE',
        'Product URL':   'PRODUCT_URL',
        'Rating':        'RATING',
        'Total Ratings': 'TOTAL_RATINGS',
        'Discount':      'DISCOUNT',
        'Jumia Express': 'JUMIA_EXPRESS',
        'Shop Global':   'SHOP_GLOBAL',
        'Stock':         'STOCK',
        'Tags':          'TAGS',
    }
    df = df.rename(columns=col_map)

    # Derive a simple CATEGORY_CODE from the last breadcrumb segment
    if 'CATEGORY' in df.columns:
        df['CATEGORY_CODE'] = df['CATEGORY'].astype(str).apply(
            lambda x: re.sub(r'[^a-z0-9]', '_', x.split('>')[-1].strip().lower())
                      if x and x != 'nan' else ''
        )

    # Synthetic columns so the rest of the app doesn't break if it
    # ever inspects these on a post-QC DataFrame
    df.setdefault('ACTIVE_STATUS_COUNTRY', 'UNKNOWN')
    df['_IS_MULTI_COUNTRY'] = False
    df['PARENTSKU']         = df.get('PRODUCT_SET_SID', pd.Series(dtype=str))
    df['COLOR']             = ''
    df['COLOR_FAMILY']      = ''
    df['GLOBAL_SALE_PRICE'] = ''

    return df


# -------------------------------------------------
# INDIVIDUAL CHECKS
# Each function receives the normalised DataFrame and returns a
# filtered copy with an extra 'Comment_Detail' column.
# -------------------------------------------------

def _empty(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=df.columns)


def check_duplicate_sku(df: pd.DataFrame) -> pd.DataFrame:
    if 'PRODUCT_SET_SID' not in df.columns:
        return _empty(df)
    flagged = df[df.duplicated(subset=['PRODUCT_SET_SID'], keep='first')].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Duplicate SKU in file"
    return flagged


def check_brand_in_name(df: pd.DataFrame) -> pd.DataFrame:
    if not {'NAME', 'BRAND'}.issubset(df.columns):
        return _empty(df)
    mask = df.apply(
        lambda r: (
            str(r['BRAND']).strip().lower() not in ['', 'nan', 'fashion', 'generic']
            and str(r['BRAND']).strip().lower() in str(r['NAME']).strip().lower()
        ),
        axis=1,
    )
    flagged = df[mask].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Brand '" + flagged['BRAND'].astype(str) + "' repeated in name"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])


def check_fashion_brand(df: pd.DataFrame) -> pd.DataFrame:
    if 'BRAND' not in df.columns:
        return _empty(df)
    flagged = df[df['BRAND'].astype(str).str.strip().str.lower() == 'fashion'].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Brand is 'Fashion' — real brand should be specified"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])


def check_fake_discount(df: pd.DataFrame, multiplier_threshold: float = 10.0) -> pd.DataFrame:
    if not {'GLOBAL_PRICE', 'OLD_PRICE'}.issubset(df.columns):
        return _empty(df)
    d = df.copy()
    d['_price']     = pd.to_numeric(d['GLOBAL_PRICE'], errors='coerce')
    d['_old_price'] = pd.to_numeric(d['OLD_PRICE'],    errors='coerce')
    mask = (
        d['_price'].notna()     &
        d['_old_price'].notna() &
        (d['_price'] > 0)       &
        (d['_old_price'] > d['_price'] * multiplier_threshold)
    )
    flagged = d[mask].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = flagged.apply(
            lambda r: (
                f"Old price {float(r['_old_price']):,.0f} is "
                f"{float(r['_old_price']) / float(r['_price']):,.0f}x "
                f"current price {float(r['_price']):,.0f}"
            ),
            axis=1,
        )
    return (
        flagged
        .drop(columns=['_price', '_old_price'], errors='ignore')
        .drop_duplicates(subset=['PRODUCT_SET_SID'])
    )


def check_low_rating(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    if 'RATING' not in df.columns:
        return _empty(df)
    d = df.copy()
    d['_rating'] = pd.to_numeric(d['RATING'], errors='coerce')
    flagged = d[d['_rating'].notna() & (d['_rating'] < threshold)].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = (
            "Rating " + flagged['_rating'].round(1).astype(str) +
            " is below " + str(threshold)
        )
    return (
        flagged
        .drop(columns=['_rating'], errors='ignore')
        .drop_duplicates(subset=['PRODUCT_SET_SID'])
    )


def check_no_ratings(df: pd.DataFrame) -> pd.DataFrame:
    if 'RATING' not in df.columns:
        return _empty(df)
    d = df.copy()
    d['_rating'] = pd.to_numeric(d['RATING'], errors='coerce')
    flagged = d[d['_rating'].isna()].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Product has no customer ratings"
    return (
        flagged
        .drop(columns=['_rating'], errors='ignore')
        .drop_duplicates(subset=['PRODUCT_SET_SID'])
    )


def check_single_word_name(df: pd.DataFrame) -> pd.DataFrame:
    if 'NAME' not in df.columns:
        return _empty(df)
    flagged = df[df['NAME'].astype(str).str.split().str.len() == 1].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Product name is a single word"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])


def check_unnecessary_words(df: pd.DataFrame, pattern: re.Pattern) -> pd.DataFrame:
    if pattern is None or 'NAME' not in df.columns:
        return _empty(df)
    mask = df['NAME'].astype(str).str.lower().str.contains(pattern, na=False)
    flagged = df[mask].copy()
    if not flagged.empty:
        def _get_matches(text):
            matches = pattern.findall(str(text))
            return ", ".join(set(m.lower() for m in matches if isinstance(m, str)))
        flagged['Comment_Detail'] = "Unnecessary words: " + flagged['NAME'].apply(_get_matches)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])


# -------------------------------------------------
# CHECK RUNNER
# -------------------------------------------------

def _compile_pattern(words: List[str]):
    if not words:
        return None
    pat = '|'.join(r'\b' + re.escape(w) + r'\b' for w in sorted(words, key=len, reverse=True))
    return re.compile(pat, re.IGNORECASE)


def run_checks(df: pd.DataFrame, support_files: Dict) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Execute every post-QC check against *df* (already normalised).

    Returns
    -------
    summary_df : pd.DataFrame
        One row per unique SKU — flagged rows carry a non-empty 'Flag' value.
    results    : dict  {check_name: flagged_dataframe}
    """
    unnecessary_pattern = _compile_pattern(support_files.get('unnecessary_words', []))

    check_registry = [
        ("Duplicate SKU",             check_duplicate_sku,      {}),
        ("Brand Repeated in Name",    check_brand_in_name,      {}),
        ("Fashion Brand",             check_fashion_brand,      {}),
        ("Fake Discount",             check_fake_discount,      {'multiplier_threshold': 10.0}),
        ("Low Rating (< 3.0)",        check_low_rating,         {'threshold': 3.0}),
        ("No Ratings",                check_no_ratings,         {}),
        ("Single-word Name",          check_single_word_name,   {}),
        ("Unnecessary Words in Name", check_unnecessary_words,  {'pattern': unnecessary_pattern}),
    ]

    results: Dict[str, pd.DataFrame] = {}
    for name, func, kwargs in check_registry:
        try:
            res = func(df, **kwargs)
            results[name] = res if (not res.empty and 'PRODUCT_SET_SID' in res.columns) else _empty(df)
        except Exception as exc:
            logger.error(f"Post-QC check '{name}' failed: {exc}")
            results[name] = _empty(df)

    # Build the summary (one row per unique SKU, first flag wins)
    rows = []
    processed: set = set()
    for name, _, _ in check_registry:
        res = results.get(name, pd.DataFrame())
        if res.empty or 'PRODUCT_SET_SID' not in res.columns:
            continue
        for _, r in res.iterrows():
            sid = str(r['PRODUCT_SET_SID']).strip()
            if sid in processed:
                continue
            processed.add(sid)
            rows.append({
                'SKU':       sid,
                'Name':      r.get('NAME', ''),
                'Brand':     r.get('BRAND', ''),
                'Seller':    r.get('SELLER_NAME', ''),
                'Flag':      name,
                'Comment':   r.get('Comment_Detail', ''),
                'Price':     r.get('GLOBAL_PRICE', ''),
                'Old Price': r.get('OLD_PRICE', ''),
                'Rating':    r.get('RATING', ''),
                'Image URL': r.get('MAIN_IMAGE', ''),
            })

    # Remaining clean rows
    for _, r in df[~df['PRODUCT_SET_SID'].astype(str).str.strip().isin(processed)].iterrows():
        sid = str(r['PRODUCT_SET_SID']).strip()
        if sid not in processed:
            rows.append({
                'SKU':       sid,
                'Name':      r.get('NAME', ''),
                'Brand':     r.get('BRAND', ''),
                'Seller':    r.get('SELLER_NAME', ''),
                'Flag':      '',
                'Comment':   '',
                'Price':     r.get('GLOBAL_PRICE', ''),
                'Old Price': r.get('OLD_PRICE', ''),
                'Rating':    r.get('RATING', ''),
                'Image URL': r.get('MAIN_IMAGE', ''),
            })
            processed.add(sid)

    return pd.DataFrame(rows), results


# -------------------------------------------------
# EXPORT HELPER
# -------------------------------------------------

def build_export(summary: pd.DataFrame, results: Dict[str, pd.DataFrame]) -> bytes:
    """Return raw bytes for an .xlsx workbook with a Summary sheet + one sheet per check."""
    out = BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        summary.to_excel(writer, sheet_name='Summary', index=False)
        wb  = writer.book
        ws  = writer.sheets['Summary']
        red_fmt   = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        green_fmt = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        if 'Flag' in summary.columns:
            flag_col = summary.columns.get_loc('Flag')
            ws.conditional_format(1, flag_col, len(summary), flag_col,
                {'type': 'cell', 'criteria': '!=', 'value': '""', 'format': red_fmt})
            ws.conditional_format(1, flag_col, len(summary), flag_col,
                {'type': 'cell', 'criteria': '==', 'value': '""', 'format': green_fmt})

        for check_name, res in results.items():
            if res.empty or 'PRODUCT_SET_SID' not in res.columns:
                continue
            sheet_name = check_name[:31]   # Excel tab-name limit
            res.to_excel(writer, sheet_name=sheet_name, index=False)

    out.seek(0)
    return out.getvalue()


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

def render_post_qc_section(support_files: Dict) -> None:
    """
    Called from app.py when a post-QC file is detected.
    Reads/writes st.session_state keys:
        post_qc_summary, post_qc_results, post_qc_data
    """
    summary  = st.session_state.post_qc_summary
    results  = st.session_state.post_qc_results
    data_pq  = st.session_state.post_qc_data

    flagged_df = summary[summary['Flag'] != '']
    clean_df   = summary[summary['Flag'] == '']
    flag_rate  = len(flagged_df) / len(summary) * 100 if len(summary) > 0 else 0

    # ── Mode banner ──────────────────────────────────────────────────────
    st.info(
        "📊 **Post-QC Mode** — this file was detected as a live Jumia product "
        "listing. Running post-listing quality checks.",
        icon="ℹ️",
    )

    # ── Metrics ──────────────────────────────────────────────────────────
    st.header(":material/bar_chart: Post-QC Results", anchor=False)
    with st.container(border=True):
        n_cols  = 5 if st.session_state.get('layout_mode') == 'wide' else 3
        p_cols  = st.columns(n_cols)
        metrics = [
            ("Total Products", data_pq['PRODUCT_SET_SID'].nunique(), JUMIA_COLORS['dark_gray']),
            ("Issues Found",   len(flagged_df),                       JUMIA_COLORS['jumia_red']),
            ("Clean",          len(clean_df),                         JUMIA_COLORS['success_green']),
            ("Issue Rate",     f"{flag_rate:.1f}%",                   JUMIA_COLORS['primary_orange']),
            ("Checks Run",     len(results),                          JUMIA_COLORS['medium_gray']),
        ]
        for i, (label, value, color) in enumerate(metrics):
            with p_cols[i % n_cols]:
                st.markdown(
                    f"""<div class="metric-card-inner" style='text-align:center;padding:18px 12px;
background:{JUMIA_COLORS['light_gray']};border-radius:8px;border-left:4px solid {color};'>
<div class="metric-card-value" style='font-size:28px;font-weight:700;color:{color};margin-bottom:4px;'>{value}</div>
<div class="metric-card-label" style='font-size:11px;color:{JUMIA_COLORS['medium_gray']};
text-transform:uppercase;letter-spacing:0.6px;font-weight:600;'>{label}</div>
</div>""",
                    unsafe_allow_html=True,
                )

    # ── Per-check expanders ───────────────────────────────────────────────
    st.subheader(":material/flag: Issues Breakdown", anchor=False)
    any_issues = False

    for check_name in CHECK_ORDER:
        res = results.get(check_name, pd.DataFrame())
        if res.empty or 'PRODUCT_SET_SID' not in res.columns:
            continue
        any_issues = True
        count = res['PRODUCT_SET_SID'].nunique()

        # Choose display columns depending on the check type
        if check_name == "Fake Discount":
            disp_cols = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'GLOBAL_PRICE', 'OLD_PRICE',
                         'DISCOUNT', 'SELLER_NAME', 'Comment_Detail']
        elif check_name in ("Low Rating (< 3.0)", "No Ratings"):
            disp_cols = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'RATING',
                         'TOTAL_RATINGS', 'SELLER_NAME', 'Comment_Detail']
        else:
            disp_cols = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'SELLER_NAME', 'Comment_Detail']

        display_df = res[[c for c in disp_cols if c in res.columns]].copy()
        if 'Comment_Detail' not in display_df.columns:
            display_df['Comment_Detail'] = ''

        with st.expander(f"{check_name} ({count})"):
            s1, s2 = st.columns([1, 1])
            with s1:
                search = st.text_input(
                    "Search", placeholder="Name, Brand...",
                    key=f"pqs_{check_name}"
                )
            with s2:
                seller_opts = (
                    sorted(display_df['SELLER_NAME'].astype(str).unique())
                    if 'SELLER_NAME' in display_df.columns else []
                )
                seller_filter = st.multiselect(
                    "Filter by Seller", seller_opts,
                    key=f"pqf_{check_name}"
                )

            if search:
                display_df = display_df[
                    display_df.apply(
                        lambda x: x.astype(str).str.contains(search, case=False).any(),
                        axis=1,
                    )
                ]
            if seller_filter:
                display_df = display_df[display_df['SELLER_NAME'].isin(seller_filter)]

            col_cfg = {
                "PRODUCT_SET_SID": st.column_config.TextColumn("SKU",       pinned=True),
                "NAME":            st.column_config.TextColumn("Name",      pinned=True),
                "GLOBAL_PRICE":    st.column_config.NumberColumn("Price",     format="₦%.0f"),
                "OLD_PRICE":       st.column_config.NumberColumn("Old Price", format="₦%.0f"),
                "RATING":          st.column_config.NumberColumn("Rating",    format="%.1f"),
                "TOTAL_RATINGS":   st.column_config.NumberColumn("# Ratings", format="%d"),
                "Comment_Detail":  st.column_config.TextColumn("Detail"),
            }
            st.dataframe(
                display_df.reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
                column_config=col_cfg,
            )
            st.caption(f"{len(display_df)} rows shown")

    if not any_issues:
        st.success("No issues found — all post-QC checks passed.")

    # ── Export ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(":material/download: Export Post-QC Report", anchor=False)

    export_key = "post_qc_export"
    if export_key not in st.session_state.get('exports_cache', {}):
        if st.button("Generate Post-QC Report", type="primary", icon=":material/download:"):
            with st.spinner("Building report..."):
                xlsx_bytes = build_export(summary, results)
                if 'exports_cache' not in st.session_state:
                    st.session_state.exports_cache = {}
                st.session_state.exports_cache[export_key] = xlsx_bytes
            st.rerun()
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
        st.download_button(
            "Download Post-QC Report",
            data=st.session_state.exports_cache[export_key],
            file_name=f"PostQC_Report_{date_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            icon=":material/file_download:",
        )
        if st.button("Clear", key="clr_post_qc_export"):
            del st.session_state.exports_cache[export_key]
            st.rerun()
