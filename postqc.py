import re
import sys
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

# Display order — post-QC-only first, then all shared pre-QC checks
CHECK_ORDER = [
    "Duplicate SKU",
    "Fake Discount",
    "Low Rating (< 3.0)",
    "No Ratings",
    "Restricted brands",
    "Suspected Fake product",
    "Seller Not approved to sell Refurb",
    "Product Warranty",
    "Seller Approve to sell books",
    "Seller Approved to Sell Perfume",
    "Perfume Tester",
    "Counterfeit Sneakers",
    "Suspected counterfeit Jerseys",
    "Prohibited products",
    "Unnecessary words in NAME",
    "Single-word NAME",
    "Brand Repeated in Name",
    "Generic BRAND Issues",
    "Fashion Brand",
    "BRAND name repeated in NAME",
    "Wrong Variation",
    "Generic branded products with genuine brands",
    "Missing COLOR",
    "Missing Weight/Volume",
    "Incomplete Smartphone Name",
    "Duplicate product",
    "Wrong Category",
]

POST_QC_ONLY_CHECKS = {"Duplicate SKU", "Fake Discount", "Low Rating (< 3.0)", "No Ratings"}

# -------------------------------------------------
# FILE DETECTION
# -------------------------------------------------

def detect_file_type(df: pd.DataFrame) -> str:
    cols_lower = set(df.columns.str.strip().str.lower())
    if POST_QC_SIGNATURE_COLS.issubset(cols_lower):
        return 'post_qc'
    return 'pre_qc'

# -------------------------------------------------
# CATEGORY MAP LOADER
# -------------------------------------------------

def load_category_map(filename: str = "category_map.xlsx") -> Dict[str, str]:
    import os
    if not os.path.exists(filename):
        csv_path = filename.replace('.xlsx', '.csv')
        if os.path.exists(csv_path):
            filename = csv_path
        else:
            logger.warning(f"load_category_map: file '{filename}' not found")
            return {}
    try:
        df = pd.read_csv(filename, dtype=str) if filename.endswith('.csv') \
             else pd.read_excel(filename, engine='openpyxl', dtype=str)
        df.columns = df.columns.str.strip()

        name_col = next((c for c in df.columns if 'name' in c.lower()), None)
        code_col = next((c for c in df.columns if 'code' in c.lower()), None)
        path_col = next((c for c in df.columns if 'path' in c.lower()), None)

        if not name_col or not code_col:
            logger.warning(f"load_category_map: couldn't find name/code columns in {df.columns.tolist()}")
            return {}

        mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            code = str(row[code_col]).strip()
            if not name or not code or name.lower() == 'nan' or code.lower() == 'nan':
                continue
            code_clean = code.split('.')[0]
            mapping[name.lower()] = code_clean
            if path_col:
                path = str(row[path_col]).strip()
                if path and path.lower() != 'nan':
                    last = path.split('/')[-1].strip().lower()
                    if last and last not in mapping:
                        mapping[last] = code_clean

        logger.info(f"load_category_map: loaded {len(mapping)} entries from '{filename}'")
        return mapping
    except Exception as e:
        logger.warning(f"load_category_map({filename}): {e}")
        return {}


# -------------------------------------------------
# NORMALISATION
# -------------------------------------------------

def normalize_post_qc(df: pd.DataFrame, category_map: Dict[str, str] = None) -> pd.DataFrame:
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

    if 'CATEGORY' in df.columns:
        cmap = category_map or {}

        def resolve_code(raw: str) -> str:
            if not raw or raw == 'nan':
                return ''
            segments = [s.strip() for s in re.split(r'[>/]', raw) if s.strip()]
            for seg in reversed(segments):
                code = cmap.get(seg.lower())
                if code:
                    return code
            last = segments[-1] if segments else raw
            return re.sub(r'[^a-z0-9]', '_', last.lower())

        df['CATEGORY_CODE'] = df['CATEGORY'].astype(str).apply(resolve_code)
        resolved = df['CATEGORY_CODE'].str.match(r'^\d+$').sum()
        logger.info(f"normalize_post_qc: {resolved}/{len(df)} rows resolved to numeric category codes")

    if 'ACTIVE_STATUS_COUNTRY' not in df.columns:
        df['ACTIVE_STATUS_COUNTRY'] = 'UNKNOWN'

    df['_IS_MULTI_COUNTRY'] = False
    df['PARENTSKU']         = df.get('PRODUCT_SET_SID', pd.Series(dtype=str))
    df['COLOR']             = df['COLOR'] if 'COLOR' in df.columns else ''
    df['COLOR_FAMILY']      = ''
    df['GLOBAL_SALE_PRICE'] = ''
    if 'COUNT_VARIATIONS' not in df.columns:
        df['COUNT_VARIATIONS'] = '1'

    return df


# -------------------------------------------------
# POST-QC-ONLY CHECKS
# -------------------------------------------------

def _empty(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=df.columns)

def check_duplicate_sku(df: pd.DataFrame) -> pd.DataFrame:
    if 'PRODUCT_SET_SID' not in df.columns: return _empty(df)
    flagged = df[df.duplicated(subset=['PRODUCT_SET_SID'], keep='first')].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Duplicate SKU in file"
    return flagged

def check_fake_discount(df: pd.DataFrame, multiplier_threshold: float = 10.0) -> pd.DataFrame:
    if not {'GLOBAL_PRICE', 'OLD_PRICE'}.issubset(df.columns): return _empty(df)
    d = df.copy()
    d['_p'] = pd.to_numeric(d['GLOBAL_PRICE'], errors='coerce')
    d['_o'] = pd.to_numeric(d['OLD_PRICE'],    errors='coerce')
    mask = d['_p'].notna() & d['_o'].notna() & (d['_p'] > 0) & (d['_o'] > d['_p'] * multiplier_threshold)
    flagged = d[mask].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = flagged.apply(
            lambda r: f"Old price {float(r['_o']):,.0f} is {float(r['_o'])/float(r['_p']):,.0f}x current price {float(r['_p']):,.0f}", axis=1)
    return flagged.drop(columns=['_p', '_o'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_low_rating(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    if 'RATING' not in df.columns: return _empty(df)
    d = df.copy()
    d['_r'] = pd.to_numeric(d['RATING'], errors='coerce')
    flagged = d[d['_r'].notna() & (d['_r'] < threshold)].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Rating " + flagged['_r'].round(1).astype(str) + " below " + str(threshold)
    return flagged.drop(columns=['_r'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_no_ratings(df: pd.DataFrame) -> pd.DataFrame:
    if 'RATING' not in df.columns: return _empty(df)
    d = df.copy()
    d['_r'] = pd.to_numeric(d['RATING'], errors='coerce')
    flagged = d[d['_r'].isna()].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "No customer ratings"
    return flagged.drop(columns=['_r'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def _compile_pattern(words: List[str]):
    if not words: return None
    pat = '|'.join(r'\b' + re.escape(w) + r'\b' for w in sorted(words, key=len, reverse=True))
    return re.compile(pat, re.IGNORECASE)


# -------------------------------------------------
# SAFE IMPORT FROM STREAMLIT_APP
# Resolves the circular-import problem: streamlit_app imports postqc at
# module load time, so a normal `from streamlit_app import …` inside
# postqc would fail (the module is only half-initialised).
# Instead we look it up from sys.modules AFTER it has fully loaded.
# -------------------------------------------------

def _get_preqc_symbols():
    """
    Return (symbols_dict, True) when streamlit_app is fully loaded,
    or ({}, False) when it is not yet available.
    """
    mod = sys.modules.get('streamlit_app')
    if mod is None:
        return {}, False

    names = [
        'check_restricted_brands', 'check_suspected_fake_products',
        'check_refurb_seller_approval', 'check_product_warranty',
        'check_seller_approved_for_books', 'check_seller_approved_for_perfume',
        'check_perfume_tester', 'check_counterfeit_sneakers',
        'check_counterfeit_jerseys', 'check_prohibited_products',
        'check_unnecessary_words', 'check_single_word_name',
        'check_generic_brand_issues', 'check_fashion_brand_issues',
        'check_brand_in_name', 'check_wrong_variation',
        'check_generic_with_brand_in_name', 'check_missing_color',
        'check_weight_volume_in_name', 'check_incomplete_smartphone_name',
        'check_duplicate_products', 'check_miscellaneous_category',
        'compile_regex_patterns', 'FX_RATE',
    ]

    symbols = {}
    missing = []
    for name in names:
        val = getattr(mod, name, None)
        if val is None:
            missing.append(name)
        else:
            symbols[name] = val

    if missing:
        logger.warning(f"_get_preqc_symbols: missing from streamlit_app: {missing}")
        return symbols, False

    return symbols, True


# -------------------------------------------------
# CHECK RUNNER
# -------------------------------------------------

def run_checks(df: pd.DataFrame, support_files: Dict) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    results: Dict[str, pd.DataFrame] = {}

    # ── POST-QC-ONLY ────────────────────────────────────────────────────────
    for name, func, kwargs in [
        ("Duplicate SKU",      check_duplicate_sku,  {}),
        ("Fake Discount",      check_fake_discount,  {'multiplier_threshold': 10.0}),
        ("Low Rating (< 3.0)", check_low_rating,     {'threshold': 3.0}),
        ("No Ratings",         check_no_ratings,     {}),
    ]:
        try:
            res = func(df, **kwargs)
            results[name] = res if (not res.empty and 'PRODUCT_SET_SID' in res.columns) else _empty(df)
        except Exception as exc:
            logger.error(f"Post-QC check '{name}': {exc}")
            results[name] = _empty(df)

    # ── SHARED PRE-QC CHECKS ────────────────────────────────────────────────
    # Use sys.modules lookup to avoid circular import.
    # streamlit_app imports postqc at startup; by the time run_checks() is
    # called interactively, streamlit_app is fully initialised in sys.modules.
    symbols, _have_preqc = _get_preqc_symbols()

    if not _have_preqc:
        logger.warning("run_checks: pre-QC validators not available — running basic checks only")

    if _have_preqc:
        check_restricted_brands          = symbols['check_restricted_brands']
        check_suspected_fake_products    = symbols['check_suspected_fake_products']
        check_refurb_seller_approval     = symbols['check_refurb_seller_approval']
        check_product_warranty           = symbols['check_product_warranty']
        check_seller_approved_for_books  = symbols['check_seller_approved_for_books']
        check_seller_approved_for_perfume= symbols['check_seller_approved_for_perfume']
        check_perfume_tester             = symbols['check_perfume_tester']
        check_counterfeit_sneakers       = symbols['check_counterfeit_sneakers']
        check_counterfeit_jerseys        = symbols['check_counterfeit_jerseys']
        check_prohibited_products        = symbols['check_prohibited_products']
        check_unnecessary_words          = symbols['check_unnecessary_words']
        check_single_word_name           = symbols['check_single_word_name']
        check_generic_brand_issues       = symbols['check_generic_brand_issues']
        check_fashion_brand_issues       = symbols['check_fashion_brand_issues']
        check_brand_in_name              = symbols['check_brand_in_name']
        check_wrong_variation            = symbols['check_wrong_variation']
        check_generic_with_brand_in_name = symbols['check_generic_with_brand_in_name']
        check_missing_color              = symbols['check_missing_color']
        check_weight_volume_in_name      = symbols['check_weight_volume_in_name']
        check_incomplete_smartphone_name = symbols['check_incomplete_smartphone_name']
        check_duplicate_products         = symbols['check_duplicate_products']
        check_miscellaneous_category     = symbols['check_miscellaneous_category']
        compile_regex_patterns           = symbols['compile_regex_patterns']
        FX_RATE                          = symbols['FX_RATE']

        country_code = support_files.get('country_code', 'KE')
        country_name = support_files.get('country_name', 'Kenya')

        shared_checks = [
            ("Restricted brands",
             check_restricted_brands,
             {'country_rules': support_files.get('restricted_brands_all', {}).get(country_name, [])}),

            ("Suspected Fake product",
             check_suspected_fake_products,
             {'suspected_fake_df': support_files.get('suspected_fake', pd.DataFrame()), 'fx_rate': FX_RATE}),

            ("Seller Not approved to sell Refurb",
             check_refurb_seller_approval,
             {'refurb_data': support_files.get('refurb_data', {}), 'country_code': country_code}),

            ("Product Warranty",
             check_product_warranty,
             {'warranty_category_codes': support_files.get('warranty_category_codes', [])}),

            ("Seller Approve to sell books",
             check_seller_approved_for_books,
             {'books_data': support_files.get('books_data', {}),
              'country_code': country_code,
              'book_category_codes': support_files.get('book_category_codes', [])}),

            ("Seller Approved to Sell Perfume",
             check_seller_approved_for_perfume,
             {'perfume_category_codes': support_files.get('perfume_category_codes', []),
              'perfume_data': support_files.get('perfume_data', {}),
              'country_code': country_code}),

            ("Perfume Tester",
             check_perfume_tester,
             {'perfume_category_codes': support_files.get('perfume_category_codes', []),
              'perfume_data': support_files.get('perfume_data', {})}),

            ("Counterfeit Sneakers",
             check_counterfeit_sneakers,
             {'sneaker_category_codes': support_files.get('sneaker_category_codes', []),
              'sneaker_sensitive_brands': support_files.get('sneaker_sensitive_brands', [])}),

            ("Suspected counterfeit Jerseys",
             check_counterfeit_jerseys,
             {'jerseys_data': support_files.get('jerseys_data', {}), 'country_code': country_code}),

            ("Prohibited products",
             check_prohibited_products,
             {'prohibited_rules': support_files.get('prohibited_words_all', {}).get(country_code, [])}),

            ("Unnecessary words in NAME",
             check_unnecessary_words,
             {'pattern': compile_regex_patterns(support_files.get('unnecessary_words', []))}),

            ("Single-word NAME",
             check_single_word_name,
             {'book_category_codes': support_files.get('book_category_codes', []),
              'books_data': support_files.get('books_data', {})}),

            ("Generic BRAND Issues",
             check_generic_brand_issues,
             {'valid_category_codes_fas': support_files.get('category_fas', [])}),

            ("Fashion Brand",
             check_fashion_brand_issues,
             {'valid_category_codes_fas': support_files.get('category_fas', [])}),

            ("Brand Repeated in Name",
             check_brand_in_name,
             {}),

            ("BRAND name repeated in NAME",
             check_brand_in_name,
             {}),

            ("Wrong Variation",
             check_wrong_variation,
             {'allowed_variation_codes': list(set(
                 support_files.get('variation_allowed_codes', []) +
                 support_files.get('category_fas', [])))}),

            ("Generic branded products with genuine brands",
             check_generic_with_brand_in_name,
             {'brands_list': support_files.get('known_brands', [])}),

            ("Missing COLOR",
             check_missing_color,
             {'pattern': compile_regex_patterns(support_files.get('colors', [])),
              'color_categories': support_files.get('color_categories', []),
              'country_code': country_code}),

            ("Missing Weight/Volume",
             check_weight_volume_in_name,
             {'weight_category_codes': support_files.get('weight_category_codes', [])}),

            ("Incomplete Smartphone Name",
             check_incomplete_smartphone_name,
             {'smartphone_category_codes': support_files.get('smartphone_category_codes', [])}),

            ("Duplicate product",
             check_duplicate_products,
             {'exempt_categories': support_files.get('duplicate_exempt_codes', []),
              'known_colors': support_files.get('colors', [])}),

            ("Wrong Category",
             check_miscellaneous_category,
             {}),
        ]

        for name, func, kwargs in shared_checks:
            # Skip true duplicates (same flag name already processed)
            if name in results:
                continue
            try:
                res = func(df, **kwargs)
                results[name] = res if (not res.empty and 'PRODUCT_SET_SID' in res.columns) else _empty(df)
            except Exception as exc:
                logger.error(f"Post-QC shared check '{name}': {exc}")
                results[name] = _empty(df)

    else:
        # Minimal fallback when pre-QC import fails
        unnecessary_pattern = _compile_pattern(support_files.get('unnecessary_words', []))

        # Brand in name
        try:
            mask = df.apply(lambda r: (
                str(r.get('BRAND', '')).strip().lower() not in ['', 'nan', 'fashion', 'generic']
                and str(r.get('BRAND', '')).strip().lower() in str(r.get('NAME', '')).strip().lower()
            ), axis=1)
            flagged = df[mask].copy()
            if not flagged.empty:
                flagged['Comment_Detail'] = "Brand repeated in name"
            results["Brand Repeated in Name"] = flagged.drop_duplicates(subset=['PRODUCT_SET_SID']) if not flagged.empty else _empty(df)
        except Exception:
            results["Brand Repeated in Name"] = _empty(df)

        # Single word name
        try:
            flagged = df[df['NAME'].astype(str).str.split().str.len() == 1].copy()
            if not flagged.empty: flagged['Comment_Detail'] = "Single word name"
            results["Single-word NAME"] = flagged.drop_duplicates(subset=['PRODUCT_SET_SID']) if not flagged.empty else _empty(df)
        except Exception:
            results["Single-word NAME"] = _empty(df)

        # Unnecessary words
        try:
            if unnecessary_pattern and 'NAME' in df.columns:
                mask = df['NAME'].astype(str).str.lower().str.contains(unnecessary_pattern, na=False)
                flagged = df[mask].copy()
                if not flagged.empty: flagged['Comment_Detail'] = "Unnecessary words in name"
                results["Unnecessary words in NAME"] = flagged.drop_duplicates(subset=['PRODUCT_SET_SID']) if not flagged.empty else _empty(df)
        except Exception:
            results["Unnecessary words in NAME"] = _empty(df)

    # ── BUILD SUMMARY ────────────────────────────────────────────────────────
    rows = []
    processed: set = set()
    all_checks = CHECK_ORDER + [k for k in results if k not in CHECK_ORDER]

    for check_name in all_checks:
        res = results.get(check_name, pd.DataFrame())
        if res.empty or 'PRODUCT_SET_SID' not in res.columns: continue
        for _, r in res.iterrows():
            sid = str(r['PRODUCT_SET_SID']).strip()
            if sid in processed: continue
            processed.add(sid)
            rows.append({
                'SKU': sid, 'Name': r.get('NAME', ''), 'Brand': r.get('BRAND', ''),
                'Category': r.get('CATEGORY', ''), 'Seller': r.get('SELLER_NAME', ''),
                'Flag': check_name, 'Comment': r.get('Comment_Detail', ''),
                'Price': r.get('GLOBAL_PRICE', ''), 'Old Price': r.get('OLD_PRICE', ''),
                'Rating': r.get('RATING', ''), 'Image URL': r.get('MAIN_IMAGE', ''),
            })

    for _, r in df[~df['PRODUCT_SET_SID'].astype(str).str.strip().isin(processed)].iterrows():
        sid = str(r['PRODUCT_SET_SID']).strip()
        if sid not in processed:
            rows.append({
                'SKU': sid, 'Name': r.get('NAME', ''), 'Brand': r.get('BRAND', ''),
                'Category': r.get('CATEGORY', ''), 'Seller': r.get('SELLER_NAME', ''),
                'Flag': '', 'Comment': '',
                'Price': r.get('GLOBAL_PRICE', ''), 'Old Price': r.get('OLD_PRICE', ''),
                'Rating': r.get('RATING', ''), 'Image URL': r.get('MAIN_IMAGE', ''),
            })
            processed.add(sid)

    return pd.DataFrame(rows), results


# -------------------------------------------------
# EXPORT HELPER
# -------------------------------------------------

def build_export(summary: pd.DataFrame, results: Dict[str, pd.DataFrame]) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        summary.to_excel(writer, sheet_name='Summary', index=False)
        wb = writer.book
        ws = writer.sheets['Summary']
        red_fmt   = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        green_fmt = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        if 'Flag' in summary.columns:
            flag_col = summary.columns.get_loc('Flag')
            ws.conditional_format(1, flag_col, len(summary), flag_col,
                {'type': 'cell', 'criteria': '!=', 'value': '""', 'format': red_fmt})
            ws.conditional_format(1, flag_col, len(summary), flag_col,
                {'type': 'cell', 'criteria': '==', 'value': '""', 'format': green_fmt})
        for check_name, res in results.items():
            if res.empty or 'PRODUCT_SET_SID' not in res.columns: continue
            res.to_excel(writer, sheet_name=check_name[:31], index=False)
    out.seek(0)
    return out.getvalue()


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

def render_post_qc_section(support_files: Dict) -> None:
    summary = st.session_state.post_qc_summary
    results = st.session_state.post_qc_results
    data_pq = st.session_state.post_qc_data

    flagged_df = summary[summary['Flag'] != '']
    clean_df   = summary[summary['Flag'] == '']
    flag_rate  = len(flagged_df) / len(summary) * 100 if len(summary) > 0 else 0

    # Category code resolution banner
    total_count = len(data_pq)
    resolved_count = int(data_pq['CATEGORY_CODE'].str.match(r'^\d+$').sum()) \
        if 'CATEGORY_CODE' in data_pq.columns else 0

    st.header(":material/bar_chart: Post-QC Results", anchor=False)

    if total_count > 0:
        pct = resolved_count / total_count * 100
        if pct == 100:
            st.success(f"All {total_count} products matched to real category codes — full validation active.", icon=":material/check_circle:")
        elif pct > 0:
            st.warning(f"{resolved_count}/{total_count} products matched to category codes ({pct:.0f}%). "
                       f"Category-dependent checks may miss unresolved rows.", icon=":material/warning:")
        else:
            st.error("No products matched to category codes. Check that `category_map.xlsx` is in your app root.",
                     icon=":material/error:")

    with st.container(border=True):
        n_cols = 5 if st.session_state.get('layout_mode') == 'wide' else 3
        p_cols = st.columns(n_cols)
        active_checks = len([k for k, v in results.items() if not v.empty and 'PRODUCT_SET_SID' in v.columns])
        metrics = [
            ("Total Products", data_pq['PRODUCT_SET_SID'].nunique(), JUMIA_COLORS['dark_gray']),
            ("Issues Found",   len(flagged_df),                       JUMIA_COLORS['jumia_red']),
            ("Clean",          len(clean_df),                         JUMIA_COLORS['success_green']),
            ("Issue Rate",     f"{flag_rate:.1f}%",                   JUMIA_COLORS['primary_orange']),
            ("Checks Run",     active_checks,                         JUMIA_COLORS['medium_gray']),
        ]
        for i, (label, value, color) in enumerate(metrics):
            with p_cols[i % n_cols]:
                st.markdown(f"<div style='height:4px;background:{color};border-radius:4px 4px 0 0;'></div>",
                            unsafe_allow_html=True)
                st.metric(label=label, value=value)

    st.subheader(":material/flag: Issues Breakdown", anchor=False)
    any_issues = False

    all_checks_ordered = CHECK_ORDER + [k for k in results if k not in CHECK_ORDER]

    for check_name in all_checks_ordered:
        res = results.get(check_name, pd.DataFrame())
        if res.empty or 'PRODUCT_SET_SID' not in res.columns: continue
        any_issues = True
        count = res['PRODUCT_SET_SID'].nunique()
        tag = "" if check_name in POST_QC_ONLY_CHECKS else " 🔍"

        if check_name == "Fake Discount":
            disp_cols = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'GLOBAL_PRICE', 'OLD_PRICE', 'DISCOUNT', 'SELLER_NAME', 'Comment_Detail']
        elif check_name in ("Low Rating (< 3.0)", "No Ratings"):
            disp_cols = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'RATING', 'TOTAL_RATINGS', 'SELLER_NAME', 'Comment_Detail']
        else:
            disp_cols = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'SELLER_NAME', 'Comment_Detail']

        display_df = res[[c for c in disp_cols if c in res.columns]].copy()
        if 'Comment_Detail' not in display_df.columns: display_df['Comment_Detail'] = ''

        with st.expander(f"{check_name}{tag} ({count})"):
            s1, s2 = st.columns([1, 1])
            with s1:
                search = st.text_input("Search", placeholder="Name, Brand, Category...", key=f"pqs_{check_name}")
            with s2:
                seller_opts = sorted(display_df['SELLER_NAME'].astype(str).unique()) if 'SELLER_NAME' in display_df.columns else []
                seller_filter = st.multiselect("Filter by Seller", seller_opts, key=f"pqf_{check_name}")

            if search:
                display_df = display_df[display_df.apply(
                    lambda x: x.astype(str).str.contains(search, case=False).any(), axis=1)]
            if seller_filter:
                display_df = display_df[display_df['SELLER_NAME'].isin(seller_filter)]

            col_cfg = {
                "PRODUCT_SET_SID": st.column_config.TextColumn("SKU", pinned=True),
                "NAME":            st.column_config.TextColumn("Name", pinned=True),
                "CATEGORY":        st.column_config.TextColumn("Category"),
                "GLOBAL_PRICE":    st.column_config.NumberColumn("Price", format="%.2f"),
                "OLD_PRICE":       st.column_config.NumberColumn("Old Price", format="%.2f"),
                "RATING":          st.column_config.NumberColumn("Rating", format="%.1f"),
                "TOTAL_RATINGS":   st.column_config.NumberColumn("# Ratings", format="%d"),
                "Comment_Detail":  st.column_config.TextColumn("Detail"),
            }
            st.dataframe(display_df.reset_index(drop=True), hide_index=True,
                         use_container_width=True, column_config=col_cfg)
            st.caption(f"{len(display_df)} rows shown")

    if not any_issues:
        st.success("No issues found — all post-QC checks passed.")

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
            type="primary", icon=":material/file_download:",
        )
        if st.button("Clear", key="clr_post_qc_export"):
            del st.session_state.exports_cache[export_key]
            st.rerun()
