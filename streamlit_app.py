import math
import hashlib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import st_yled
from io import BytesIO
from datetime import datetime
import re
import logging
from typing import Dict, List, Tuple, Optional, Set
import traceback
import json
import zipfile
import os
import concurrent.futures
from dataclasses import dataclass
import base64
import requests
from PIL import Image

# -------------------------------------------------
# JUMIA THEME COLORS
# -------------------------------------------------
JUMIA_COLORS = {
    'primary_orange': '#F68B1E',
    'secondary_orange': '#FF9933',
    'jumia_red': '#E73C17',
    'dark_gray': '#313133',
    'medium_gray': '#5A5A5C',
    'light_gray': '#F5F5F5',
    'border_gray': '#E0E0E0',
    'success_green': '#4CAF50',
    'warning_yellow': '#FFC107',
    'white': '#FFFFFF',
    'black': '#000000'
}

# -------------------------------------------------
# CONSTANTS & MAPPING
# -------------------------------------------------
PRODUCTSETS_COLS = ["ProductSetSid", "ParentSKU", "Status", "Reason", "Comment", "FLAG", "SellerName"]
REJECTION_REASONS_COLS = ['CODE - REJECTION_REASON', 'COMMENT']

FULL_DATA_COLS = [
    "PRODUCT_SET_SID", "ACTIVE_STATUS_COUNTRY", "NAME", "BRAND", "CATEGORY", "CATEGORY_CODE",
    "COLOR", "COLOR_FAMILY", "MAIN_IMAGE", "VARIATION", "PARENTSKU", "SELLER_NAME", "SELLER_SKU",
    "GLOBAL_PRICE", "GLOBAL_SALE_PRICE", "TAX_CLASS", "FLAG", "LISTING_STATUS",
    "PRODUCT_WARRANTY", "WARRANTY_DURATION", "WARRANTY_ADDRESS", "WARRANTY_TYPE", "COUNT_VARIATIONS",
    "LIST_VARIATIONS"
]

# TIP 4: Slim column set for image grid — only what the grid needs
GRID_COLS = [
    'PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'CATEGORY_CODE',
    'COLOR', 'SELLER_NAME', 'MAIN_IMAGE', 'GLOBAL_PRICE', 'GLOBAL_SALE_PRICE'
]

FX_RATE = 128.0

COUNTRY_CURRENCY = {
    "Kenya":   {"code": "KES", "symbol": "KSh", "pair": "USD/KES"},
    "Uganda":  {"code": "UGX", "symbol": "USh", "pair": "USD/UGX"},
    "Nigeria": {"code": "NGN", "symbol": "₦",   "pair": "USD/NGN"},
    "Ghana":   {"code": "GHS", "symbol": "GH₵", "pair": "USD/GHS"},
    "Morocco": {"code": "MAD", "symbol": "MAD", "pair": "USD/MAD"},
}

@st.cache_data(ttl=3600)
def fetch_exchange_rate(country: str) -> float:
    cfg = COUNTRY_CURRENCY.get(country)
    if not cfg: return 1.0
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen("https://open.er-api.com/v6/latest/USD", timeout=3) as r:
            data = _json.loads(r.read())
        return float(data["rates"].get(cfg["code"], 1.0))
    except Exception:
        return {"Kenya": 128.0, "Uganda": 3750.0, "Nigeria": 1550.0, "Ghana": 15.5, "Morocco": 10.1}.get(country, 1.0)

def format_local_price(usd_price, country: str) -> str:
    try:
        price = float(usd_price)
        if price <= 0: return ""
        cfg = COUNTRY_CURRENCY.get(country, {})
        local = price * fetch_exchange_rate(country)
        symbol = cfg.get("symbol", "$")
        return f"{symbol} {local:,.0f}" if cfg.get("code") in ("KES", "UGX", "NGN") else f"{symbol} {local:,.2f}"
    except (ValueError, TypeError): return ""

SPLIT_LIMIT = 9998

NEW_FILE_MAPPING = {
    'cod_productset_sid': 'PRODUCT_SET_SID',
    "2qz3wx4ec5rv6b7hnj8kl;'[]": 'PRODUCT_SET_SID',
    'dsc_name': 'NAME', 'dsc_brand_name': 'BRAND',
    'cod_category_code': 'CATEGORY_CODE', 'dsc_category_name': 'CATEGORY',
    'dsc_shop_seller_name': 'SELLER_NAME', 'dsc_shop_active_country': 'ACTIVE_STATUS_COUNTRY',
    'cod_parent_sku': 'PARENTSKU', 'color': 'COLOR', 'colour': 'COLOR',
    'color_family': 'COLOR_FAMILY', 'colour_family': 'COLOR_FAMILY',
    'colour family': 'COLOR_FAMILY', 'color family': 'COLOR_FAMILY',
    'COLOUR FAMILY': 'COLOR_FAMILY', 'list_seller_skus': 'SELLER_SKU',
    'image1': 'MAIN_IMAGE', 'dsc_status': 'LISTING_STATUS', 'dsc_shop_email': 'SELLER_EMAIL',
    'product_warranty': 'PRODUCT_WARRANTY', 'warranty_duration': 'WARRANTY_DURATION',
    'warranty_address': 'WARRANTY_ADDRESS', 'warranty_type': 'WARRANTY_TYPE',
    'count_variations': 'COUNT_VARIATIONS', 'count variations': 'COUNT_VARIATIONS',
    'number of variations': 'COUNT_VARIATIONS', 'list_variations': 'LIST_VARIATIONS',
    'list variations': 'LIST_VARIATIONS'
}

logger = logging.getLogger(__name__)
from postqc import detect_file_type, normalize_post_qc, run_checks as run_post_qc_checks, render_post_qc_section

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
defaults = {
    'layout_mode': "wide", 'final_report': pd.DataFrame(), 'all_data_map': pd.DataFrame(),
    'grid_data': pd.DataFrame(), 'post_qc_summary': pd.DataFrame(),
    'post_qc_results': {}, 'post_qc_data': pd.DataFrame(), 'file_mode': None,
    'intersection_sids': set(), 'intersection_count': 0, 'grid_page': 0,
    'grid_items_per_page': 50, 'main_toasts': [], 'exports_cache': {},
    'display_df_cache': {}, 'do_scroll_top': False,
    'bg_executor': concurrent.futures.ThreadPoolExecutor(max_workers=3),
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

try: st.set_page_config(page_title="Product Tool", layout=st.session_state.layout_mode)
except: pass
st_yled.init()

# --- GLOBAL CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
    :root {{ --jumia-orange: {JUMIA_COLORS['primary_orange']}; --jumia-red: {JUMIA_COLORS['jumia_red']}; }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    .stButton > button {{ border-radius: 4px; font-weight: 600; transition: all 0.3s ease; }}
    .stButton > button[kind="primary"] {{ background-color: {JUMIA_COLORS['primary_orange']} !important; border: none !important; color: white !important; }}
    .stButton > button[kind="primary"]:hover {{ background-color: {JUMIA_COLORS['secondary_orange']} !important; box-shadow: 0 4px 8px rgba(246,139,30,0.3); transform: translateY(-1px); }}
    .stButton > button[kind="secondary"] {{ background-color: white !important; border: 2px solid {JUMIA_COLORS['primary_orange']} !important; color: {JUMIA_COLORS['primary_orange']} !important; }}
    ::-webkit-scrollbar {{ width: 18px !important; height: 18px !important; }}
    ::-webkit-scrollbar-track {{ background: {JUMIA_COLORS['light_gray']}; border-radius: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {JUMIA_COLORS['medium_gray']}; border-radius: 8px; border: 3px solid {JUMIA_COLORS['light_gray']}; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {JUMIA_COLORS['primary_orange']}; }}
    div[data-testid="stExpander"] {{ border: 1px solid {JUMIA_COLORS['border_gray']}; border-radius: 8px; }}
    div[data-testid="stExpander"] summary {{ background-color: {JUMIA_COLORS['light_gray']}; padding: 12px; border-radius: 8px 8px 0 0; }}
    h1, h2, h3 {{ color: {JUMIA_COLORS['dark_gray']} !important; }}
    div[data-baseweb="segmented-control"] button[aria-pressed="true"] {{ background-color: {JUMIA_COLORS['primary_orange']} !important; color: white !important; }}
    
    /* Hide bridge inputs */
    div[data-testid="stTextInput"]:has(input[placeholder="__CARD_ACT__"]) {{
        position: absolute !important; opacity: 0 !important; pointer-events: none !important;
        height: 0 !important; overflow: hidden !important; width: 1px !important;
    }}
    
    @media (prefers-color-scheme: dark) {{
        h1, h2, h3 {{ color: #F5F5F5 !important; }}
        .metric-card-inner {{ background: #2a2a2e !important; }}
        ::-webkit-scrollbar-track {{ background: #1e1e1e; }}
        ::-webkit-scrollbar-thumb {{ background: #555; border-color: #1e1e1e; }}
    }}
</style>
""", unsafe_allow_html=True)

def get_default_country():
    try:
        lang = st.context.headers.get("Accept-Language", "")
        for code, name in [("KE","Kenya"),("UG","Uganda"),("NG","Nigeria"),("GH","Ghana"),("MA","Morocco")]:
            if code in lang: return name
    except: pass
    return "Kenya"

if 'selected_country' not in st.session_state: st.session_state.selected_country = get_default_country()

if st.session_state.main_toasts:
    for msg in st.session_state.main_toasts:
        if isinstance(msg, tuple): st.toast(msg[0], icon=msg[1])
        else: st.toast(msg)
    st.session_state.main_toasts.clear()

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
def clean_category_code(code) -> str:
    try:
        if pd.isna(code): return ""
        s = str(code).strip()
        return s.split('.')[0] if '.' in s else s
    except: return str(code).strip()

def normalize_text(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).lower().strip()
    text = re.sub(r'\b(new|sale|original|genuine|authentic|official|premium|quality|best|hot|2024|2025)\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', '', text)

def create_match_key(row: pd.Series) -> str:
    return f"{normalize_text(row.get('BRAND',''))}|{normalize_text(row.get('NAME',''))}|{normalize_text(row.get('COLOR',''))}"

def df_hash(df: pd.DataFrame) -> str:
    try: return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    except: return hashlib.md5(str(df.shape).encode()).hexdigest()

COLOR_PATTERNS = {
    'red': ['red','crimson','scarlet','maroon','burgundy','wine','ruby'],
    'blue': ['blue','navy','royal','sky','azure','cobalt','sapphire'],
    'green': ['green','lime','olive','emerald','mint','forest','jade'],
    'black': ['black','onyx','ebony','jet','charcoal','midnight'],
    'white': ['white','ivory','cream','pearl','snow','alabaster'],
    'gray': ['gray','grey','silver','slate','ash','graphite'],
    'yellow': ['yellow','gold','golden','amber','lemon','mustard'],
    'orange': ['orange','tangerine','peach','coral','apricot'],
    'pink': ['pink','rose','magenta','fuchsia','salmon','blush'],
    'purple': ['purple','violet','lavender','plum','mauve','lilac'],
    'brown': ['brown','tan','beige','khaki','chocolate','coffee','bronze'],
    'multicolor': ['multicolor','multicolour','multi-color','rainbow','mixed']
}
COLOR_VARIANT_TO_BASE = {v: k for k, variants in COLOR_PATTERNS.items() for v in variants}

@dataclass
class ProductAttributes:
    base_name: str; colors: Set[str]; sizes: Set[str]; storage: Set[str]
    memory: Set[str]; quantities: Set[str]; raw_name: str
    def get_base_key(self) -> str: return self.base_name.lower()

def extract_colors(text: str, explicit_color: Optional[str] = None) -> Set[str]:
    colors = set()
    text_lower = str(text).lower() if text else ""
    if explicit_color and pd.notna(explicit_color):
        for variant, base in COLOR_VARIANT_TO_BASE.items():
            if variant in str(explicit_color).lower(): colors.add(base)
    for variant, base in COLOR_VARIANT_TO_BASE.items():
        if re.search(r'\b' + re.escape(variant) + r'\b', text_lower): colors.add(base)
    return colors

def extract_sizes(text: str) -> Set[str]:
    sizes = set()
    text_lower = str(text).lower() if text else ""
    for pattern, size in {r'\bxxs\b|2xs':'xxs', r'\bxs\b|xsmall':'xs', r'\bs\b|small':'small',
                          r'\bm\b|medium':'medium', r'\bl\b|large':'large', r'\bxl\b|xlarge':'xl',
                          r'\bxxl\b|2xl':'xxl', r'\bxxxl\b|3xl':'xxxl'}.items():
        if re.search(pattern, text_lower): sizes.add(size)
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s*(?:inch|inches|")\b', text_lower): sizes.add(f"{m.group(1)}inch")
    return sizes

def extract_storage(text: str) -> Set[str]:
    return {f"{m.group(1)}{'tb' if 'tb' in m.group(0) else 'gb'}"
            for m in re.finditer(r'\b(\d+)\s*(?:gb|tb)\b', str(text).lower() if text else "")}

def extract_memory(text: str) -> Set[str]:
    return {f"{m.group(1)}gb"
            for m in re.finditer(r'\b(\d+)\s*(?:gb|mb)\s*(?:ram|memory|ddr)\b', str(text).lower() if text else "")
            if 2 <= int(m.group(1)) <= 128}

def extract_quantities(text: str) -> Set[str]:
    quantities = set()
    for pat in [r'\b(\d+)[- ]?pack\b', r'\bpack\s+of\s+(\d+)\b', r'\b(\d+)[- ]?(?:pieces?|pcs?)\b']:
        for m in re.finditer(pat, str(text).lower() if text else ""): quantities.add(f"{m.group(1)}pack")
    return quantities

def remove_attributes(text: str) -> str:
    base = str(text).lower() if text else ""
    for variant in COLOR_VARIANT_TO_BASE: base = re.sub(r'\b' + re.escape(variant) + r'\b', '', base)
    base = re.sub(r'\b(?:xxs|xs|small|medium|large|xl|xxl|xxxl)\b', '', base)
    base = re.sub(r'\b\d+\s*(?:gb|tb|inch|inches|"|ram|memory|ddr|pack|piece|pcs)\b', '', base)
    for w in ['new','original','genuine','authentic','official','premium','quality','best','hot','sale','promo','deal']:
        base = re.sub(r'\b' + w + r'\b', '', base)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', base)).strip()

def extract_product_attributes(name: str, explicit_color=None, brand=None) -> ProductAttributes:
    name_str = str(name).strip() if pd.notna(name) else ""
    attrs = ProductAttributes(base_name="", colors=extract_colors(name_str, explicit_color),
                               sizes=extract_sizes(name_str), storage=extract_storage(name_str),
                               memory=extract_memory(name_str), quantities=extract_quantities(name_str),
                               raw_name=name_str)
    base_name = remove_attributes(name_str)
    if brand and pd.notna(brand):
        bl = str(brand).lower().strip()
        if bl not in base_name and bl not in ['generic', 'fashion']: base_name = f"{bl} {base_name}"
    attrs.base_name = base_name.strip()
    return attrs

# -------------------------------------------------
# FILE LOADING
# -------------------------------------------------
def load_txt_file(filename: str) -> List[str]:
    try:
        if not os.path.exists(os.path.abspath(filename)): return []
        with open(filename, 'r', encoding='utf-8') as f: return [l.strip() for l in f if l.strip()]
    except: return []

@st.cache_data(ttl=3600)
def load_excel_file(filename: str, column: Optional[str] = None):
    try:
        if not os.path.exists(filename): return [] if column else pd.DataFrame()
        df = pd.read_excel(filename, engine='openpyxl', dtype=str)
        df.columns = df.columns.str.strip()
        if column and column in df.columns: return df[column].apply(clean_category_code).tolist()
        return df
    except: return [] if column else pd.DataFrame()

def safe_excel_read(filename: str, sheet_name, usecols=None) -> pd.DataFrame:
    if not os.path.exists(filename): return pd.DataFrame()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols=usecols, engine='openpyxl', dtype=str)
        return df.dropna(how='all')
    except Exception as e:
        logger.error(f"Error reading '{sheet_name}' from {filename}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_prohibited_from_local() -> Dict[str, List[Dict]]:
    FILE_NAME = "Prohibbited.xlsx"
    result = {}
    for tab in ["KE","UG","NG","GH","MA"]:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if df.empty: result[tab] = []; continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            kw_col = next((c for c in df.columns if 'keyword' in c or 'prohibited' in c or 'name' in c), df.columns[0])
            cat_col = next((c for c in df.columns if 'cat' in c), None)
            rules = []
            for _, row in df.iterrows():
                kw = str(row.get(kw_col,'')).strip().lower()
                if not kw or kw in ('nan','keywords'): continue
                cats = set()
                if cat_col:
                    cats_raw = str(row.get(cat_col,'')).strip()
                    if cats_raw and cats_raw.lower() != 'nan':
                        cats.update([clean_category_code(c.strip()) for c in re.split(r'[,\n]+', cats_raw) if c.strip()])
                rules.append({'keyword': kw, 'categories': cats})
            result[tab] = rules
        except Exception as e:
            logger.error(f"Failed to load {FILE_NAME} tab {tab}: {e}")
            result[tab] = []
    return result

@st.cache_data(ttl=3600)
def load_restricted_brands_from_local() -> Dict[str, List[Dict]]:
    FILE_NAME = "Restricted_Brands.xlsx"
    result = {}
    for country_name, tab_name in {"Kenya":"KE","Uganda":"UG","Nigeria":"NG","Ghana":"GH","Morocco":"MA"}.items():
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab_name)
            if df.empty: result[country_name] = []; continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            brand_dict = {}
            for _, row in df.iterrows():
                brand = str(row.get('brand','')).strip()
                if not brand or brand.lower() == 'nan': continue
                bl = brand.lower()
                if bl not in brand_dict:
                    brand_dict[bl] = {'brand_raw': brand, 'sellers': set(), 'categories': set(), 'variations': set(), 'has_blank_category': False}
                sellers_raw = str(row.get('approved sellers','')).strip().lower()
                if sellers_raw not in ('nan',''):
                    brand_dict[bl]['sellers'].update([s.strip() for s in sellers_raw.split(',') if s.strip()])
                cats_raw = str(row.get('categories','')).strip()
                if cats_raw in ('nan',''):
                    brand_dict[bl]['has_blank_category'] = True
                else:
                    brand_dict[bl]['categories'].update([clean_category_code(c.strip()) for c in cats_raw.split(',') if c.strip()])
                vars_raw = str(row.get('variations','')).strip().lower()
                if vars_raw not in ('nan',''):
                    brand_dict[bl]['variations'].update([v.strip() for v in vars_raw.split(',') if v.strip()])
            rules = []
            for bl, d in brand_dict.items():
                if d['has_blank_category']: d['categories'] = set()
                rules.append({'brand': bl, 'brand_raw': d['brand_raw'], 'sellers': d['sellers'],
                              'categories': d['categories'], 'variations': list(d['variations'])})
            result[country_name] = rules
        except Exception as e:
            logger.error(f"Failed to load {FILE_NAME} tab {tab_name}: {e}")
            result[country_name] = []
    return result

@st.cache_data(ttl=3600)
def load_refurb_data_from_local() -> dict:
    FILE_NAME = "Refurb.xlsx"
    result = {"sellers": {}, "categories": {"Phones": set(), "Laptops": set()}, "keywords": set()}
    for tab in ["KE","UG","NG","GH","MA"]:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab, usecols=[0,1])
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                result["sellers"][tab] = {
                    "Phones": set(df.iloc[:,0].dropna().astype(str).str.strip().str.lower()) - {"","nan","phones"},
                    "Laptops": set(df.iloc[:,1].dropna().astype(str).str.strip().str.lower()) - {"","nan","laptops"}
                }
        except Exception as e:
            logger.error(f"Refurb sellers tab {tab}: {e}")
            result["sellers"][tab] = {"Phones": set(), "Laptops": set()}
    for sheet in ["Categories", "Categries"]:
        try:
            df_c = safe_excel_read(FILE_NAME, sheet_name=sheet, usecols=[0,1])
            if not df_c.empty:
                df_c.columns = [str(c).strip() for c in df_c.columns]
                result["categories"]["Phones"] = {clean_category_code(c) for c in df_c.iloc[:,0].dropna().astype(str) if c.strip().lower() not in ("phones","phone","nan")}
                result["categories"]["Laptops"] = {clean_category_code(c) for c in df_c.iloc[:,1].dropna().astype(str) if c.strip().lower() not in ("laptops","laptop","nan")}
                break
        except: pass
    try:
        df_n = safe_excel_read(FILE_NAME, sheet_name="Name", usecols=[0])
        if not df_n.empty:
            result["keywords"] = {k for k in df_n.iloc[:,0].dropna().astype(str).str.strip().str.lower()
                                   if k and k not in ("name","keyword","keywords","words","nan")}
    except: result["keywords"] = {"refurb","refurbished","renewed"}
    return result

@st.cache_data(ttl=3600)
def load_perfume_data_from_local() -> Dict:
    FILE_NAME = "Perfume.xlsx"
    result = {"sellers": {}, "keywords": set(), "category_codes": set()}
    for tab in ["KE","UG","NG","GH","MA"]:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                sc = next((c for c in df.columns if 'seller' in c.lower()), df.columns[0])
                result["sellers"][tab] = set(df[sc].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["","nan","sellername","seller name","seller"])]))
        except: result["sellers"][tab] = set()
    try:
        df_kw = safe_excel_read(FILE_NAME, sheet_name="Keywords")
        if not df_kw.empty:
            df_kw.columns = [str(c).strip() for c in df_kw.columns]
            kc = next((c for c in df_kw.columns if 'brand' in c.lower() or 'keyword' in c.lower()), df_kw.columns[0])
            result["keywords"] = set(df_kw[kc].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["","nan","brand","keyword","keywords"])]))
    except: pass
    try:
        df_c = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_c.empty:
            df_c.columns = [str(c).strip() for c in df_c.columns]
            cc = next((c for c in df_c.columns if 'cat' in c.lower()), df_c.columns[0])
            result["category_codes"] = set(df_c[cc].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["","nan","categories","category"])]))
    except: pass
    return result

@st.cache_data(ttl=3600)
def load_books_data_from_local() -> Dict:
    FILE_NAME = "Books_sellers.xlsx"
    result = {"sellers": {}, "category_codes": set()}
    for tab in ["KE","UG","NG","GH","MA"]:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                sc = next((c for c in df.columns if 'seller' in c.lower()), df.columns[0])
                result["sellers"][tab] = set(df[sc].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["","nan","sellername","seller name","seller"])]))
        except: result["sellers"][tab] = set()
    try:
        df_c = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_c.empty:
            df_c.columns = [str(c).strip() for c in df_c.columns]
            cc = next((c for c in df_c.columns if 'cat' in c.lower()), df_c.columns[0])
            result["category_codes"] = set(df_c[cc].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["","nan","categories","category"])]))
    except: pass
    return result

@st.cache_data(ttl=3600)
def load_jerseys_from_local() -> Dict:
    FILE_NAME = "Jersey_validation.xlsx"
    result: Dict = {"keywords": {t: set() for t in ["KE","UG","NG","GH","MA"]},
                    "exempted": {t: set() for t in ["KE","UG","NG","GH","MA"]}, "categories": set()}
    for tab in ["KE","UG","NG","GH","MA"]:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                kc = next((c for c in df.columns if "keyword" in c.lower()), df.columns[0])
                result["keywords"][tab] = set(df[kc].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["","nan","keywords","keyword"])]))
                ec = next((c for c in df.columns if "exempt" in c.lower() or "seller" in c.lower()), None)
                if ec: result["exempted"][tab] = set(df[ec].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["","nan","exempted sellers","seller"])]))
        except Exception as e: logger.error(f"Jerseys {tab}: {e}")
    try:
        df_c = safe_excel_read(FILE_NAME, sheet_name="categories")
        if not df_c.empty:
            df_c.columns = [str(c).strip().lower() for c in df_c.columns]
            cc = next((c for c in df_c.columns if "cat" in c), df_c.columns[0])
            result["categories"] = set(df_c[cc].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["","nan","categories","category"])]))
    except Exception as e: logger.error(f"Jerseys categories: {e}")
    return result

@st.cache_data(ttl=3600)
def load_suspected_fake_from_local() -> pd.DataFrame:
    try:
        if os.path.exists('suspected_fake.xlsx'): return pd.read_excel('suspected_fake.xlsx', sheet_name=0, engine='openpyxl', dtype=str)
    except Exception as e: logger.warning(f"suspected_fake: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_flags_mapping(filename="reason.xlsx") -> Dict[str, Tuple[str, str]]:
    default_mapping = {
        'Restricted brands': ('1000024 - Product does not have a license to be sold via Jumia (Not Authorized)', "Missing license for this item. Raise a claim via Vendor Center."),
        'Suspected Fake product': ('1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)', "Product confirmed counterfeit."),
        'Seller Not approved to sell Refurb': ('1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim', "Contact Seller Support for Refurbished approval."),
        'Product Warranty': ('1000013 - Kindly Provide Product Warranty Details', "Valid warranty required in Description/Warranty tabs."),
        'Seller Approve to sell books': ('1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim', "Contact Seller Support for Book category approval."),
        'Seller Approved to Sell Perfume': ('1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim', "Contact Seller Support for Perfume approval."),
        'Counterfeit Sneakers': ('1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)', "Sneaker confirmed counterfeit."),
        'Suspected counterfeit Jerseys': ('1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)', "Jersey confirmed counterfeit."),
        'Prohibited products': ('1000007 - Other Reason', "Listing of this product is prohibited."),
        'Unnecessary words in NAME': ('1000008 - Kindly Improve Product Name Description', "Avoid unnecessary words in title."),
        'Single-word NAME': ('1000008 - Kindly Improve Product Name Description', "Update product title format: Name – Type – Color."),
        'Generic BRAND Issues': ('1000007 - Other Reason', "Use correct brand instead of Generic/Fashion."),
        'Fashion brand issues': ('1000007 - Other Reason', "Use correct brand instead of Fashion."),
        'BRAND name repeated in NAME': ('1000007 - Other Reason', "Brand name should not be repeated in product name."),
        'Generic branded products with genuine brands': ('1000007 - Other Reason', "Use the displayed brand on the product instead of Generic."),
        'Missing COLOR': ('1000005 - Kindly confirm the actual product colour', "Product color must be mentioned in title/color tab."),
        'Duplicate product': ('1000007 - Other Reason', "This product is a duplicate."),
        'Wrong Variation': ('1000039 - Product Poorly Created. Each Variation Of This Product Should Be Created Uniquely (Not Authorized) (Not Authorized)', "Create different SKUs instead of variations."),
        'Missing Weight/Volume': ('1000008 - Kindly Improve Product Name Description', "Include weight or volume (e.g., '1kg', '500ml')."),
        'Incomplete Smartphone Name': ('1000008 - Kindly Improve Product Name Description', "Include memory/storage details (e.g., '128GB')."),
        'Wrong Category': ('1000004 - Wrong Category', "Assigned to Wrong Category. Please use correct category."),
        'Poor images': ('1000042 - Kindly follow our product image upload guideline.', "Poor Image Quality")
    }
    try:
        if os.path.exists(filename):
            df = pd.read_excel(filename, engine='openpyxl', dtype=str)
            df.columns = df.columns.str.strip().str.lower()
            if 'flag' in df.columns and 'reason' in df.columns and 'comment' in df.columns:
                custom = {str(r['flag']).strip(): (str(r['reason']).strip(), str(r['comment']).strip())
                          for _, r in df.iterrows() if str(r.get('flag','')).strip().lower() not in ('nan','')}
                if custom: return custom
    except Exception as e: logger.error(f"load_flags_mapping: {e}")
    return default_mapping

# TIP 2: Cache reasons_df separately so it doesn't reload on every export render
@st.cache_data(ttl=3600)
def get_reasons_df() -> pd.DataFrame:
    return load_excel_file('reasons.xlsx')

@st.cache_data(ttl=3600)
def load_all_support_files() -> Dict:
    def safe_load_txt(f): return load_txt_file(f) if os.path.exists(f) else []
    fashion_paths = []
    try:
        fash_file = 'fashion brands.xlsx'
        if os.path.exists(fash_file):
            df_fash = pd.read_excel(fash_file, dtype=str)
            df_fash.columns = df_fash.columns.astype(str).str.strip().str.lower()
            path_col = next((col for col in df_fash.columns if 'path' in col), None)
            if path_col: fashion_paths = df_fash[path_col].dropna().astype(str).tolist()
    except Exception as e: logger.error(f"Fashion brands: {e}")
    return {
        'postqc_fashion_cats': fashion_paths,
        'blacklisted_words': safe_load_txt('blacklisted.txt'),
        'book_category_codes': safe_load_txt('Books_cat.txt'),
        'books_data': load_books_data_from_local(),
        'perfume_category_codes': safe_load_txt('Perfume_cat.txt'),
        'perfume_data': load_perfume_data_from_local(),
        'sneaker_category_codes': safe_load_txt('Sneakers_Cat.txt'),
        'sneaker_sensitive_brands': [b.lower() for b in safe_load_txt('Sneakers_Sensitive.txt')],
        'sensitive_words': [w.lower() for w in safe_load_txt('sensitive_words.txt')],
        'unnecessary_words': [w.lower() for w in safe_load_txt('unnecessary.txt')],
        'colors': [c.lower() for c in safe_load_txt('colors.txt')],
        'color_categories': safe_load_txt('color_cats.txt'),
        'category_fas': safe_load_txt('Fashion_cat.txt'),
        'flags_mapping': load_flags_mapping(),
        'jerseys_data': load_jerseys_from_local(),
        'warranty_category_codes': safe_load_txt('warranty.txt'),
        'suspected_fake': load_suspected_fake_from_local(),
        'duplicate_exempt_codes': safe_load_txt('duplicate_exempt.txt'),
        'restricted_brands_all': load_restricted_brands_from_local(),
        'prohibited_words_all': load_prohibited_from_local(),
        'known_brands': safe_load_txt('brands.txt'),
        'variation_allowed_codes': safe_load_txt('variation.txt'),
        'weight_category_codes': safe_load_txt('weight.txt'),
        'smartphone_category_codes': safe_load_txt('smartphones.txt'),
        'refurb_data': load_refurb_data_from_local(),
    }

@st.cache_data(ttl=3600)
def load_support_files_lazy(): return load_all_support_files()

@st.cache_data(ttl=3600)
def compile_regex_patterns(words: List[str]) -> re.Pattern:
    if not words: return None
    pattern = '|'.join(r'\b' + re.escape(w) + r'\b' for w in sorted(words, key=len, reverse=True))
    return re.compile(pattern, re.IGNORECASE)

class CountryValidator:
    COUNTRY_CONFIG = {
        "Kenya": {"code": "KE", "skip_validations": []},
        "Uganda": {"code": "UG", "skip_validations": ["Counterfeit Sneakers", "Product Warranty", "Generic BRAND Issues"]},
        "Nigeria": {"code": "NG", "skip_validations": []},
        "Ghana": {"code": "GH", "skip_validations": []},
        "Morocco": {"code": "MA", "skip_validations": []}
    }
    def __init__(self, country: str):
        self.country = country
        self.config = self.COUNTRY_CONFIG.get(country, self.COUNTRY_CONFIG["Kenya"])
        self.code = self.config["code"]
        self.skip_validations = self.config["skip_validations"]
    def should_skip_validation(self, name: str) -> bool: return name in self.skip_validations
    def ensure_status_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and 'Status' not in df.columns: df['Status'] = 'Approved'
        return df

# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------
def standardize_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    map_lower = {k.lower(): v for k, v in NEW_FILE_MAPPING.items()}
    renamed = {col: map_lower.get(col.lower(), col.upper()) for col in df.columns}
    df = df.rename(columns=renamed)
    for col in ['ACTIVE_STATUS_COUNTRY','CATEGORY_CODE','BRAND','TAX_CLASS','NAME','SELLER_NAME']:
        if col in df.columns: df[col] = df[col].astype(str)
    return df

def validate_input_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = [f"Missing: {f}" for f in ['PRODUCT_SET_SID','NAME','BRAND','CATEGORY_CODE','ACTIVE_STATUS_COUNTRY'] if f not in df.columns]
    return len(errors) == 0, errors

MULTI_COUNTRY_VALUES = {'MULTIPLE', 'MULTI'}

def filter_by_country(df: pd.DataFrame, cv: CountryValidator) -> Tuple[pd.DataFrame, List[str]]:
    if 'ACTIVE_STATUS_COUNTRY' not in df.columns: return df, []
    s = df['ACTIVE_STATUS_COUNTRY'].astype(str).str.strip().str.upper().str.replace(r'^JUMIA-', '', regex=True)
    df['ACTIVE_STATUS_COUNTRY'] = s
    if cv.code == 'NG':
        is_ng = df['ACTIVE_STATUS_COUNTRY'] == 'NG'
        is_multi = df['ACTIVE_STATUS_COUNTRY'].isin(MULTI_COUNTRY_VALUES)
        filtered = df[is_ng | is_multi].copy()
        filtered['_IS_MULTI_COUNTRY'] = is_multi[filtered.index]
    else:
        filtered = df[df['ACTIVE_STATUS_COUNTRY'] == cv.code].copy()
        filtered['_IS_MULTI_COUNTRY'] = False
    detected_names = []
    if filtered.empty:
        codes = [c for c in df['ACTIVE_STATUS_COUNTRY'].unique() if str(c).strip().lower() != 'nan']
        em = {"KE":"Kenya","UG":"Uganda","NG":"Nigeria","GH":"Ghana","MA":"Morocco"}
        detected_names = [em.get(c, f"'{c}'") for c in codes]
    return filtered, detected_names

def propagate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for col in ['COLOR_FAMILY','PRODUCT_WARRANTY','WARRANTY_DURATION','WARRANTY_ADDRESS','WARRANTY_TYPE','COUNT_VARIATIONS','LIST_VARIATIONS']:
        if col not in df.columns: df[col] = pd.NA
        df[col] = df.groupby('PRODUCT_SET_SID')[col].transform(lambda x: x.ffill().bfill())
    return df

# -------------------------------------------------
# VALIDATION CHECKS
# -------------------------------------------------
def check_miscellaneous_category(data: pd.DataFrame) -> pd.DataFrame:
    if 'CATEGORY' not in data.columns: return pd.DataFrame(columns=data.columns)
    flagged = data[data['CATEGORY'].astype(str).str.contains("miscellaneous", case=False, na=False)].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Category contains 'Miscellaneous'"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_restricted_brands(data: pd.DataFrame, country_rules: List[Dict]) -> pd.DataFrame:
    if not {'NAME','BRAND','SELLER_NAME','CATEGORY_CODE'}.issubset(data.columns) or not country_rules: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d['_nl'] = d['NAME'].astype(str).str.lower().fillna('')
    d['_bl'] = d['BRAND'].astype(str).str.lower().str.strip().fillna('')
    d['_sl'] = d['SELLER_NAME'].astype(str).str.lower().str.strip().fillna('')
    d['_cc'] = d['CATEGORY_CODE'].apply(clean_category_code)
    flagged_idx = set(); cm = {}; md = {}
    for rule in country_rules:
        bn = rule['brand']; br = rule['brand_raw']
        bp = r'(?<!\w)' + re.escape(bn) + r'(?!\w)'
        mb = (d['_bl'] == bn); mn = d['_nl'].str.contains(bp, regex=True, na=False)
        mm = mb | mn
        for i in d[mb].index: md[i] = ('main_brand', br)
        for i in d[mn & ~mb].index: md[i] = ('main_name', br)
        if rule['variations']:
            sv = sorted(rule['variations'], key=len, reverse=True)
            vp = r'(?<!\w)(' + '|'.join([re.escape(v) for v in sv]) + r')(?!\w)'
            vbm = d['_bl'].str.contains(vp, regex=True, na=False)
            vnm = d['_nl'].str.contains(vp, regex=True, na=False)
            for i in d[vbm | vnm].index:
                if i not in md:
                    txt = d.loc[i,'_bl'] if vbm[i] else d.loc[i,'_nl']
                    for v in sv:
                        if v in txt: md[i] = ('variation', f"{br} (as '{v}')"); break
            mm = mm | vbm | vnm
        if not mm.any(): continue
        cm2 = d[mm]
        if rule['categories']: cm2 = cm2[cm2['_cc'].isin(rule['categories'])]
        if cm2.empty: continue
        rej = cm2[~cm2['_sl'].isin(rule['sellers'])]
        for i in rej.index:
            flagged_idx.add(i)
            mt, mi = md.get(i, ('unknown', br))
            ss = "Seller not in approved list" if rule['sellers'] else "No sellers approved"
            cm[i] = f"Restricted Brand: {mi} - {ss}"
    if not flagged_idx: return pd.DataFrame(columns=data.columns)
    result = data.loc[list(flagged_idx)].copy()
    result['Comment_Detail'] = result.index.map(cm)
    return result.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_prohibited_products(data: pd.DataFrame, prohibited_rules: List[Dict]) -> pd.DataFrame:
    if not {'NAME','CATEGORY_CODE'}.issubset(data.columns) or not prohibited_rules: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d['_nl'] = d['NAME'].astype(str).str.lower().fillna('')
    d['_cc'] = d['CATEGORY_CODE'].apply(clean_category_code)
    fi = set(); cm = {}; nr = {}
    for rule in prohibited_rules:
        kw = rule['keyword']; tc = rule['categories']
        pat = re.compile(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', re.IGNORECASE)
        mm = d['_nl'].str.contains(pat, regex=True, na=False)
        if not mm.any(): continue
        cur = d[mm]
        if tc: cur = cur[cur['_cc'].isin(tc)]
        if cur.empty: continue
        for i in cur.index:
            fi.add(i)
            ec = cm.get(i, "Prohibited:")
            if kw not in ec: cm[i] = f"{ec} {kw},"
            nr[i] = pat.sub(lambda m: f"[!]{m.group(0)}[!]", str(d.loc[i,'NAME']))
    if not fi: return pd.DataFrame(columns=data.columns)
    result = data.loc[list(fi)].copy()
    result['Comment_Detail'] = result.index.map(lambda i: cm[i].rstrip(','))
    for i, n in nr.items(): result.loc[i,'NAME'] = n
    return result.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_suspected_fake_products(data: pd.DataFrame, suspected_fake_df: pd.DataFrame, fx_rate: float) -> pd.DataFrame:
    if not all(c in data.columns for c in ['CATEGORY_CODE','BRAND','GLOBAL_SALE_PRICE','GLOBAL_PRICE']) or suspected_fake_df.empty: return pd.DataFrame(columns=data.columns)
    try:
        ref = suspected_fake_df.copy(); bcp = {}
        for brand in [c for c in ref.columns if c not in ['Unnamed: 0','Brand','Price'] and pd.notna(c)]:
            try:
                pt = pd.to_numeric(ref[brand].iloc[0], errors='coerce')
                if pd.isna(pt) or pt <= 0: continue
            except: continue
            for cat in ref[brand].iloc[1:].dropna():
                cb = str(cat).strip().split('.')[0]
                if cb and cb.lower() != 'nan': bcp[(brand.strip().lower(), cb)] = pt
        if not bcp: return pd.DataFrame(columns=data.columns)
        d = data.copy()
        d['ptu'] = pd.to_numeric(d['GLOBAL_SALE_PRICE'].where(d['GLOBAL_SALE_PRICE'].notna() & (pd.to_numeric(d['GLOBAL_SALE_PRICE'], errors='coerce') > 0), d['GLOBAL_PRICE']), errors='coerce').fillna(0)
        d['BL'] = d['BRAND'].astype(str).str.strip().str.lower()
        d['CB'] = d['CATEGORY_CODE'].apply(clean_category_code)
        d['is_fake'] = [p < bcp.get((b,c),-1) for p,b,c in zip(d['ptu'].values, d['BL'].values, d['CB'].values)]
        return d[d['is_fake']][data.columns].drop_duplicates(subset=['PRODUCT_SET_SID'])
    except: return pd.DataFrame(columns=data.columns)

def check_refurb_seller_approval(data: pd.DataFrame, refurb_data: dict, country_code: str) -> pd.DataFrame:
    if not {'PRODUCT_SET_SID','CATEGORY_CODE','SELLER_NAME','NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    phone_cats = refurb_data.get("categories",{}).get("Phones",set())
    laptop_cats = refurb_data.get("categories",{}).get("Laptops",set())
    keywords = refurb_data.get("keywords",set())
    sellers = refurb_data.get("sellers",{}).get(country_code,{})
    if (not phone_cats and not laptop_cats) or not keywords: return pd.DataFrame(columns=data.columns)
    kp = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r')\b', re.IGNORECASE)
    d = data.copy()
    d['_cat'] = d['CATEGORY_CODE'].apply(clean_category_code)
    d['_sel'] = d['SELLER_NAME'].astype(str).str.strip().str.lower()
    d['_nm'] = d['NAME'].astype(str).str.strip()
    ip = d['_cat'].isin(phone_cats); il = d['_cat'].isin(laptop_cats)
    hk = d['_nm'].str.contains(kp, na=False)
    ap = sellers.get("Phones",set()); al = sellers.get("Laptops",set())
    na = ((ip & ~d['_sel'].isin(ap)) | (il & ~d['_sel'].isin(al)))
    flagged = d[(ip|il) & hk & na].copy()
    if not flagged.empty:
        def bc(row):
            pt = "Phone" if row['_cat'] in phone_cats else "Laptop"
            m = kp.search(row['_nm'])
            return f"Unapproved {pt} refurb seller — keyword '{m.group(0) if m else '?'}' in name"
        flagged['Comment_Detail'] = flagged.apply(bc, axis=1)
    return flagged.drop(columns=['_cat','_sel','_nm'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_product_warranty(data: pd.DataFrame, warranty_category_codes: List[str]) -> pd.DataFrame:
    d = data.copy()
    for c in ['PRODUCT_WARRANTY','WARRANTY_DURATION']:
        if c not in d.columns: d[c] = ""
        d[c] = d[c].astype(str).fillna('').str.strip()
    if not warranty_category_codes: return pd.DataFrame(columns=d.columns)
    d['CC'] = d['CATEGORY_CODE'].apply(clean_category_code)
    target = d[d['CC'].isin([clean_category_code(c) for c in warranty_category_codes])]
    if target.empty: return pd.DataFrame(columns=d.columns)
    def ip(s): return (s != 'nan') & (s != '') & (s != 'none') & (s != 'nat') & (s != 'n/a')
    return target[~(ip(target['PRODUCT_WARRANTY']) | ip(target['WARRANTY_DURATION']))].drop(columns=['CC'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_seller_approved_for_books(data: pd.DataFrame, books_data: Dict, country_code: str, book_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','SELLER_NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    cc = books_data.get('category_codes') or set(clean_category_code(c) for c in book_category_codes)
    if not cc: return pd.DataFrame(columns=data.columns)
    ap = books_data.get('sellers',{}).get(country_code,set())
    if not ap: return pd.DataFrame(columns=data.columns)
    books = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(cc)].copy()
    if books.empty: return pd.DataFrame(columns=data.columns)
    flagged = books[~books['SELLER_NAME'].astype(str).str.strip().str.lower().isin(ap)].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Seller not approved to sell books: " + flagged['SELLER_NAME'].astype(str)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_seller_approved_for_perfume(data: pd.DataFrame, perfume_category_codes: List[str], perfume_data: Dict, country_code: str) -> pd.DataFrame:
    if not {'CATEGORY_CODE','SELLER_NAME','BRAND','NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    cc = perfume_data.get('category_codes') or set(clean_category_code(c) for c in perfume_category_codes)
    perfume = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(cc)].copy()
    if perfume.empty: return pd.DataFrame(columns=data.columns)
    keywords = perfume_data.get('keywords',set())
    ap = perfume_data.get('sellers',{}).get(country_code,set())
    bl = perfume['BRAND'].astype(str).str.strip().str.lower()
    nl = perfume['NAME'].astype(str).str.strip().str.lower()
    GP = {'designers collection','smart collection','generic','original','fashion'}
    if keywords:
        kp = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r')\b', re.IGNORECASE)
        sm = bl.isin(GP) & nl.apply(lambda x: bool(kp.search(x)))
    else: sm = pd.Series([False]*len(perfume), index=perfume.index)
    if ap:
        bsm = bl.apply(lambda x: bool(kp.search(x))) if keywords else pd.Series([False]*len(perfume), index=perfume.index)
        na = ~perfume['SELLER_NAME'].astype(str).str.strip().str.lower().isin(ap)
        fm = (sm | bsm) & na
    else: fm = sm
    flagged = perfume[fm].copy()
    if not flagged.empty:
        def desc(row):
            b, n = str(row['BRAND']).strip(), str(row['NAME']).strip()[:40]
            return f"Sneaky brand in name: '{n}'" if b.lower() in GP else f"Sensitive brand '{b}' — seller not approved"
        flagged['Comment_Detail'] = flagged.apply(desc, axis=1)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_counterfeit_sneakers(data: pd.DataFrame, sneaker_category_codes: List[str], sneaker_sensitive_brands: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','NAME','BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    snk = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in sneaker_category_codes))].copy()
    if snk.empty: return pd.DataFrame(columns=data.columns)
    bl = snk['BRAND'].astype(str).str.strip().str.lower()
    nl = snk['NAME'].astype(str).str.strip().str.lower()
    return snk[bl.isin(['generic','fashion']) & nl.apply(lambda x: any(b in x for b in sneaker_sensitive_brands))].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_counterfeit_jerseys(data: pd.DataFrame, jerseys_data: Dict, country_code: str) -> pd.DataFrame:
    if not {"CATEGORY_CODE","NAME","SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    cats = jerseys_data.get("categories",set())
    kws = jerseys_data.get("keywords",{}).get(country_code,set())
    ex = jerseys_data.get("exempted",{}).get(country_code,set())
    if not cats or not kws: return pd.DataFrame(columns=data.columns)
    kp = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
    d = data.copy()
    d["_cat"] = d["CATEGORY_CODE"].apply(clean_category_code)
    d["_sel"] = d["SELLER_NAME"].astype(str).str.strip().str.lower()
    d["_nm"] = d["NAME"].astype(str).str.strip()
    flagged = d[d["_cat"].isin(cats) & d["_nm"].str.contains(kp, na=False) & ~d["_sel"].isin(ex)].copy()
    if not flagged.empty:
        def bc(row):
            m = kp.search(row["_nm"])
            return f"Suspected counterfeit jersey — keyword '{m.group(0) if m else '?'}' (cat: {row['_cat']})"
        flagged["Comment_Detail"] = flagged.apply(bc, axis=1)
    return flagged.drop(columns=["_cat","_sel","_nm"], errors="ignore").drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_unnecessary_words(data: pd.DataFrame, pattern: re.Pattern) -> pd.DataFrame:
    if not {'NAME'}.issubset(data.columns) or pattern is None: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    flagged = d[d['NAME'].astype(str).str.strip().str.lower().str.contains(pattern, na=False)].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Unnecessary: " + flagged['NAME'].apply(lambda t: ", ".join(set(m.lower() for m in pattern.findall(str(t)) if isinstance(m,str))) if not pd.isna(t) else "")
        flagged['NAME'] = flagged['NAME'].apply(lambda t: pattern.sub(lambda m: f"[*]{m.group(0)}[*]", str(t)) if not pd.isna(t) else t)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_single_word_name(data: pd.DataFrame, book_category_codes: List[str], books_data: Dict = None) -> pd.DataFrame:
    if not {'CATEGORY_CODE','NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    cc = (books_data or {}).get('category_codes') or set(clean_category_code(c) for c in book_category_codes)
    nb = data[~data['CATEGORY_CODE'].apply(clean_category_code).isin(cc)]
    return nb[nb['NAME'].astype(str).str.split().str.len() == 1].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_generic_brand_issues(data: pd.DataFrame, valid_category_codes_fas: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    return data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in valid_category_codes_fas)) & (data['BRAND'].astype(str).str.lower() == 'generic')].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_fashion_brand_issues(data: pd.DataFrame, valid_category_codes_fas: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    return data[(data['BRAND'].astype(str).str.strip().str.lower() == 'fashion') & (~data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in valid_category_codes_fas)))].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_brand_in_name(data: pd.DataFrame) -> pd.DataFrame:
    if not {'BRAND','NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    mask = [b in n if b and b != 'nan' else False
            for b, n in zip(data['BRAND'].astype(str).str.strip().str.lower().values,
                            data['NAME'].astype(str).str.strip().str.lower().values)]
    return data[mask].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_wrong_variation(data: pd.DataFrame, allowed_variation_codes: List[str]) -> pd.DataFrame:
    d = data.copy()
    if 'COUNT_VARIATIONS' not in d.columns: d['COUNT_VARIATIONS'] = 1
    if 'CATEGORY_CODE' not in d.columns: return pd.DataFrame(columns=data.columns)
    d['cc'] = d['CATEGORY_CODE'].apply(clean_category_code)
    d['qv'] = pd.to_numeric(d['COUNT_VARIATIONS'], errors='coerce').fillna(1).astype(int)
    flagged = d[(d['qv'] >= 3) & (~d['cc'].isin(set(clean_category_code(c) for c in allowed_variation_codes)))].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Variations: " + flagged['qv'].astype(str) + ", Category: " + flagged['cc']
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_generic_with_brand_in_name(data: pd.DataFrame, brands_list: List[str]) -> pd.DataFrame:
    if not {'NAME','BRAND'}.issubset(data.columns) or not brands_list: return pd.DataFrame(columns=data.columns)
    mask = (data['BRAND'].astype(str).str.strip().str.lower() == 'generic')
    if 'CATEGORY' in data.columns: mask = mask & ~data['CATEGORY'].astype(str).str.lower().str.contains(r'\b(case|cases|cover|covers)\b', regex=True, na=False)
    gen = data[mask].copy()
    if gen.empty: return pd.DataFrame(columns=data.columns)
    sb = sorted([str(b).strip().lower() for b in brands_list if b], key=len, reverse=True)
    def detect(n):
        nc = re.sub(r'\s+', ' ', re.sub(r"['\.\-]", ' ', str(n).lower())).strip()
        for b in sb:
            bc = re.sub(r'\s+', ' ', re.sub(r"['\.\-]", ' ', b)).strip()
            if nc.startswith(bc) and (len(nc) == len(bc) or not nc[len(bc)].isalnum()): return b.title()
        return None
    gen['Detected_Brand'] = [detect(n) for n in gen['NAME'].values]
    flagged = gen[gen['Detected_Brand'].notna()].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Detected Brand: " + flagged['Detected_Brand']
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_missing_color(data: pd.DataFrame, pattern: re.Pattern, color_categories: List[str], country_code: str) -> pd.DataFrame:
    if not {'CATEGORY_CODE','NAME'}.issubset(data.columns) or pattern is None: return pd.DataFrame(columns=data.columns)
    target = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in color_categories))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    hc = 'COLOR' in data.columns
    names = target['NAME'].astype(str).values
    colors = target['COLOR'].astype(str).str.strip().str.lower().values if hc else ['']*len(target)
    mask = [not pattern.search(n) and (not hc or c in ['nan','','none','null']) for n, c in zip(names, colors)]
    return target[mask].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_weight_volume_in_name(data: pd.DataFrame, weight_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','NAME'}.issubset(data.columns) or not weight_category_codes: return pd.DataFrame(columns=data.columns)
    target = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in weight_category_codes))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    pat = re.compile(r'\b\d+(?:\.\d+)?\s*(?:kg|kgs|g|gm|gms|grams|mg|mcg|ml|l|ltr|liter|litres|litre|cl|oz|ounces|lb|lbs|tablets|capsules|sachets|count|ct|sticks|iu|teabags|pieces|pcs|pack|packs)\b', re.IGNORECASE)
    return target[~target['NAME'].apply(lambda n: bool(pat.search(str(n))))].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_incomplete_smartphone_name(data: pd.DataFrame, smartphone_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','NAME'}.issubset(data.columns) or not smartphone_category_codes: return pd.DataFrame(columns=data.columns)
    target = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in smartphone_category_codes))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    pat = re.compile(r'\b\d+\s*(gb|tb)\b', re.IGNORECASE)
    flagged = target[~target['NAME'].apply(lambda n: bool(pat.search(str(n))))].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Name missing Storage/Memory spec (e.g., 64GB)"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

# FIX 4: O(n) duplicate check
def check_duplicate_products(data: pd.DataFrame, exempt_categories: List[str] = None, similarity_threshold: float = 0.70, known_colors: List[str] = None, **kwargs) -> pd.DataFrame:
    if not {'NAME','SELLER_NAME','BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    if exempt_categories and 'CATEGORY_CODE' in d.columns:
        d = d[~d['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in exempt_categories))]
    if d.empty: return pd.DataFrame(columns=data.columns)
    d['_nn'] = d['NAME'].astype(str).apply(lambda x: re.sub(r'\s+', '', normalize_text(x)))
    d['_nb'] = d['BRAND'].astype(str).str.lower().str.strip()
    d['_ns'] = d['SELLER_NAME'].astype(str).str.lower().str.strip()
    d['_dk'] = d['_ns'] + '|' + d['_nb'] + '|' + d['_nn']
    seen: dict = {}; rej: set = set(); details: dict = {}
    for _, row in d.iterrows():
        key = row['_dk']; sid = str(row['PRODUCT_SET_SID'])
        if key in seen:
            rej.add(sid)
            details[sid] = str(row['NAME'])[:40]
        else: seen[key] = sid
    if not rej: return pd.DataFrame(columns=data.columns)
    rdf = d[d['PRODUCT_SET_SID'].astype(str).isin(rej)].copy()
    rdf['Comment_Detail'] = rdf['PRODUCT_SET_SID'].apply(lambda s: f"Duplicate: '{details.get(str(s), '')}'" )
    base_cols = data.columns.tolist()
    return rdf[base_cols + ['Comment_Detail']].drop_duplicates(subset=['PRODUCT_SET_SID'])

# -------------------------------------------------
# MASTER VALIDATION (FIX 2: no st.write in threads)
# -------------------------------------------------
def validate_products(data: pd.DataFrame, support_files: Dict, cv: CountryValidator, data_has_warranty_cols: bool, common_sids=None, skip_validators=None):
    data['PRODUCT_SET_SID'] = data['PRODUCT_SET_SID'].astype(str).str.strip()
    fm = support_files['flags_mapping']
    crr = support_files.get('restricted_brands_all',{}).get(cv.country,[])
    cpw = support_files.get('prohibited_words_all',{}).get(cv.code,[])
    validations = [
        ("Wrong Category", check_miscellaneous_category, {}),
        ("Restricted brands", check_restricted_brands, {'country_rules': crr}),
        ("Suspected Fake product", check_suspected_fake_products, {'suspected_fake_df': support_files['suspected_fake'], 'fx_rate': FX_RATE}),
        ("Seller Not approved to sell Refurb", check_refurb_seller_approval, {'refurb_data': support_files.get('refurb_data',{}), 'country_code': cv.code}),
        ("Product Warranty", check_product_warranty, {'warranty_category_codes': support_files['warranty_category_codes']}),
        ("Seller Approve to sell books", check_seller_approved_for_books, {'books_data': support_files.get('books_data',{}), 'country_code': cv.code, 'book_category_codes': support_files['book_category_codes']}),
        ("Seller Approved to Sell Perfume", check_seller_approved_for_perfume, {'perfume_category_codes': support_files['perfume_category_codes'], 'perfume_data': support_files.get('perfume_data',{}), 'country_code': cv.code}),
        ("Counterfeit Sneakers", check_counterfeit_sneakers, {'sneaker_category_codes': support_files['sneaker_category_codes'], 'sneaker_sensitive_brands': support_files['sneaker_sensitive_brands']}),
        ("Suspected counterfeit Jerseys", check_counterfeit_jerseys, {'jerseys_data': support_files.get('jerseys_data',{}), 'country_code': cv.code}),
        ("Prohibited products", check_prohibited_products, {'prohibited_rules': cpw}),
        ("Unnecessary words in NAME", check_unnecessary_words, {'pattern': compile_regex_patterns(support_files['unnecessary_words'])}),
        ("Single-word NAME", check_single_word_name, {'book_category_codes': support_files['book_category_codes'], 'books_data': support_files.get('books_data',{})}),
        ("Generic BRAND Issues", check_generic_brand_issues, {}),
        ("Fashion brand issues", check_fashion_brand_issues, {}),
        ("BRAND name repeated in NAME", check_brand_in_name, {}),
        ("Wrong Variation", check_wrong_variation, {'allowed_variation_codes': list(set(support_files.get('variation_allowed_codes',[]) + support_files.get('category_fas',[])))}),
        ("Generic branded products with genuine brands", check_generic_with_brand_in_name, {'brands_list': support_files.get('known_brands',[])}),
        ("Missing COLOR", check_missing_color, {'pattern': compile_regex_patterns(support_files['colors']), 'color_categories': support_files['color_categories'], 'country_code': cv.code}),
        ("Missing Weight/Volume", check_weight_volume_in_name, {'weight_category_codes': support_files.get('weight_category_codes',[])}),
        ("Incomplete Smartphone Name", check_incomplete_smartphone_name, {'smartphone_category_codes': support_files.get('smartphone_category_codes',[])}),
        ("Duplicate product", check_duplicate_products, {'exempt_categories': support_files.get('duplicate_exempt_codes',[]), 'known_colors': support_files['colors']}),
    ]
    results = {}; dup_groups = {}
    if {'NAME','BRAND','SELLER_NAME','COLOR'}.issubset(data.columns):
        dt = data.copy()
        dt['dk'] = dt[['NAME','BRAND','SELLER_NAME','COLOR']].apply(lambda r: tuple(str(v).strip().lower() for v in r), axis=1)
        for k, v in dt.groupby('dk')['PRODUCT_SET_SID'].apply(list).items():
            if len(v) > 1:
                for sid in v: dup_groups[sid] = v
    restricted_keys = {}; validation_errors = []
    with st.spinner("Validating products..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            ftn = {}
            for name, func, kwargs in validations:
                if skip_validators and name in skip_validators: continue
                if cv.should_skip_validation(name): continue
                ckw = {'data': data, **kwargs}
                if name in ["Generic BRAND Issues","Fashion brand issues"]: ckw['valid_category_codes_fas'] = support_files.get('category_fas',[])
                ftn[executor.submit(func, **ckw)] = name
            for future in concurrent.futures.as_completed(ftn):
                name = ftn[future]
                try:
                    res = future.result()
                    if not res.empty and 'PRODUCT_SET_SID' in res.columns:
                        res = res.loc[:, ~res.columns.duplicated()].copy()
                        res['PRODUCT_SET_SID'] = res['PRODUCT_SET_SID'].astype(str).str.strip()
                        if name in ["Seller Approve to sell books","Seller Approved to Sell Perfume","Counterfeit Sneakers","Seller Not approved to sell Refurb","Restricted brands"]:
                            res['match_key'] = res.apply(create_match_key, axis=1)
                            restricted_keys.setdefault(name, set()).update(res['match_key'].unique())
                        exp = set()
                        for sid in set(res['PRODUCT_SET_SID'].unique()): exp.update(dup_groups.get(sid, [sid]))
                        fr2 = data[data['PRODUCT_SET_SID'].isin(exp)].copy()
                        if 'Comment_Detail' in res.columns: fr2['Comment_Detail'] = res['Comment_Detail']
                        results[name] = pd.concat([results[name], fr2]).drop_duplicates(subset=['PRODUCT_SET_SID']) if name in results and not results[name].empty else fr2
                    else:
                        if name not in results: results[name] = pd.DataFrame(columns=data.columns)
                except Exception as e:
                    logger.error(f"Error in {name}: {e}")
                    validation_errors.append((name, str(e)))
                    if name not in results: results[name] = pd.DataFrame(columns=data.columns)
    if validation_errors:
        st.warning(f"{len(validation_errors)} validation checks encountered errors.")
        with st.expander("View Errors"):
            for en, em in validation_errors: st.error(f"**{en}**: {em}")
    if restricted_keys:
        data['match_key'] = data.apply(create_match_key, axis=1)
        for fn, keys in restricted_keys.items():
            extra = data[data['match_key'].isin(keys)].copy()
            results[fn] = pd.concat([results.get(fn, pd.DataFrame()), extra]).drop_duplicates(subset=['PRODUCT_SET_SID'])
    rows = []; processed = set()
    for name, _, _ in validations:
        if name not in results or results[name].empty or 'PRODUCT_SET_SID' not in results[name].columns: continue
        res = results[name]
        ri = fm.get(name, ("1000007 - Other Reason", f"Flagged by {name}"))
        res['PRODUCT_SET_SID'] = res['PRODUCT_SET_SID'].astype(str).str.strip()
        flagged = pd.merge(res[['PRODUCT_SET_SID','Comment_Detail']] if 'Comment_Detail' in res.columns else res[['PRODUCT_SET_SID']], data, on='PRODUCT_SET_SID', how='left')
        if 'Comment_Detail' not in flagged.columns and 'Comment_Detail' in res.columns:
            flagged['Comment_Detail'] = res['Comment_Detail'].iloc[:,0] if isinstance(res['Comment_Detail'], pd.DataFrame) else res['Comment_Detail']
        for _, r in flagged.iterrows():
            sid = str(r['PRODUCT_SET_SID']).strip()
            if sid in processed: continue
            processed.add(sid)
            det = r.get('Comment_Detail','')
            rows.append({'ProductSetSid': sid, 'ParentSKU': r.get('PARENTSKU',''), 'Status': 'Rejected',
                         'Reason': ri[0], 'Comment': f"{ri[1]} ({det})" if pd.notna(det) and det else ri[1],
                         'FLAG': name, 'SellerName': r.get('SELLER_NAME','')})
    for _, r in data[~data['PRODUCT_SET_SID'].astype(str).str.strip().isin(processed)].iterrows():
        sid = str(r['PRODUCT_SET_SID']).strip()
        if sid not in processed:
            rows.append({'ProductSetSid': sid, 'ParentSKU': r.get('PARENTSKU',''), 'Status': 'Approved',
                         'Reason': '', 'Comment': '', 'FLAG': '', 'SellerName': r.get('SELLER_NAME','')})
            processed.add(sid)
    final_df = pd.DataFrame(rows)
    for c in ["ProductSetSid","ParentSKU","Status","Reason","Comment","FLAG","SellerName"]:
        if c not in final_df.columns: final_df[c] = ""
    return cv.ensure_status_column(final_df), results

# FIX 1: Cached validate_products
@st.cache_data(show_spinner=False, ttl=3600)
def cached_validate_products(data_hash: str, _data: pd.DataFrame, _support_files: Dict, country_code: str, data_has_warranty_cols: bool):
    country_name = next((k for k, v in CountryValidator.COUNTRY_CONFIG.items() if v['code'] == country_code), "Kenya")
    return validate_products(_data, _support_files, CountryValidator(country_name), data_has_warranty_cols)

# -------------------------------------------------
# EXPORTS
# -------------------------------------------------
def to_excel_base(df, sheet, cols, writer, format_rules=False):
    df_p = df.copy()
    for c in cols:
        if c not in df_p.columns: df_p[c] = pd.NA
    df_w = df_p[[c for c in cols if c in df_p.columns]]
    df_w.to_excel(writer, index=False, sheet_name=sheet)
    if format_rules and 'Status' in df_w.columns:
        wb = writer.book; ws = writer.sheets[sheet]
        idx = df_w.columns.get_loc('Status')
        ws.conditional_format(1, idx, len(df_w), idx, {'type':'cell','criteria':'equal','value':'"Rejected"','format': wb.add_format({'bg_color':'#FFC7CE','font_color':'#9C0006'})})
        ws.conditional_format(1, idx, len(df_w), idx, {'type':'cell','criteria':'equal','value':'"Approved"','format': wb.add_format({'bg_color':'#C6EFCE','font_color':'#006100'})})

def write_excel_single(df, sheet_name, cols, auxiliary_df=None, aux_sheet_name=None, aux_cols=None, format_status=False, full_data_stats=False):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        to_excel_base(df, sheet_name, cols, writer, format_rules=format_status)
        if auxiliary_df is not None and not auxiliary_df.empty: to_excel_base(auxiliary_df, aux_sheet_name, aux_cols, writer)
        if full_data_stats and 'SELLER_NAME' in df.columns and 'Status' in df.columns:
            ws = writer.book.add_worksheet('Sellers Data')
            fmt = writer.book.add_format({'bold':True,'bg_color':'#E6F0FA','border':1,'align':'center'})
            df2 = df.copy()
            df2['R'] = (df2['Status'] == 'Rejected').astype(int)
            df2['A'] = (df2['Status'] == 'Approved').astype(int)
            summ = df2.groupby('SELLER_NAME').agg(Rejected=('R','sum'), Approved=('A','sum')).reset_index().sort_values('Rejected', ascending=False)
            summ.insert(0, 'Rank', range(1, len(summ)+1))
            ws.write(0, 0, "Sellers Summary (This File)", fmt)
            summ.to_excel(writer, sheet_name='Sellers Data', startrow=1, index=False)
    output.seek(0)
    return output

def generate_smart_export(df, filename_prefix, export_type='simple', auxiliary_df=None):
    cols = FULL_DATA_COLS + [c for c in ["Status","Reason","Comment","FLAG","SellerName"] if c not in FULL_DATA_COLS] if export_type == 'full' else PRODUCTSETS_COLS
    if len(df) <= SPLIT_LIMIT:
        data = write_excel_single(df, "ProductSets", cols, auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS, True, export_type == 'full')
        return data, f"{filename_prefix}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    zb = BytesIO()
    with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(0, len(df), SPLIT_LIMIT):
            chunk = df.iloc[i:i+SPLIT_LIMIT]
            ed = write_excel_single(chunk, "ProductSets", cols, auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS, True, export_type == 'full')
            zf.writestr(f"{filename_prefix}_Part_{(i//SPLIT_LIMIT)+1}.xlsx", ed.getvalue())
    zb.seek(0)
    return zb, f"{filename_prefix}.zip", "application/zip"

def prepare_full_data_merged(data_df, final_report_df):
    try:
        d, r = data_df.copy(), final_report_df.copy()
        d['PRODUCT_SET_SID'] = d['PRODUCT_SET_SID'].astype(str).str.strip()
        r['ProductSetSid'] = r['ProductSetSid'].astype(str).str.strip()
        merged = pd.merge(d, r[["ProductSetSid","Status","Reason","Comment","FLAG","SellerName"]], left_on="PRODUCT_SET_SID", right_on="ProductSetSid", how='left')
        if 'ProjectSetSid' in merged.columns: merged.drop(columns=['ProjectSetSid'], inplace=True)
        return merged
    except: return pd.DataFrame()

# -------------------------------------------------
# FLAG EXPANDER (FIX 3: cached display df merge)
# -------------------------------------------------
def strip_html(text): return re.sub('<[^<]+?>', '', text) if isinstance(text, str) else text

def render_flag_expander(title, df_flagged, data, data_has_warranty_cols_check, support_files, country_validator):
    cache_key = f"display_df_{title}"
    base_cols = ['PRODUCT_SET_SID','NAME','BRAND','CATEGORY','COLOR','GLOBAL_SALE_PRICE','GLOBAL_PRICE','PARENTSKU','SELLER_NAME']
    disp_cols = base_cols.copy()
    if title == "Wrong Variation":
        if 'COUNT_VARIATIONS' in data.columns: disp_cols.append('COUNT_VARIATIONS')
        if 'LIST_VARIATIONS' in data.columns: disp_cols.append('LIST_VARIATIONS')
    if cache_key not in st.session_state.display_df_cache:
        df_display = pd.merge(df_flagged[['ProductSetSid']], data, left_on='ProductSetSid', right_on='PRODUCT_SET_SID', how='left')[[c for c in disp_cols if c in data.columns]]
        st.session_state.display_df_cache[cache_key] = df_display
    df_display = st.session_state.display_df_cache[cache_key].copy()
    c1, c2 = st.columns([1,1])
    with c1: search_term = st.text_input("Search", placeholder="Name, Brand...", key=f"s_{title}")
    with c2: seller_filter = st.multiselect("Filter by Seller", sorted(df_display['SELLER_NAME'].astype(str).unique()), key=f"f_{title}")
    if search_term: df_display = df_display[df_display.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    if seller_filter: df_display = df_display[df_display['SELLER_NAME'].isin(seller_filter)]
    df_display = df_display.reset_index(drop=True)
    if 'NAME' in df_display.columns: df_display['NAME'] = df_display['NAME'].apply(strip_html)
    event = st.dataframe(df_display, hide_index=True, use_container_width=True, selection_mode="multi-row", on_select="rerun",
        column_config={
            "PRODUCT_SET_SID": st.column_config.TextColumn(pinned=True),
            "NAME": st.column_config.TextColumn(pinned=True),
            "GLOBAL_SALE_PRICE": st.column_config.NumberColumn("Sale Price (USD)", format="$%.2f"),
            "GLOBAL_PRICE": st.column_config.NumberColumn("Price (USD)", format="$%.2f"),
        }, key=f"df_{title}")
    selected_indices = [i for i in list(event.selection.rows) if i < len(df_display)]
    st.caption(f"{len(selected_indices)} of {len(df_display)} rows selected")
    has_sel = len(selected_indices) > 0
    _fm = support_files['flags_mapping']
    _reason_options = ["Wrong Category","Restricted brands","Suspected Fake product","Seller Not approved to sell Refurb","Product Warranty","Seller Approve to sell books","Seller Approved to Sell Perfume","Counterfeit Sneakers","Suspected counterfeit Jerseys","Prohibited products","Unnecessary words in NAME","Single-word NAME","Generic BRAND Issues","Fashion brand issues","BRAND name repeated in NAME","Wrong Variation","Generic branded products with genuine brands","Missing COLOR","Missing Weight/Volume","Incomplete Smartphone Name","Duplicate product","Poor images","Other Reason (Custom)"]
    bc1, bc2 = st.columns([1,1])
    with bc1:
        if st.button("✓ Approve Selected", key=f"approve_sel_{title}", type="primary", use_container_width=True, disabled=not has_sel):
            sids = df_display.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
            subset = data[data['PRODUCT_SET_SID'].isin(sids)]
            dh = df_hash(subset) + country_validator.code
            new_report, _ = cached_validate_products(dh, subset, support_files, country_validator.code, data_has_warranty_cols_check)
            msg_approved = 0; msg_moved = {}
            for sid in sids:
                nr = new_report[new_report['ProductSetSid'] == sid]
                if nr.empty or not str(nr.iloc[0]['FLAG']):
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status','Reason','Comment','FLAG']] = ['Approved','','','Approved by User']
                    msg_approved += 1
                else:
                    nf = str(nr.iloc[0]['FLAG'])
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status','Reason','Comment','FLAG']] = ['Rejected', nr.iloc[0]['Reason'], nr.iloc[0]['Comment'], nf]
                    msg_moved[nf] = msg_moved.get(nf, 0) + 1
            if msg_approved > 0: st.session_state.main_toasts.append(f"{msg_approved} items Approved!")
            for flag, count in msg_moved.items(): st.session_state.main_toasts.append(f"{count} items re-flagged: {flag}")
            st.session_state.exports_cache.clear()
            st.session_state.display_df_cache.clear()
            st.rerun()
    with bc2:
        with st.popover("↓ Reject As...", use_container_width=True, disabled=not has_sel):
            chosen = st.selectbox("Reason", _reason_options, key=f"rej_reason_dd_{title}", label_visibility="collapsed")
            if chosen == "Other Reason (Custom)":
                custom_cmt = st.text_area("Custom comment", placeholder="Type reason...", key=f"custom_comment_{title}", height=80)
                if st.button("Apply Custom", key=f"apply_custom_{title}", type="primary", use_container_width=True, disabled=not has_sel):
                    sids = df_display.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(sids), ['Status','Reason','Comment','FLAG']] = ['Rejected','1000007 - Other Reason', custom_cmt.strip() or "Other Reason", 'Other Reason (Custom)']
                    st.session_state.main_toasts.append(f"{len(sids)} items rejected with custom reason.")
                    st.session_state.exports_cache.clear(); st.session_state.display_df_cache.clear(); st.rerun()
            else:
                rc, rcmt = _fm.get(chosen, ('1000007 - Other Reason', chosen))
                st.caption(f"Code: {rc[:40]}...")
                if st.button("Apply Rejection", key=f"apply_dd_{title}", type="primary", use_container_width=True, disabled=not has_sel):
                    sids = df_display.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(sids), ['Status','Reason','Comment','FLAG']] = ['Rejected', rc, rcmt, chosen]
                    st.session_state.main_toasts.append(f"{len(sids)} items rejected as '{chosen}'.")
                    st.session_state.exports_cache.clear(); st.session_state.display_df_cache.clear(); st.rerun()

# -------------------------------------------------
# FIX 5: Image quality — caches failures to avoid retry storms
# -------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def analyze_image_quality_cached(url: str) -> List[str]:
    if not url or not url.startswith("http"): return []
    try:
        resp = requests.get(url, timeout=2, stream=True)
        if resp.status_code != 200: return ["Unreachable"]  # Cache the failure
        img = Image.open(resp.raw)
        w, h = img.size
        warnings = []
        if w < 300 or h < 300: warnings.append("Low Resolution")
        ratio = h / w if w > 0 else 1
        if ratio > 1.5: warnings.append("Tall (Screenshot?)")
        elif ratio < 0.6: warnings.append("Wide Aspect")
        return warnings
    except Exception:
        return ["Unreachable"]  # Cache the failure — stops retrying bad URLs

# -------------------------------------------------
# FAST GRID: JS-driven selection (zero reruns on click)
# -------------------------------------------------
def _process_card_bridge_action(action_bridge: str, support_files: dict) -> bool:
    """Process all card actions received from the JS bridge."""
    if not action_bridge or ':' not in action_bridge: return False
    ci = action_bridge.index(':')
    action_type = action_bridge[:ci]
    target = action_bridge[ci+1:]
    fm = support_files['flags_mapping']
    if action_type == 'RESTORE':
        sid = target
        st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status','Reason','Comment','FLAG']] = ['Approved','','','Approved by User']
        st.session_state.pop(f"quick_rej_{sid}", None)
        st.session_state.pop(f"quick_rej_reason_{sid}", None)
        st.session_state.exports_cache.clear(); st.session_state.display_df_cache.clear()
        st.session_state.main_toasts.append("Item restored!")
        return True
    card_action_map = {
        'REJECT_POOR_IMAGE': 'Poor images',
        'REJECT_WRONG_CAT': 'Wrong Category',
        'REJECT_FAKE': 'Suspected Fake product',
        'REJECT_BRAND': 'Restricted brands',
        'REJECT_PROHIBITED': 'Prohibited products',
        'REJECT_COLOR': 'Missing COLOR',
        'REJECT_WRONG_BRAND': 'Generic branded products with genuine brands',
    }
    if action_type in card_action_map:
        sid = target
        flag_name = card_action_map[action_type]
        code, cmt = fm.get(flag_name, ('1000007 - Other Reason', flag_name))
        st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status','Reason','Comment','FLAG']] = ['Rejected', code, cmt, flag_name]
        st.session_state[f"quick_rej_{sid}"] = True
        st.session_state[f"quick_rej_reason_{sid}"] = flag_name
        st.session_state.exports_cache.clear(); st.session_state.display_df_cache.clear()
        st.session_state.main_toasts.append(f"Rejected as '{flag_name}'")
        return True
    if action_type.startswith('BATCH_REJECT_'):
        flag_name = action_type[len('BATCH_REJECT_'):]
        sids = [s.strip() for s in target.split(',') if s.strip()]
        if not sids: return False
        code, cmt = fm.get(flag_name, ('1000007 - Other Reason', flag_name))
        st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(sids), ['Status','Reason','Comment','FLAG']] = ['Rejected', code, cmt, flag_name]
        for s in sids:
            st.session_state[f"quick_rej_{s}"] = True
            st.session_state[f"quick_rej_reason_{s}"] = flag_name
        st.session_state.exports_cache.clear(); st.session_state.display_df_cache.clear()
        st.session_state.main_toasts.append((f"Batch rejected {len(sids)} as '{flag_name}'", "✅"))
        return True
    if action_type == 'NAV_PREV':
        st.session_state.grid_page = max(0, st.session_state.grid_page - 1)
        st.session_state.do_scroll_top = True; return True
    if action_type == 'NAV_NEXT':
        st.session_state.grid_page += 1
        st.session_state.do_scroll_top = True; return True
    return False

def build_fast_grid_html(page_data, flags_mapping, country, page_warnings, rejected_state, cols_per_row, current_page, total_pages) -> str:
    O = "#F68B1E"; R = "#E73C17"; G = "#4CAF50"; DG = "#313133"
    batch_options = [
        ("Poor Image Quality", "Poor images"),
        ("Wrong Category", "Wrong Category"),
        ("Suspected Fake", "Suspected Fake product"),
        ("Restricted Brand", "Restricted brands"),
        ("Wrong Brand", "Generic branded products with genuine brands"),
    ]
    batch_options_html = "".join([f'<option value="{val}">{label}</option>' for label, val in batch_options])

    html_parts = []
    # Inject CSS and JS (pure client-side selection, zero Python reruns until an action is triggered)
    html_parts.append(f"""
    <style>
        .fg-card {{ border: 1px solid #eee; border-radius: 8px; padding: 10px; background: #fff; position: relative; transition: all 0.2s; }}
        .fg-card.selected {{ border-color: {G}; box-shadow: 0 0 0 3px rgba(76,175,80,0.2); background: rgba(76,175,80,0.05); }}
        .fg-check {{ position: absolute; bottom: 10px; right: 10px; width: 24px; height: 24px; border-radius: 50%; background: rgba(0,0,0,0.2); display: flex; align-items: center; justify-content: center; color: transparent; font-weight: bold; transition: all 0.2s; pointer-events: none; }}
        .fg-card.selected .fg-check {{ background: {G}; color: white; }}
        .fg-img {{ width: 100%; aspect-ratio: 1; object-fit: contain; border-radius: 6px; cursor: pointer; }}
        .fg-meta {{ font-size: 11px; margin-top: 8px; line-height: 1.4; }}
        .fg-brand {{ color: {O}; font-weight: bold; margin: 4px 0; }}
        .fg-actions {{ display: flex; gap: 4px; margin-top: 8px; }}
        .fg-btn {{ flex: 1; padding: 6px; font-size: 11px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; color: white; text-align: center; }}
        .fg-btn-poor {{ background: {O}; }}
        .fg-btn-undo {{ background: {DG}; }}
        .fg-overlay {{ position: absolute; inset: 0; background: rgba(255,255,255,0.85); display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 10; border-radius: 8px; }}
        .fg-rej-badge {{ background: {R}; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px; margin-bottom: 8px; text-align: center; }}
        .fg-controls {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding: 12px; background: #f9f9f9; border-radius: 8px; border: 1px solid #eee; }}
        .fg-batch-select {{ padding: 6px; border-radius: 4px; border: 1px solid #ccc; outline: none; }}
    </style>
    <script>
        let selectedSids = new Set();
        
        function sendBridge(action) {{
            // Target the hidden Streamlit text input across the iframe boundary
            const inputs = window.parent.document.querySelectorAll('input');
            let bridgeInput = null;
            for (let i = 0; i < inputs.length; i++) {{
                if (inputs[i].placeholder === "__CARD_ACT__") {{
                    bridgeInput = inputs[i];
                    break;
                }}
            }}
            if (bridgeInput) {{
                const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                nativeSetter.call(bridgeInput, action);
                bridgeInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }} else {{
                console.error("Bridge input not found. Make sure the hidden text_input is rendered.");
            }}
        }}

        function toggleCard(sid) {{
            const el = document.getElementById('card-' + sid);
            if(selectedSids.has(sid)) {{
                selectedSids.delete(sid);
                el.classList.remove('selected');
            }} else {{
                selectedSids.add(sid);
                el.classList.add('selected');
            }}
            document.getElementById('sel-count').innerText = selectedSids.size;
        }}

        function selectAll(sids) {{
            sids.forEach(sid => {{
                selectedSids.add(sid);
                let el = document.getElementById('card-' + sid);
                if(el) el.classList.add('selected');
            }});
            document.getElementById('sel-count').innerText = selectedSids.size;
        }}

        function deselectAll() {{
            selectedSids.clear();
            document.querySelectorAll('.fg-card').forEach(el => el.classList.remove('selected'));
            document.getElementById('sel-count').innerText = '0';
        }}

        function doBatch() {{
            if(selectedSids.size === 0) return;
            const reason = document.getElementById('batch-reason').value;
            const sids = Array.from(selectedSids).join(',');
            sendBridge('BATCH_REJECT_' + reason + ':' + sids);
        }}
    </script>
    <div class="fg-controls">
        <div>
            <button onclick="sendBridge('NAV_PREV:1')" {'disabled' if current_page == 0 else ''} style="padding: 6px 12px; cursor: pointer;">&larr; Prev</button>
            <span style="margin: 0 15px; font-weight: bold; font-size: 13px;">Page {current_page + 1} of {total_pages}</span>
            <button onclick="sendBridge('NAV_NEXT:1')" {'disabled' if current_page >= total_pages - 1 else ''} style="padding: 6px 12px; cursor: pointer;">Next &rarr;</button>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <button onclick='selectAll({json.dumps(page_data["PRODUCT_SET_SID"].tolist())})' style="padding: 6px; cursor: pointer;">Select All</button>
            <button onclick="deselectAll()" style="padding: 6px; cursor: pointer;">Deselect All</button>
            <span style="font-size: 13px; font-weight: bold;"><span id="sel-count">0</span> selected</span>
            <select id="batch-reason" class="fg-batch-select">{batch_options_html}</select>
            <button onclick="doBatch()" style="background: {R}; color: white; border: none; padding: 7px 14px; border-radius: 4px; font-weight: bold; cursor: pointer;">Batch Reject</button>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat({cols_per_row}, 1fr); gap: 15px; margin-bottom: 20px;">
    """)

    # Render individual cards
    for _, row in page_data.iterrows():
        sid = str(row['PRODUCT_SET_SID'])
        img_url = str(row.get('MAIN_IMAGE', '')).strip()
        if not img_url.startswith('http'): img_url = "https://via.placeholder.com/150?text=No+Image"
        
        name = str(row.get('NAME', ''))
        short_name = name[:35] + "..." if len(name) > 35 else name
        brand = str(row.get('BRAND', 'Unknown Brand'))
        cat = str(row.get('CATEGORY', 'Unknown Category'))
        seller = str(row.get('SELLER_NAME', 'Unknown Seller'))
        
        is_rej = sid in rejected_state
        rej_reason = rejected_state.get(sid, "")
        
        warnings = page_warnings.get(sid, [])
        warn_html = "".join([f'<div style="background:rgba(255,193,7,0.95); color:#313133; font-size:10px; font-weight:800; padding:4px 8px; border-radius:12px; margin-bottom:4px; box-shadow:0 2px 4px rgba(0,0,0,0.2);">{w}</div>' for w in warnings])
        warn_container = f'<div style="position:absolute; top:8px; right:8px; display:flex; flex-direction:column; z-index:5;">{warn_html}</div>' if warnings else ""

        if is_rej:
            html_parts.append(f"""
            <div style="position: relative; border: 1px solid #ccc; border-radius: 8px; padding: 10px; background: #fafafa;">
                <img src="{img_url}" class="fg-img" style="filter: grayscale(100%); opacity: 0.4;">
                <div class="fg-overlay">
                    <div class="fg-rej-badge">REJECTED</div>
                    <div style="font-size: 11px; font-weight: bold; text-align: center; color: {R}; padding: 0 10px;">{rej_reason}</div>
                    <button class="fg-btn fg-btn-undo" onclick="sendBridge('RESTORE:{sid}')" style="margin-top: 12px; padding: 6px 16px;">Undo Rejection</button>
                </div>
                <div class="fg-meta">
                    <div style="font-weight: 600;">{short_name}</div>
                    <div class="fg-brand">{brand}</div>
                </div>
            </div>
            """)
        else:
            html_parts.append(f"""
            <div class="fg-card" id="card-{sid}">
                <div onclick="toggleCard('{sid}')" style="position:relative;">
                    {warn_container}
                    <img src="{img_url}" class="fg-img" loading="lazy">
                    <div class="fg-check">✓</div>
                </div>
                <div class="fg-meta">
                    <div title="{name}" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 600;">{short_name}</div>
                    <div class="fg-brand">{brand}</div>
                    <div style="color: #666; font-size: 10px;">{cat}</div>
                    <div style="color: #999; font-size: 9px; margin-top: 4px; border-top: 1px dashed #eee; padding-top: 4px;">{seller}</div>
                </div>
                <div class="fg-actions">
                    <button class="fg-btn fg-btn-poor" onclick="sendBridge('REJECT_POOR_IMAGE:{sid}')">Poor Img</button>
                    <select onchange="if(this.value) sendBridge(this.value + ':{sid}')" style="flex:1; font-size:11px; border:1px solid #ccc; border-radius:4px; outline:none;">
                        <option value="">More...</option>
                        <option value="REJECT_WRONG_CAT">Wrong Category</option>
                        <option value="REJECT_FAKE">Fake Product</option>
                        <option value="REJECT_BRAND">Restricted Brand</option>
                        <option value="REJECT_PROHIBITED">Prohibited</option>
                        <option value="REJECT_COLOR">Wrong Color</option>
                        <option value="REJECT_WRONG_BRAND">Wrong Brand</option>
                    </select>
                </div>
            </div>
            """)

    html_parts.append("</div>")
    return "".join(html_parts)

# ==========================================
# APP UI & DATA PIPELINE
# ==========================================
st.header(":material/upload_file: Upload Files", anchor=False)

current_country = st.session_state.get('selected_country', 'Kenya')
country_choice = st.segmented_control("Country", ["Kenya", "Uganda", "Nigeria", "Ghana", "Morocco"], default=current_country)

if country_choice: st.session_state.selected_country = country_choice
else: country_choice = current_country

country_validator = CountryValidator(st.session_state.selected_country)

uploaded_files = st.file_uploader("Upload CSV or XLSX files", type=['csv', 'xlsx'], accept_multiple_files=True, key="daily_files")

if uploaded_files:
    current_file_signature = hashlib.md5(str(sorted([f.name + str(f.size) for f in uploaded_files])).encode()).hexdigest()
    process_signature = f"{current_file_signature}_{country_validator.code}"
else: process_signature = "empty"

if st.session_state.get('last_processed_files') != process_signature:
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.grid_data = pd.DataFrame()
    st.session_state.post_qc_summary = pd.DataFrame()
    st.session_state.post_qc_results = {}
    st.session_state.file_mode = None
    st.session_state.grid_page = 0
    st.session_state.exports_cache = {}
    st.session_state.display_df_cache = {}
    
    # Clean up old quick reject keys
    for k in list(st.session_state.keys()):
        if k.startswith(("quick_rej_", "grid_chk_", "toast_", "card_action_")):
            del st.session_state[k]

    if process_signature == "empty": 
        st.session_state.last_processed_files = "empty"
    else:
        # OPTIMIZATION: Check for Parquet cache first
        cache_data_path = f"cache_{process_signature}_data.parquet"
        cache_report_path = f"cache_{process_signature}_report.parquet"
        
        if os.path.exists(cache_data_path) and os.path.exists(cache_report_path):
            st.toast("Loaded from Parquet cache instantly! 🚀", icon="⚡")
            st.session_state.all_data_map = pd.read_parquet(cache_data_path)
            st.session_state.final_report = pd.read_parquet(cache_report_path)
            st.session_state.file_mode = 'pre_qc'
            st.session_state.last_processed_files = process_signature
        else:
            try:
                all_dfs = []
                detected_modes = []
                for uf in uploaded_files:
                    uf.seek(0)
                    if uf.name.endswith('.xlsx'): raw_data = pd.read_excel(uf, engine='openpyxl', dtype=str)
                    else:
                        try:
                            raw_data = pd.read_csv(uf, dtype=str)
                            if len(raw_data.columns) <= 1:
                                uf.seek(0)
                                raw_data = pd.read_csv(uf, sep=';', encoding='ISO-8859-1', dtype=str)
                        except:
                            uf.seek(0)
                            raw_data = pd.read_csv(uf, sep=';', encoding='ISO-8859-1', dtype=str)
                    detected_modes.append(detect_file_type(raw_data))
                    all_dfs.append(raw_data)

                file_mode = detected_modes[0] if detected_modes else 'pre_qc'
                st.session_state.file_mode = file_mode

                if file_mode == 'post_qc':
                    norm_dfs = [normalize_post_qc(df) for df in all_dfs]
                    merged = pd.concat(norm_dfs, ignore_index=True).drop_duplicates(subset=['PRODUCT_SET_SID'])
                    summary_df, results = run_post_qc_checks(merged, support_files)
                    st.session_state.post_qc_summary = summary_df
                    st.session_state.post_qc_results = results
                    st.session_state.post_qc_data = merged
                    st.session_state.last_processed_files = process_signature
                else:
                    std_dfs = [standardize_input_data(df) for df in all_dfs]
                    merged_data = pd.concat(std_dfs, ignore_index=True)
                    data_prop = propagate_metadata(merged_data)
                    is_valid, errors = validate_input_schema(data_prop)
                    
                    if is_valid:
                        data_filtered, det_names = filter_by_country(data_prop, country_validator)
                        if data_filtered.empty: 
                            st.error(f"No {country_validator.country} products found.", icon=":material/error:")
                            st.stop()
                            
                        actual_counts = data_filtered.groupby('PRODUCT_SET_SID')['PRODUCT_SET_SID'].transform('count')
                        if 'COUNT_VARIATIONS' in data_filtered.columns:
                            file_counts = pd.to_numeric(data_filtered['COUNT_VARIATIONS'], errors='coerce').fillna(1)
                            data_filtered['COUNT_VARIATIONS'] = actual_counts.combine(file_counts, max)
                        else: 
                            data_filtered['COUNT_VARIATIONS'] = actual_counts
                            
                        data = data_filtered.drop_duplicates(subset=['PRODUCT_SET_SID'], keep='first')
                        data_has_warranty = all(c in data.columns for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION'])
                        
                        for c in ['NAME', 'BRAND', 'COLOR', 'SELLER_NAME', 'CATEGORY_CODE', 'LIST_VARIATIONS']:
                            if c in data.columns: data[c] = data[c].astype(str).fillna('')
                        if 'COLOR_FAMILY' not in data.columns: data['COLOR_FAMILY'] = ""

                        data_hash = df_hash(data) + country_validator.code
                        final_report, _ = cached_validate_products(data_hash, data, support_files, country_validator.code, data_has_warranty)

                        st.session_state.final_report = final_report
                        st.session_state.all_data_map = data
                        
                        # Save Parquet Cache
                        try:
                            data.to_parquet(cache_data_path)
                            final_report.to_parquet(cache_report_path)
                        except Exception as e:
                            logger.warning(f"Failed to write parquet cache: {e}")

                        st.session_state.last_processed_files = process_signature
                    else:
                        for e in errors: st.error(e)
                        st.session_state.last_processed_files = "error"
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.session_state.last_processed_files = "error"

# ==========================================
# POST-QC RESULTS SECTION
# ==========================================
if uploaded_files and st.session_state.file_mode == 'post_qc' and not st.session_state.post_qc_summary.empty:
    render_post_qc_section(support_files)

# ==========================================
# RESULTS SECTION
# ==========================================
if uploaded_files and not st.session_state.final_report.empty and st.session_state.file_mode != 'post_qc':
    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    app_df = fr[fr['Status'] == 'Approved']
    rej_df = fr[fr['Status'] == 'Rejected']

    st.header(":material/bar_chart: Validation Results", anchor=False)
    with st.container(border=True):
        cols = st.columns(5 if st.session_state.layout_mode == "wide" else 3)
        is_nigeria = st.session_state.get('selected_country') == 'Nigeria'
        multi_count = int(data['_IS_MULTI_COUNTRY'].sum()) if '_IS_MULTI_COUNTRY' in data.columns else 0

        metrics_config = [
            ("Total Products", len(data), JUMIA_COLORS['dark_gray']),
            ("Approved", len(app_df), JUMIA_COLORS['success_green']),
            ("Rejected", len(rej_df), JUMIA_COLORS['jumia_red']),
            ("Rejection Rate", f"{(len(rej_df)/len(data)*100) if len(data)>0 else 0:.1f}%", JUMIA_COLORS['primary_orange']),
            ("Multi-Country SKUs" if is_nigeria else "Common SKUs", multi_count if is_nigeria else st.session_state.intersection_count, JUMIA_COLORS['warning_yellow'] if is_nigeria else JUMIA_COLORS['medium_gray']),
        ]
        for i, (label, value, color) in enumerate(metrics_config):
            with cols[i % len(cols)]:
                st.markdown(f"""<div class="metric-card-inner" style='text-align: center; padding: 18px 12px; background: {JUMIA_COLORS['light_gray']}; border-radius: 8px; border-left: 4px solid {color};'><div class="metric-card-value" style='font-size: 28px; font-weight: 700; color: {color}; margin-bottom: 4px;'>{value}</div><div class="metric-card-label" style='font-size: 11px; color: {JUMIA_COLORS['medium_gray']}; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;'>{label}</div></div>""", unsafe_allow_html=True)

    st.subheader(":material/flag: Flags Breakdown", anchor=False)
    if not rej_df.empty:
        for title in rej_df['FLAG'].unique():
            df_flagged = rej_df[rej_df['FLAG'] == title]
            with st.expander(f"{title} ({len(df_flagged)})"):
                render_flag_expander(title, df_flagged, data, all(c in data.columns for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION']), support_files, country_validator)
    else: st.success("All products passed validation — no rejections found.")


# ==========================================
# SECTION 2: FAST IMAGE REVIEW GRID
# ==========================================
@st.fragment
def render_image_grid():
    if st.session_state.final_report.empty or st.session_state.file_mode == 'post_qc':
        return

    st.markdown("---")
    st.header(":material/pageview: Manual Image & Category Review", anchor=False)

    # 1. Hidden Bridge Input
    action_bridge = st.text_input("bridge", key="card_action_bridge", label_visibility="collapsed", placeholder="__CARD_ACT__")
    
    # 2. Process pending bridge actions
    if action_bridge:
        if _process_card_bridge_action(action_bridge, support_files):
            # Clear bridge via session state to avoid infinite loops
            st.session_state.card_action_bridge = ""
            st.rerun()

    fr = st.session_state.final_report
    quick_rej_sids = [k.replace("quick_rej_", "") for k in st.session_state.keys() if k.startswith("quick_rej_") and "reason" not in k]
    mask = (fr['Status'] == 'Approved') | (fr['ProductSetSid'].isin(quick_rej_sids))
    valid_grid_df = fr[mask]

    c_rev_1, c_rev_2, c_rev_3 = st.columns([1.5, 1.5, 2])
    with c_rev_1: search_n = st.text_input("Search by Name", placeholder="Product name...")
    with c_rev_2: search_sc = st.text_input("Search by Seller / Category", placeholder="Seller or Category...")
    with c_rev_3: st.session_state.grid_items_per_page = st.select_slider("Items per page", options=[20, 50, 100, 200], value=st.session_state.grid_items_per_page)

    # Slim grid merge
    review_data = pd.merge(valid_grid_df[['ProductSetSid']], 
                           st.session_state.all_data_map[[c for c in GRID_COLS if c in st.session_state.all_data_map.columns]], 
                           left_on='ProductSetSid', right_on='PRODUCT_SET_SID', how='left')

    if search_n: review_data = review_data[review_data['NAME'].astype(str).str.contains(search_n, case=False, na=False)]
    if search_sc:
        mc = review_data['CATEGORY'].astype(str).str.contains(search_sc, case=False, na=False) if 'CATEGORY' in review_data.columns else False
        ms = review_data['SELLER_NAME'].astype(str).str.contains(search_sc, case=False, na=False)
        review_data = review_data[mc | ms]

    items_per_page = st.session_state.grid_items_per_page
    total_pages = max(1, (len(review_data) + items_per_page - 1) // items_per_page)
    if st.session_state.grid_page >= total_pages: st.session_state.grid_page = 0

    page_data = review_data.iloc[st.session_state.grid_page * items_per_page : (st.session_state.grid_page + 1) * items_per_page]

    # Pre-fetch image warnings
    page_warnings = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_sid = {executor.submit(analyze_image_quality_cached, str(r.get('MAIN_IMAGE', '')).strip()): str(r['PRODUCT_SET_SID']) for _, r in page_data.iterrows()}
        for future in concurrent.futures.as_completed(future_to_sid):
            warns = future.result()
            if warns: page_warnings[future_to_sid[future]] = warns

    rejected_state = {sid: st.session_state[f"quick_rej_reason_{sid}"] 
                      for sid in page_data['PRODUCT_SET_SID'] if st.session_state.get(f"quick_rej_{sid}")}

    cols_per_row = 3 if st.session_state.layout_mode == "centered" else 4

    grid_html = build_fast_grid_html(page_data, support_files['flags_mapping'], st.session_state.selected_country, 
                                     page_warnings, rejected_state, cols_per_row, 
                                     st.session_state.grid_page, total_pages)

    components.html(grid_html, height=1200, scrolling=True)

    if st.session_state.get('do_scroll_top', False):
        st.components.v1.html("<script>window.parent.document.querySelector('.main').scrollTo({top: 0, behavior: 'smooth'});</script>", height=0)
        st.session_state.do_scroll_top = False


# ==========================================
# SECTION 3: EXPORTS
# ==========================================
@st.fragment
def render_exports_section():
    if st.session_state.final_report.empty or st.session_state.file_mode == 'post_qc':
        return

    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    app_df = fr[fr['Status'] == 'Approved']
    rej_df = fr[fr['Status'] == 'Rejected']
    c_code = st.session_state.selected_country[:2].upper()
    date_str = datetime.now().strftime('%Y-%m-%d')
    reasons_df = support_files.get('reasons', pd.DataFrame())

    st.markdown("---")
    st.markdown(f"""<div style='background: linear-gradient(135deg, {JUMIA_COLORS['primary_orange']}, {JUMIA_COLORS['secondary_orange']}); padding: 20px 24px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(246, 139, 30, 0.25);'><h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>Download Reports</h2><p style='color: rgba(255,255,255,0.9); margin: 6px 0 0 0; font-size: 13px;'>Export validation results in Excel or ZIP format</p></div>""", unsafe_allow_html=True)

    exports_config = [
        ("Final Report",  fr,     'assignment',   'Complete validation report with all statuses', lambda df: generate_smart_export(df, f"{c_code}_Final_{date_str}", 'simple', reasons_df)),
        ("Rejected Only", rej_df, 'block',        'Products that failed validation', lambda df: generate_smart_export(df, f"{c_code}_Rejected_{date_str}", 'simple', reasons_df)),
        ("Approved Only", app_df, 'check_circle', 'Products that passed validation', lambda df: generate_smart_export(df, f"{c_code}_Approved_{date_str}", 'simple', reasons_df)),
        ("Full Data",     data,   'database',     'Complete dataset with validation flags', lambda df: generate_smart_export(prepare_full_data_merged(df, fr), f"{c_code}_Full_{date_str}", 'full')),
    ]

    cols_count = 4 if st.session_state.layout_mode == "wide" else 2
    for i in range(0, len(exports_config), cols_count):
        cols = st.columns(cols_count)
        for j, col in enumerate(cols):
            if i + j < len(exports_config):
                title, df, icon, desc, func = exports_config[i + j]
                with col:
                    with st.container(border=True):
                        st.markdown(f"""<div style='text-align: center; margin-bottom: 15px;'><div style='font-size: 48px; margin-bottom: 8px;' class='material-symbols-outlined'>{icon}</div><div style='font-size: 18px; font-weight: 700;'>{title}</div><div style='font-size: 11px; margin-top: 4px; opacity: 0.7;'>{desc}</div><div style='background: {JUMIA_COLORS['light_gray']}; color: {JUMIA_COLORS['primary_orange']}; padding: 8px; border-radius: 6px; margin-top: 12px; font-weight: 600;'>{len(df):,} rows</div></div>""", unsafe_allow_html=True)
                        export_key = title
                        if export_key not in st.session_state.exports_cache:
                            if st.button("Generate", key=f"gen_{title}", type="primary", use_container_width=True, icon=":material/download:"):
                                with st.spinner(f"Generating {title}..."):
                                    res, fname, mime = func(df)
                                    st.session_state.exports_cache[export_key] = {"data": res.getvalue(), "fname": fname, "mime": mime}
                                st.rerun()
                        else:
                            cache = st.session_state.exports_cache[export_key]
                            st.download_button("Download", data=cache["data"], file_name=cache["fname"], mime=cache["mime"], use_container_width=True, type="primary", icon=":material/file_download:", key=f"dl_{title}")
                            if st.button("Clear", key=f"clr_{title}", use_container_width=True):
                                del st.session_state.exports_cache[export_key]
                                st.rerun()

# Call fragmented blocks
render_image_grid()
render_exports_section()
