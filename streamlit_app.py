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
import time
import hashlib

# --- IMPORTS FOR IMAGE ADVISOR ---
import requests
from PIL import Image

try:
    from postqc import detect_file_type, normalize_post_qc, run_checks as run_post_qc_checks, render_post_qc_section
except ImportError:
    pass

# -------------------------------------------------
# JUMIA THEME COLORS & GLOBAL CSS
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
GRID_COLS = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'SELLER_NAME', 'MAIN_IMAGE', 'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE', 'COLOR']

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
        import urllib.request, json
        url = f"https://open.er-api.com/v6/latest/USD"
        with urllib.request.urlopen(url, timeout=3) as r: data = json.loads(r.read())
        rate = data["rates"].get(cfg["code"], 1.0)
        return float(rate)
    except Exception:
        fallbacks = {"Kenya": 128.0, "Uganda": 3750.0, "Nigeria": 1550.0, "Ghana": 15.5, "Morocco": 10.1}
        return fallbacks.get(country, 1.0)

def format_local_price(usd_price, country: str) -> str:
    try:
        price = float(usd_price)
        if price <= 0: return ""
        cfg = COUNTRY_CURRENCY.get(country, {})
        rate = fetch_exchange_rate(country)
        local = price * rate
        symbol = cfg.get("symbol", "$")
        if cfg.get("code") in ("KES", "UGX", "NGN"): return f"{symbol} {local:,.0f}"
        else: return f"{symbol} {local:,.2f}"
    except (ValueError, TypeError): return ""

SPLIT_LIMIT = 9998

NEW_FILE_MAPPING = {
    'cod_productset_sid': 'PRODUCT_SET_SID',
    "2qz3wx4ec5rv6b7hnj8kl;'[]": 'PRODUCT_SET_SID',
    'dsc_name': 'NAME',
    'dsc_brand_name': 'BRAND',
    'cod_category_code': 'CATEGORY_CODE',
    'dsc_category_name': 'CATEGORY',
    'dsc_shop_seller_name': 'SELLER_NAME',
    'dsc_shop_active_country': 'ACTIVE_STATUS_COUNTRY',
    'cod_parent_sku': 'PARENTSKU',
    'color': 'COLOR',
    'colour': 'COLOR',
    'color_family': 'COLOR_FAMILY',
    'colour_family': 'COLOR_FAMILY',
    'colour family': 'COLOR_FAMILY',
    'color family': 'COLOR_FAMILY',
    'COLOUR FAMILY': 'COLOR_FAMILY',
    'list_seller_skus': 'SELLER_SKU',
    'image1': 'MAIN_IMAGE',
    'dsc_status': 'LISTING_STATUS',
    'dsc_shop_email': 'SELLER_EMAIL',
    'product_warranty': 'PRODUCT_WARRANTY',
    'warranty_duration': 'WARRANTY_DURATION',
    'warranty_address': 'WARRANTY_ADDRESS',
    'warranty_type': 'WARRANTY_TYPE',
    'count_variations': 'COUNT_VARIATIONS',
    'count variations': 'COUNT_VARIATIONS',
    'number of variations': 'COUNT_VARIATIONS',
    'list_variations': 'LIST_VARIATIONS',
    'list variations': 'LIST_VARIATIONS'
}

logger = logging.getLogger(__name__)

# -------------------------------------------------
# INITIALIZATION & CONTEXT
# -------------------------------------------------
if 'layout_mode' not in st.session_state: st.session_state.layout_mode = "wide"
if 'final_report' not in st.session_state: st.session_state.final_report = pd.DataFrame()
if 'all_data_map' not in st.session_state: st.session_state.all_data_map = pd.DataFrame()
if 'post_qc_summary' not in st.session_state: st.session_state.post_qc_summary = pd.DataFrame()
if 'post_qc_results' not in st.session_state: st.session_state.post_qc_results = {}
if 'post_qc_data' not in st.session_state: st.session_state.post_qc_data = pd.DataFrame()
if 'file_mode' not in st.session_state: st.session_state.file_mode = None
if 'intersection_sids' not in st.session_state: st.session_state.intersection_sids = set()
if 'intersection_count' not in st.session_state: st.session_state.intersection_count = 0
if 'grid_page' not in st.session_state: st.session_state.grid_page = 0
if 'grid_items_per_page' not in st.session_state: st.session_state.grid_items_per_page = 50
if 'main_toasts' not in st.session_state: st.session_state.main_toasts = []
if 'exports_cache' not in st.session_state: st.session_state.exports_cache = {}
if 'bg_executor' not in st.session_state: st.session_state.bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
if 'do_scroll_top' not in st.session_state: st.session_state.do_scroll_top = False
if 'display_df_cache' not in st.session_state: st.session_state.display_df_cache = {}

try: st.set_page_config(page_title="Product Tool", layout=st.session_state.layout_mode)
except: pass

st_yled.init()

# --- GLOBAL CSS ---
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

        :root {{
            --jumia-orange: {JUMIA_COLORS['primary_orange']};
            --jumia-red: {JUMIA_COLORS['jumia_red']};
            --jumia-dark: {JUMIA_COLORS['dark_gray']};
        }}
        header[data-testid="stHeader"] {{ background: transparent !important; }}
        div[data-testid="stStatusWidget"] {{ z-index: 9999999 !important; }}
        .stButton > button {{ border-radius: 4px; font-weight: 600; transition: all 0.3s ease; }}
        .stButton > button[kind="primary"] {{ background-color: {JUMIA_COLORS['primary_orange']} !important; border: none !important; color: white !important; }}
        .stButton > button[kind="primary"]:hover {{ background-color: {JUMIA_COLORS['secondary_orange']} !important; box-shadow: 0 4px 8px rgba(246, 139, 30, 0.3); transform: translateY(-1px); }}
        .stButton > button[kind="secondary"] {{ background-color: white !important; border: 2px solid {JUMIA_COLORS['primary_orange']} !important; color: {JUMIA_COLORS['primary_orange']} !important; }}
        .stButton > button[kind="secondary"]:hover {{ background-color: {JUMIA_COLORS['light_gray']} !important; }}
        div[data-testid="stMetricValue"] {{ color: {JUMIA_COLORS['dark_gray']}; font-weight: 700; }}
        div[data-testid="stMetricLabel"] {{ color: {JUMIA_COLORS['medium_gray']}; }}

        ::-webkit-scrollbar {{ width: 18px !important; height: 18px !important; }}
        ::-webkit-scrollbar-track {{ background: {JUMIA_COLORS['light_gray']}; border-radius: 8px; }}
        ::-webkit-scrollbar-thumb {{ background: {JUMIA_COLORS['medium_gray']}; border-radius: 8px; border: 3px solid {JUMIA_COLORS['light_gray']}; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {JUMIA_COLORS['primary_orange']}; }}
        * {{ scrollbar-width: auto; scrollbar-color: {JUMIA_COLORS['medium_gray']} {JUMIA_COLORS['light_gray']}; }}

        div[data-baseweb="slider"] div[role="slider"] {{ height: 24px !important; width: 24px !important; border: 4px solid {JUMIA_COLORS['primary_orange']} !important; cursor: pointer !important; }}
        div[data-baseweb="slider"] > div > div {{ height: 12px !important; }}

        @media (prefers-color-scheme: dark) {{
            div[data-testid="stMetricValue"] {{ color: #F5F5F5 !important; }}
            div[data-testid="stMetricLabel"] {{ color: #B0B0B0 !important; }}
            h1, h2, h3 {{ color: #F5F5F5 !important; }}
            div[data-testid="stExpander"] summary {{ background-color: #2a2a2e !important; color: #F5F5F5 !important; }}
            div[data-testid="stExpander"] summary p, div[data-testid="stExpander"] summary span, div[data-testid="stExpander"] summary div {{ color: #F5F5F5 !important; }}
            div[data-testid="stDataFrame"] * {{ color: #F5F5F5 !important; }}
            .stDataFrame th {{ background-color: #2a2a2e !important; color: #F5F5F5 !important; }}
            .metric-card-inner {{ background: #2a2a2e !important; }}
            .metric-card-value {{ color: inherit !important; }}
            .metric-card-label {{ color: #B0B0B0 !important; }}
            .color-badge {{ background: #3a3a3e !important; border-color: #555 !important; color: #E0E0E0 !important; }}
            div[style*="position: sticky"], div[style*="position:sticky"] {{ background-color: #0e1117 !important; border-bottom-color: #2a2a2e !important; }}
            .stCaption, div[data-testid="stCaptionContainer"] p {{ color: #B0B0B0 !important; }}
            .prod-meta-text {{ color: #B0B0B0 !important; }}
            .prod-brand-text {{ color: {JUMIA_COLORS['secondary_orange']} !important; }}
            ::-webkit-scrollbar-track {{ background: #1e1e1e; border-color: #1e1e1e; }}
            ::-webkit-scrollbar-thumb {{ background: #555; border-color: #1e1e1e; }}
            ::-webkit-scrollbar-thumb:hover {{ background: {JUMIA_COLORS['primary_orange']}; }}
        }}

        div[data-testid="stExpander"] {{ border: 1px solid {JUMIA_COLORS['border_gray']}; border-radius: 8px; }}
        div[data-testid="stExpander"] summary {{ background-color: {JUMIA_COLORS['light_gray']}; padding: 12px; border-radius: 8px 8px 0 0; }}
        h1, h2, h3 {{ color: {JUMIA_COLORS['dark_gray']} !important; }}
        div[data-baseweb="segmented-control"] button {{ border-radius: 4px; }}
        div[data-baseweb="segmented-control"] button[aria-pressed="true"] {{ background-color: {JUMIA_COLORS['primary_orange']} !important; color: white !important; }}
        input[type="checkbox"]:checked {{ background-color: {JUMIA_COLORS['primary_orange']} !important; border-color: {JUMIA_COLORS['primary_orange']} !important; }}
        div[data-testid="stCheckbox"] {{ margin-top: 5px; margin-bottom: 5px; }}

        .prod-name-tip {{ position: relative; cursor: default; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; display: block; margin-bottom: 5px; }}
        .prod-name-tip::after {{ content: attr(data-full); display: none; position: absolute; bottom: 120%; left: 0; background: {JUMIA_COLORS['dark_gray']}; color: #fff; padding: 7px 11px; border-radius: 6px; font-size: 12px; font-weight: 500; white-space: normal; max-width: 260px; line-height: 1.45; z-index: 9999; box-shadow: 0 4px 14px rgba(0,0,0,0.28); pointer-events: none; }}
        .prod-name-tip:hover::after {{ display: block; }}

        .color-badge {{ display: inline-block; padding: 2px 9px; border-radius: 12px; font-size: 10px; font-weight: 600; letter-spacing: 0.3px; border: 1px solid {JUMIA_COLORS['border_gray']}; background: {JUMIA_COLORS['light_gray']}; color: {JUMIA_COLORS['dark_gray']}; margin-bottom: 5px; cursor: default; transition: background 0.18s, color 0.18s, border-color 0.18s; }}
        .color-badge:hover {{ background: {JUMIA_COLORS['primary_orange']}; color: #fff; border-color: {JUMIA_COLORS['primary_orange']}; }}

        .batch-info-box {{ padding: 10px 14px; border-radius: 6px; border-left: 4px solid {JUMIA_COLORS['primary_orange']}; margin-top: 10px; }}
        div[data-testid="stVerticalBlock"] div[data-testid="stPopover"] {{ overflow: visible !important; }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{ overflow: visible !important; }}

        .rejected-card-overlay {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(231, 60, 23, 0.9); color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 12px; white-space: nowrap; display: flex; align-items: center; }}
        
        div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button {{ white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; font-size: 11px !important; padding: 6px 4px !important; min-height: 36px !important; line-height: 1.2 !important; }}
    </style>
""", unsafe_allow_html=True)

def get_default_country():
    try:
        lang = st.context.headers.get("Accept-Language", "")
        if "KE" in lang: return "Kenya"
        if "UG" in lang: return "Uganda"
        if "NG" in lang: return "Nigeria"
        if "GH" in lang: return "Ghana"
        if "MA" in lang: return "Morocco"
    except: pass
    return "Kenya"

if 'selected_country' not in st.session_state: st.session_state.selected_country = get_default_country()

if st.session_state.main_toasts:
    for msg in st.session_state.main_toasts:
        if isinstance(msg, tuple): st.toast(msg[0], icon=msg[1])
        else: st.toast(msg)
    st.session_state.main_toasts.clear()

# -------------------------------------------------
# UTILITIES & EXTRACTION
# -------------------------------------------------
def clean_category_code(code) -> str:
    try:
        if pd.isna(code): return ""
        s = str(code).strip()
        if '.' in s: s = s.split('.')[0]
        return s
    except: return str(code).strip()

def normalize_text(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).lower().strip()
    noise = r'\b(new|sale|original|genuine|authentic|official|premium|quality|best|hot|2024|2025)\b'
    text = re.sub(noise, '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def create_match_key(row: pd.Series) -> str:
    name = normalize_text(row.get('NAME', ''))
    brand = normalize_text(row.get('BRAND', ''))
    color = normalize_text(row.get('COLOR', ''))
    return f"{brand}|{name}|{color}"

def df_hash(df: pd.DataFrame) -> str:
    try:
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    except Exception:
        return hashlib.md5(str(df.shape).encode()).hexdigest()

COLOR_PATTERNS = {
    'red': ['red', 'crimson', 'scarlet', 'maroon', 'burgundy', 'wine', 'ruby'],
    'blue': ['blue', 'navy', 'royal', 'sky', 'azure', 'cobalt', 'sapphire'],
    'green': ['green', 'lime', 'olive', 'emerald', 'mint', 'forest', 'jade'],
    'black': ['black', 'onyx', 'ebony', 'jet', 'charcoal', 'midnight'],
    'white': ['white', 'ivory', 'cream', 'pearl', 'snow', 'alabaster'],
    'gray': ['gray', 'grey', 'silver', 'slate', 'ash', 'graphite'],
    'yellow': ['yellow', 'gold', 'golden', 'amber', 'lemon', 'mustard'],
    'orange': ['orange', 'tangerine', 'peach', 'coral', 'apricot'],
    'pink': ['pink', 'rose', 'magenta', 'fuchsia', 'salmon', 'blush'],
    'purple': ['purple', 'violet', 'lavender', 'plum', 'mauve', 'lilac'],
    'brown': ['brown', 'tan', 'beige', 'khaki', 'chocolate', 'coffee', 'bronze'],
    'multicolor': ['multicolor', 'multicolour', 'multi-color', 'rainbow', 'mixed']
}

COLOR_VARIANT_TO_BASE = {}
for base_color, variants in COLOR_PATTERNS.items():
    for variant in variants: COLOR_VARIANT_TO_BASE[variant] = base_color

@dataclass
class ProductAttributes:
    base_name: str; colors: Set[str]; sizes: Set[str]; storage: Set[str]; memory: Set[str]; quantities: Set[str]; raw_name: str
    def get_variant_key(self) -> str:
        parts = [self.base_name]
        if self.colors: parts.append("_color_" + "_".join(sorted(self.colors)))
        if self.sizes: parts.append("_size_" + "_".join(sorted(self.sizes)))
        if self.storage: parts.append("_storage_" + "_".join(sorted(self.storage)))
        if self.memory: parts.append("_memory_" + "_".join(sorted(self.memory)))
        if self.quantities: parts.append("_qty_" + "_".join(sorted(self.quantities)))
        return "|".join(parts).lower()
    def get_base_key(self) -> str: return self.base_name.lower()

def extract_colors(text: str, explicit_color: Optional[str] = None) -> Set[str]:
    colors = set()
    text_lower = str(text).lower() if text else ""
    if explicit_color and pd.notna(explicit_color):
        color_lower = str(explicit_color).lower().strip()
        for variant, base in COLOR_VARIANT_TO_BASE.items():
            if variant in color_lower: colors.add(base)
    for variant, base in COLOR_VARIANT_TO_BASE.items():
        if re.search(r'\b' + re.escape(variant) + r'\b', text_lower): colors.add(base)
    return colors

def extract_sizes(text: str) -> Set[str]:
    sizes = set()
    text_lower = str(text).lower() if text else ""
    size_map = { r'\bxxs\b|2xs': 'xxs', r'\bxs\b|xsmall': 'xs', r'\bs\b|small': 'small', r'\bm\b|medium': 'medium', r'\bl\b|large': 'large', r'\bxl\b|xlarge': 'xl', r'\bxxl\b|2xl': 'xxl', r'\bxxxl\b|3xl': 'xxxl' }
    for pattern, size in size_map.items():
        if re.search(pattern, text_lower): sizes.add(size)
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\s*(?:inch|inches|")\b', text_lower): sizes.add(f"{match.group(1)}inch")
    return sizes

def extract_storage(text: str) -> Set[str]:
    storage = set()
    for match in re.finditer(r'\b(\d+)\s*(?:gb|tb)\b', str(text).lower() if text else ""): storage.add(f"{match.group(1)}{'tb' if 'tb' in match.group(0) else 'gb'}")
    return storage

def extract_memory(text: str) -> Set[str]:
    memory = set()
    for match in re.finditer(r'\b(\d+)\s*(?:gb|mb)\s*(?:ram|memory|ddr)\b', str(text).lower() if text else ""):
        if 2 <= int(match.group(1)) <= 128: memory.add(f"{match.group(1)}gb")
    return memory

def extract_quantities(text: str) -> Set[str]:
    quantities = set()
    patterns = [r'\b(\d+)[- ]?pack\b', r'\bpack\s+of\s+(\d+)\b', r'\b(\d+)[- ]?(?:pieces?|pcs?)\b']
    for pattern in patterns:
        for match in re.finditer(pattern, str(text).lower() if text else ""): quantities.add(f"{match.group(1)}pack")
    return quantities

def remove_attributes(text: str) -> str:
    base = str(text).lower() if text else ""
    for variant in COLOR_VARIANT_TO_BASE.keys(): base = re.sub(r'\b' + re.escape(variant) + r'\b', '', base)
    base = re.sub(r'\b(?:xxs|xs|small|medium|large|xl|xxl|xxxl)\b', '', base)
    base = re.sub(r'\b\d+\s*(?:gb|tb|inch|inches|"|ram|memory|ddr|pack|piece|pcs)\b', '', base)
    for word in ['new', 'original', 'genuine', 'authentic', 'official', 'premium', 'quality', 'best', 'hot', 'sale', 'promo', 'deal']:
        base = re.sub(r'\b' + word + r'\b', '', base)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', base)).strip()

def extract_product_attributes(name: str, explicit_color: Optional[str] = None, brand: Optional[str] = None) -> ProductAttributes:
    name_str = str(name).strip() if pd.notna(name) else ""
    attrs = ProductAttributes(base_name="", colors=extract_colors(name_str, explicit_color), sizes=extract_sizes(name_str), storage=extract_storage(name_str), memory=extract_memory(name_str), quantities=extract_quantities(name_str), raw_name=name_str)
    base_name = remove_attributes(name_str)
    if brand and pd.notna(brand):
        brand_lower = str(brand).lower().strip()
        if brand_lower not in base_name and brand_lower not in ['generic', 'fashion']: base_name = f"{brand_lower} {base_name}"
    attrs.base_name = base_name.strip()
    return attrs

# -------------------------------------------------
# LOCAL EXCEL DATA LOADING HELPERS
# -------------------------------------------------
def load_txt_file(filename: str) -> List[str]:
    try:
        if not os.path.exists(os.path.abspath(filename)): return []
        with open(filename, 'r', encoding='utf-8') as f: return [line.strip() for line in f if line.strip()]
    except Exception: return []

@st.cache_data(ttl=3600)
def load_excel_file(filename: str, column: Optional[str] = None):
    try:
        if not os.path.exists(filename): return [] if column else pd.DataFrame()
        df = pd.read_excel(filename, engine='openpyxl', dtype=str)
        df.columns = df.columns.str.strip()
        if column and column in df.columns: return df[column].apply(clean_category_code).tolist()
        return df
    except Exception: return [] if column else pd.DataFrame()

def safe_excel_read(filename: str, sheet_name, usecols=None) -> pd.DataFrame:
    if not os.path.exists(filename):
        logger.warning(f"Local file not found: {filename}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols=usecols, engine='openpyxl', dtype=str)
        return df.dropna(how='all')
    except Exception as e:
        logger.error(f"Error reading tab '{sheet_name}' from {filename}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_prohibited_from_local() -> Dict[str, List[Dict]]:
    FILE_NAME = "Prohibbited.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    prohibited_by_country = {}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if df.empty:
                prohibited_by_country[tab] = []
                continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            keyword_col = next((c for c in df.columns if 'keyword' in c or 'prohibited' in c or 'name' in c), df.columns[0])
            category_col = next((c for c in df.columns if 'cat' in c), None)
            country_rules = []
            for _, row in df.iterrows():
                keyword = str(row.get(keyword_col, '')).strip().lower()
                if not keyword or keyword == 'nan' or keyword == 'keywords': continue
                categories = set()
                if category_col:
                    cats_raw = str(row.get(category_col, '')).strip()
                    if cats_raw and cats_raw.lower() != 'nan':
                        split_cats = re.split(r'[,\n]+', cats_raw)
                        categories.update([clean_category_code(c.strip()) for c in split_cats if c.strip()])
                country_rules.append({'keyword': keyword, 'categories': categories})
            prohibited_by_country[tab] = country_rules
        except Exception as e:
            logger.error(f"⚠️ **File Error:** Failed to load {FILE_NAME} - Tab **{tab}**. (Error: {e})")
            prohibited_by_country[tab] = []
    return prohibited_by_country

@st.cache_data(ttl=3600)
def load_restricted_brands_from_local() -> Dict[str, List[Dict]]:
    FILE_NAME = "Restricted_Brands.xlsx"
    COUNTRY_TABS = {"Kenya": "KE", "Uganda": "UG", "Nigeria": "NG", "Ghana": "GH", "Morocco": "MA"}
    config_by_country = {}
    for country_name, tab_name in COUNTRY_TABS.items():
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab_name)
            if df.empty:
                config_by_country[country_name] = []
                continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            brand_dict = {}
            for _, row in df.iterrows():
                brand = str(row.get('brand', '')).strip()
                if not brand or brand.lower() == 'nan': continue
                b_lower = brand.lower()
                if b_lower not in brand_dict:
                    brand_dict[b_lower] = {'brand_raw': brand, 'sellers': set(), 'categories': set(), 'variations': set(), 'has_blank_category': False}
                sellers_raw = str(row.get('approved sellers', '')).strip().lower()
                if sellers_raw != 'nan' and sellers_raw:
                    brand_dict[b_lower]['sellers'].update([s.strip() for s in sellers_raw.split(',') if s.strip()])
                cats_raw = str(row.get('categories', '')).strip()
                if cats_raw == 'nan' or not cats_raw:
                    brand_dict[b_lower]['has_blank_category'] = True
                else:
                    brand_dict[b_lower]['categories'].update([clean_category_code(c.strip()) for c in cats_raw.split(',') if c.strip()])
                vars_raw = str(row.get('variations', '')).strip().lower()
                if vars_raw != 'nan' and vars_raw:
                    brand_dict[b_lower]['variations'].update([v.strip() for v in vars_raw.split(',') if v.strip()])
            country_rules = []
            for b_lower, data in brand_dict.items():
                if data['has_blank_category']:
                    data['categories'] = set()
                country_rules.append({'brand': b_lower, 'brand_raw': data['brand_raw'], 'sellers': data['sellers'], 'categories': data['categories'], 'variations': list(data['variations'])})
            config_by_country[country_name] = country_rules
        except Exception as e:
            logger.error(f"⚠️ **File Error:** Failed to load {FILE_NAME} - Tab **{tab_name}**. (Error: {e})")
            config_by_country[country_name] = []
    return config_by_country

@st.cache_data(ttl=3600)
def load_refurb_data_from_local() -> dict:
    FILE_NAME = "Refurb.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    result = {"sellers": {}, "categories": {"Phones": set(), "Laptops": set()}, "keywords": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab, usecols=[0, 1])
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                phones_set = set(df.iloc[:, 0].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "phones"}
                laptops_set = set(df.iloc[:, 1].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "laptops"}
                result["sellers"][tab] = {"Phones": phones_set, "Laptops": laptops_set}
        except Exception as e:
            logger.error(f"⚠️ **File Error:** Failed to load {FILE_NAME} - Sellers Tab **{tab}**. (Error: {e})")
            result["sellers"][tab] = {"Phones": set(), "Laptops": set()}
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories", usecols=[0, 1])
        if df_cats.empty: df_cats = safe_excel_read(FILE_NAME, sheet_name="Categries", usecols=[0, 1])
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            result["categories"]["Phones"] = {clean_category_code(c) for c in df_cats.iloc[:, 0].dropna().astype(str) if c.strip() and c.strip().lower() not in ("phones", "phone", "nan")}
            result["categories"]["Laptops"] = {clean_category_code(c) for c in df_cats.iloc[:, 1].dropna().astype(str) if c.strip() and c.strip().lower() not in ("laptops", "laptop", "nan")}
    except Exception as e: logger.error(f"Error loading {FILE_NAME}: {e}")
    try:
        df_names = safe_excel_read(FILE_NAME, sheet_name="Name", usecols=[0])
        if not df_names.empty:
            first_col = df_names.columns[0]
            result["keywords"] = {k for k in df_names[first_col].dropna().astype(str).str.strip().str.lower() if k and k not in ("name", "keyword", "keywords", "words", "nan")}
    except Exception as e:
        logger.error(f"Error loading {FILE_NAME}: {e}")
        result["keywords"] = {"refurb", "refurbished", "renewed"}
    return result

@st.cache_data(ttl=3600)
def load_perfume_data_from_local() -> Dict:
    FILE_NAME = "Perfume.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    result = {"sellers": {}, "keywords": set(), "category_codes": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                seller_col = next((c for c in df.columns if 'seller' in c.lower()), df.columns[0])
                sellers = set(df[seller_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "sellername", "seller name", "seller"])]))
                result["sellers"][tab] = sellers
        except Exception as e:
            logger.error(f"Error: {e}")
            result["sellers"][tab] = set()
    try:
        df_kw = safe_excel_read(FILE_NAME, sheet_name="Keywords")
        if not df_kw.empty:
            df_kw.columns = [str(c).strip() for c in df_kw.columns]
            kw_col = next((c for c in df_kw.columns if 'brand' in c.lower() or 'keyword' in c.lower()), df_kw.columns[0])
            keywords = set(df_kw[kw_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "brand", "keyword", "keywords"])]))
            result["keywords"] = keywords
    except Exception as e: result["keywords"] = set()
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if 'cat' in c.lower()), df_cats.columns[0])
            cat_codes = set(df_cats[cat_col].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])]))
            result["category_codes"] = cat_codes
    except Exception as e: result["category_codes"] = set()
    return result

@st.cache_data(ttl=3600)
def load_books_data_from_local() -> Dict:
    FILE_NAME = "Books_sellers.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    result = {"sellers": {}, "category_codes": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                seller_col = next((c for c in df.columns if 'seller' in c.lower()), df.columns[0])
                sellers = set(df[seller_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "sellername", "seller name", "seller"])]))
                result["sellers"][tab] = sellers
        except Exception as e: result["sellers"][tab] = set()
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if 'cat' in c.lower()), df_cats.columns[0])
            cat_codes = set(df_cats[cat_col].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])]))
            result["category_codes"] = cat_codes
    except Exception as e: result["category_codes"] = set()
    return result

@st.cache_data(ttl=3600)
def load_jerseys_from_local() -> Dict:
    FILE_NAME = "Jersey_validation.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    result: Dict = {"keywords": {tab: set() for tab in COUNTRY_TABS}, "exempted": {tab: set() for tab in COUNTRY_TABS}, "categories": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                kw_col = next((c for c in df.columns if "keyword" in c.lower()), df.columns[0])
                keywords = set(df[kw_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "keywords", "keyword"])]))
                result["keywords"][tab] = keywords
                ex_col = next((c for c in df.columns if "exempt" in c.lower() or "seller" in c.lower()), None)
                if ex_col:
                    exempted = set(df[ex_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "exempted sellers", "seller"])]))
                    result["exempted"][tab] = exempted
        except Exception as e: logger.error(f"Error: {e}")
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip().lower() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if "cat" in c), df_cats.columns[0])
            result["categories"] = set(df_cats[cat_col].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])]))
    except Exception as e: logger.error(f"Error: {e}")
    return result

@st.cache_data(ttl=3600)
def load_suspected_fake_from_local() -> pd.DataFrame:
    try:
        if os.path.exists('suspected_fake.xlsx'): return pd.read_excel('suspected_fake.xlsx', sheet_name=0, engine='openpyxl', dtype=str)
    except Exception as e: logger.warning(f"Error: {e}")
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
        'Generic BRAND Issues': ('1000007 - Other Reason', "Use correct brand instead of Generic/Fashion. Apply for brand approval if needed."),
        'Fashion brand issues': ('1000007 - Other Reason', "Use correct brand instead of Fashion. Apply for brand approval if needed."),
        'BRAND name repeated in NAME': ('1000007 - Other Reason', "Brand name should not be repeated in product name."),
        'Generic branded products with genuine brands': ('1000007 - Other Reason', "Use the displayed brand on the product instead of Generic."),
        'Missing COLOR': ('1000005 - Kindly confirm the actual product colour', "Product color must be mentioned in title/color tab."),
        'Duplicate product': ('1000007 - Other Reason', "This product is a duplicate."),
        'Wrong Variation': ('1000039 - Product Poorly Created. Each Variation Of This Product Should Be Created Uniquely (Not Authorized) (Not Authorized)', "Create different SKUs instead of variations (variations only for sizes)."),
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
                custom_mapping = {}
                for _, row in df.iterrows():
                    flag = str(row['flag']).strip()
                    reason = str(row['reason']).strip()
                    comment = str(row['comment']).strip()
                    if flag and flag.lower() != 'nan': custom_mapping[flag] = (reason, comment)
                if custom_mapping: return custom_mapping
    except Exception as e: logger.error(f"Error loading external {filename}: {e}")
    return default_mapping

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
            if path_col:
                fashion_paths = df_fash[path_col].dropna().astype(str).tolist()
    except Exception as e:
        logger.error(f"Error loading fashion brands: {e}")

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
        'reasons': load_excel_file('reasons.xlsx'),
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

    def should_skip_validation(self, validation_name: str) -> bool: return validation_name in self.skip_validations
    def ensure_status_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and 'Status' not in df.columns: df['Status'] = 'Approved'
        return df

# -------------------------------------------------
# DATA PREPROCESSING
# -------------------------------------------------
def standardize_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    map_lower = {k.lower(): v for k, v in NEW_FILE_MAPPING.items()}
    renamed = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in map_lower: renamed[col] = map_lower[col_lower]
        else: renamed[col] = col.upper()
    df = df.rename(columns=renamed)
    for col in ['ACTIVE_STATUS_COUNTRY', 'CATEGORY_CODE', 'BRAND', 'TAX_CLASS', 'NAME', 'SELLER_NAME']:
        if col in df.columns: df[col] = df[col].astype(str)
    return df

def validate_input_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = [f"Missing: {f}" for f in ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY_CODE', 'ACTIVE_STATUS_COUNTRY'] if f not in df.columns]
    return len(errors) == 0, errors

MULTI_COUNTRY_VALUES = {'MULTIPLE', 'MULTI'}

def filter_by_country(df: pd.DataFrame, country_validator: CountryValidator) -> Tuple[pd.DataFrame, List[str]]:
    if 'ACTIVE_STATUS_COUNTRY' not in df.columns: return df, []
    s = df['ACTIVE_STATUS_COUNTRY'].astype(str).str.strip().str.upper().str.replace(r'^JUMIA-', '', regex=True)
    df['ACTIVE_STATUS_COUNTRY'] = s

    if country_validator.code == 'NG':
        is_ng = df['ACTIVE_STATUS_COUNTRY'] == 'NG'
        is_multi = df['ACTIVE_STATUS_COUNTRY'].isin(MULTI_COUNTRY_VALUES)
        filtered = df[is_ng | is_multi].copy()
        filtered['_IS_MULTI_COUNTRY'] = is_multi[filtered.index]
    else:
        filtered = df[df['ACTIVE_STATUS_COUNTRY'] == country_validator.code].copy()
        filtered['_IS_MULTI_COUNTRY'] = False

    detected_names = []
    if filtered.empty:
        detected_codes = [c for c in df['ACTIVE_STATUS_COUNTRY'].unique() if str(c).strip() and str(c).strip().lower() != 'nan']
        emoji_map = {"KE": "Kenya", "UG": "Uganda", "NG": "Nigeria", "GH": "Ghana", "MA": "Morocco"}
        detected_names = [emoji_map.get(c, f"'{c}'") for c in detected_codes]
    return filtered, detected_names

def propagate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for col in ['COLOR_FAMILY', 'PRODUCT_WARRANTY', 'WARRANTY_DURATION', 'WARRANTY_ADDRESS', 'WARRANTY_TYPE', 'COUNT_VARIATIONS', 'LIST_VARIATIONS']:
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
    if not {'NAME', 'BRAND', 'SELLER_NAME', 'CATEGORY_CODE'}.issubset(data.columns) or not country_rules: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d['_name_lower'] = d['NAME'].astype(str).str.lower().fillna('')
    d['_brand_lower'] = d['BRAND'].astype(str).str.lower().str.strip().fillna('')
    d['_seller_lower'] = d['SELLER_NAME'].astype(str).str.lower().str.strip().fillna('')
    d['_cat_clean'] = d['CATEGORY_CODE'].apply(clean_category_code)
    flagged_indices = set()
    comment_map = {}
    match_details = {}
    for rule in country_rules:
        brand_name = rule['brand']
        brand_raw = rule['brand_raw']
        brand_pattern = r'(?<!\w)' + re.escape(brand_name) + r'(?!\w)'
        main_brand_matches = (d['_brand_lower'] == brand_name)
        main_name_matches = d['_name_lower'].str.contains(brand_pattern, regex=True, na=False)
        current_match_mask = main_brand_matches | main_name_matches
        for idx in d[main_brand_matches].index: match_details[idx] = ('main_brand', brand_raw)
        for idx in d[main_name_matches & ~main_brand_matches].index: match_details[idx] = ('main_name', brand_raw)
        if rule['variations']:
            sorted_vars = sorted(rule['variations'], key=len, reverse=True)
            var_pattern = r'(?<!\w)(' + '|'.join([re.escape(v) for v in sorted_vars]) + r')(?!\w)'
            var_brand_matches = d['_brand_lower'].str.contains(var_pattern, regex=True, na=False)
            var_name_matches = d['_name_lower'].str.contains(var_pattern, regex=True, na=False)
            for idx in d[var_brand_matches | var_name_matches].index:
                if idx not in match_details:
                    text_to_check = d.loc[idx, '_brand_lower'] if var_brand_matches[idx] else d.loc[idx, '_name_lower']
                    for var in sorted_vars:
                        if var in text_to_check:
                            match_details[idx] = ('variation', f"{brand_raw} (as '{var}')")
                            break
            current_match_mask = current_match_mask | var_brand_matches | var_name_matches
        if not current_match_mask.any(): continue
        current_match = d[current_match_mask]
        if rule['categories']: current_match = current_match[current_match['_cat_clean'].isin(rule['categories'])]
        if current_match.empty: continue
        rejected = current_match[~current_match['_seller_lower'].isin(rule['sellers'])]
        if not rejected.empty:
            for idx in rejected.index:
                flagged_indices.add(idx)
                match_type, match_info = match_details.get(idx, ('unknown', brand_raw))
                seller_status = "Seller not in approved list" if rule['sellers'] else "No sellers approved"
                comment_map[idx] = f"Restricted Brand: {match_info} - {seller_status}"
    if not flagged_indices: return pd.DataFrame(columns=data.columns)
    result = data.loc[list(flagged_indices)].copy()
    result['Comment_Detail'] = result.index.map(comment_map)
    return result.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_prohibited_products(data: pd.DataFrame, prohibited_rules: List[Dict]) -> pd.DataFrame:
    if not {'NAME', 'CATEGORY_CODE'}.issubset(data.columns) or not prohibited_rules: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d['_name_lower'] = d['NAME'].astype(str).str.lower().fillna('')
    d['_cat_clean'] = d['CATEGORY_CODE'].apply(clean_category_code)
    flagged_indices = set()
    comment_map = {}
    name_replacements = {}
    for rule in prohibited_rules:
        keyword = rule['keyword']
        target_cats = rule['categories']
        pattern = re.compile(r'(?<!\w)' + re.escape(keyword) + r'(?!\w)', re.IGNORECASE)
        match_mask = d['_name_lower'].str.contains(pattern, regex=True, na=False)
        if not match_mask.any(): continue
        current_match = d[match_mask]
        if target_cats: current_match = current_match[current_match['_cat_clean'].isin(target_cats)]
        if current_match.empty: continue
        for idx in current_match.index:
            flagged_indices.add(idx)
            existing_comment = comment_map.get(idx, "Prohibited:")
            if keyword not in existing_comment: comment_map[idx] = f"{existing_comment} {keyword},"
            raw_name = str(d.loc[idx, 'NAME'])
            highlighted = pattern.sub(lambda m: f"[!]{m.group(0)}[!]", raw_name)
            name_replacements[idx] = highlighted
    if not flagged_indices: return pd.DataFrame(columns=data.columns)
    result = data.loc[list(flagged_indices)].copy()
    result['Comment_Detail'] = result.index.map(lambda i: comment_map[i].rstrip(','))
    for idx, new_name in name_replacements.items(): result.loc[idx, 'NAME'] = new_name
    return result.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_suspected_fake_products(data: pd.DataFrame, suspected_fake_df: pd.DataFrame, fx_rate: float) -> pd.DataFrame:
    if not all(c in data.columns for c in ['CATEGORY_CODE', 'BRAND', 'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE']) or suspected_fake_df.empty: return pd.DataFrame(columns=data.columns)
    try:
        ref_data = suspected_fake_df.copy()
        brand_cat_price = {}
        for brand in [c for c in ref_data.columns if c not in ['Unnamed: 0', 'Brand', 'Price'] and pd.notna(c)]:
            try:
                pt = pd.to_numeric(ref_data[brand].iloc[0], errors='coerce')
                if pd.isna(pt) or pt <= 0: continue
            except: continue
            for cat in ref_data[brand].iloc[1:].dropna():
                cat_base = str(cat).strip().split('.')[0]
                if cat_base and cat_base.lower() != 'nan': brand_cat_price[(brand.strip().lower(), cat_base)] = pt
        if not brand_cat_price: return pd.DataFrame(columns=data.columns)
        d = data.copy()
        d['price_to_use'] = pd.to_numeric(d['GLOBAL_SALE_PRICE'].where(d['GLOBAL_SALE_PRICE'].notna() & (pd.to_numeric(d['GLOBAL_SALE_PRICE'], errors='coerce') > 0), d['GLOBAL_PRICE']), errors='coerce').fillna(0)
        d['BRAND_LOWER'] = d['BRAND'].astype(str).str.strip().str.lower()
        d['CAT_BASE'] = d['CATEGORY_CODE'].apply(clean_category_code)
        prices = d['price_to_use'].values
        brands = d['BRAND_LOWER'].values
        cats = d['CAT_BASE'].values
        d['is_fake'] = [p < brand_cat_price.get((b, c), -1) for p, b, c in zip(prices, brands, cats)]
        return d[d['is_fake'] == True][data.columns].drop_duplicates(subset=['PRODUCT_SET_SID'])
    except: return pd.DataFrame(columns=data.columns)

def check_refurb_seller_approval(data: pd.DataFrame, refurb_data: dict, country_code: str) -> pd.DataFrame:
    required = {'PRODUCT_SET_SID', 'CATEGORY_CODE', 'SELLER_NAME', 'NAME'}
    if not required.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    phone_cats = refurb_data.get("categories", {}).get("Phones", set())
    laptop_cats = refurb_data.get("categories", {}).get("Laptops", set())
    keywords = refurb_data.get("keywords", set())
    sellers = refurb_data.get("sellers", {}).get(country_code, {})
    if not phone_cats and not laptop_cats: return pd.DataFrame(columns=data.columns)
    if not keywords: return pd.DataFrame(columns=data.columns)
    kw_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r')\b', re.IGNORECASE)
    d = data.copy()
    d['_cat'] = d['CATEGORY_CODE'].apply(clean_category_code)
    d['_seller'] = d['SELLER_NAME'].astype(str).str.strip().str.lower()
    d['_name'] = d['NAME'].astype(str).str.strip()
    is_phone = d['_cat'].isin(phone_cats)
    is_laptop = d['_cat'].isin(laptop_cats)
    in_scope = is_phone | is_laptop
    has_keyword = d['_name'].str.contains(kw_pattern, na=False)
    approved_phones = sellers.get("Phones", set())
    approved_laptops = sellers.get("Laptops", set())
    not_approved = ((is_phone & ~d['_seller'].isin(approved_phones)) | (is_laptop & ~d['_seller'].isin(approved_laptops)))
    flagged = d[in_scope & has_keyword & not_approved].copy()
    if not flagged.empty:
        def build_comment(row):
            ptype = "Phone" if row['_cat'] in phone_cats else "Laptop"
            match = kw_pattern.search(row['_name'])
            kw_found = match.group(0) if match else "?"
            return f"Unapproved {ptype} refurb seller — keyword '{kw_found}' in name (cat: {row['_cat']})"
        flagged['Comment_Detail'] = flagged.apply(build_comment, axis=1)
    flagged = flagged.drop(columns=['_cat', '_seller', '_name'], errors='ignore')
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_product_warranty(data: pd.DataFrame, warranty_category_codes: List[str]) -> pd.DataFrame:
    d = data.copy()
    for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION']:
        if c not in d.columns: d[c] = ""
        d[c] = d[c].astype(str).fillna('').str.strip()
    if not warranty_category_codes: return pd.DataFrame(columns=d.columns)
    d['CAT_CLEAN'] = d['CATEGORY_CODE'].apply(clean_category_code)
    target = d[d['CAT_CLEAN'].isin([clean_category_code(c) for c in warranty_category_codes])]
    if target.empty: return pd.DataFrame(columns=d.columns)
    def is_present(s): return (s != 'nan') & (s != '') & (s != 'none') & (s != 'nat') & (s != 'n/a')
    mask = ~(is_present(target['PRODUCT_WARRANTY']) | is_present(target['WARRANTY_DURATION']))
    return target[mask].drop(columns=['CAT_CLEAN'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_seller_approved_for_books(data: pd.DataFrame, books_data: Dict, country_code: str, book_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'SELLER_NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    category_codes = books_data.get('category_codes') or set(clean_category_code(c) for c in book_category_codes)
    if not category_codes: return pd.DataFrame(columns=data.columns)
    approved_sellers = books_data.get('sellers', {}).get(country_code, set())
    if not approved_sellers: return pd.DataFrame(columns=data.columns)
    books = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(category_codes)].copy()
    if books.empty: return pd.DataFrame(columns=data.columns)
    not_approved = ~books['SELLER_NAME'].astype(str).str.strip().str.lower().isin(approved_sellers)
    flagged = books[not_approved].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Seller not approved to sell books: " + flagged['SELLER_NAME'].astype(str)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_seller_approved_for_perfume(data: pd.DataFrame, perfume_category_codes: List[str], perfume_data: Dict, country_code: str) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'SELLER_NAME', 'BRAND', 'NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    sheet_cat_codes = perfume_data.get('category_codes')
    cat_codes = sheet_cat_codes if sheet_cat_codes else set(clean_category_code(c) for c in perfume_category_codes)
    perfume = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(cat_codes)].copy()
    if perfume.empty: return pd.DataFrame(columns=data.columns)
    keywords = perfume_data.get('keywords', set())
    approved_sellers = perfume_data.get('sellers', {}).get(country_code, set())
    has_seller_list = bool(approved_sellers)
    b_lower = perfume['BRAND'].astype(str).str.strip().str.lower()
    n_lower = perfume['NAME'].astype(str).str.strip().str.lower()
    GENERIC_PLACEHOLDERS = {'designers collection', 'smart collection', 'generic', 'original', 'fashion'}
    if keywords:
        kw_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r')\b', re.IGNORECASE)
        sneaky_mask = b_lower.isin(GENERIC_PLACEHOLDERS) & n_lower.apply(lambda x: bool(kw_pattern.search(x)))
    else: sneaky_mask = pd.Series([False] * len(perfume), index=perfume.index)
    if has_seller_list:
        brand_sens_mask = b_lower.apply(lambda x: bool(kw_pattern.search(x))) if keywords else pd.Series([False]*len(perfume), index=perfume.index)
        needs_approval = sneaky_mask | brand_sens_mask
        not_approved = ~perfume['SELLER_NAME'].astype(str).str.strip().str.lower().isin(approved_sellers)
        flagged_mask = needs_approval & not_approved
    else: flagged_mask = sneaky_mask
    flagged = perfume[flagged_mask].copy()
    if not flagged.empty:
        def describe(row):
            b, n = str(row['BRAND']).strip(), str(row['NAME']).strip()[:40]
            if b.lower() in GENERIC_PLACEHOLDERS: return f"Sneaky brand in name: '{n}'"
            return f"Sensitive brand '{b}' — seller not approved"
        flagged['Comment_Detail'] = flagged.apply(describe, axis=1)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_counterfeit_sneakers(data: pd.DataFrame, sneaker_category_codes: List[str], sneaker_sensitive_brands: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME', 'BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    sneakers = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in sneaker_category_codes))].copy()
    if sneakers.empty: return pd.DataFrame(columns=data.columns)
    b_lower, n_lower = sneakers['BRAND'].astype(str).str.strip().str.lower(), sneakers['NAME'].astype(str).str.strip().str.lower()
    return sneakers[b_lower.isin(['generic', 'fashion']) & n_lower.apply(lambda x: any(b in x for b in sneaker_sensitive_brands))].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_counterfeit_jerseys(data: pd.DataFrame, jerseys_data: Dict, country_code: str) -> pd.DataFrame:
    if not {"CATEGORY_CODE", "NAME", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    categories = jerseys_data.get("categories", set())
    keywords   = jerseys_data.get("keywords",   {}).get(country_code, set())
    exempted   = jerseys_data.get("exempted",   {}).get(country_code, set())
    if not categories or not keywords: return pd.DataFrame(columns=data.columns)
    kw_pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
    d = data.copy()
    d["_cat"]    = d["CATEGORY_CODE"].apply(clean_category_code)
    d["_seller"] = d["SELLER_NAME"].astype(str).str.strip().str.lower()
    d["_name"]   = d["NAME"].astype(str).str.strip()
    in_scope     = d["_cat"].isin(categories)
    has_keyword  = d["_name"].str.contains(kw_pattern, na=False)
    not_exempted = ~d["_seller"].isin(exempted)
    flagged = d[in_scope & has_keyword & not_exempted].copy()
    if not flagged.empty:
        def build_comment(row):
            match = kw_pattern.search(row["_name"])
            kw_found = match.group(0) if match else "?"
            return f"Suspected counterfeit jersey — keyword '{kw_found}' (cat: {row['_cat']})"
        flagged["Comment_Detail"] = flagged.apply(build_comment, axis=1)
    return flagged.drop(columns=["_cat", "_seller", "_name"], errors="ignore").drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_unnecessary_words(data: pd.DataFrame, pattern: re.Pattern) -> pd.DataFrame:
    if not {'NAME'}.issubset(data.columns) or pattern is None: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    mask = d['NAME'].astype(str).str.strip().str.lower().str.contains(pattern, na=False)
    flagged = d[mask].copy()
    if not flagged.empty:
        def get_matches(text):
            if pd.isna(text): return ""
            matches = pattern.findall(str(text))
            return ", ".join(set(m.lower() for m in matches if isinstance(m, str)))
        def highlight_matches(text):
            if pd.isna(text): return text
            return pattern.sub(lambda m: f"[*]{m.group(0)}[*]", str(text))
        flagged['Comment_Detail'] = "Unnecessary: " + flagged['NAME'].apply(get_matches)
        flagged['NAME'] = flagged['NAME'].apply(highlight_matches)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_single_word_name(data: pd.DataFrame, book_category_codes: List[str], books_data: Dict = None) -> pd.DataFrame:
    if not {'CATEGORY_CODE','NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    cat_codes = (books_data or {}).get('category_codes') or set(clean_category_code(c) for c in book_category_codes)
    non_books = data[~data['CATEGORY_CODE'].apply(clean_category_code).isin(cat_codes)]
    return non_books[non_books['NAME'].astype(str).str.split().str.len() == 1].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_generic_brand_issues(data: pd.DataFrame, valid_category_codes_fas: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    return data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in valid_category_codes_fas)) & (data['BRAND'].astype(str).str.lower() == 'generic')].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_fashion_brand_issues(data: pd.DataFrame, valid_category_codes_fas: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    return data[(data['BRAND'].astype(str).str.strip().str.lower() == 'fashion') & (~data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in valid_category_codes_fas)))].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_brand_in_name(data: pd.DataFrame) -> pd.DataFrame:
    if not {'BRAND','NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    brands = data['BRAND'].astype(str).str.strip().str.lower().values
    names = data['NAME'].astype(str).str.strip().str.lower().values
    mask = [b in n if b and b != 'nan' else False for b, n in zip(brands, names)]
    return data[mask].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_wrong_variation(data: pd.DataFrame, allowed_variation_codes: List[str]) -> pd.DataFrame:
    d = data.copy()
    if 'COUNT_VARIATIONS' not in d.columns: d['COUNT_VARIATIONS'] = 1
    if 'CATEGORY_CODE' not in d.columns: return pd.DataFrame(columns=data.columns)
    d['cat_clean'] = d['CATEGORY_CODE'].apply(clean_category_code)
    d['qty_var'] = pd.to_numeric(d['COUNT_VARIATIONS'], errors='coerce').fillna(1).astype(int)
    flagged = d[(d['qty_var'] >= 3) & (~d['cat_clean'].isin(set(clean_category_code(c) for c in allowed_variation_codes)))].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Variations: " + flagged['qty_var'].astype(str) + ", Category: " + flagged['cat_clean']
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_generic_with_brand_in_name(data: pd.DataFrame, brands_list: List[str]) -> pd.DataFrame:
    if not {'NAME', 'BRAND'}.issubset(data.columns) or not brands_list: return pd.DataFrame(columns=data.columns)
    mask = (data['BRAND'].astype(str).str.strip().str.lower() == 'generic')
    if 'CATEGORY' in data.columns: mask = mask & ~data['CATEGORY'].astype(str).str.lower().str.contains(r'\b(case|cases|cover|covers)\b', regex=True, na=False)
    gen = data[mask].copy()
    if gen.empty: return pd.DataFrame(columns=data.columns)
    sorted_b = sorted([str(b).strip().lower() for b in brands_list if b], key=len, reverse=True)
    def detect(n):
        nc = re.sub(r'\s+', ' ', re.sub(r"['\.\-]", ' ', str(n).lower())).strip()
        for b in sorted_b:
            bc = re.sub(r'\s+', ' ', re.sub(r"['\.\-]", ' ', b)).strip()
            if nc.startswith(bc) and (len(nc) == len(bc) or not nc[len(bc)].isalnum()): return b.title()
        return None
    gen['Detected_Brand'] = [detect(n) for n in gen['NAME'].values]
    flagged = gen[gen['Detected_Brand'].notna()].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Detected Brand: " + flagged['Detected_Brand']
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_missing_color(data: pd.DataFrame, pattern: re.Pattern, color_categories: List[str], country_code: str) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns) or pattern is None: return pd.DataFrame(columns=data.columns)
    target = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in color_categories))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    has_color = 'COLOR' in data.columns
    names = target['NAME'].astype(str).values
    colors = target['COLOR'].astype(str).str.strip().str.lower().values if has_color else [''] * len(target)
    mask = []
    for n, c in zip(names, colors):
        if pattern.search(n): mask.append(False)
        elif has_color and c not in ['nan', '', 'none', 'null']: mask.append(False)
        else: mask.append(True)
    return target[mask].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_weight_volume_in_name(data: pd.DataFrame, weight_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns) or not weight_category_codes: return pd.DataFrame(columns=data.columns)
    target = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in weight_category_codes))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    pat = re.compile(r'\b\d+(?:\.\d+)?\s*(?:kg|kgs|g|gm|gms|grams|mg|mcg|ml|l|ltr|liter|litres|litre|cl|oz|ounces|lb|lbs|tablets|capsules|sachets|count|ct|sticks|iu|teabags|pieces|pcs|pack|packs)\b', re.IGNORECASE)
    return target[~target['NAME'].apply(lambda n: bool(pat.search(str(n))))].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_incomplete_smartphone_name(data: pd.DataFrame, smartphone_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns) or not smartphone_category_codes: return pd.DataFrame(columns=data.columns)
    target = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in smartphone_category_codes))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    pat = re.compile(r'\b\d+\s*(gb|tb)\b', re.IGNORECASE)
    flagged = target[~target['NAME'].apply(lambda n: bool(pat.search(str(n))))].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Name missing Storage/Memory spec (e.g., 64GB)"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_duplicate_products(data: pd.DataFrame, exempt_categories: List[str] = None, similarity_threshold: float = 0.70, known_colors: List[str] = None, **kwargs) -> pd.DataFrame:
    if not {'NAME', 'SELLER_NAME', 'BRAND'}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    d = data.copy()
    if exempt_categories and 'CATEGORY_CODE' in d.columns:
        d = d[~d['CATEGORY_CODE'].apply(clean_category_code).isin(
            set(clean_category_code(c) for c in exempt_categories))]
    if d.empty:
        return pd.DataFrame(columns=data.columns)

    d['_norm_name'] = d['NAME'].astype(str).apply(lambda x: re.sub(r'\s+', '', normalize_text(x)))
    d['_norm_brand'] = d['BRAND'].astype(str).str.lower().str.strip()
    d['_norm_seller'] = d['SELLER_NAME'].astype(str).str.lower().str.strip()
    d['_dedup_key'] = d['_norm_seller'] + '|' + d['_norm_brand'] + '|' + d['_norm_name']

    seen_keys: dict = {}
    rej: set = set()
    details: dict = {}

    for _, row in d.iterrows():
        key = row['_dedup_key']
        sid = str(row['PRODUCT_SET_SID'])
        if key in seen_keys:
            rej.add(sid)
            details[sid] = {'base': str(row['NAME'])[:40]}
        else:
            seen_keys[key] = sid

    if not rej:
        return pd.DataFrame(columns=data.columns)

    rdf = d[d['PRODUCT_SET_SID'].astype(str).isin(rej)].copy()
    rdf['Comment_Detail'] = rdf['PRODUCT_SET_SID'].apply(
        lambda sid: f"Duplicate: '{details[str(sid)]['base']}'" if str(sid) in details else "Duplicate detected"
    )
    base_cols = data.columns.tolist()
    extra_cols = [c for c in ['Comment_Detail'] if c not in base_cols]
    return rdf[base_cols + extra_cols].drop_duplicates(subset=['PRODUCT_SET_SID'])

# -------------------------------------------------
# MASTER VALIDATION RUNNER
# -------------------------------------------------
def validate_products(data: pd.DataFrame, support_files: Dict, country_validator: CountryValidator, data_has_warranty_cols: bool, common_sids: Optional[set] = None, skip_validators: Optional[List[str]] = None):
    data['PRODUCT_SET_SID'] = data['PRODUCT_SET_SID'].astype(str).str.strip()
    flags_mapping = support_files['flags_mapping']
    country_restricted_rules = support_files.get('restricted_brands_all', {}).get(country_validator.country, [])
    country_prohibited_words = support_files.get('prohibited_words_all', {}).get(country_validator.code, [])
    validations = [
        ("Wrong Category", check_miscellaneous_category, {}),
        ("Restricted brands", check_restricted_brands, {'country_rules': country_restricted_rules}),
        ("Suspected Fake product", check_suspected_fake_products, {'suspected_fake_df': support_files['suspected_fake'], 'fx_rate': FX_RATE}),
        ("Seller Not approved to sell Refurb", check_refurb_seller_approval, {'refurb_data': support_files.get('refurb_data', {}), 'country_code': country_validator.code}),
        ("Product Warranty", check_product_warranty, {'warranty_category_codes': support_files['warranty_category_codes']}),
        ("Seller Approve to sell books", check_seller_approved_for_books, {'books_data': support_files.get('books_data', {}), 'country_code': country_validator.code, 'book_category_codes': support_files['book_category_codes']}),
        ("Seller Approved to Sell Perfume", check_seller_approved_for_perfume, {'perfume_category_codes': support_files['perfume_category_codes'], 'perfume_data': support_files.get('perfume_data', {}), 'country_code': country_validator.code}),
        ("Counterfeit Sneakers", check_counterfeit_sneakers, {'sneaker_category_codes': support_files['sneaker_category_codes'], 'sneaker_sensitive_brands': support_files['sneaker_sensitive_brands']}),
        ("Suspected counterfeit Jerseys", check_counterfeit_jerseys, {'jerseys_data': support_files.get('jerseys_data', {}), 'country_code': country_validator.code}),
        ("Prohibited products", check_prohibited_products, {'prohibited_rules': country_prohibited_words}),
        ("Unnecessary words in NAME", check_unnecessary_words, {'pattern': compile_regex_patterns(support_files['unnecessary_words'])}),
        ("Single-word NAME", check_single_word_name, {'book_category_codes': support_files['book_category_codes'], 'books_data': support_files.get('books_data', {})}),
        ("Generic BRAND Issues", check_generic_brand_issues, {}),
        ("Fashion brand issues", check_fashion_brand_issues, {}),
        ("BRAND name repeated in NAME", check_brand_in_name, {}),
        ("Wrong Variation", check_wrong_variation, {'allowed_variation_codes': list(set(support_files.get('variation_allowed_codes', []) + support_files.get('category_fas', [])))}),
        ("Generic branded products with genuine brands", check_generic_with_brand_in_name, {'brands_list': support_files.get('known_brands', [])}),
        ("Missing COLOR", check_missing_color, {'pattern': compile_regex_patterns(support_files['colors']), 'color_categories': support_files['color_categories'], 'country_code': country_validator.code}),
        ("Missing Weight/Volume", check_weight_volume_in_name, {'weight_category_codes': support_files.get('weight_category_codes', [])}),
        ("Incomplete Smartphone Name", check_incomplete_smartphone_name, {'smartphone_category_codes': support_files.get('smartphone_category_codes', [])}),
        ("Duplicate product", check_duplicate_products, {'exempt_categories': support_files.get('duplicate_exempt_codes', []), 'known_colors': support_files['colors']}),
    ]
    results = {}
    dup_groups = {}
    if {'NAME','BRAND','SELLER_NAME','COLOR'}.issubset(data.columns):
        dt = data.copy()
        dt['dup_key'] = dt[['NAME','BRAND','SELLER_NAME','COLOR']].apply(lambda r: tuple(str(v).strip().lower() for v in r), axis=1)
        for k, v in dt.groupby('dup_key')['PRODUCT_SET_SID'].apply(list).items():
            if len(v) > 1:
                for sid in v: dup_groups[sid] = v
    restricted_keys = {}
    validation_errors = []

    with st.spinner("Validating products... This may take a moment."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_name = {}
            for i, (name, func, kwargs) in enumerate(validations):
                if skip_validators and name in skip_validators: continue
                if country_validator.should_skip_validation(name): continue
                ckwargs = {'data': data, **kwargs}
                if name in ["Generic BRAND Issues", "Fashion brand issues"]: ckwargs['valid_category_codes_fas'] = support_files.get('category_fas', [])
                future_to_name[executor.submit(func, **ckwargs)] = name
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    res = future.result()
                    if not res.empty and 'PRODUCT_SET_SID' in res.columns:
                        res = res.loc[:, ~res.columns.duplicated()].copy()
                        res['PRODUCT_SET_SID'] = res['PRODUCT_SET_SID'].astype(str).str.strip()
                        if name in ["Seller Approve to sell books", "Seller Approved to Sell Perfume", "Counterfeit Sneakers", "Seller Not approved to sell Refurb", "Restricted brands"]:
                            res['match_key'] = res.apply(create_match_key, axis=1)
                            restricted_keys.setdefault(name, set()).update(res['match_key'].unique())
                        expanded_sids = set()
                        for sid in set(res['PRODUCT_SET_SID'].unique()): expanded_sids.update(dup_groups.get(sid, [sid]))
                        final_res = data[data['PRODUCT_SET_SID'].isin(expanded_sids)].copy()
                        if 'Comment_Detail' in res.columns: final_res['Comment_Detail'] = res['Comment_Detail']
                        if name in results and not results[name].empty: results[name] = pd.concat([results[name], final_res]).drop_duplicates(subset=['PRODUCT_SET_SID'])
                        else: results[name] = final_res
                    else:
                        if name not in results: results[name] = pd.DataFrame(columns=data.columns)
                except Exception as e:
                    logger.error(f"Error in {name}: {e}")
                    validation_errors.append((name, str(e)))
                    if name not in results: results[name] = pd.DataFrame(columns=data.columns)

    if validation_errors:
        st.warning(f"{len(validation_errors)} validation checks encountered errors.")
        with st.expander("View Error Details"):
            for e_name, e_msg in validation_errors: st.error(f"**{e_name}**: {e_msg}")
    if restricted_keys:
        data['match_key'] = data.apply(create_match_key, axis=1)
        for fname, keys in restricted_keys.items():
            extra = data[data['match_key'].isin(keys)].copy()
            results[fname] = pd.concat([results.get(fname, pd.DataFrame()), extra]).drop_duplicates(subset=['PRODUCT_SET_SID'])
    rows = []
    processed = set()
    for name, _, _ in validations:
        if name not in results or results[name].empty or 'PRODUCT_SET_SID' not in results[name].columns: continue
        res = results[name]
        rinfo = flags_mapping.get(name, ("1000007 - Other Reason", f"Flagged by {name}"))
        res['PRODUCT_SET_SID'] = res['PRODUCT_SET_SID'].astype(str).str.strip()
        flagged = pd.merge(res[['PRODUCT_SET_SID', 'Comment_Detail']] if 'Comment_Detail' in res.columns else res[['PRODUCT_SET_SID']], data, on='PRODUCT_SET_SID', how='left')
        if 'Comment_Detail' not in flagged.columns and 'Comment_Detail' in res.columns:
            if isinstance(res['Comment_Detail'], pd.DataFrame): flagged['Comment_Detail'] = res['Comment_Detail'].iloc[:, 0]
            else: flagged['Comment_Detail'] = res['Comment_Detail']
        for _, r in flagged.iterrows():
            sid = str(r['PRODUCT_SET_SID']).strip()
            if sid in processed: continue
            processed.add(sid)
            det = r.get('Comment_Detail', '')
            comment_str = f"{rinfo[1]} ({det})" if pd.notna(det) and det else rinfo[1]
            rows.append({'ProductSetSid': sid, 'ParentSKU': r.get('PARENTSKU', ''), 'Status': 'Rejected', 'Reason': rinfo[0], 'Comment': comment_str, 'FLAG': name, 'SellerName': r.get('SELLER_NAME', '')})
    for _, r in data[~data['PRODUCT_SET_SID'].astype(str).str.strip().isin(processed)].iterrows():
        sid = str(r['PRODUCT_SET_SID']).strip()
        if sid not in processed:
            rows.append({'ProductSetSid': sid, 'ParentSKU': r.get('PARENTSKU', ''), 'Status': 'Approved', 'Reason': "", 'Comment': "", 'FLAG': "", 'SellerName': r.get('SELLER_NAME', '')})
            processed.add(sid)
    final_df = pd.DataFrame(rows)
    for c in ["ProductSetSid", "ParentSKU", "Status", "Reason", "Comment", "FLAG", "SellerName"]:
        if c not in final_df.columns: final_df[c] = ""
    return country_validator.ensure_status_column(final_df), results

@st.cache_data(show_spinner=False, ttl=3600)
def cached_validate_products(data_hash: str, _data: pd.DataFrame, _support_files: Dict, country_code: str, data_has_warranty_cols: bool):
    country_name = next(
        (k for k, v in CountryValidator.COUNTRY_CONFIG.items() if v['code'] == country_code),
        "Kenya"
    )
    cv = CountryValidator(country_name)
    return validate_products(_data, _support_files, cv, data_has_warranty_cols)


# -------------------------------------------------
# EXPORTS UTILITIES
# -------------------------------------------------
def to_excel_base(df, sheet, cols, writer, format_rules=False):
    df_p = df.copy()
    for c in cols:
        if c not in df_p.columns: df_p[c] = pd.NA
    df_to_write = df_p[[c for c in cols if c in df_p.columns]]
    df_to_write.to_excel(writer, index=False, sheet_name=sheet)
    if format_rules and 'Status' in df_to_write.columns:
        wb = writer.book
        ws = writer.sheets[sheet]
        rf = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        gf = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        idx = df_to_write.columns.get_loc('Status')
        ws.conditional_format(1, idx, len(df_to_write), idx, {'type': 'cell', 'criteria': 'equal', 'value': '"Rejected"', 'format': rf})
        ws.conditional_format(1, idx, len(df_to_write), idx, {'type': 'cell', 'criteria': 'equal', 'value': '"Approved"', 'format': gf})

def write_excel_single(df, sheet_name, cols, auxiliary_df=None, aux_sheet_name=None, aux_cols=None, format_status=False, full_data_stats=False):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        to_excel_base(df, sheet_name, cols, writer, format_rules=format_status)
        if auxiliary_df is not None and not auxiliary_df.empty: to_excel_base(auxiliary_df, aux_sheet_name, aux_cols, writer)
        if full_data_stats and 'SELLER_NAME' in df.columns and 'Status' in df.columns:
            ws = writer.book.add_worksheet('Sellers Data')
            fmt = writer.book.add_format({'bold': True, 'bg_color': '#E6F0FA', 'border': 1, 'align': 'center'})
            df['Rejected_Count'] = (df['Status'] == 'Rejected').astype(int)
            df['Approved_Count'] = (df['Status'] == 'Approved').astype(int)
            summ = df.groupby('SELLER_NAME').agg(Rejected=('Rejected_Count', 'sum'), Approved=('Approved_Count', 'sum')).reset_index().sort_values('Rejected', ascending=False)
            summ.insert(0, 'Rank', range(1, len(summ) + 1))
            ws.write(0, 0, "Sellers Summary (This File)", fmt)
            summ.to_excel(writer, sheet_name='Sellers Data', startrow=1, index=False)
    output.seek(0)
    return output

def generate_smart_export(df, filename_prefix, export_type='simple', auxiliary_df=None):
    cols = FULL_DATA_COLS + [c for c in ["Status", "Reason", "Comment", "FLAG", "SellerName"] if c not in FULL_DATA_COLS] if export_type == 'full' else PRODUCTSETS_COLS
    if len(df) <= SPLIT_LIMIT:
        data = write_excel_single(df, "ProductSets", cols, auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS, True, export_type == 'full')
        return data, f"{filename_prefix}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        zb = BytesIO()
        with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(0, len(df), SPLIT_LIMIT):
                chunk = df.iloc[i : i + SPLIT_LIMIT]
                excel_data = write_excel_single(chunk, "ProductSets", cols, auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS, True, export_type == 'full')
                zf.writestr(f"{filename_prefix}_Part_{(i//SPLIT_LIMIT)+1}.xlsx", excel_data.getvalue())
        zb.seek(0)
        return zb, f"{filename_prefix}.zip", "application/zip"

def prepare_full_data_merged(data_df, final_report_df):
    try:
        d_cp, r_cp = data_df.copy(), final_report_df.copy()
        d_cp['PRODUCT_SET_SID'] = d_cp['PRODUCT_SET_SID'].astype(str).str.strip()
        r_cp['ProductSetSid'] = r_cp['ProductSetSid'].astype(str).str.strip()
        merged = pd.merge(d_cp, r_cp[["ProductSetSid", "Status", "Reason", "Comment", "FLAG", "SellerName"]], left_on="PRODUCT_SET_SID", right_on="ProductSetSid", how='left')
        if 'ProductSetSid' in merged.columns: merged.drop(columns=['ProductSetSid'], inplace=True)
        return merged
    except Exception: return pd.DataFrame()


# -------------------------------------------------
# UTILITIES FOR BRIDGE COMPONENT
# -------------------------------------------------
def apply_rejection(sids: list, reason_code: str, comment: str, flag_name: str):
    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(sids), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', reason_code, comment, flag_name]
    st.session_state.exports_cache.clear()
    st.session_state.display_df_cache.clear()

def restore_single_item(sid):
    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status', 'Reason', 'Comment', 'FLAG']] = ['Approved', '', '', 'Approved by User']
    st.session_state.pop(f"quick_rej_{sid}", None)
    st.session_state.pop(f"quick_rej_reason_{sid}", None)
    st.session_state.exports_cache.clear()
    st.session_state.display_df_cache.clear()
    st.session_state.main_toasts.append("Restored item to previous state!")

REASON_MAP = {
    "REJECT_POOR_IMAGE": "Poor images",
    "REJECT_WRONG_CAT": "Wrong Category",
    "REJECT_FAKE": "Suspected Fake product",
    "REJECT_BRAND": "Restricted brands",
    "REJECT_PROHIBITED": "Prohibited products",
    "REJECT_COLOR": "Missing COLOR",
    "REJECT_WRONG_BRAND": "Generic branded products with genuine brands",
    "OTHER_CUSTOM": "Other Reason (Custom)"
}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  BRIDGE ACTION PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
def _process_card_bridge_action(action_str: str, support_files: dict) -> bool:
    if not action_str:
        return False
    try:
        colon = action_str.index(":")
        action  = action_str[:colon]
        payload = action_str[colon + 1:]
    except ValueError:
        return False

    try:
        if action == "NAV_PREV":
            st.session_state.grid_page = max(0, st.session_state.grid_page - 1)
            st.session_state.do_scroll_top = True
            return True

        if action == "NAV_NEXT":
            st.session_state.grid_page += 1
            st.session_state.do_scroll_top = True
            return True

        if action == "BATCH_COMMIT":
            try:
                pending: dict = json.loads(payload) 
            except json.JSONDecodeError:
                return False

            if not pending:
                return False

            reason_groups: dict[str, list] = {}
            for sid, reason_key in pending.items():
                reason_groups.setdefault(reason_key, []).append(sid)

            total = 0
            for reason_key, sids in reason_groups.items():
                flag_name = REASON_MAP.get(reason_key, "Other Reason (Custom)")
                code, cmt  = support_files["flags_mapping"].get(
                    flag_name, ("1000007 - Other Reason", "Manual rejection")
                )
                apply_rejection(sids, code, cmt, flag_name)
                for s in sids:
                    st.session_state[f"quick_rej_{s}"]        = True
                    st.session_state[f"quick_rej_reason_{s}"] = flag_name
                total += len(sids)

            st.session_state.main_toasts.append(
                (f"Rejected {total} product(s)", "✅")
            )
            return True

        if action == "RESTORE":
            restore_single_item(payload.strip())
            return True

    except Exception as e:
        logger.error(f"Bridge action error: {e}")

    return False


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HTML GRID BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_fast_grid_html(
    page_data: pd.DataFrame,
    flags_mapping: dict,
    country: str,
    page_warnings: dict,
    rejected_state: dict,
    cols_per_row: int,
    current_page: int,
    total_pages: int,
) -> str:

    O  = JUMIA_COLORS["primary_orange"]
    G  = JUMIA_COLORS["success_green"]
    R  = JUMIA_COLORS["jumia_red"]
    DG = JUMIA_COLORS["dark_gray"]

    batch_options = [
        ("Poor Image Quality",  "REJECT_POOR_IMAGE"),
        ("Wrong Category",      "REJECT_WRONG_CAT"),
        ("Suspected Fake",      "REJECT_FAKE"),
        ("Restricted Brand",    "REJECT_BRAND"),
        ("Wrong Brand",         "REJECT_WRONG_BRAND"),
        ("Prohibited Product",  "REJECT_PROHIBITED"),
    ]
    batch_opts_html = "".join(f'<option value="{v}">{l}</option>' for l, v in batch_options)

    committed_json = json.dumps(rejected_state)
    all_sids_json  = json.dumps(page_data["PRODUCT_SET_SID"].astype(str).tolist())

    cards_data = []
    for _, row in page_data.iterrows():
        sid      = str(row["PRODUCT_SET_SID"])
        img_url  = str(row.get("MAIN_IMAGE", "")).strip()
        if not img_url.startswith("http"):
            img_url = "https://via.placeholder.com/150?text=No+Image"
        cards_data.append({
            "sid": sid, "img": img_url,
            "name": str(row.get("NAME", "")),
            "brand": str(row.get("BRAND", "Unknown Brand")),
            "cat": str(row.get("CATEGORY", "Unknown Category")),
            "seller": str(row.get("SELLER_NAME", "Unknown Seller")),
            "warnings": page_warnings.get(sid, []),
        })
    cards_json = json.dumps(cards_data)

    prev_disabled = "disabled" if current_page <= 0 else ""
    next_disabled = "disabled" if current_page >= total_pages - 1 else ""

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;font-family:sans-serif;}}
  body{{background:#f5f5f5;padding:10px;}}
  .toolbar{{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#fff;border:1px solid #e0e0e0;border-radius:8px;margin-bottom:14px;position:sticky;top:0;z-index:200;gap:10px;flex-wrap:wrap;}}
  .toolbar button{{padding:6px 13px;border:1px solid #ccc;border-radius:4px;background:#fff;cursor:pointer;font-size:12px;font-weight:600;}}
  .toolbar button:disabled{{opacity:.4;cursor:default;}}
  .toolbar button.danger{{background:{R};color:#fff;border-color:{R};}}
  .toolbar button.nav{{background:{DG};color:#fff;border-color:{DG};}}
  .toolbar select{{padding:6px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px;outline:none;}}
  .page-info{{font-size:13px;font-weight:700;color:{DG};white-space:nowrap;}}
  .sel-count{{font-size:13px;font-weight:700;color:{O};}}
  .grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}
  .card{{border:2px solid #e0e0e0;border-radius:8px;padding:10px;background:#fff;position:relative;transition:border-color .15s,box-shadow .15s;cursor:pointer;}}
  .card.selected{{border-color:{G};box-shadow:0 0 0 3px rgba(76,175,80,.2);background:rgba(76,175,80,.04);}}
  .card.pending-rej{{border-color:{R};opacity:.55;box-shadow:0 0 0 3px rgba(231,60,23,.15);}}
  .card.committed-rej{{border-color:#aaa;opacity:.45;}}
  .card-img-wrap{{position:relative;}}
  .card-img{{width:100%;aspect-ratio:1;object-fit:contain;border-radius:6px;display:block;}}
  .card.pending-rej .card-img, .card.committed-rej .card-img{{filter:grayscale(80%);}}
  .tick{{position:absolute;bottom:6px;right:6px;width:22px;height:22px;border-radius:50%;background:rgba(0,0,0,.18);display:flex;align-items:center;justify-content:center;color:transparent;font-size:13px;font-weight:900;transition:all .15s;pointer-events:none;}}
  .card.selected .tick{{background:{G};color:#fff;}}
  .warn-wrap{{position:absolute;top:6px;right:6px;display:flex;flex-direction:column;gap:3px;z-index:5;pointer-events:none;}}
  .warn-badge{{background:rgba(255,193,7,.95);color:#313133;font-size:9px;font-weight:800;padding:3px 7px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.2);}}
  .overlay{{display:none;position:absolute;inset:0;background:rgba(255,255,255,.88);border-radius:6px;flex-direction:column;align-items:center;justify-content:center;z-index:20;gap:6px;padding:8px;text-align:center;}}
  .card.pending-rej .overlay, .card.committed-rej .overlay{{display:flex;}}
  .rej-badge{{background:{R};color:#fff;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:700;}}
  .rej-label{{font-size:10px;font-weight:600;color:{R};max-width:120px;}}
  .undo-btn{{margin-top:4px;padding:5px 12px;background:{DG};color:#fff;border:none;border-radius:4px;font-size:11px;font-weight:700;cursor:pointer;}}
  .meta{{font-size:11px;margin-top:8px;line-height:1.4;}}
  .meta .nm{{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .meta .br{{color:{O};font-weight:700;margin:2px 0;}}
  .meta .ct{{color:#666;font-size:10px;}}
  .meta .sl{{color:#999;font-size:9px;margin-top:4px;border-top:1px dashed #eee;padding-top:4px;}}
  .acts{{display:flex;gap:4px;margin-top:8px;}}
  .act-btn{{flex:1;padding:6px;font-size:11px;border:none;border-radius:4px;cursor:pointer;font-weight:700;color:#fff;}}
  .act-poor{{background:{O};}}
  .act-more{{flex:1;font-size:11px;border:1px solid #ccc;border-radius:4px;outline:none;cursor:pointer;background:#fff;}}
</style>
</head>
<body>
<div class="toolbar">
  <div style="display:flex;align-items:center;gap:8px;">
    <button class="nav" id="btn-prev" {prev_disabled} onclick="navAction('NAV_PREV')">&#8592; Prev</button>
    <span class="page-info">Page {current_page + 1} / {total_pages}</span>
    <button class="nav" id="btn-next" {next_disabled} onclick="navAction('NAV_NEXT')">Next &#8594;</button>
  </div>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <button onclick="selectAll()">Select All</button>
    <button onclick="deselectAll()">Deselect All</button>
    <span class="sel-count"><span id="sel-count">0</span> selected</span>
    <select id="batch-reason">{batch_opts_html}</select>
    <button class="danger" onclick="doBatchReject()">&#10007; Batch Reject</button>
    <button style="background:{G};color:#fff;border-color:{G};" onclick="commitAll()">&#10003; Commit All</button>
  </div>
</div>
<div class="grid" id="card-grid"></div>

<script>
const CARDS = {cards_json};
const COMMITTED = {committed_json};
const ALL_SIDS = {all_sids_json};

let selectedSids = new Set();
let pendingRejections = {{}};

function sendBridge(action) {{
  let input = window.parent.document.querySelector('input[placeholder="__CARD_ACT__"]');
  if (!input) {{
    for (const el of window.parent.document.querySelectorAll('input')) {{
      if (el.placeholder === '__CARD_ACT__') {{ input = el; break; }}
    }}
  }}
  if (!input) {{
    for (const frame of window.parent.document.querySelectorAll('iframe')) {{
      try {{
        const doc = frame.contentDocument || frame.contentWindow.document;
        input = doc.querySelector('input[placeholder="__CARD_ACT__"]');
        if (input) break;
      }} catch(e) {{}}
    }}
  }}
  if (!input) return;
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
  setter.call(input, action);
  input.dispatchEvent(new Event('input', {{bubbles:true}}));
}}

function renderCard(card) {{
  const {{sid, img, name, brand, cat, seller, warnings}} = card;
  const isCommitted = sid in COMMITTED;
  const isPending = sid in pendingRejections;
  const isSelected = selectedSids.has(sid);

  let cls = 'card';
  if (isCommitted) cls += ' committed-rej';
  else if (isPending) cls += ' pending-rej';
  else if (isSelected) cls += ' selected';

  const shortName = name.length > 38 ? name.slice(0,38)+'…' : name;
  const warnHtml = warnings.map(w => `<span class="warn-badge">${{w}}</span>`).join('');
  const overlayLabel = isPending ? pendingRejections[sid].replace(/_/g,' ') : (isCommitted ? COMMITTED[sid] : '');

  const overlayHtml = (isPending || isCommitted) ? `
    <div class="overlay">
      <div class="rej-badge">REJECTED</div>
      <div class="rej-label">${{overlayLabel}}</div>
      ${{isCommitted ? `<button class="undo-btn" onclick="doRestore('${{sid}}')">Undo (saved)</button>` : `<button class="undo-btn" onclick="undoPending('${{sid}}')">Undo (pending)</button>`}}
    </div>` : '';

  const actionsHtml = (!isPending && !isCommitted) ? `
    <div class="acts">
      <button class="act-btn act-poor" onclick="rejectCard(event,'${{sid}}','REJECT_POOR_IMAGE')">Poor Img</button>
      <select class="act-more" onchange="if(this.value){{rejectCard(event,'${{sid}}',this.value);this.value=''}}">
        <option value="">More…</option>
        <option value="REJECT_WRONG_CAT">Wrong Category</option>
        <option value="REJECT_FAKE">Fake Product</option>
        <option value="REJECT_BRAND">Restricted Brand</option>
        <option value="REJECT_PROHIBITED">Prohibited</option>
        <option value="REJECT_COLOR">Wrong Color</option>
        <option value="REJECT_WRONG_BRAND">Wrong Brand</option>
      </select>
    </div>` : '';

  return `
  <div class="${{cls}}" id="card-${{sid}}" data-sid="${{sid}}">
    <div class="card-img-wrap" onclick="toggleSelect('${{sid}}')">
      <div class="warn-wrap">${{warnHtml}}</div>
      <img class="card-img" src="${{img}}" loading="lazy">
      <div class="tick">✓</div>
    </div>
    ${{overlayHtml}}
    <div class="meta">
      <div class="nm" title="${{name}}">${{shortName}}</div>
      <div class="br">${{brand}}</div>
      <div class="ct">${{cat}}</div>
      <div class="sl">${{seller}}</div>
    </div>
    ${{actionsHtml}}
  </div>`;
}}

function renderAll() {{ document.getElementById('card-grid').innerHTML = CARDS.map(renderCard).join(''); }}
function toggleSelect(sid) {{ if (sid in pendingRejections || sid in COMMITTED) return; if (selectedSids.has(sid)) selectedSids.delete(sid); else selectedSids.add(sid); refreshCard(sid); document.getElementById('sel-count').textContent = selectedSids.size; }}
function rejectCard(evt, sid, reasonKey) {{ evt.stopPropagation(); pendingRejections[sid] = reasonKey; selectedSids.delete(sid); refreshCard(sid); document.getElementById('sel-count').textContent = selectedSids.size; }}
function undoPending(sid) {{ delete pendingRejections[sid]; refreshCard(sid); }}
function doRestore(sid) {{ sendBridge('RESTORE:' + sid); }}
function refreshCard(sid) {{ const card = CARDS.find(c => c.sid === sid); if (!card) return; const el = document.getElementById('card-' + sid); if (!el) return; el.outerHTML = renderCard(card); }}
function selectAll() {{ ALL_SIDS.forEach(sid => {{ if (!(sid in pendingRejections) && !(sid in COMMITTED)) selectedSids.add(sid); }}); renderAll(); document.getElementById('sel-count').textContent = selectedSids.size; }}
function deselectAll() {{ selectedSids.clear(); renderAll(); document.getElementById('sel-count').textContent = 0; }}
function doBatchReject() {{ const reasonKey = document.getElementById('batch-reason').value; if (selectedSids.size === 0) {{ alert('Select at least one product first.'); return; }} selectedSids.forEach(sid => {{ pendingRejections[sid] = reasonKey; }}); selectedSids.clear(); renderAll(); document.getElementById('sel-count').textContent = 0; commitAll(); }}
function commitAll() {{ if (Object.keys(pendingRejections).length === 0) return; sendBridge('BATCH_COMMIT:' + JSON.stringify(pendingRejections)); pendingRejections = {{}}; }}
function navAction(direction) {{ const hasPending = Object.keys(pendingRejections).length > 0; if (hasPending) {{ sendBridge('BATCH_COMMIT:' + JSON.stringify(pendingRejections)); pendingRejections = {{}}; setTimeout(() => sendBridge(direction + ':1'), 120); }} else {{ sendBridge(direction + ':1'); }} }}
renderAll();
</script>
</body>
</html>
"""
    return html

# -------------------------------------------------
# UI COMPONENTS
# -------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def analyze_image_quality_cached(url: str) -> List[str]:
    if not url or not str(url).startswith("http"):
        return []
    warnings = []
    try:
        resp = requests.get(url, timeout=2, stream=True)
        if resp.status_code == 200:
            img = Image.open(resp.raw)
            w, h = img.size
            if w < 300 or h < 300: warnings.append("Low Resolution")
            ratio = h / w if w > 0 else 1
            if ratio > 1.5: warnings.append("Tall (Screenshot?)")
            elif ratio < 0.6: warnings.append("Wide Aspect")
    except Exception: pass
    return warnings

@st.dialog("Confirm Bulk Approval")
def bulk_approve_dialog(sids_to_process, title, subset_data, data_has_warranty_cols_check, support_files, country_validator):
    st.warning(f"You are about to approve **{len(sids_to_process)}** items from `{title}`.")
    if st.button("Confirm Approval", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            data_hash = df_hash(subset_data) + country_validator.code + "_skip_" + title
            new_report, _ = cached_validate_products(data_hash, subset_data, support_files, country_validator.code, data_has_warranty_cols_check)
            msg_moved, msg_approved = {}, 0
            for sid in sids_to_process:
                new_row = new_report[new_report['ProductSetSid'] == sid]
                if new_row.empty or not str(new_row.iloc[0]['FLAG']):
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status', 'Reason', 'Comment', 'FLAG']] = ['Approved', '', '', 'Approved by User']
                    msg_approved += 1
                else:
                    new_flag = str(new_row.iloc[0]['FLAG'])
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', new_row.iloc[0]['Reason'], new_row.iloc[0]['Comment'], new_flag]
                    msg_moved[new_flag] = msg_moved.get(new_flag, 0) + 1
            if msg_approved > 0: st.session_state.main_toasts.append(f"{msg_approved} items successfully Approved!")
            for flag, count in msg_moved.items(): st.session_state.main_toasts.append(f"{count} items re-flagged as: {flag}")
            st.session_state.exports_cache.clear()
            st.session_state.display_df_cache.clear()
        st.rerun()

def render_flag_expander(title, df_flagged_sids, data, data_has_warranty_cols_check, support_files, country_validator):
    cache_key = f"display_df_{title}"
    base_display_cols = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'COLOR', 'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE', 'PARENTSKU', 'SELLER_NAME']
    current_display_cols = base_display_cols.copy()
    if title == "Wrong Variation":
        if 'COUNT_VARIATIONS' in data.columns: current_display_cols.append('COUNT_VARIATIONS')
        if 'LIST_VARIATIONS' in data.columns: current_display_cols.append('LIST_VARIATIONS')

    if cache_key not in st.session_state.display_df_cache:
        df_display = pd.merge(
            df_flagged_sids[['ProductSetSid']],
            data,
            left_on='ProductSetSid', right_on='PRODUCT_SET_SID', how='left'
        )[[c for c in current_display_cols if c in data.columns]]
        st.session_state.display_df_cache[cache_key] = df_display
    else:
        df_display = st.session_state.display_df_cache[cache_key]

    c1, c2 = st.columns([1, 1])
    with c1: search_term = st.text_input("Search", placeholder="Name, Brand...", key=f"s_{title}")
    with c2: seller_filter = st.multiselect("Filter by Seller", sorted(df_display['SELLER_NAME'].astype(str).unique()), key=f"f_{title}")

    df_view = df_display.copy()
    if search_term: df_view = df_view[df_view.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    if seller_filter: df_view = df_view[df_view['SELLER_NAME'].isin(seller_filter)]
    df_view = df_view.reset_index(drop=True)
    if 'NAME' in df_view.columns:
        def strip_html(text): return re.sub('<[^<]+?>', '', text) if isinstance(text, str) else text
        df_view['NAME'] = df_view['NAME'].apply(strip_html)

    event = st.dataframe(
        df_view, hide_index=True, use_container_width=True, selection_mode="multi-row", on_select="rerun",
        column_config={
            "PRODUCT_SET_SID": st.column_config.TextColumn(pinned=True),
            "NAME": st.column_config.TextColumn(pinned=True),
            "GLOBAL_SALE_PRICE": st.column_config.NumberColumn("Sale Price (USD)", format="$%.2f"),
            "GLOBAL_PRICE": st.column_config.NumberColumn("Price (USD)", format="$%.2f"),
        }, key=f"df_{title}"
    )
    raw_selected_indices = list(event.selection.rows)
    selected_indices = [i for i in raw_selected_indices if i < len(df_view)]
    st.caption(f"{len(selected_indices)} of {len(df_view)} rows selected")
    has_selection = len(selected_indices) > 0

    _fm = support_files['flags_mapping']
    _reason_options = [
        "Wrong Category", "Restricted brands", "Suspected Fake product", "Seller Not approved to sell Refurb",
        "Product Warranty", "Seller Approve to sell books", "Seller Approved to Sell Perfume", "Counterfeit Sneakers",
        "Suspected counterfeit Jerseys", "Prohibited products", "Unnecessary words in NAME", "Single-word NAME",
        "Generic BRAND Issues", "Fashion brand issues", "BRAND name repeated in NAME", "Wrong Variation",
        "Generic branded products with genuine brands", "Missing COLOR", "Missing Weight/Volume",
        "Incomplete Smartphone Name", "Duplicate product", "Poor images", "Other Reason (Custom)",
    ]

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("✓ Approve Selected", key=f"approve_sel_{title}", type="primary", use_container_width=True, disabled=not has_selection):
            sids_to_process = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
            subset = data[data['PRODUCT_SET_SID'].isin(sids_to_process)]
            bulk_approve_dialog(sids_to_process, title, subset, data_has_warranty_cols_check, support_files, country_validator)
    with btn_col2:
        with st.popover("↓ Reject As...", use_container_width=True, disabled=not has_selection):
            st.markdown("<p style='font-size:12px;font-weight:700;margin:0 0 8px 0;'>Select rejection reason:</p>", unsafe_allow_html=True)
            chosen_reason = st.selectbox("Reason", _reason_options, key=f"rej_reason_dd_{title}", label_visibility="collapsed")
            if chosen_reason == "Other Reason (Custom)":
                custom_comment = st.text_area("Custom comment", placeholder="Type your rejection reason here...", key=f"custom_comment_{title}", height=80)
                if st.button("Apply Custom Rejection", key=f"apply_custom_{title}", type="primary", use_container_width=True, disabled=not has_selection):
                    to_reject = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    final_comment = custom_comment.strip() if custom_comment.strip() else "Other Reason"
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(to_reject), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', '1000007 - Other Reason', final_comment, 'Other Reason (Custom)']
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected with custom reason.")
                    st.session_state.exports_cache.clear()
                    st.session_state.display_df_cache.clear()
                    st.rerun()
            else:
                _rcode, _rcmt = _fm.get(chosen_reason, ('1000007 - Other Reason', chosen_reason))
                st.caption(f"Code: {_rcode[:40]}...")
                if st.button("Apply Rejection", key=f"apply_dd_{title}", type="primary", use_container_width=True, disabled=not has_selection):
                    to_reject = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(to_reject), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', _rcode, _rcmt, chosen_reason]
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected as '{chosen_reason}'.")
                    st.session_state.exports_cache.clear()
                    st.session_state.display_df_cache.clear()
                    st.rerun()


# ==========================================
# SECTION 1: UPLOAD & VALIDATION
# ==========================================
st.header(":material/upload_file: Upload Files", anchor=False)

current_country = st.session_state.get('selected_country', 'Kenya')
country_choice = st.segmented_control("Country", ["Kenya", "Uganda", "Nigeria", "Ghana", "Morocco"], default=current_country)

if country_choice: st.session_state.selected_country = country_choice
else: country_choice = current_country

country_validator = CountryValidator(st.session_state.selected_country)

uploaded_files = st.file_uploader("Upload CSV or XLSX files", type=['csv', 'xlsx'], accept_multiple_files=True, key="daily_files")

if uploaded_files:
    current_file_signature = sorted([f.name + str(f.size) for f in uploaded_files])
    process_signature = str(current_file_signature) + f"_{country_validator.code}"
else: process_signature = "empty"

if st.session_state.get('last_processed_files') != process_signature:
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.post_qc_summary = pd.DataFrame()
    st.session_state.post_qc_results = {}
    st.session_state.post_qc_data = pd.DataFrame()
    st.session_state.file_mode = None
    st.session_state.intersection_sids = set()
    st.session_state.intersection_count = 0
    st.session_state.grid_page = 0
    st.session_state.exports_cache = {}
    st.session_state.display_df_cache = {}
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith(("quick_rej_", "grid_chk_", "toast_"))]
    for k in keys_to_delete: del st.session_state[k]

    if process_signature == "empty": st.session_state.last_processed_files = "empty"
    else:
        try:
            all_dfs = []
            file_sids_sets = []
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
                merged = pd.concat(norm_dfs, ignore_index=True)
                merged_dedup = merged.drop_duplicates(subset=['PRODUCT_SET_SID'], keep='first')
                with st.spinner("Running Post-QC checks..."):
                    summary_df, results = run_post_qc_checks(merged_dedup, support_files)
                st.session_state.post_qc_summary = summary_df
                st.session_state.post_qc_results = results
                st.session_state.post_qc_data = merged_dedup
                st.session_state.last_processed_files = process_signature
            else:
                std_dfs = []
                for raw_data in all_dfs:
                    std_data = standardize_input_data(raw_data)
                    if 'PRODUCT_SET_SID' in std_data.columns:
                        std_data['PRODUCT_SET_SID'] = std_data['PRODUCT_SET_SID'].astype(str).str.strip()
                        file_sids_sets.append(set(std_data['PRODUCT_SET_SID'].unique()))
                    std_dfs.append(std_data)
                merged_data = pd.concat(std_dfs, ignore_index=True)
                if len(file_sids_sets) > 1: st.session_state.intersection_sids = set.intersection(*file_sids_sets)
                else: st.session_state.intersection_sids = set()
                st.session_state.intersection_count = len(st.session_state.intersection_sids)
                data_prop = propagate_metadata(merged_data)
                is_valid, errors = validate_input_schema(data_prop)
                if is_valid:
                    data_filtered, det_names = filter_by_country(data_prop, country_validator)
                    if data_filtered.empty: 
                        st.error(f"No {country_validator.country} products found. Detected countries: {', '.join(det_names) if det_names else 'None'}", icon=":material/error:")
                        st.stop()
                    actual_counts = data_filtered.groupby('PRODUCT_SET_SID')['PRODUCT_SET_SID'].transform('count')
                    if 'COUNT_VARIATIONS' in data_filtered.columns:
                        file_counts = pd.to_numeric(data_filtered['COUNT_VARIATIONS'], errors='coerce').fillna(1)
                        data_filtered['COUNT_VARIATIONS'] = actual_counts.combine(file_counts, max)
                    else: data_filtered['COUNT_VARIATIONS'] = actual_counts
                    data = data_filtered.drop_duplicates(subset=['PRODUCT_SET_SID'], keep='first')
                    if '_IS_MULTI_COUNTRY' not in data.columns: data['_IS_MULTI_COUNTRY'] = False
                    data_has_warranty = all(c in data.columns for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION'])
                    for c in ['NAME', 'BRAND', 'COLOR', 'SELLER_NAME', 'CATEGORY_CODE', 'LIST_VARIATIONS']:
                        if c in data.columns: data[c] = data[c].astype(str).fillna('')
                    if 'COLOR_FAMILY' not in data.columns: data['COLOR_FAMILY'] = ""

                    data_hash = df_hash(data) + country_validator.code
                    final_report, _ = cached_validate_products(data_hash, data, support_files, country_validator.code, data_has_warranty)

                    st.session_state.final_report = final_report
                    st.session_state.all_data_map = data
                    st.session_state.last_processed_files = process_signature
                else:
                    for e in errors: st.error(e)
                    st.session_state.last_processed_files = "error"
        except Exception as e:
            st.error(f"Processing error: {e}")
            st.code(traceback.format_exc())
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
# SECTION 2: MANUAL IMAGE REVIEW
# ==========================================
@st.fragment
def render_image_grid():
    if st.session_state.final_report.empty or st.session_state.file_mode == "post_qc":
        return

    st.markdown("---")
    st.header(":material/pageview: Manual Image & Category Review", anchor=False)

    action_bridge = st.text_input(
        "bridge",
        key="card_action_bridge",
        label_visibility="collapsed",
        placeholder="__CARD_ACT__",
    )

    if action_bridge:
        changed = _process_card_bridge_action(action_bridge, support_files)
        st.session_state["card_action_bridge"] = ""
        if changed:
            st.rerun(scope="fragment")
        return

    fr = st.session_state.final_report
    committed_rej_sids = {
        k.replace("quick_rej_", "")
        for k in st.session_state.keys()
        if k.startswith("quick_rej_") and "reason" not in k
    }
    mask = (fr["Status"] == "Approved") | (fr["ProductSetSid"].isin(committed_rej_sids))
    valid_grid_df = fr[mask]

    c1, c2, c3 = st.columns([1.5, 1.5, 2])
    with c1: search_n  = st.text_input("Search by Name", placeholder="Product name…")
    with c2: search_sc = st.text_input("Search by Seller/Category", placeholder="Seller or Category…")
    with c3:
        st.session_state.grid_items_per_page = st.select_slider(
            "Items per page", options=[20, 50, 100, 200],
            value=st.session_state.grid_items_per_page,
        )

    available_cols = [c for c in GRID_COLS if c in st.session_state.all_data_map.columns]
    review_data = pd.merge(
        valid_grid_df[["ProductSetSid"]],
        st.session_state.all_data_map[available_cols],
        left_on="ProductSetSid", right_on="PRODUCT_SET_SID", how="left",
    )

    if search_n:
        review_data = review_data[
            review_data["NAME"].astype(str).str.contains(search_n, case=False, na=False)
        ]
    if search_sc:
        mc = (review_data["CATEGORY"].astype(str).str.contains(search_sc, case=False, na=False)
              if "CATEGORY" in review_data.columns else pd.Series(False, index=review_data.index))
        ms = review_data["SELLER_NAME"].astype(str).str.contains(search_sc, case=False, na=False)
        review_data = review_data[mc | ms]

    ipp = st.session_state.grid_items_per_page
    total_pages = max(1, (len(review_data) + ipp - 1) // ipp)
    if st.session_state.grid_page >= total_pages:
        st.session_state.grid_page = 0

    page_start = st.session_state.grid_page * ipp
    page_data  = review_data.iloc[page_start : page_start + ipp]

    page_warnings: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        future_to_sid = {
            ex.submit(analyze_image_quality_cached, str(r.get("MAIN_IMAGE", "")).strip()):
            str(r["PRODUCT_SET_SID"])
            for _, r in page_data.iterrows()
        }
        for future in concurrent.futures.as_completed(future_to_sid):
            warns = future.result()
            if warns:
                page_warnings[future_to_sid[future]] = warns

    rejected_state = {
        sid: st.session_state[f"quick_rej_reason_{sid}"]
        for sid in page_data["PRODUCT_SET_SID"].astype(str)
        if st.session_state.get(f"quick_rej_{sid}")
    }

    cols_per_row = 3 if st.session_state.layout_mode == "centered" else 4
    grid_html = build_fast_grid_html(
        page_data, support_files["flags_mapping"], st.session_state.selected_country,
        page_warnings, rejected_state, cols_per_row,
        st.session_state.grid_page, total_pages,
    )
    components.html(grid_html, height=1300, scrolling=True)

    if st.session_state.get("do_scroll_top", False):
        st.components.v1.html(
            "<script>window.parent.document.querySelector('.main').scrollTo({top:0,behavior:'smooth'});</script>",
            height=0,
        )
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

# Call the fragmented sections
render_image_grid()
render_exports_section()
