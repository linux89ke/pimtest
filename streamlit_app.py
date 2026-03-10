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

# --- IMPORTS FOR IMAGE ADVISOR ---
import requests
from PIL import Image

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
    if not cfg:
        return 1.0
    try:
        import urllib.request, json
        url = f"https://open.er-api.com/v6/latest/USD"
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read())
        rate = data["rates"].get(cfg["code"], 1.0)
        return float(rate)
    except Exception:
        fallbacks = {"Kenya": 128.0, "Uganda": 3750.0, "Nigeria": 1550.0, "Ghana": 15.5, "Morocco": 10.1}
        return fallbacks.get(country, 1.0)

def format_local_price(usd_price, country: str) -> str:
    try:
        price = float(usd_price)
        if price <= 0:
            return ""
        cfg = COUNTRY_CURRENCY.get(country, {})
        rate = fetch_exchange_rate(country)
        local = price * rate
        symbol = cfg.get("symbol", "$")
        if cfg.get("code") in ("KES", "UGX", "NGN"):
            return f"{symbol} {local:,.0f}"
        else:
            return f"{symbol} {local:,.2f}"
    except (ValueError, TypeError):
        return ""

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
    # ── COLOR aliases (English & British spelling) ──
    'color': 'COLOR',
    'colour': 'COLOR',
    # ── COLOR_FAMILY aliases ──
    'color_family': 'COLOR_FAMILY',
    'colour_family': 'COLOR_FAMILY',
    'colour family': 'COLOR_FAMILY',
    'color family': 'COLOR_FAMILY',
    'colour_family': 'COLOR_FAMILY',
    'COLOUR FAMILY': 'COLOR_FAMILY',
    # ── rest of mapping ──
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
if 'intersection_sids' not in st.session_state: st.session_state.intersection_sids = set()
if 'intersection_count' not in st.session_state: st.session_state.intersection_count = 0
if 'grid_page' not in st.session_state: st.session_state.grid_page = 0
if 'grid_items_per_page' not in st.session_state: st.session_state.grid_items_per_page = 50
if 'main_toasts' not in st.session_state: st.session_state.main_toasts = []
if 'exports_cache' not in st.session_state: st.session_state.exports_cache = {}
if 'image_advisor_cache' not in st.session_state: st.session_state.image_advisor_cache = {}
if 'bg_executor' not in st.session_state: st.session_state.bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
if 'do_scroll_top' not in st.session_state: st.session_state.do_scroll_top = False

try:
    st.set_page_config(page_title="Product Tool", layout=st.session_state.layout_mode)
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

        /* --- SCROLLBAR ENHANCEMENTS --- */
        ::-webkit-scrollbar {{
            width: 18px !important;  
            height: 18px !important; 
        }}
        ::-webkit-scrollbar-track {{
            background: {JUMIA_COLORS['light_gray']};
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: {JUMIA_COLORS['medium_gray']}; 
            border-radius: 8px;
            border: 3px solid {JUMIA_COLORS['light_gray']}; 
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {JUMIA_COLORS['primary_orange']};
        }}
        * {{
            scrollbar-width: auto;
            scrollbar-color: {JUMIA_COLORS['medium_gray']} {JUMIA_COLORS['light_gray']};
        }}

        /* --- SLIDER ENHANCEMENTS (Items per page) --- */
        div[data-baseweb="slider"] div[role="slider"] {{
            height: 24px !important;
            width: 24px !important;
            border: 4px solid {JUMIA_COLORS['primary_orange']} !important;
            cursor: pointer !important;
        }}
        div[data-baseweb="slider"] > div > div {{
            height: 12px !important;
        }}

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
            
            /* Dark Mode Scrollbars */
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
        
        /* Prevent product card buttons from wrapping */
        div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button {{
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            font-size: 11px !important;
            padding: 6px 4px !important;
            min-height: 36px !important;
            line-height: 1.2 !important;
        }}
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

if 'selected_country' not in st.session_state:
    st.session_state.selected_country = get_default_country()

if st.session_state.main_toasts:
    for msg in st.session_state.main_toasts:
        if isinstance(msg, tuple):
            st.toast(msg[0], icon=msg[1])
        else:
            st.toast(msg)
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

# --- LOCAL: PROHIBITED PRODUCTS ---
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
            msg = f"⚠️ **File Error:** Failed to load {FILE_NAME} - Tab **{tab}**. (Error: {e})"
            logger.error(msg)
            prohibited_by_country[tab] = []
    return prohibited_by_country

# --- LOCAL: RESTRICTED BRANDS ---
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
            msg = f"⚠️ **File Error:** Failed to load {FILE_NAME} - Tab **{tab_name}**. (Error: {e})"
            logger.error(msg)
            config_by_country[country_name] = []
    return config_by_country

# --- LOCAL: REFURBISHED DATA ---
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
            msg = f"⚠️ **File Error:** Failed to load {FILE_NAME} - Sellers Tab **{tab}**. (Error: {e})"
            logger.error(msg)
            result["sellers"][tab] = {"Phones": set(), "Laptops": set()}
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories", usecols=[0, 1])
        if df_cats.empty:
            df_cats = safe_excel_read(FILE_NAME, sheet_name="Categries", usecols=[0, 1])
            
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            result["categories"]["Phones"] = {clean_category_code(c) for c in df_cats.iloc[:, 0].dropna().astype(str) if c.strip() and c.strip().lower() not in ("phones", "phone", "nan")}
            result["categories"]["Laptops"] = {clean_category_code(c) for c in df_cats.iloc[:, 1].dropna().astype(str) if c.strip() and c.strip().lower() not in ("laptops", "laptop", "nan")}
    except Exception as e:
        logger.error(f"Error loading {FILE_NAME}: {e}")
    try:
        df_names = safe_excel_read(FILE_NAME, sheet_name="Name", usecols=[0])
        if not df_names.empty:
            first_col = df_names.columns[0]
            result["keywords"] = {k for k in df_names[first_col].dropna().astype(str).str.strip().str.lower() if k and k not in ("name", "keyword", "keywords", "words", "nan")}
    except Exception as e:
        logger.error(f"Error loading {FILE_NAME}: {e}")
        result["keywords"] = {"refurb", "refurbished", "renewed"}
    return result

# --- LOCAL: PERFUME DATA ---
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
    except Exception as e:
        logger.error(f"Error: {e}")
        result["keywords"] = set()

    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if 'cat' in c.lower()), df_cats.columns[0])
            cat_codes = set(df_cats[cat_col].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])]))
            result["category_codes"] = cat_codes
    except Exception as e:
        logger.error(f"Error: {e}")
        result["category_codes"] = set()
    return result

# --- LOCAL: BOOKS DATA ---
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
        except Exception as e:
            logger.error(f"Error: {e}")
            result["sellers"][tab] = set()

    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if 'cat' in c.lower()), df_cats.columns[0])
            cat_codes = set(df_cats[cat_col].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])]))
            result["category_codes"] = cat_codes
    except Exception as e:
        logger.error(f"Error: {e}")
        result["category_codes"] = set()
    return result

# --- LOCAL: JERSEYS DATA ---
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
        except Exception as e:
            logger.error(f"Error: {e}")

    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip().lower() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if "cat" in c), df_cats.columns[0])
            result["categories"] = set(df_cats[cat_col].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])]))
    except Exception as e:
        logger.error(f"Error: {e}")
    return result

# --- LOCAL: SUSPECTED FAKE DATA ---
@st.cache_data(ttl=3600)
def load_suspected_fake_from_local() -> pd.DataFrame:
    try:
        if os.path.exists('suspected_fake.xlsx'):
            return pd.read_excel('suspected_fake.xlsx', sheet_name=0, engine='openpyxl', dtype=str)
    except Exception as e:
        logger.warning(f"Error: {e}")
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
                    if flag and flag.lower() != 'nan':
                        custom_mapping[flag] = (reason, comment)
                if custom_mapping: return custom_mapping
    except Exception as e:
        logger.error(f"Error loading external {filename}: {e}")
    return default_mapping

@st.cache_data(ttl=3600)
def load_all_support_files() -> Dict:
    def safe_load_txt(f): return load_txt_file(f) if os.path.exists(f) else []
    return {
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
def load_support_files_lazy():
    return load_all_support_files()

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
        if col_lower in map_lower:
            renamed[col] = map_lower[col_lower]
        else:
            renamed[col] = col.upper()
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
    if not {'NAME', 'BRAND', 'SELLER_NAME', 'CATEGORY_CODE'}.issubset(data.columns) or not country_rules:
        return pd.DataFrame(columns=data.columns)
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
        for idx in d[main_brand_matches].index:
            match_details[idx] = ('main_brand', brand_raw)
        for idx in d[main_name_matches & ~main_brand_matches].index:
            match_details[idx] = ('main_name', brand_raw)
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
        if rule['categories']:
            current_match = current_match[current_match['_cat_clean'].isin(rule['categories'])]
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
    if not {'NAME', 'CATEGORY_CODE'}.issubset(data.columns) or not prohibited_rules:
        return pd.DataFrame(columns=data.columns)
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
        if target_cats:
            current_match = current_match[current_match['_cat_clean'].isin(target_cats)]
        if current_match.empty: continue
        for idx in current_match.index:
            flagged_indices.add(idx)
            existing_comment = comment_map.get(idx, "Prohibited:")
            if keyword not in existing_comment:
                comment_map[idx] = f"{existing_comment} {keyword},"
            raw_name = str(d.loc[idx, 'NAME'])
            highlighted = pattern.sub(lambda m: f"[!]{m.group(0)}[!]", raw_name)
            name_replacements[idx] = highlighted
    if not flagged_indices: return pd.DataFrame(columns=data.columns)
    result = data.loc[list(flagged_indices)].copy()
    result['Comment_Detail'] = result.index.map(lambda i: comment_map[i].rstrip(','))
    for idx, new_name in name_replacements.items():
        result.loc[idx, 'NAME'] = new_name
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
    if not flagged.empty:
        flagged['Comment_Detail'] = "Seller not approved to sell books: " + flagged['SELLER_NAME'].astype(str)
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
    else:
        sneaky_mask = pd.Series([False] * len(perfume), index=perfume.index)
    if has_seller_list:
        if keywords: brand_sens_mask = b_lower.apply(lambda x: bool(kw_pattern.search(x))) if keywords else pd.Series([False]*len(perfume), index=perfume.index)
        else: brand_sens_mask = pd.Series([False] * len(perfume), index=perfume.index)
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
    if not flagged.empty:
        flagged['Comment_Detail'] = "Variations: " + flagged['qty_var'].astype(str) + ", Category: " + flagged['cat_clean']
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_generic_with_brand_in_name(data: pd.DataFrame, brands_list: List[str]) -> pd.DataFrame:
    if not {'NAME', 'BRAND'}.issubset(data.columns) or not brands_list: return pd.DataFrame(columns=data.columns)
    mask = (data['BRAND'].astype(str).str.strip().str.lower() == 'generic')
    if 'CATEGORY' in data.columns:
        mask = mask & ~data['CATEGORY'].astype(str).str.lower().str.contains(r'\b(case|cases|cover|covers)\b', regex=True, na=False)
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
    if not flagged.empty:
        flagged['Comment_Detail'] = "Detected Brand: " + flagged['Detected_Brand']
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
    if not flagged.empty:
        flagged['Comment_Detail'] = "Name missing Storage/Memory spec (e.g., 64GB)"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_duplicate_products(data: pd.DataFrame, exempt_categories: List[str] = None, similarity_threshold: float = 0.70, known_colors: List[str] = None, **kwargs) -> pd.DataFrame:
    dt = int(similarity_threshold * 100) if similarity_threshold <= 1 else int(similarity_threshold)
    if not {'NAME', 'SELLER_NAME', 'BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    if exempt_categories and 'CATEGORY_CODE' in d.columns:
        d = d[~d['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in exempt_categories))]
    if d.empty: return pd.DataFrame(columns=data.columns)
    names = d['NAME'].values
    colors = d['COLOR'].values if 'COLOR' in d.columns else [None] * len(d)
    brands = d['BRAND'].values if 'BRAND' in d.columns else [None] * len(d)
    d['_attrs'] = [extract_product_attributes(n, c, b) for n, c, b in zip(names, colors, brands)]
    d['_base'] = d['_attrs'].apply(lambda x: x.get_base_key())
    d['_var'] = d['_attrs'].apply(lambda x: x.get_variant_key().replace(x.get_base_key(), ''))
    d['_seller'] = d['SELLER_NAME'].astype(str).str.strip().str.lower()
    rej = set()
    details = {}
    try:
        import importlib.util
        if importlib.util.find_spec('sklearn') is None: raise ImportError("sklearn not found")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        for _, group in d.groupby(['_seller', '_var']):
            if len(group) < 2: continue
            sids, bases, attrs = group['PRODUCT_SET_SID'].tolist(), group['_base'].tolist(), group['_attrs'].tolist()
            try:
                sim = cosine_similarity(TfidfVectorizer(analyzer='word').fit_transform(bases))
            except ValueError:
                sim = [[1.0]*len(bases) for _ in range(len(bases))]
            tsim = (dt - 30) / 70.0
            for i in range(len(sids)):
                if str(sids[i]) in rej: continue
                for j in range(i + 1, len(sids)):
                    if str(sids[j]) in rej: continue
                    if sim[i][j] >= tsim:
                        rej.add(str(sids[j]))
                        vdesc = []
                        if attrs[i].colors: vdesc.append(f"Color: {','.join(attrs[i].colors)}")
                        if attrs[i].sizes: vdesc.append(f"Size: {','.join(attrs[i].sizes)}")
                        if attrs[i].storage: vdesc.append(f"Storage: {','.join(attrs[i].storage)}")
                        if attrs[i].memory: vdesc.append(f"Memory: {','.join(attrs[i].memory)}")
                        if attrs[i].quantities: vdesc.append(f"Qty: {','.join(attrs[i].quantities)}")
                        details[str(sids[j])] = {'base': bases[i][:40], 'variant': ",".join(vdesc) or "Same specs", 'score': (sim[i][j]*70)+30}
    except ImportError:
        for _, group in d.groupby(['_seller', '_base', '_var']):
            if len(group) < 2: continue
            sids = group['PRODUCT_SET_SID'].tolist()
            for j in range(1, len(sids)):
                rej.add(str(sids[j]))
                details[str(sids[j])] = {'base': group['_base'].tolist()[0][:40], 'variant': 'Same specs', 'score': 100}
    if not rej: return pd.DataFrame(columns=data.columns)
    rdf = d[d['PRODUCT_SET_SID'].astype(str).isin(rej)].copy()
    def apply_dup_comment(sid):
        if str(sid) in details: return f"Duplicate: Base '{details[str(sid)]['base']}', {details[str(sid)]['variant']}, Conf: {details[str(sid)]['score']:.0f}%"
        return "Duplicate detected"
    rdf['Comment_Detail'] = rdf['PRODUCT_SET_SID'].apply(apply_dup_comment)
    cols = data.columns.tolist()
    if 'Comment_Detail' not in cols: cols.append('Comment_Detail')
    return rdf[cols].drop_duplicates(subset=['PRODUCT_SET_SID'])

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
                for sid in v:
                    dup_groups[sid] = v
    restricted_keys = {}
    validation_errors = []
    with st.status("Validating products...", expanded=True) as status:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_name = {}
            for i, (name, func, kwargs) in enumerate(validations):
                if skip_validators and name in skip_validators: continue
                if country_validator.should_skip_validation(name): continue
                ckwargs = {'data': data, **kwargs}
                if name in ["Generic BRAND Issues", "Fashion brand issues"]:
                    ckwargs['valid_category_codes_fas'] = support_files.get('category_fas', [])
                future_to_name[executor.submit(func, **ckwargs)] = name
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                st.write(f"Completed: {name}")
                try:
                    res = future.result()
                    if not res.empty and 'PRODUCT_SET_SID' in res.columns:
                        res = res.loc[:, ~res.columns.duplicated()].copy()
                        res['PRODUCT_SET_SID'] = res['PRODUCT_SET_SID'].astype(str).str.strip()
                        if name in ["Seller Approve to sell books", "Seller Approved to Sell Perfume", "Counterfeit Sneakers", "Seller Not approved to sell Refurb", "Restricted brands"]:
                            res['match_key'] = res.apply(create_match_key, axis=1)
                            restricted_keys.setdefault(name, set()).update(res['match_key'].unique())
                        expanded_sids = set()
                        for sid in set(res['PRODUCT_SET_SID'].unique()):
                            expanded_sids.update(dup_groups.get(sid, [sid]))
                        final_res = data[data['PRODUCT_SET_SID'].isin(expanded_sids)].copy()
                        if 'Comment_Detail' in res.columns:
                            final_res['Comment_Detail'] = res['Comment_Detail']
                        if name in results and not results[name].empty:
                            results[name] = pd.concat([results[name], final_res]).drop_duplicates(subset=['PRODUCT_SET_SID'])
                        else:
                            results[name] = final_res
                    else:
                        if name not in results:
                            results[name] = pd.DataFrame(columns=data.columns)
                except Exception as e:
                    logger.error(f"Error in {name}: {e}")
                    validation_errors.append((name, str(e)))
                    if name not in results:
                        results[name] = pd.DataFrame(columns=data.columns)
        status.update(label="Validation complete!", state="complete", expanded=False)
    if validation_errors:
        st.warning(f"{len(validation_errors)} validation checks encountered errors.")
        with st.expander("View Error Details"):
            for e_name, e_msg in validation_errors:
                st.error(f"**{e_name}**: {e_msg}")
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
            if isinstance(res['Comment_Detail'], pd.DataFrame):
                flagged['Comment_Detail'] = res['Comment_Detail'].iloc[:, 0]
            else:
                flagged['Comment_Detail'] = res['Comment_Detail']
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
        if auxiliary_df is not None and not auxiliary_df.empty:
            to_excel_base(auxiliary_df, aux_sheet_name, aux_cols, writer)
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
        d_cp = data_df.copy()
        r_cp = final_report_df.copy()
        d_cp['PRODUCT_SET_SID'] = d_cp['PRODUCT_SET_SID'].astype(str).str.strip()
        r_cp['ProductSetSid'] = r_cp['ProductSetSid'].astype(str).str.strip()
        merged = pd.merge(d_cp, r_cp[["ProductSetSid", "Status", "Reason", "Comment", "FLAG", "SellerName"]], left_on="PRODUCT_SET_SID", right_on="ProductSetSid", how='left')
        if 'ProductSetSid' in merged.columns:
            merged.drop(columns=['ProductSetSid'], inplace=True)
        return merged
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------
# UI COMPONENTS & FRAGMENTS
# -------------------------------------------------
def apply_rejection(sids: list, reason_code: str, comment: str, flag_name: str):
    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(sids), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', reason_code, comment, flag_name]
    st.session_state.exports_cache.clear()

def restore_single_item(sid):
    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status', 'Reason', 'Comment', 'FLAG']] = ['Approved', '', '', 'Approved by User']
    st.session_state.pop(f"quick_rej_{sid}", None)
    st.session_state.pop(f"quick_rej_reason_{sid}", None)
    st.session_state.exports_cache.clear()
    st.session_state.main_toasts.append("Restored item to previous state!")

def quick_reject_item(sid, reason_code, comment, flag_name, toast_name):
    apply_rejection([sid], reason_code, comment, flag_name)
    st.session_state[f"quick_rej_{sid}"] = True
    st.session_state[f"quick_rej_reason_{sid}"] = flag_name
    st.session_state[f"toast_{sid}"] = f"{toast_name} Rejected: {flag_name}"

def strip_html(text):
    if not isinstance(text, str): return text
    return re.sub('<[^<]+?>', '', text)

def analyze_image_quality(url: str, cache_dict: dict) -> List[str]:
    if not url or not str(url).startswith("http"): return []
    if url in cache_dict: return cache_dict[url]
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
    cache_dict[url] = warnings
    return warnings

def render_product_card(row, flags_mapping, country: str = 'Kenya', advisor_warnings=None):
    if advisor_warnings is None: advisor_warnings = []
    sid = str(row['PRODUCT_SET_SID'])
    toast_key = f"toast_{sid}"
    raw_name = str(row.get('NAME', ''))
    name_text = raw_name[:35] + ("..." if len(raw_name) > 35 else "")
    raw_name_attr = (raw_name.replace('&', '&amp;').replace('"', '&quot;').replace("'", '&#39;').replace('<', '&lt;').replace('>', '&gt;'))
    toast_name = raw_name[:25] + "..." if len(raw_name) > 25 else raw_name
    cat_text = str(row.get('CATEGORY', 'Unknown Category'))
    brand_text = str(row.get('BRAND', 'Unknown Brand')).strip()
    seller_text = str(row.get('SELLER_NAME', 'Unknown Seller'))
    color_val = str(row.get('COLOR', '')).strip()
    color_display = color_val if color_val and color_val.lower() not in ['nan', '', 'none', 'null'] else ''

    if toast_key in st.session_state:
        msg = st.session_state.pop(toast_key)
        if isinstance(msg, tuple): st.toast(msg[0], icon=msg[1])
        else: st.toast(msg)

    img_url = str(row.get('MAIN_IMAGE', '')).strip()
    if not img_url.startswith('http'): img_url = "https://via.placeholder.com/150?text=No+Image"

    is_rejected = st.session_state.get(f"quick_rej_{sid}")
    reason = st.session_state.get(f"quick_rej_reason_{sid}")
    color_badge_html = (f"<span class='color-badge'>{color_display}</span><br>" if color_display else "")

    def _to_float(v):
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError): return None
            
    raw_price = _to_float(row.get('GLOBAL_SALE_PRICE')) or _to_float(row.get('GLOBAL_PRICE')) or 0
    price_str = format_local_price(raw_price, country)
    price_overlay_html = (f"<div style='position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,0.72);color:#fff;font-size:12px;font-weight:700;padding:3px 8px;border-radius:6px;letter-spacing:0.3px;z-index:5;'>{price_str}</div>" if price_str else "")

    img_code, img_cmt = flags_mapping.get('Poor images', ('1000042', 'Poor Image Quality'))
    cat_code, cat_cmt = flags_mapping.get('Wrong Category', ('1000004', 'Wrong Category'))
    fake_code, fake_cmt = flags_mapping.get('Suspected Fake product', ('1000023', 'Fake Product'))
    brnd_code, brnd_cmt = flags_mapping.get('Restricted brands', ('1000024', 'Restricted brands'))
    proh_code, proh_cmt = flags_mapping.get('Prohibited products', ('1000007', 'Prohibited product'))
    color_code, color_cmt = flags_mapping.get('Missing COLOR', ('1000005', 'Missing/Wrong Color'))
    wrong_brand_code, wrong_brand_cmt = flags_mapping.get('Generic branded products with genuine brands', ('1000007', 'Use the displayed brand on the product instead of Generic.'))

    with st.container(border=True):
        if is_rejected:
            st.markdown(f"""<div style="position: relative;"><img src="{img_url}" loading="lazy" style="width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background-color: #FFFFFF; border-radius: 8px; border: 1px solid #eee; opacity: 0.4; filter: grayscale(100%);"><div class="rejected-card-overlay"><span class="material-symbols-outlined" style="margin-right: 6px; font-size: 16px;">block</span> REJECTED</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size: 13px; line-height: 1.4; margin: 8px 0;"><span class="prod-name-tip" data-full="{raw_name_attr}">{name_text}</span>{color_badge_html}<div class="prod-meta-text" style="font-size: 11px; margin-bottom: 4px;">{cat_text}</div><div class="prod-brand-text" style="font-weight: 600; margin-bottom: 8px;">{brand_text}</div><div class="prod-meta-text" style="font-size: 10px; padding-top: 8px; border-top: 1px dashed #E0E0E0;">{seller_text}</div></div>""", unsafe_allow_html=True)
            col_msg, col_btn = st.columns([3.5, 1])
            with col_msg:
                st.markdown(f"""<div style="background: linear-gradient(135deg, {JUMIA_COLORS['jumia_red']}, #FF6B6B); color: white; padding: 8px 12px; border-radius: 6px; font-weight: bold; font-size: 11px; text-align: center;">{reason}</div>""", unsafe_allow_html=True)
            with col_btn:
                st.button(":material/undo:", key=f"res_{sid}", on_click=restore_single_item, args=(sid,), use_container_width=True, help="Undo rejection")
        else:
            col_chk, col_zm = st.columns([3, 1.5], vertical_alignment="center")
            with col_chk:
                st.checkbox("Select", key=f"grid_chk_{sid}")
            with col_zm:
                with st.popover(":material/search:", use_container_width=True):
                    st.image(img_url, use_container_width=True)
                    st.caption(raw_name)

            is_checked = st.session_state.get(f"grid_chk_{sid}", False)
            warnings_html = ""
            if advisor_warnings:
                badges = "".join([f'<div style="background:rgba(255,193,7,0.95);color:#313133;font-size:10px;font-weight:800;padding:4px 8px;border-radius:12px;box-shadow:0 2px 4px rgba(0,0,0,0.2);display:flex;align-items:center;margin-bottom:4px;"><span class="material-symbols-outlined" style="font-size:13px;margin-right:4px;">warning</span>{w}</div>' for w in advisor_warnings])
                warnings_html = f'<div style="position:absolute;top:8px;right:8px;display:flex;flex-direction:column;z-index:10;">{badges}</div>'

            border_style = "3px solid #4CAF50" if is_checked else "1px solid #eee"
            box_shadow = "0 0 0 3px rgba(76,175,80,0.2), 0 4px 16px rgba(76,175,80,0.15)" if is_checked else "none"
            green_overlay = ("<div style='position:absolute;inset:0;background:rgba(76,175,80,0.1);border-radius:8px;pointer-events:none;z-index:2;'></div>" if is_checked else "")
            tick_html = ("<div style='position:absolute;bottom:10px;right:10px;width:28px;height:28px;background:#4CAF50;border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,0.3);z-index:10;'><span class='material-symbols-outlined' style='color:#fff;font-size:18px;line-height:1;font-weight:bold;'>check</span></div>" if is_checked else "")
            img_div_id = f"imgclick-{sid}"

            st.markdown(
                f'<div id="{img_div_id}" style="position:relative;cursor:pointer;border-radius:10px;border:{border_style};box-shadow:{box_shadow};transition:border 0.15s ease,box-shadow 0.15s ease;overflow:hidden;margin-bottom:8px;">'
                f'{warnings_html}{green_overlay}'
                f'<img src="{img_url}" loading="lazy" style="width:100%;aspect-ratio: 1 / 1;object-fit:contain;background-color:#FFFFFF;border-radius:8px;display:block;">'
                f'{price_overlay_html}{tick_html}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""<div style="font-size:13px;line-height:1.4;margin:8px 0;">
<span class='prod-name-tip' data-full="{raw_name_attr}">{name_text}</span>{color_badge_html}
<div class='prod-meta-text' style="font-size:11px;margin-bottom:4px;">{cat_text}</div>
<div class='prod-brand-text' style="color:{JUMIA_COLORS['primary_orange']};font-weight:600;margin-bottom:8px;">{brand_text}</div>
<div class='prod-meta-text' style="font-size:10px;padding-top:8px;border-top:1px dashed #E0E0E0;">{seller_text}</div>
</div>""", unsafe_allow_html=True)

            col_img, col_more = st.columns([1.2, 0.8], gap="small")
            with col_img:
                st.button("Poor Image", key=f"btn_img_{sid}", use_container_width=True, on_click=quick_reject_item, args=(sid, img_code, img_cmt, 'Poor images', toast_name), type="primary", help="Reject: Poor Image Quality")
            with col_more:
                with st.popover("✕ More", use_container_width=True):
                    st.markdown("<p style='font-size:12px;font-weight:700;margin:0 0 8px 0;'>Select rejection reason:</p>", unsafe_allow_html=True)
                    st.button("Wrong Category",     key=f"cat_{sid}",   use_container_width=True, on_click=quick_reject_item, args=(sid, cat_code,         cat_cmt,         'Wrong Category',                                toast_name))
                    st.button("Fake Product",       key=f"fake_{sid}",  use_container_width=True, on_click=quick_reject_item, args=(sid, fake_code,        fake_cmt,        'Suspected Fake product',                        toast_name))
                    st.button("Restricted Brand",   key=f"brnd_{sid}",  use_container_width=True, on_click=quick_reject_item, args=(sid, brnd_code,        brnd_cmt,        'Restricted brands',                             toast_name))
                    st.button("Prohibited Product", key=f"proh_{sid}",  use_container_width=True, on_click=quick_reject_item, args=(sid, proh_code,        proh_cmt,        'Prohibited products',                           toast_name))
                    st.button("Wrong Color",        key=f"colr_{sid}",  use_container_width=True, on_click=quick_reject_item, args=(sid, color_code,       color_cmt,       'Missing COLOR',                                 toast_name))
                    st.button("Wrong Brand",        key=f"wbrnd_{sid}", use_container_width=True, on_click=quick_reject_item, args=(sid, wrong_brand_code, wrong_brand_cmt, 'Generic branded products with genuine brands',  toast_name))
                    st.divider()
                    st.markdown("<p style='font-size:11px;font-weight:700;margin:0 0 4px 0;'>Other Reason (Custom)</p>", unsafe_allow_html=True)
                    custom_cmt_input = st.text_area("Custom comment", placeholder="Type rejection reason...", key=f"custom_cmt_{sid}", height=70, label_visibility="collapsed")
                    def _apply_custom(sid=sid, toast_name=toast_name):
                        cmt = st.session_state.get(f"custom_cmt_{sid}", "").strip()
                        if not cmt:
                            st.session_state.main_toasts.append(("Please enter a custom comment.", "⚠️"))
                            return
                        quick_reject_item(sid, "1000007 - Other Reason", cmt, "Other Reason (Custom)", toast_name)
                    st.button("Apply Custom Rejection", key=f"custom_apply_{sid}", use_container_width=True, type="primary", on_click=_apply_custom)

@st.dialog("Confirm Bulk Approval")
def bulk_approve_dialog(sids_to_process, title, subset_data, data_has_warranty_cols_check, support_files, country_validator):
    st.warning(f"You are about to approve **{len(sids_to_process)}** items from `{title}`.")
    if st.button("Confirm Approval", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            new_report, _ = validate_products(subset_data, support_files, country_validator, data_has_warranty_cols_check, skip_validators=[title])
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
            if msg_approved > 0:
                st.session_state.main_toasts.append(f"{msg_approved} items successfully Approved!")
            for flag, count in msg_moved.items():
                st.session_state.main_toasts.append(f"{count} items re-flagged as: {flag}")
            st.session_state.exports_cache.clear()
        st.rerun()

def render_flag_expander(title, df_display, subset_data, data_has_warranty_cols_check, support_files, country_validator):
    c1, c2 = st.columns([1, 1])
    with c1: search_term = st.text_input("Search", placeholder="Name, Brand...", key=f"s_{title}")
    with c2: seller_filter = st.multiselect("Filter by Seller", sorted(df_display['SELLER_NAME'].astype(str).unique()), key=f"f_{title}")
    if search_term: df_display = df_display[df_display.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    if seller_filter: df_display = df_display[df_display['SELLER_NAME'].isin(seller_filter)]
    df_display = df_display.reset_index(drop=True)
    if 'NAME' in df_display.columns: df_display['NAME'] = df_display['NAME'].apply(strip_html)
    event = st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        selection_mode="multi-row",
        on_select="rerun",
        column_config={
            "PRODUCT_SET_SID": st.column_config.TextColumn(pinned=True),
            "NAME": st.column_config.TextColumn(pinned=True),
            # ── PRICE COLUMNS ──────────────────────────────────────────────
            "GLOBAL_SALE_PRICE": st.column_config.NumberColumn(
                "Sale Price (USD)",
                format="$%.2f",
                help="Global sale price in USD",
            ),
            "GLOBAL_PRICE": st.column_config.NumberColumn(
                "Price (USD)",
                format="$%.2f",
                help="Global listed price in USD",
            ),
        },
        key=f"df_{title}"
    )
    raw_selected_indices = list(event.selection.rows)
    selected_indices = [i for i in raw_selected_indices if i < len(df_display)]
    st.caption(f"{len(selected_indices)} of {len(df_display)} rows selected")
    has_selection = len(selected_indices) > 0

    # Build ordered rejection reason options from flags_mapping
    _fm = support_files['flags_mapping']
    _reason_options = [
        "Wrong Category",
        "Restricted brands",
        "Suspected Fake product",
        "Seller Not approved to sell Refurb",
        "Product Warranty",
        "Seller Approve to sell books",
        "Seller Approved to Sell Perfume",
        "Counterfeit Sneakers",
        "Suspected counterfeit Jerseys",
        "Prohibited products",
        "Unnecessary words in NAME",
        "Single-word NAME",
        "Generic BRAND Issues",
        "Fashion brand issues",
        "BRAND name repeated in NAME",
        "Wrong Variation",
        "Generic branded products with genuine brands",
        "Missing COLOR",
        "Missing Weight/Volume",
        "Incomplete Smartphone Name",
        "Duplicate product",
        "Poor images",
        "Other Reason (Custom)",
    ]

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("✓ Approve Selected", key=f"approve_sel_{title}", type="primary", use_container_width=True, disabled=not has_selection):
            sids_to_process = df_display.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
            bulk_approve_dialog(sids_to_process, title, subset_data[subset_data['PRODUCT_SET_SID'].isin(sids_to_process)], data_has_warranty_cols_check, support_files, country_validator)
    with btn_col2:
        with st.popover("↓ Reject As...", use_container_width=True, disabled=not has_selection):
            st.markdown("<p style='font-size:12px;font-weight:700;margin:0 0 8px 0;'>Select rejection reason:</p>", unsafe_allow_html=True)
            chosen_reason = st.selectbox(
                "Reason", _reason_options,
                key=f"rej_reason_dd_{title}",
                label_visibility="collapsed"
            )
            if chosen_reason == "Other Reason (Custom)":
                custom_comment = st.text_area(
                    "Custom comment", placeholder="Type your rejection reason here...",
                    key=f"custom_comment_{title}", height=80
                )
                if st.button("Apply Custom Rejection", key=f"apply_custom_{title}", type="primary",
                             use_container_width=True, disabled=not has_selection):
                    to_reject = df_display.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    final_comment = custom_comment.strip() if custom_comment.strip() else "Other Reason"
                    st.session_state.final_report.loc[
                        st.session_state.final_report['ProductSetSid'].isin(to_reject),
                        ['Status', 'Reason', 'Comment', 'FLAG']
                    ] = ['Rejected', '1000007 - Other Reason', final_comment, 'Other Reason (Custom)']
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected with custom reason.")
                    st.session_state.exports_cache.clear()
                    st.rerun()
            else:
                _rcode, _rcmt = _fm.get(chosen_reason, ('1000007 - Other Reason', chosen_reason))
                st.caption(f"Code: {_rcode[:40]}...")
                if st.button("Apply Rejection", key=f"apply_dd_{title}", type="primary",
                             use_container_width=True, disabled=not has_selection):
                    to_reject = df_display.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    st.session_state.final_report.loc[
                        st.session_state.final_report['ProductSetSid'].isin(to_reject),
                        ['Status', 'Reason', 'Comment', 'FLAG']
                    ] = ['Rejected', _rcode, _rcmt, chosen_reason]
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected as '{chosen_reason}'.")
                    st.session_state.exports_cache.clear()
                    st.rerun()

# -------------------------------------------------
# APP INITIALIZATION
# -------------------------------------------------
try:
    support_files = load_support_files_lazy()
except Exception as e:
    st.error(f"Failed to load configs: {e}")
    st.stop()

def get_image_base64(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception: return ""
    return ""

logo_base64 = get_image_base64("jumia logo.png") or get_image_base64("jumia_logo.png")
logo_html = f"<img src='data:image/png;base64,{logo_base64}' style='height: 42px; margin-right: 15px;'>" if logo_base64 else "<span class='material-symbols-outlined' style='font-size: 42px; margin-right: 15px;'>verified_user</span>"

# --- BACK TO TOP BUTTON (FLOATING) ---
st.markdown(f"""
    <div class="back-to-top" onclick="window.parent.document.querySelector('.main').scrollTo({{top: 0, behavior: 'smooth'}});" title="Back to Top">
        <span class="material-symbols-outlined">arrow_upward</span>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='background: linear-gradient(135deg, {JUMIA_COLORS['primary_orange']}, {JUMIA_COLORS['secondary_orange']}); padding: 25px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(246, 139, 30, 0.3);'>
    <h1 style='color: white; margin: 0; font-size: 36px; display: flex; align-items: center;'>{logo_html}Product Validation Tool</h1>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("System Status")
    if st.button("🔄 Clear Cache & Reload Data", use_container_width=True, type="secondary", help="Forces a reload of all support rules from local files."):
        st.cache_data.clear()
        st.rerun()
        
    st.markdown("---")
    st.header("Display Settings")
    new_mode = "wide" if "Wide" in st.radio("Layout Mode", ["Centered", "Wide"], index=1 if st.session_state.layout_mode == "wide" else 0) else "centered"
    if new_mode != st.session_state.layout_mode:
        st.session_state.layout_mode = new_mode
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
else:
    process_signature = "empty"

if st.session_state.get('last_processed_files') != process_signature:
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.intersection_sids = set()
    st.session_state.intersection_count = 0
    st.session_state.grid_page = 0
    st.session_state.exports_cache = {}
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith(("quick_rej_", "grid_chk_", "toast_"))]
    for k in keys_to_delete: del st.session_state[k]
    
    if process_signature == "empty":
        st.session_state.last_processed_files = "empty"
    else:
        try:
            all_dfs = []
            file_sids_sets = []
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
                std_data = standardize_input_data(raw_data)
                if 'PRODUCT_SET_SID' in std_data.columns:
                    std_data['PRODUCT_SET_SID'] = std_data['PRODUCT_SET_SID'].astype(str).str.strip()
                    file_sids_sets.append(set(std_data['PRODUCT_SET_SID'].unique()))
                all_dfs.append(std_data)
            merged_data = pd.concat(all_dfs, ignore_index=True)
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
                # Ensure the multi-country flag column survives deduplication
                if '_IS_MULTI_COUNTRY' not in data.columns:
                    data['_IS_MULTI_COUNTRY'] = False
                data_has_warranty = all(c in data.columns for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION'])
                for c in ['NAME', 'BRAND', 'COLOR', 'SELLER_NAME', 'CATEGORY_CODE', 'LIST_VARIATIONS']:
                    if c in data.columns: data[c] = data[c].astype(str).fillna('')
                if 'COLOR_FAMILY' not in data.columns: data['COLOR_FAMILY'] = ""
                final_report, _ = validate_products(data, support_files, country_validator, data_has_warranty, None)
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
# RESULTS SECTION
# ==========================================
if uploaded_files and not st.session_state.final_report.empty:
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
            (
                "Multi-Country SKUs" if is_nigeria else "Common SKUs",
                multi_count if is_nigeria else st.session_state.intersection_count,
                JUMIA_COLORS['warning_yellow'] if is_nigeria else JUMIA_COLORS['medium_gray']
            ),
        ]
        for i, (label, value, color) in enumerate(metrics_config):
            with cols[i % len(cols)]:
                st.markdown(f"""
<div class="metric-card-inner" style='text-align: center; padding: 18px 12px; background: {JUMIA_COLORS['light_gray']}; border-radius: 8px; border-left: 4px solid {color};'>
    <div class="metric-card-value" style='font-size: 28px; font-weight: 700; color: {color}; margin-bottom: 4px;'>{value}</div>
    <div class="metric-card-label" style='font-size: 11px; color: {JUMIA_COLORS['medium_gray']}; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;'>{label}</div>
</div>
""", unsafe_allow_html=True)

    st.subheader(":material/flag: Flags Breakdown", anchor=False)
    if not rej_df.empty:
        # ── base_display_cols now includes price columns ──────────────────
        base_display_cols = [
            'PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'COLOR',
            'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE',
            'PARENTSKU', 'SELLER_NAME'
        ]
        for title in rej_df['FLAG'].unique():
            df_flagged = rej_df[rej_df['FLAG'] == title]
            current_display_cols = base_display_cols.copy()
            if title == "Wrong Variation":
                if 'COUNT_VARIATIONS' in data.columns: current_display_cols.append('COUNT_VARIATIONS')
                if 'LIST_VARIATIONS' in data.columns: current_display_cols.append('LIST_VARIATIONS')
            # Safe merge
            df_display = pd.merge(
                df_flagged[['ProductSetSid']],
                data,
                left_on='ProductSetSid',
                right_on='PRODUCT_SET_SID',
                how='left'
            )[[c for c in current_display_cols if c in data.columns]]
            with st.expander(f"{title} ({len(df_display)})"):
                render_flag_expander(title, df_display, data, all(c in data.columns for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION']), support_files, country_validator)
    else:
        st.success("All products passed validation — no rejections found.")

# ==========================================
# SECTION 2: MANUAL IMAGE REVIEW
# ==========================================
if not st.session_state.final_report.empty:
    st.markdown("---")
    st.header(":material/pageview: Manual Image & Category Review", anchor=False)

    components.html(
        """
        <script>
        const doc = window.parent.document;
        if (!doc.getElementById("global-img-click-listener")) {
            let scriptTag = doc.createElement("script");
            scriptTag.id = "global-img-click-listener";
            scriptTag.text = `
                document.addEventListener("click", function(e) {
                    let target = e.target.closest('div[id^="imgclick-"]');
                    if (target) {
                        e.preventDefault();
                        e.stopPropagation();
                        let card = target.closest('[data-testid="stVerticalBlockBorderWrapper"]') || target.closest('[data-testid="stVerticalBlock"]');
                        if (card) {
                            let cb = card.querySelector('input[type="checkbox"]');
                            if (cb) cb.click();
                        }
                    }
                });
            `;
            doc.body.appendChild(scriptTag);
        }
        </script>
        """,
        height=0, width=0
    )

    fr = st.session_state.final_report
    quick_rej_sids = [k.replace("quick_rej_", "") for k in st.session_state.keys() if k.startswith("quick_rej_") and "reason" not in k]
    mask = (fr['Status'] == 'Approved') | (fr['ProductSetSid'].isin(quick_rej_sids))
    valid_grid_df = fr[mask]

    c_rev_1, c_rev_2, c_rev_3 = st.columns([1.5, 1.5, 2])
    with c_rev_1: search_n = st.text_input("Search by Name", placeholder="Product name...")
    with c_rev_2: search_sc = st.text_input("Search by Seller / Category", placeholder="Seller or Category...")
    with c_rev_3: st.session_state.grid_items_per_page = st.select_slider("Items per page", options=[20, 50, 100, 200], value=st.session_state.grid_items_per_page)

    def get_batch_labels():
        return {
            "Poor Image Quality": "Poor images",
            "Wrong Category": "Wrong Category",
            "Suspected Fake": "Suspected Fake product",
            "Restricted Brand": "Restricted brands",
            "Wrong Brand": "Generic branded products with genuine brands",
            "Other Reason (Custom)": "Other Reason (Custom)",
        }

    with st.container(border=True):
        st.markdown(f"<p style='font-weight: 700; margin: 0 0 10px 0;'>Batch Rejection Mode</p>", unsafe_allow_html=True)
        grid_reason = st.segmented_control("Select rejection reason for batch actions:", list(get_batch_labels().keys()), default="Poor Image Quality", label_visibility="collapsed")
        if not grid_reason: grid_reason = "Poor Image Quality"

        grid_custom_comment = ""
        if grid_reason == "Other Reason (Custom)":
            grid_custom_comment = st.text_area(
                "Custom rejection comment",
                placeholder="Type your rejection reason here...",
                key="grid_custom_comment",
                height=80,
            )
            if not grid_custom_comment.strip():
                st.warning("Please enter a custom comment before rejecting.", icon="⚠️")

        active_label = grid_reason if grid_reason != "Other Reason (Custom)" else f"Other: {grid_custom_comment[:40]}..." if grid_custom_comment.strip() else "Other Reason (Custom) — enter comment above"
        st.markdown(f"""<div class='batch-info-box' style='background: var(--background-color, {JUMIA_COLORS['light_gray']});'><span style='font-size: 12px; font-weight: 600;'>Active Reason:</span><span style='font-size: 13px; font-weight: 700; margin-left: 8px;'>{active_label}</span></div>""", unsafe_allow_html=True)

    review_data = pd.merge(valid_grid_df[['ProductSetSid']], st.session_state.all_data_map, left_on='ProductSetSid', right_on='PRODUCT_SET_SID', how='left')

    if search_n: review_data = review_data[review_data['NAME'].astype(str).str.contains(search_n, case=False, na=False)]
    if search_sc:
        mc = review_data['CATEGORY'].astype(str).str.contains(search_sc, case=False, na=False) if 'CATEGORY' in review_data.columns else False
        ms = review_data['SELLER_NAME'].astype(str).str.contains(search_sc, case=False, na=False)
        review_data = review_data[mc | ms]

    items_per_page = st.session_state.grid_items_per_page
    total_pages = max(1, (len(review_data) + items_per_page - 1) // items_per_page)
    if st.session_state.grid_page >= total_pages: st.session_state.grid_page = 0

    page_data = review_data.iloc[st.session_state.grid_page * items_per_page : (st.session_state.grid_page + 1) * items_per_page]
    cur_sids = page_data['PRODUCT_SET_SID'].tolist()

    def cb_process_batch(show_warning, sids, active_reason):
        flagged = [s for s in sids if st.session_state.get(f"grid_chk_{s}")]
        if not flagged:
            if show_warning: st.session_state.main_toasts.append(("No items selected.", "⚠️"))
            return
        if active_reason == "Other Reason (Custom)":
            custom_cmt = st.session_state.get("grid_custom_comment", "").strip()
            if not custom_cmt:
                st.session_state.main_toasts.append(("Please enter a custom comment first.", "⚠️"))
                return
            code, cmt, flag_key = "1000007 - Other Reason", custom_cmt, "Other Reason (Custom)"
        else:
            flag_key = get_batch_labels()[active_reason]
            code, cmt = support_files['flags_mapping'].get(flag_key, ("1000007 - Other Reason", "Manual rejection"))
        st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(flagged), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', code, cmt, flag_key]
        for s in flagged:
            st.session_state[f"quick_rej_{s}"] = True
            st.session_state[f"quick_rej_reason_{s}"] = flag_key
            st.session_state.pop(f"grid_chk_{s}", None)
        st.session_state.exports_cache.clear()
        st.session_state.main_toasts.append((f"Batch rejected {len(flagged)} items", "✅"))

    def cb_prev(sids, active_reason): 
        cb_process_batch(False, sids, active_reason)
        st.session_state.grid_page -= 1
        st.session_state.do_scroll_top = True

    def cb_next(sids, active_reason): 
        cb_process_batch(False, sids, active_reason)
        st.session_state.grid_page += 1
        st.session_state.do_scroll_top = True

    def cb_sel_all(sids):
        for s in sids: st.session_state[f"grid_chk_{s}"] = True
    def cb_desel_all(sids):
        for s in sids: st.session_state.pop(f"grid_chk_{s}", None)
    def cb_reject_all(sids, active_reason): cb_process_batch(True, sids, active_reason)

    with st_yled.sticky_header(background_color="#FFFFFF", padding="15px 10px", key="sticky-nav-top"):
        col_pg1, col_pg2, col_pg3, col_sel, col_desel, col_rej, col_spacer = st.columns([0.8, 1.2, 0.8, 0.6, 0.6, 1.5, 1.5])
        col_pg1.button(":material/arrow_back: Prev", key="pt", disabled=(st.session_state.grid_page == 0), use_container_width=True, on_click=cb_prev, args=(cur_sids, grid_reason))
        col_pg2.markdown(f"<p style='text-align: center; margin: 0; padding: 10px 0; font-weight: 600;'>Page {st.session_state.grid_page + 1} of {total_pages}</p>", unsafe_allow_html=True)
        col_pg3.button("Next :material/arrow_forward:", key="nt", disabled=(st.session_state.grid_page >= total_pages - 1), use_container_width=True, on_click=cb_next, args=(cur_sids, grid_reason))
        col_sel.button("Select All", key="sel_all_top", use_container_width=True, on_click=cb_sel_all, args=(cur_sids,))
        col_desel.button("Deselect All", key="desel_all_top", use_container_width=True, on_click=cb_desel_all, args=(cur_sids,))
        col_rej.button(f":material/block: Reject All — {grid_reason}", key="reject_active_top", type="primary", use_container_width=True, on_click=cb_reject_all, args=(cur_sids, grid_reason))

    if st.session_state.get('do_scroll_top', False):
        components.html("""
            <script>
                const doc = window.parent.document;
                const main = doc.querySelector('.main') || doc.querySelector('[data-testid="stAppViewContainer"]');
                if (main) { main.scrollTo({top: 0, behavior: 'smooth'}); }
            </script>
        """, height=0, width=0)
        st.session_state.do_scroll_top = False

    page_warnings = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_sid = {}
        for _, r in page_data.iterrows():
            url = str(r.get('MAIN_IMAGE', '')).strip()
            sid = str(r['PRODUCT_SET_SID'])
            if url in st.session_state.image_advisor_cache:
                page_warnings[sid] = st.session_state.image_advisor_cache[url]
            else:
                future_to_sid[executor.submit(analyze_image_quality, url, st.session_state.image_advisor_cache)] = sid
        for future in concurrent.futures.as_completed(future_to_sid):
            sid = future_to_sid[future]
            try: page_warnings[sid] = future.result()
            except Exception: page_warnings[sid] = []

    cols_per_row = 3 if st.session_state.layout_mode == "centered" else 4
    
    for i in range(0, len(page_data), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (_, row) in enumerate(page_data.iloc[i:i+cols_per_row].iterrows()):
            with cols[j]:
                sid_for_row = str(row['PRODUCT_SET_SID'])
                render_product_card(row, support_files['flags_mapping'], st.session_state.get('selected_country', 'Kenya'), page_warnings.get(sid_for_row, []))

    if st.session_state.grid_page < total_pages - 1:
        next_page_data = review_data.iloc[(st.session_state.grid_page + 1) * items_per_page : (st.session_state.grid_page + 2) * items_per_page]
        next_urls = [str(u).strip() for u in next_page_data['MAIN_IMAGE'].values if str(u).startswith('http')]
        if next_urls:
            js_array = json.dumps(next_urls)
            components.html(f"""<script>setTimeout(function() {{ var urls = {js_array}; urls.forEach(function(url) {{ var img = new Image(); img.src = url; }}); }}, 1500);</script>""", height=0, width=0)
            for url in next_urls:
                if url not in st.session_state.image_advisor_cache: st.session_state.bg_executor.submit(analyze_image_quality, url, st.session_state.image_advisor_cache)

    st.divider()

    with st.container():
        col_pg1_b, col_pg2_b, col_pg3_b, col_sel_b, col_desel_b, col_rej_b, col_spacer_b = st.columns([0.8, 1.2, 0.8, 0.6, 0.6, 1.5, 1.5])
        col_pg1_b.button(":material/arrow_back: Prev", key="pb", disabled=(st.session_state.grid_page == 0), use_container_width=True, on_click=cb_prev, args=(cur_sids, grid_reason))
        col_pg2_b.markdown(f"<p style='text-align: center; margin: 0; padding: 10px 0; font-weight: 600;'>Page {st.session_state.grid_page + 1} of {total_pages}</p>", unsafe_allow_html=True)
        col_pg3_b.button("Next :material/arrow_forward:", key="nb", disabled=(st.session_state.grid_page >= total_pages - 1), use_container_width=True, on_click=cb_next, args=(cur_sids, grid_reason))
        col_sel_b.button("Select All", key="sel_all_bot", use_container_width=True, on_click=cb_sel_all, args=(cur_sids,))
        col_desel_b.button("Deselect All", key="desel_all_bot", use_container_width=True, on_click=cb_desel_all, args=(cur_sids,))
        col_rej_b.button(f":material/block: Reject All — {grid_reason}", key="reject_active_bot", type="primary", use_container_width=True, on_click=cb_reject_all, args=(cur_sids, grid_reason))

# ==========================================
# SECTION 3: EXPORTS
# ==========================================
if not st.session_state.final_report.empty:
    st.markdown("---")
    st.markdown(f"""
<div style='background: linear-gradient(135deg, {JUMIA_COLORS['primary_orange']}, {JUMIA_COLORS['secondary_orange']}); padding: 20px 24px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(246, 139, 30, 0.25);'>
    <h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>Download Reports</h2>
    <p style='color: rgba(255,255,255,0.9); margin: 6px 0 0 0; font-size: 13px;'>Export validation results in Excel or ZIP format</p>
</div>
""", unsafe_allow_html=True)

    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    app_df = fr[fr['Status'] == 'Approved']
    rej_df = fr[fr['Status'] == 'Rejected']
    c_code = st.session_state.selected_country[:2].upper()
    date_str = datetime.now().strftime('%Y-%m-%d')
    reasons_df = support_files.get('reasons', pd.DataFrame())

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
                        st.markdown(f"""
<div style='text-align: center; margin-bottom: 15px;'>
    <div style='font-size: 48px; margin-bottom: 8px;' class='material-symbols-outlined'>{icon}</div>
    <div style='font-size: 18px; font-weight: 700;'>{title}</div>
    <div style='font-size: 11px; margin-top: 4px; opacity: 0.7;'>{desc}</div>
    <div style='background: {JUMIA_COLORS['light_gray']}; color: {JUMIA_COLORS['primary_orange']}; padding: 8px; border-radius: 6px; margin-top: 12px; font-weight: 600;'>{len(df):,} rows</div>
</div>
""", unsafe_allow_html=True)
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
