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

GRID_COLS = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'SELLER_NAME', 'MAIN_IMAGE',
             'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE', 'COLOR']

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
        import urllib.request
        url = "https://open.er-api.com/v6/latest/USD"
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read())
        return float(data["rates"].get(cfg["code"], 1.0))
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

from postqc import detect_file_type, normalize_post_qc, run_checks as run_post_qc_checks, render_post_qc_section

# -------------------------------------------------
# INITIALIZATION & CONTEXT
# -------------------------------------------------
if 'layout_mode'            not in st.session_state: st.session_state.layout_mode            = "wide"
if 'final_report'           not in st.session_state: st.session_state.final_report           = pd.DataFrame()
if 'all_data_map'           not in st.session_state: st.session_state.all_data_map           = pd.DataFrame()
if 'post_qc_summary'        not in st.session_state: st.session_state.post_qc_summary        = pd.DataFrame()
if 'post_qc_results'        not in st.session_state: st.session_state.post_qc_results        = {}
if 'post_qc_data'           not in st.session_state: st.session_state.post_qc_data           = pd.DataFrame()
if 'file_mode'              not in st.session_state: st.session_state.file_mode              = None
if 'intersection_sids'      not in st.session_state: st.session_state.intersection_sids      = set()
if 'intersection_count'     not in st.session_state: st.session_state.intersection_count     = 0
if 'grid_page'              not in st.session_state: st.session_state.grid_page              = 0
if 'grid_items_per_page'    not in st.session_state: st.session_state.grid_items_per_page    = 50
if 'main_toasts'            not in st.session_state: st.session_state.main_toasts            = []
if 'exports_cache'          not in st.session_state: st.session_state.exports_cache          = {}
if 'do_scroll_top'          not in st.session_state: st.session_state.do_scroll_top          = False
if 'display_df_cache'       not in st.session_state: st.session_state.display_df_cache       = {}

try:
    st.set_page_config(page_title="Product Tool", layout=st.session_state.layout_mode)
except Exception:
    pass

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

        div[data-testid="stExpander"] {{ border: 1px solid {JUMIA_COLORS['border_gray']}; border-radius: 8px; }}
        div[data-testid="stExpander"] summary {{ background-color: {JUMIA_COLORS['light_gray']}; padding: 12px; border-radius: 8px 8px 0 0; }}
        h1, h2, h3 {{ color: {JUMIA_COLORS['dark_gray']} !important; }}
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
    except Exception:
        pass
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
    except Exception:
        return str(code).strip()

def normalize_text(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).lower().strip()
    noise = r'\b(new|sale|original|genuine|authentic|official|premium|quality|best|hot|2024|2025)\b'
    text = re.sub(noise, '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def create_match_key(row: pd.Series) -> str:
    name  = normalize_text(row.get('NAME', ''))
    brand = normalize_text(row.get('BRAND', ''))
    color = normalize_text(row.get('COLOR', ''))
    return f"{brand}|{name}|{color}"

def df_hash(df: pd.DataFrame) -> str:
    try:
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    except Exception:
        return hashlib.md5(str(df.shape).encode()).hexdigest()

COLOR_PATTERNS = {
    'red':        ['red', 'crimson', 'scarlet', 'maroon', 'burgundy', 'wine', 'ruby'],
    'blue':       ['blue', 'navy', 'royal', 'sky', 'azure', 'cobalt', 'sapphire'],
    'green':      ['green', 'lime', 'olive', 'emerald', 'mint', 'forest', 'jade'],
    'black':      ['black', 'onyx', 'ebony', 'jet', 'charcoal', 'midnight'],
    'white':      ['white', 'ivory', 'cream', 'pearl', 'snow', 'alabaster'],
    'gray':       ['gray', 'grey', 'silver', 'slate', 'ash', 'graphite'],
    'yellow':     ['yellow', 'gold', 'golden', 'amber', 'lemon', 'mustard'],
    'orange':     ['orange', 'tangerine', 'peach', 'coral', 'apricot'],
    'pink':       ['pink', 'rose', 'magenta', 'fuchsia', 'salmon', 'blush'],
    'purple':     ['purple', 'violet', 'lavender', 'plum', 'mauve', 'lilac'],
    'brown':      ['brown', 'tan', 'beige', 'khaki', 'chocolate', 'coffee', 'bronze'],
    'multicolor': ['multicolor', 'multicolour', 'multi-color', 'rainbow', 'mixed']
}

# -------------------------------------------------
# LOCAL EXCEL DATA LOADING HELPERS
# -------------------------------------------------
def load_txt_file(filename: str) -> List[str]:
    try:
        if not os.path.exists(os.path.abspath(filename)): return []
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []

@st.cache_data(ttl=3600)
def load_excel_file(filename: str, column: Optional[str] = None):
    try:
        if not os.path.exists(filename):
            return [] if column else pd.DataFrame()
        df = pd.read_excel(filename, engine='openpyxl', dtype=str)
        df.columns = df.columns.str.strip()
        if column and column in df.columns:
            return df[column].apply(clean_category_code).tolist()
        return df
    except Exception:
        return [] if column else pd.DataFrame()

def safe_excel_read(filename: str, sheet_name, usecols=None) -> pd.DataFrame:
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols=usecols,
                           engine='openpyxl', dtype=str)
        return df.dropna(how='all')
    except Exception:
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
            keyword_col = next(
                (c for c in df.columns if 'keyword' in c or 'prohibited' in c or 'name' in c),
                df.columns[0]
            )
            category_col = next((c for c in df.columns if 'cat' in c), None)
            country_rules = []
            for _, row in df.iterrows():
                keyword = str(row.get(keyword_col, '')).strip().lower()
                if not keyword or keyword in ('nan', 'keywords'):
                    continue
                categories: Set[str] = set()
                if category_col:
                    cats_raw = str(row.get(category_col, '')).strip()
                    if cats_raw and cats_raw.lower() != 'nan':
                        split_cats = re.split(r'[,\n]+', cats_raw)
                        categories.update(
                            [clean_category_code(c.strip()) for c in split_cats if c.strip()]
                        )
                country_rules.append({'keyword': keyword, 'categories': categories})
            prohibited_by_country[tab] = country_rules
        except Exception:
            prohibited_by_country[tab] = []
    return prohibited_by_country

@st.cache_data(ttl=3600)
def load_restricted_brands_from_local() -> Dict[str, List[Dict]]:
    FILE_NAME = "Restricted_Brands.xlsx"
    COUNTRY_TABS = {"Kenya": "KE", "Uganda": "UG", "Nigeria": "NG", "Ghana": "GH", "Morocco": "MA"}
    config_by_country: Dict[str, List[Dict]] = {}
    for country_name, tab_name in COUNTRY_TABS.items():
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab_name)
            if df.empty:
                config_by_country[country_name] = []
                continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            brand_dict: Dict[str, Dict] = {}
            for _, row in df.iterrows():
                brand = str(row.get('brand', '')).strip()
                if not brand or brand.lower() == 'nan':
                    continue
                b_lower = brand.lower()
                if b_lower not in brand_dict:
                    brand_dict[b_lower] = {
                        'brand_raw': brand, 'sellers': set(), 'categories': set(),
                        'variations': set(), 'has_blank_category': False
                    }
                sellers_raw = str(row.get('approved sellers', '')).strip().lower()
                if sellers_raw not in ('nan', ''):
                    brand_dict[b_lower]['sellers'].update(
                        [s.strip() for s in sellers_raw.split(',') if s.strip()]
                    )
                cats_raw = str(row.get('categories', '')).strip()
                if cats_raw in ('nan', ''):
                    brand_dict[b_lower]['has_blank_category'] = True
                else:
                    brand_dict[b_lower]['categories'].update(
                        [clean_category_code(c.strip()) for c in cats_raw.split(',') if c.strip()]
                    )
                vars_raw = str(row.get('variations', '')).strip().lower()
                if vars_raw not in ('nan', ''):
                    brand_dict[b_lower]['variations'].update(
                        [v.strip() for v in vars_raw.split(',') if v.strip()]
                    )
            country_rules = []
            for b_lower, data in brand_dict.items():
                if data['has_blank_category']:
                    data['categories'] = set()
                country_rules.append({
                    'brand': b_lower, 'brand_raw': data['brand_raw'],
                    'sellers': data['sellers'], 'categories': data['categories'],
                    'variations': list(data['variations'])
                })
            config_by_country[country_name] = country_rules
        except Exception:
            config_by_country[country_name] = []
    return config_by_country

@st.cache_data(ttl=3600)
def load_suspected_fake_from_local() -> pd.DataFrame:
    try:
        if os.path.exists('suspected_fake.xlsx'):
            return pd.read_excel('suspected_fake.xlsx', sheet_name=0, engine='openpyxl', dtype=str)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_flags_mapping(filename: str = "reason.xlsx") -> Dict[str, Tuple[str, str]]:
    default_mapping = {
        'Restricted brands':          ('1000024 - Product does not have a license to be sold via Jumia (Not Authorized)', "Missing license for this item. Raise a claim via Vendor Center."),
        'Suspected Fake product':     ('1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)', "Product confirmed counterfeit."),
        'Seller Not approved to sell Refurb': ('1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim', "Contact Seller Support for Refurbished approval."),
        'Product Warranty':           ('1000013 - Kindly Provide Product Warranty Details', "Valid warranty required in Description/Warranty tabs."),
        'Seller Approve to sell books': ('1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim', "Contact Seller Support for Book category approval."),
        'Seller Approved to Sell Perfume': ('1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim', "Contact Seller Support for Perfume approval."),
        'Counterfeit Sneakers':       ('1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)', "Sneaker confirmed counterfeit."),
        'Suspected counterfeit Jerseys': ('1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)', "Jersey confirmed counterfeit."),
        'Prohibited products':        ('1000007 - Other Reason', "Listing of this product is prohibited."),
        'Unnecessary words in NAME':  ('1000008 - Kindly Improve Product Name Description', "Avoid unnecessary words in title."),
        'Single-word NAME':           ('1000008 - Kindly Improve Product Name Description', "Update product title format: Name – Type – Color."),
        'Generic BRAND Issues':       ('1000007 - Other Reason', "Use correct brand instead of Generic/Fashion. Apply for brand approval if needed."),
        'Fashion brand issues':       ('1000007 - Other Reason', "Use correct brand instead of Fashion. Apply for brand approval if needed."),
        'BRAND name repeated in NAME': ('1000007 - Other Reason', "Brand name should not be repeated in product name."),
        'Generic branded products with genuine brands': ('1000007 - Other Reason', "Use the displayed brand on the product instead of Generic."),
        'Missing COLOR':              ('1000005 - Kindly confirm the actual product colour', "Product color must be mentioned in title/color tab."),
        'Duplicate product':          ('1000007 - Other Reason', "This product is a duplicate."),
        'Wrong Variation':            ('1000039 - Product Poorly Created. Each Variation Of This Product Should Be Created Uniquely (Not Authorized)', "Create different SKUs instead of variations (variations only for sizes)."),
        'Missing Weight/Volume':      ('1000008 - Kindly Improve Product Name Description', "Include weight or volume (e.g., '1kg', '500ml')."),
        'Incomplete Smartphone Name': ('1000008 - Kindly Improve Product Name Description', "Include memory/storage details (e.g., '128GB')."),
        'Wrong Category':             ('1000004 - Wrong Category', "Assigned to Wrong Category. Please use correct category."),
        'Poor images':                ('1000042 - Kindly follow our product image upload guideline.', "Poor Image Quality"),
        'Other Reason (Custom)':      ('1000007 - Other Reason', "Manual rejection."),
    }
    try:
        if os.path.exists(filename):
            df = pd.read_excel(filename, engine='openpyxl', dtype=str)
            df.columns = df.columns.str.strip().str.lower()
            if 'flag' in df.columns and 'reason' in df.columns and 'comment' in df.columns:
                custom_mapping = {}
                for _, row in df.iterrows():
                    flag    = str(row['flag']).strip()
                    reason  = str(row['reason']).strip()
                    comment = str(row['comment']).strip()
                    if flag and flag.lower() != 'nan':
                        custom_mapping[flag] = (reason, comment)
                if custom_mapping:
                    return custom_mapping
    except Exception as e:
        logger.error(f"Error loading external {filename}: {e}")
    return default_mapping

@st.cache_data(ttl=3600)
def load_all_support_files() -> Dict:
    def safe_load_txt(f):
        return load_txt_file(f) if os.path.exists(f) else []
    return {
        'blacklisted_words':       safe_load_txt('blacklisted.txt'),
        'book_category_codes':     safe_load_txt('Books_cat.txt'),
        'perfume_category_codes':  safe_load_txt('Perfume_cat.txt'),
        'sneaker_category_codes':  safe_load_txt('Sneakers_Cat.txt'),
        'sneaker_sensitive_brands': [b.lower() for b in safe_load_txt('Sneakers_Sensitive.txt')],
        'sensitive_words':         [w.lower() for w in safe_load_txt('sensitive_words.txt')],
        'unnecessary_words':       [w.lower() for w in safe_load_txt('unnecessary.txt')],
        'colors':                  [c.lower() for c in safe_load_txt('colors.txt')],
        'color_categories':        safe_load_txt('color_cats.txt'),
        'category_fas':            safe_load_txt('Fashion_cat.txt'),
        'reasons':                 load_excel_file('reasons.xlsx'),
        'flags_mapping':           load_flags_mapping(),
        'warranty_category_codes': safe_load_txt('warranty.txt'),
        'suspected_fake':          load_suspected_fake_from_local(),
        'duplicate_exempt_codes':  safe_load_txt('duplicate_exempt.txt'),
        'restricted_brands_all':   load_restricted_brands_from_local(),
        'prohibited_words_all':    load_prohibited_from_local(),
        'known_brands':            safe_load_txt('brands.txt'),
        'variation_allowed_codes': safe_load_txt('variation.txt'),
        'weight_category_codes':   safe_load_txt('weight.txt'),
        'smartphone_category_codes': safe_load_txt('smartphones.txt'),
    }

@st.cache_data(ttl=3600)
def load_support_files_lazy():
    return load_all_support_files()

@st.cache_data(ttl=3600)
def compile_regex_patterns(words: List[str]) -> Optional[re.Pattern]:
    if not words:
        return None
    pattern = '|'.join(r'\b' + re.escape(w) + r'\b' for w in sorted(words, key=len, reverse=True))
    return re.compile(pattern, re.IGNORECASE)

class CountryValidator:
    COUNTRY_CONFIG = {
        "Kenya":   {"code": "KE", "skip_validations": []},
        "Uganda":  {"code": "UG", "skip_validations": ["Counterfeit Sneakers", "Product Warranty", "Generic BRAND Issues"]},
        "Nigeria": {"code": "NG", "skip_validations": []},
        "Ghana":   {"code": "GH", "skip_validations": []},
        "Morocco": {"code": "MA", "skip_validations": []},
    }

    def __init__(self, country: str):
        self.country          = country
        self.config           = self.COUNTRY_CONFIG.get(country, self.COUNTRY_CONFIG["Kenya"])
        self.code             = self.config["code"]
        self.skip_validations = self.config["skip_validations"]

    def should_skip_validation(self, validation_name: str) -> bool:
        return validation_name in self.skip_validations

    def ensure_status_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and 'Status' not in df.columns:
            df['Status'] = 'Approved'
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
        renamed[col] = map_lower[col_lower] if col_lower in map_lower else col.upper()
    df = df.rename(columns=renamed)
    for col in ['ACTIVE_STATUS_COUNTRY', 'CATEGORY_CODE', 'BRAND', 'TAX_CLASS', 'NAME', 'SELLER_NAME']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def validate_input_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = [
        f"Missing: {f}" for f in
        ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY_CODE', 'ACTIVE_STATUS_COUNTRY']
        if f not in df.columns
    ]
    return len(errors) == 0, errors

MULTI_COUNTRY_VALUES = {'MULTIPLE', 'MULTI'}

def filter_by_country(df: pd.DataFrame, country_validator: CountryValidator) -> Tuple[pd.DataFrame, List[str]]:
    if 'ACTIVE_STATUS_COUNTRY' not in df.columns:
        return df, []
    s = df['ACTIVE_STATUS_COUNTRY'].astype(str).str.strip().str.upper().str.replace(r'^JUMIA-', '', regex=True)
    df['ACTIVE_STATUS_COUNTRY'] = s

    if country_validator.code == 'NG':
        is_ng    = df['ACTIVE_STATUS_COUNTRY'] == 'NG'
        is_multi = df['ACTIVE_STATUS_COUNTRY'].isin(MULTI_COUNTRY_VALUES)
        filtered = df[is_ng | is_multi].copy()
        filtered['_IS_MULTI_COUNTRY'] = is_multi[filtered.index]
    else:
        filtered = df[df['ACTIVE_STATUS_COUNTRY'] == country_validator.code].copy()
        filtered['_IS_MULTI_COUNTRY'] = False

    detected_names = []
    if filtered.empty:
        detected_codes = [
            c for c in df['ACTIVE_STATUS_COUNTRY'].unique()
            if str(c).strip() and str(c).strip().lower() != 'nan'
        ]
        emoji_map = {"KE": "Kenya", "UG": "Uganda", "NG": "Nigeria", "GH": "Ghana", "MA": "Morocco"}
        detected_names = [emoji_map.get(c, f"'{c}'") for c in detected_codes]
    return filtered, detected_names

def propagate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ['COLOR_FAMILY', 'PRODUCT_WARRANTY', 'WARRANTY_DURATION', 'WARRANTY_ADDRESS',
                'WARRANTY_TYPE', 'COUNT_VARIATIONS', 'LIST_VARIATIONS']:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df.groupby('PRODUCT_SET_SID')[col].transform(lambda x: x.ffill().bfill())
    return df

# -------------------------------------------------
# VALIDATION CHECKS (stubs – replace with full implementations)
# -------------------------------------------------
def check_miscellaneous_category(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=data.columns)

def check_restricted_brands(data: pd.DataFrame, country_rules: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(columns=data.columns)

def check_prohibited_products(data: pd.DataFrame, prohibited_rules: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(columns=data.columns)

def check_suspected_fake_products(
    data: pd.DataFrame, suspected_fake_df: pd.DataFrame, fx_rate: float
) -> pd.DataFrame:
    return pd.DataFrame(columns=data.columns)

# -------------------------------------------------
# MASTER VALIDATION RUNNER
# -------------------------------------------------
def validate_products(
    data: pd.DataFrame,
    support_files: Dict,
    country_validator: CountryValidator,
    data_has_warranty_cols: bool,
    common_sids: Optional[set] = None,
    skip_validators: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    data['PRODUCT_SET_SID'] = data['PRODUCT_SET_SID'].astype(str).str.strip()
    rows = []
    for _, r in data.iterrows():
        sid = str(r['PRODUCT_SET_SID']).strip()
        rows.append({
            'ProductSetSid': sid,
            'ParentSKU':     r.get('PARENTSKU', ''),
            'Status':        'Approved',
            'Reason':        '',
            'Comment':       '',
            'FLAG':          '',
            'SellerName':    r.get('SELLER_NAME', ''),
        })
    return country_validator.ensure_status_column(pd.DataFrame(rows)), {}

@st.cache_data(show_spinner=False, ttl=3600)
def cached_validate_products(
    data_hash: str,
    _data: pd.DataFrame,
    _support_files: Dict,
    country_code: str,
    data_has_warranty_cols: bool,
) -> Tuple[pd.DataFrame, Dict]:
    country_name = next(
        (k for k, v in CountryValidator.COUNTRY_CONFIG.items() if v['code'] == country_code),
        "Kenya"
    )
    return validate_products(_data, _support_files, CountryValidator(country_name), data_has_warranty_cols)

# -------------------------------------------------
# EXPORTS UTILITIES
# -------------------------------------------------
def generate_smart_export(
    df: pd.DataFrame,
    filename_prefix: str,
    export_type: str = 'simple',
    auxiliary_df: Optional[pd.DataFrame] = None,
) -> Tuple[BytesIO, str, str]:
    if export_type == 'full':
        cols = FULL_DATA_COLS + [
            c for c in ["Status", "Reason", "Comment", "FLAG", "SellerName"]
            if c not in FULL_DATA_COLS
        ]
    else:
        cols = PRODUCTSETS_COLS

    export_df = df[[c for c in cols if c in df.columns]]
    zb = BytesIO()
    with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{filename_prefix}.csv", export_df.to_csv(index=False))
    zb.seek(0)
    return zb, f"{filename_prefix}.zip", "application/zip"

def prepare_full_data_merged(data_df: pd.DataFrame, final_report_df: pd.DataFrame) -> pd.DataFrame:
    try:
        d_cp = data_df.copy()
        r_cp = final_report_df.copy()
        d_cp['PRODUCT_SET_SID']  = d_cp['PRODUCT_SET_SID'].astype(str).str.strip()
        r_cp['ProductSetSid']    = r_cp['ProductSetSid'].astype(str).str.strip()
        merged = pd.merge(
            d_cp,
            r_cp[["ProductSetSid", "Status", "Reason", "Comment", "FLAG", "SellerName"]],
            left_on="PRODUCT_SET_SID", right_on="ProductSetSid", how='left'
        )
        if 'ProjectSetSid' in merged.columns:
            merged.drop(columns=['ProjectSetSid'], inplace=True)
        if 'ProductSetSid' in merged.columns:
            merged.drop(columns=['ProductSetSid'], inplace=True)
        return merged
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------
# IMAGE QUALITY CHECK (cached for 24 h)
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
            if w < 300 or h < 300:
                warnings.append("Low Res")
            ratio = h / w if w > 0 else 1
            if ratio > 1.5:
                warnings.append("Tall")
            elif ratio < 0.6:
                warnings.append("Wide")
    except Exception:
        pass
    return warnings

# -------------------------------------------------
# REJECTION HELPERS
# -------------------------------------------------
def apply_rejection(sids: List[str], reason_code: str, comment: str, flag_name: str) -> None:
    mask = st.session_state.final_report['ProductSetSid'].isin(sids)
    st.session_state.final_report.loc[mask, ['Status', 'Reason', 'Comment', 'FLAG']] = [
        'Rejected', reason_code, comment, flag_name
    ]
    st.session_state.exports_cache.clear()

def restore_single_item(sid: str) -> None:
    mask = st.session_state.final_report['ProductSetSid'] == sid
    st.session_state.final_report.loc[mask, ['Status', 'Reason', 'Comment', 'FLAG']] = [
        'Approved', '', '', 'Approved by User'
    ]
    st.session_state.pop(f"quick_rej_{sid}", None)
    st.session_state.pop(f"quick_rej_reason_{sid}", None)
    st.session_state.exports_cache.clear()
    st.session_state.main_toasts.append(("Item restored to Approved.", "↩️"))

# Maps JS reason keys → human-readable flag names (must match keys in flags_mapping)
REASON_MAP = {
    "REJECT_POOR_IMAGE":  "Poor images",
    "REJECT_WRONG_CAT":   "Wrong Category",
    "REJECT_FAKE":        "Suspected Fake product",
    "REJECT_BRAND":       "Restricted brands",
    "REJECT_PROHIBITED":  "Prohibited products",
    "REJECT_COLOR":       "Missing COLOR",
    "REJECT_WRONG_BRAND": "Generic branded products with genuine brands",
    "OTHER_CUSTOM":       "Other Reason (Custom)",
}

# =============================================================================
# BRIDGE ACTION PROCESSOR
# =============================================================================
def _process_card_bridge_action(action_str: str, support_files: Dict) -> bool:
    """
    Handles events sent from the JS HTML component via the hidden text_input bridge.

    Supported action formats
    ─────────────────────────
    NAV_PREV:<json_pending>   – apply any pending rejections then go to previous page
    NAV_NEXT:<json_pending>   – apply any pending rejections then go to next page
    BATCH_COMMIT:<json>       – commit {sid: reason_key, …} to Python state
    RESTORE:<sid>             – undo a committed rejection for one product
    """
    if not action_str:
        return False

    # Split only on the FIRST colon so JSON payloads are not mangled
    try:
        colon   = action_str.index(":")
        action  = action_str[:colon]
        payload = action_str[colon + 1:]
    except ValueError:
        return False

    def _apply_pending(pending: Dict[str, str]) -> int:
        """Apply a {sid: reason_key} dict; returns total items processed."""
        if not pending:
            return 0
        # Group by reason_key for efficiency
        reason_groups: Dict[str, List[str]] = {}
        for sid, reason_key in pending.items():
            reason_groups.setdefault(reason_key, []).append(sid)

        total = 0
        for reason_key, sids in reason_groups.items():
            flag_name = REASON_MAP.get(reason_key, "Other Reason (Custom)")
            code, cmt = support_files["flags_mapping"].get(
                flag_name, ("1000007 - Other Reason", "Manual rejection")
            )
            apply_rejection(sids, code, cmt, flag_name)
            for s in sids:
                st.session_state[f"quick_rej_{s}"]        = True
                st.session_state[f"quick_rej_reason_{s}"] = flag_name
            total += len(sids)
        return total

    try:
        # ── NAV_PREV / NAV_NEXT ───────────────────────────────────────────────
        # Payload is JSON of any pending rejections to commit atomically
        # before navigating (avoids the two-event race condition).
        if action in ("NAV_PREV", "NAV_NEXT"):
            try:
                pending = json.loads(payload) if payload.strip() else {}
            except json.JSONDecodeError:
                pending = {}

            committed = _apply_pending(pending)
            if committed:
                st.session_state.main_toasts.append(
                    (f"Committed {committed} rejection(s) before navigation.", "✅")
                )

            if action == "NAV_PREV":
                st.session_state.grid_page = max(0, st.session_state.grid_page - 1)
            else:
                st.session_state.grid_page += 1

            st.session_state.do_scroll_top = True
            return True

        # ── BATCH_COMMIT ──────────────────────────────────────────────────────
        if action == "BATCH_COMMIT":
            try:
                pending = json.loads(payload)
            except json.JSONDecodeError:
                return False

            total = _apply_pending(pending)
            if total:
                st.session_state.main_toasts.append(
                    (f"Rejected {total} product(s).", "✅")
                )
            return True  # always return True so bridge clears even for empty dict

        # ── RESTORE ───────────────────────────────────────────────────────────
        if action == "RESTORE":
            restore_single_item(payload.strip())
            return True

    except Exception as e:
        logger.error(f"Bridge action error: {e}\n{traceback.format_exc()}")

    return False


# =============================================================================
# HTML GRID BUILDER
# =============================================================================
def build_fast_grid_html(
    page_data:      pd.DataFrame,
    flags_mapping:  Dict,
    country:        str,
    page_warnings:  Dict[str, List[str]],
    rejected_state: Dict[str, str],   # {sid: reason_label} already committed in Python
    cols_per_row:   int,
    current_page:   int,
    total_pages:    int,
) -> str:
    # Pull colours directly from the module-level dict (no import needed)
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
    batch_opts_html = "".join(
        f'<option value="{v}">{label}</option>' for label, v in batch_options
    )

    # Serialise Python data into JS constants
    # json.dumps handles all quoting/escaping so product names with
    # backticks, quotes, or ${…} cannot break the JS template literals.
    committed_json = json.dumps(rejected_state)
    all_sids_json  = json.dumps(page_data["PRODUCT_SET_SID"].astype(str).tolist())

    cards_data = []
    for _, row in page_data.iterrows():
        sid     = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip()
        if not img_url.startswith("http"):
            img_url = "https://via.placeholder.com/150?text=No+Image"
        cards_data.append({
            "sid":      sid,
            "img":      img_url,
            "name":     str(row.get("NAME",        "")),
            "brand":    str(row.get("BRAND",       "Unknown Brand")),
            "cat":      str(row.get("CATEGORY",    "Unknown Category")),
            "seller":   str(row.get("SELLER_NAME", "Unknown Seller")),
            "warnings": page_warnings.get(sid, []),
        })
    cards_json = json.dumps(cards_data)

    prev_disabled = "disabled" if current_page <= 0               else ""
    next_disabled = "disabled" if current_page >= total_pages - 1 else ""

    # ── The full HTML / CSS / JS block ───────────────────────────────────────
    # Python f-string rules: {{ → {  and  }} → }  in the output HTML.
    # All CSS and JS braces that should stay as-is must be doubled here.
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;font-family:sans-serif;}}
  body{{background:#f5f5f5;padding:10px;}}

  /* toolbar */
  .toolbar{{
    display:flex;justify-content:space-between;align-items:center;
    padding:10px 14px;background:#fff;border:1px solid #e0e0e0;
    border-radius:8px;margin-bottom:14px;
    position:sticky;top:0;z-index:200;gap:10px;flex-wrap:wrap;
  }}
  .toolbar button{{
    padding:6px 13px;border:1px solid #ccc;border-radius:4px;
    background:#fff;cursor:pointer;font-size:12px;font-weight:600;
  }}
  .toolbar button:disabled{{opacity:.4;cursor:default;}}
  .toolbar button.danger{{background:{R};color:#fff;border-color:{R};}}
  .toolbar button.nav{{background:{DG};color:#fff;border-color:{DG};}}
  .toolbar button.commit{{background:{G};color:#fff;border-color:{G};}}
  .toolbar select{{padding:6px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px;outline:none;}}
  .page-info{{font-size:13px;font-weight:700;color:{DG};white-space:nowrap;}}
  .sel-count{{font-size:13px;font-weight:700;color:{O};}}

  /* grid */
  .grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}

  /* card */
  .card{{
    border:2px solid #e0e0e0;border-radius:8px;padding:10px;
    background:#fff;position:relative;transition:border-color .15s,box-shadow .15s;
  }}
  .card.selected{{
    border-color:{G};
    box-shadow:0 0 0 3px rgba(76,175,80,.2);
    background:rgba(76,175,80,.04);
  }}
  .card.pending-rej{{
    border-color:{R};opacity:.6;
    box-shadow:0 0 0 3px rgba(231,60,23,.15);
  }}
  .card.committed-rej{{border-color:#bbb;opacity:.45;}}

  .card-img-wrap{{position:relative;cursor:pointer;}}
  .card-img{{width:100%;aspect-ratio:1;object-fit:contain;border-radius:6px;display:block;}}
  .card.pending-rej .card-img,
  .card.committed-rej .card-img{{filter:grayscale(80%);}}

  /* selection tick */
  .tick{{
    position:absolute;bottom:6px;right:6px;width:22px;height:22px;
    border-radius:50%;background:rgba(0,0,0,.18);
    display:flex;align-items:center;justify-content:center;
    color:transparent;font-size:13px;font-weight:900;
    transition:all .15s;pointer-events:none;
  }}
  .card.selected .tick{{background:{G};color:#fff;}}

  /* warning badges */
  .warn-wrap{{
    position:absolute;top:6px;right:6px;
    display:flex;flex-direction:column;gap:3px;z-index:5;pointer-events:none;
  }}
  .warn-badge{{
    background:rgba(255,193,7,.95);color:#313133;
    font-size:9px;font-weight:800;padding:3px 7px;
    border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.2);
  }}

  /* rejection overlay */
  .overlay{{
    display:none;position:absolute;inset:0;
    background:rgba(255,255,255,.9);border-radius:6px;
    flex-direction:column;align-items:center;justify-content:center;
    z-index:20;gap:6px;padding:8px;text-align:center;
  }}
  .card.pending-rej .overlay,
  .card.committed-rej .overlay{{display:flex;}}
  .rej-badge{{
    background:{R};color:#fff;padding:3px 10px;
    border-radius:10px;font-size:11px;font-weight:700;
  }}
  .rej-label{{font-size:10px;font-weight:600;color:{R};max-width:130px;line-height:1.3;}}
  .undo-btn{{
    margin-top:4px;padding:5px 12px;background:{DG};color:#fff;
    border:none;border-radius:4px;font-size:11px;font-weight:700;cursor:pointer;
  }}

  /* meta */
  .meta{{font-size:11px;margin-top:8px;line-height:1.4;}}
  .meta .nm{{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .meta .br{{color:{O};font-weight:700;margin:2px 0;}}
  .meta .ct{{color:#666;font-size:10px;}}
  .meta .sl{{color:#999;font-size:9px;margin-top:4px;border-top:1px dashed #eee;padding-top:4px;}}

  /* action row */
  .acts{{display:flex;gap:4px;margin-top:8px;}}
  .act-btn{{
    flex:1;padding:6px;font-size:11px;border:none;
    border-radius:4px;cursor:pointer;font-weight:700;color:#fff;
  }}
  .act-poor{{background:{O};}}
  .act-more{{
    flex:1;font-size:11px;border:1px solid #ccc;
    border-radius:4px;outline:none;cursor:pointer;background:#fff;
  }}
  .pending-dot{{
    display:inline-block;width:8px;height:8px;
    border-radius:50%;background:{R};margin-left:6px;
    vertical-align:middle;
  }}
</style>
</head>
<body>

<!-- TOOLBAR -->
<div class="toolbar">
  <div style="display:flex;align-items:center;gap:8px;">
    <button class="nav" {prev_disabled}
      onclick="navAction('NAV_PREV')">&#8592; Prev</button>
    <span class="page-info">Page {current_page + 1} / {total_pages}</span>
    <button class="nav" {next_disabled}
      onclick="navAction('NAV_NEXT')">Next &#8594;</button>
  </div>

  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <button onclick="selectAll()">Select All</button>
    <button onclick="deselectAll()">Deselect All</button>
    <span class="sel-count"><span id="sel-count">0</span> selected</span>
    <span id="pending-indicator" style="display:none;font-size:12px;color:{R};font-weight:700;">
      <span id="pending-count">0</span> pending<span class="pending-dot"></span>
    </span>
    <select id="batch-reason">{batch_opts_html}</select>
    <button class="danger" onclick="doBatchReject()">&#10007; Batch Reject</button>
    <button class="commit" onclick="commitAll()">&#10003; Commit</button>
  </div>
</div>

<!-- CARD GRID (populated by JS) -->
<div class="grid" id="card-grid"></div>

<script>
// ── Data injected from Python (all values JSON-safe) ──────────────────────────
const CARDS     = {cards_json};
const COMMITTED = {committed_json};   // {{sid: label}} already saved in Python
const ALL_SIDS  = {all_sids_json};

// ── Client-only state (never sent to Python until Commit / Nav) ───────────────
let selectedSids      = new Set();
let pendingRejections = {{}};   // {{sid: reason_key}} – not yet in Python

// ── Utility: escape HTML special chars (prevents XSS from product names) ──────
function esc(s) {{
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}

// ── Bridge: send an action string to Streamlit ────────────────────────────────
function sendBridge(action) {{
  let input = null;

  // Strategy 1: direct placeholder selector in parent document
  input = window.parent.document.querySelector('input[placeholder="__CARD_ACT__"]');

  // Strategy 2: iterate every input in parent document
  if (!input) {{
    const inputs = window.parent.document.querySelectorAll('input[type="text"], input:not([type])');
    for (const el of inputs) {{
      if (el.placeholder === '__CARD_ACT__') {{ input = el; break; }}
    }}
  }}

  // Strategy 3: search sibling iframes (handles Streamlit fragment isolation)
  if (!input) {{
    const frames = window.parent.document.querySelectorAll('iframe');
    for (const frame of frames) {{
      if (frame === window.frameElement) continue;
      try {{
        const doc = frame.contentDocument || frame.contentWindow.document;
        const el  = doc.querySelector('input[placeholder="__CARD_ACT__"]');
        if (el) {{ input = el; break; }}
      }} catch (e) {{
        // cross-origin – skip silently
      }}
    }}
  }}

  if (!input) {{
    console.error('[Bridge] Could not locate bridge input. Action lost:', action);
    return;
  }}

  // Use native value setter so React/Streamlit detects the change
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(input, action);
  input.dispatchEvent(new Event('input', {{bubbles: true}}));
}}

// ── Card rendering ────────────────────────────────────────────────────────────
function renderCard(card) {{
  const {{sid, img, name, brand, cat, seller, warnings}} = card;
  const isCommitted = COMMITTED.hasOwnProperty(sid);
  const isPending   = pendingRejections.hasOwnProperty(sid);
  const isSelected  = selectedSids.has(sid);

  let cls = 'card';
  if      (isCommitted) cls += ' committed-rej';
  else if (isPending)   cls += ' pending-rej';
  else if (isSelected)  cls += ' selected';

  const shortName = name.length > 40 ? name.slice(0, 40) + '…' : name;

  const warnHtml = warnings.map(w => `<span class="warn-badge">${{esc(w)}}</span>`).join('');
  const warnWrap = warnHtml ? `<div class="warn-wrap">${{warnHtml}}</div>` : '';

  // Overlay for pending and committed rejections
  let overlayHtml = '';
  if (isPending || isCommitted) {{
    const label = isPending
      ? pendingRejections[sid].replace(/_/g, ' ')
      : COMMITTED[sid];
    const undoBtn = isPending
      ? `<button class="undo-btn" onclick="undoPending('${{sid}}')">Undo (not saved)</button>`
      : `<button class="undo-btn" onclick="doRestore('${{sid}}')">Undo (saved)</button>`;
    overlayHtml = `
      <div class="overlay">
        <div class="rej-badge">REJECTED</div>
        <div class="rej-label">${{esc(label)}}</div>
        ${{undoBtn}}
      </div>`;
  }}

  // Action buttons – only shown when card is not rejected
  let actionsHtml = '';
  if (!isPending && !isCommitted) {{
    actionsHtml = `
      <div class="acts">
        <button class="act-btn act-poor"
          onclick="rejectCard(event,'${{sid}}','REJECT_POOR_IMAGE')">Poor Img</button>
        <select class="act-more"
          onchange="if(this.value){{rejectCard(event,'${{sid}}',this.value);this.value=''}}">
          <option value="">More…</option>
          <option value="REJECT_WRONG_CAT">Wrong Category</option>
          <option value="REJECT_FAKE">Fake Product</option>
          <option value="REJECT_BRAND">Restricted Brand</option>
          <option value="REJECT_PROHIBITED">Prohibited</option>
          <option value="REJECT_COLOR">Wrong Color</option>
          <option value="REJECT_WRONG_BRAND">Wrong Brand</option>
        </select>
      </div>`;
  }}

  return `
  <div class="${{cls}}" id="card-${{sid}}" data-sid="${{sid}}">
    <div class="card-img-wrap" onclick="toggleSelect('${{sid}}')">
      ${{warnWrap}}
      <img class="card-img" src="${{esc(img)}}" loading="lazy"
           onerror="this.src='https://via.placeholder.com/150?text=Error'">
      <div class="tick">&#10003;</div>
    </div>
    ${{overlayHtml}}
    <div class="meta">
      <div class="nm" title="${{esc(name)}}">${{esc(shortName)}}</div>
      <div class="br">${{esc(brand)}}</div>
      <div class="ct">${{esc(cat)}}</div>
      <div class="sl">${{esc(seller)}}</div>
    </div>
    ${{actionsHtml}}
  </div>`;
}}

function renderAll() {{
  document.getElementById('card-grid').innerHTML = CARDS.map(renderCard).join('');
  updatePendingIndicator();
}}

function refreshCard(sid) {{
  const card = CARDS.find(c => c.sid === sid);
  if (!card) return;
  const el = document.getElementById('card-' + sid);
  if (!el) return;
  // Create a temporary container so we can replace the element safely
  const tmp = document.createElement('div');
  tmp.innerHTML = renderCard(card);
  el.replaceWith(tmp.firstElementChild);
  updatePendingIndicator();
}}

// ── Pending indicator ─────────────────────────────────────────────────────────
function updatePendingIndicator() {{
  const n = Object.keys(pendingRejections).length;
  const ind = document.getElementById('pending-indicator');
  if (n > 0) {{
    document.getElementById('pending-count').textContent = n;
    ind.style.display = 'inline';
  }} else {{
    ind.style.display = 'none';
  }}
  document.getElementById('sel-count').textContent = selectedSids.size;
}}

// ── Interactions ──────────────────────────────────────────────────────────────
function toggleSelect(sid) {{
  // Cannot select already-rejected cards
  if (pendingRejections.hasOwnProperty(sid) || COMMITTED.hasOwnProperty(sid)) return;
  if (selectedSids.has(sid)) selectedSids.delete(sid);
  else selectedSids.add(sid);
  refreshCard(sid);
}}

function rejectCard(evt, sid, reasonKey) {{
  evt.stopPropagation();
  pendingRejections[sid] = reasonKey;
  selectedSids.delete(sid);
  refreshCard(sid);
}}

function undoPending(sid) {{
  delete pendingRejections[sid];
  refreshCard(sid);
}}

function doRestore(sid) {{
  // Committed rejections need a Python round-trip
  sendBridge('RESTORE:' + sid);
}}

function selectAll() {{
  ALL_SIDS.forEach(sid => {{
    if (!pendingRejections.hasOwnProperty(sid) && !COMMITTED.hasOwnProperty(sid))
      selectedSids.add(sid);
  }});
  renderAll();
}}

function deselectAll() {{
  selectedSids.clear();
  renderAll();
}}

function doBatchReject() {{
  if (selectedSids.size === 0) {{
    alert('Select at least one product first.');
    return;
  }}
  const reasonKey = document.getElementById('batch-reason').value;
  selectedSids.forEach(sid => {{ pendingRejections[sid] = reasonKey; }});
  selectedSids.clear();
  renderAll();
  // Immediately commit so the user sees feedback right away
  commitAll();
}}

function commitAll() {{
  const pending = Object.assign({{}}, pendingRejections);
  if (Object.keys(pending).length === 0) return;
  pendingRejections = {{}};  // optimistic clear
  updatePendingIndicator();
  sendBridge('BATCH_COMMIT:' + JSON.stringify(pending));
}}

// ── Navigation: commit pending atomically with the nav event ──────────────────
function navAction(direction) {{
  const pending = Object.assign({{}}, pendingRejections);
  pendingRejections = {{}};
  // Send ONE event: direction:{{...pendingJSON}}
  // Python handler applies rejections AND navigates in a single rerun.
  sendBridge(direction + ':' + JSON.stringify(pending));
}}

// ── Init ──────────────────────────────────────────────────────────────────────
renderAll();
</script>
</body>
</html>"""
    return html


# =============================================================================
# IMAGE GRID FRAGMENT
# =============================================================================
@st.fragment
def render_image_grid() -> None:
    if st.session_state.final_report.empty or st.session_state.file_mode == "post_qc":
        return

    st.markdown("---")
    st.header(":material/pageview: Manual Image & Category Review", anchor=False)

    # ── Hidden bridge input ───────────────────────────────────────────────────
    # label_visibility="collapsed" hides the label visually but still renders
    # the <input> element that the JS sendBridge() function targets.
    action_bridge = st.text_input(
        "bridge",
        key="card_action_bridge",
        label_visibility="collapsed",
        placeholder="__CARD_ACT__",
    )

    # ── Process any incoming bridge event ─────────────────────────────────────
    if action_bridge:
        # Clear BEFORE processing so a crash doesn't cause an infinite loop
        st.session_state["card_action_bridge"] = ""
        changed = _process_card_bridge_action(action_bridge, support_files)
        if changed:
            # Rerun only this fragment (Streamlit ≥ 1.37); fall back to full rerun
            try:
                st.rerun(scope="fragment")
            except TypeError:
                st.rerun()
        return  # Skip rendering until the clean rerun arrives

    # ── Build the reviewable set ──────────────────────────────────────────────
    fr = st.session_state.final_report
    committed_rej_sids = {
        k.replace("quick_rej_", "")
        for k in st.session_state.keys()
        if k.startswith("quick_rej_") and "reason" not in k
    }
    mask = (fr["Status"] == "Approved") | (fr["ProductSetSid"].isin(committed_rej_sids))
    valid_grid_df = fr[mask]

    if valid_grid_df.empty:
        st.info("No products to review on this page.")
        return

    # ── Search & pagination controls ──────────────────────────────────────────
    c1, c2, c3 = st.columns([1.5, 1.5, 2])
    with c1:
        search_n  = st.text_input("Search by Name",            placeholder="Product name…")
    with c2:
        search_sc = st.text_input("Search by Seller/Category", placeholder="Seller or category…")
    with c3:
        st.session_state.grid_items_per_page = st.select_slider(
            "Items per page", options=[20, 50, 100, 200],
            value=st.session_state.grid_items_per_page,
        )

    # ── Merge with product data ───────────────────────────────────────────────
    available_cols = [c for c in GRID_COLS if c in st.session_state.all_data_map.columns]
    if "PRODUCT_SET_SID" not in available_cols:
        st.warning("PRODUCT_SET_SID not found in data. Cannot render grid.")
        return

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
        has_cat = "CATEGORY" in review_data.columns
        mc = (
            review_data["CATEGORY"].astype(str).str.contains(search_sc, case=False, na=False)
            if has_cat
            else pd.Series(False, index=review_data.index)
        )
        ms = review_data["SELLER_NAME"].astype(str).str.contains(search_sc, case=False, na=False)
        review_data = review_data[mc | ms]

    # ── Pagination ────────────────────────────────────────────────────────────
    ipp         = st.session_state.grid_items_per_page
    total_pages = max(1, (len(review_data) + ipp - 1) // ipp)
    if st.session_state.grid_page >= total_pages:
        st.session_state.grid_page = 0

    page_start = st.session_state.grid_page * ipp
    page_data  = review_data.iloc[page_start : page_start + ipp].copy()

    st.caption(
        f"Showing {len(page_data)} of {len(review_data)} products  •  "
        f"Page {st.session_state.grid_page + 1} / {total_pages}"
    )

    # ── Parallel image quality checks ─────────────────────────────────────────
    page_warnings: Dict[str, List[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_sid = {
            executor.submit(
                analyze_image_quality_cached,
                str(row.get("MAIN_IMAGE", "")).strip()
            ): str(row["PRODUCT_SET_SID"])
            for _, row in page_data.iterrows()
        }
        for future in concurrent.futures.as_completed(future_to_sid):
            try:
                warns = future.result()
                if warns:
                    page_warnings[future_to_sid[future]] = warns
            except Exception:
                pass

    # ── Committed rejections on this page ────────────────────────────────────
    rejected_state: Dict[str, str] = {
        sid: st.session_state[f"quick_rej_reason_{sid}"]
        for sid in page_data["PRODUCT_SET_SID"].astype(str)
        if st.session_state.get(f"quick_rej_{sid}")
    }

    # ── Render the HTML grid ──────────────────────────────────────────────────
    cols_per_row = 3 if st.session_state.layout_mode == "centered" else 4
    grid_html = build_fast_grid_html(
        page_data,
        support_files["flags_mapping"],
        st.session_state.selected_country,
        page_warnings,
        rejected_state,
        cols_per_row,
        st.session_state.grid_page,
        total_pages,
    )
    components.html(grid_html, height=1350, scrolling=True)

    # ── Smooth scroll to top after navigation ────────────────────────────────
    if st.session_state.get("do_scroll_top", False):
        st.components.v1.html(
            "<script>"
            "try{window.parent.document.querySelector('.main')"
            ".scrollTo({top:0,behavior:'smooth'});}catch(e){}"
            "</script>",
            height=0,
        )
        st.session_state.do_scroll_top = False


# =============================================================================
# EXPORTS FRAGMENT
# =============================================================================
@st.fragment
def render_exports_section() -> None:
    if st.session_state.final_report.empty or st.session_state.file_mode == "post_qc":
        return

    fr      = st.session_state.final_report
    data    = st.session_state.all_data_map
    app_df  = fr[fr["Status"] == "Approved"]
    rej_df  = fr[fr["Status"] == "Rejected"]
    c_code  = st.session_state.selected_country[:2].upper()
    date_str = datetime.now().strftime("%Y-%m-%d")

    st.markdown("---")
    st.markdown(
        f"<div style='background:linear-gradient(135deg,{JUMIA_COLORS['primary_orange']},"
        f"{JUMIA_COLORS['secondary_orange']});padding:20px 24px;border-radius:10px;"
        f"margin-bottom:20px;box-shadow:0 4px 12px rgba(246,139,30,.25);'>"
        f"<h2 style='color:white;margin:0;font-size:24px;font-weight:700;'>Download Reports</h2>"
        f"</div>",
        unsafe_allow_html=True,
    )

    exports_config = [
        ("Final Report",  fr,    "assignment",   "Complete validation report with all statuses",
         lambda df: generate_smart_export(df, f"{c_code}_Final_{date_str}",    "simple")),
        ("Rejected Only", rej_df, "block",        "Products that failed validation",
         lambda df: generate_smart_export(df, f"{c_code}_Rejected_{date_str}", "simple")),
        ("Approved Only", app_df, "check_circle", "Products that passed validation",
         lambda df: generate_smart_export(df, f"{c_code}_Approved_{date_str}", "simple")),
        ("Full Data",     data,  "database",     "Complete dataset with validation flags",
         lambda df: generate_smart_export(
             prepare_full_data_merged(df, fr), f"{c_code}_Full_{date_str}", "full"
         )),
    ]

    cols_count = 4 if st.session_state.layout_mode == "wide" else 2
    for i in range(0, len(exports_config), cols_count):
        cols = st.columns(cols_count)
        for j, col in enumerate(cols):
            if i + j >= len(exports_config):
                break
            title, df, icon, desc, func = exports_config[i + j]
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='text-align:center;margin-bottom:15px;'>"
                        f"<div style='font-size:48px;margin-bottom:8px;'"
                        f" class='material-symbols-outlined'>{icon}</div>"
                        f"<div style='font-size:18px;font-weight:700;'>{title}</div>"
                        f"<div style='font-size:12px;color:#666;margin-top:4px;'>{desc}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    export_key = title
                    if export_key not in st.session_state.exports_cache:
                        if st.button("Generate", key=f"gen_{title}", type="primary",
                                     use_container_width=True):
                            with st.spinner(f"Generating {title}…"):
                                res, fname, mime = func(df)
                                st.session_state.exports_cache[export_key] = {
                                    "data": res.getvalue(), "fname": fname, "mime": mime
                                }
                            st.rerun()
                    else:
                        cache = st.session_state.exports_cache[export_key]
                        st.download_button(
                            "⬇ Download", data=cache["data"],
                            file_name=cache["fname"], mime=cache["mime"],
                            use_container_width=True, type="primary",
                            key=f"dl_{title}",
                        )
                        if st.button("Clear", key=f"clr_{title}", use_container_width=True):
                            del st.session_state.exports_cache[export_key]
                            st.rerun()


# =============================================================================
# APP INITIALISATION
# =============================================================================
try:
    support_files = load_support_files_lazy()
except Exception as e:
    st.error(f"Failed to load support files: {e}")
    st.stop()

def get_image_base64(path: str) -> str:
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
    return ""

logo_b64  = get_image_base64("jumia logo.png") or get_image_base64("jumia_logo.png")
logo_html = (
    f"<img src='data:image/png;base64,{logo_b64}' style='height:42px;margin-right:15px;'>"
    if logo_b64
    else "<span class='material-symbols-outlined' style='font-size:42px;margin-right:15px;'>verified_user</span>"
)

st.markdown(
    f"<div style='background:linear-gradient(135deg,{JUMIA_COLORS['primary_orange']},"
    f"{JUMIA_COLORS['secondary_orange']});padding:25px;border-radius:12px;"
    f"margin-bottom:20px;box-shadow:0 4px 12px rgba(246,139,30,.3);'>"
    f"<h1 style='color:white;margin:0;font-size:36px;display:flex;align-items:center;'>"
    f"{logo_html}Product Validation Tool</h1></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("System Status")
    if st.button("🔄 Clear Cache & Reload Data", use_container_width=True, type="secondary"):
        st.cache_data.clear()
        st.session_state.display_df_cache = {}
        st.rerun()

# =============================================================================
# SECTION 1 – UPLOAD & VALIDATION
# =============================================================================
st.header(":material/upload_file: Upload Files", anchor=False)

current_country = st.session_state.get("selected_country", "Kenya")
country_choice  = st.segmented_control(
    "Country", ["Kenya", "Uganda", "Nigeria", "Ghana", "Morocco"],
    default=current_country,
)
if country_choice:
    st.session_state.selected_country = country_choice
else:
    country_choice = current_country

country_validator = CountryValidator(st.session_state.selected_country)

uploaded_files = st.file_uploader(
    "Upload CSV or XLSX files", type=["csv", "xlsx"],
    accept_multiple_files=True, key="daily_files",
)

if uploaded_files:
    current_file_signature = hashlib.md5(
        str(sorted([f.name + str(f.size) for f in uploaded_files])).encode()
    ).hexdigest()
    process_signature = f"{current_file_signature}_{country_validator.code}"
else:
    process_signature = "empty"

if st.session_state.get("last_processed_files") != process_signature:
    # Reset all state for new files / country change
    st.session_state.final_report      = pd.DataFrame()
    st.session_state.all_data_map      = pd.DataFrame()
    st.session_state.grid_data         = pd.DataFrame()
    st.session_state.post_qc_summary   = pd.DataFrame()
    st.session_state.post_qc_results   = {}
    st.session_state.file_mode         = None
    st.session_state.grid_page         = 0
    st.session_state.exports_cache     = {}
    st.session_state.display_df_cache  = {}
    # Clean up per-card rejection keys
    for k in list(st.session_state.keys()):
        if k.startswith(("quick_rej_", "grid_chk_", "toast_", "card_action_")):
            del st.session_state[k]

    if process_signature == "empty":
        st.session_state.last_processed_files = "empty"
    else:
        cache_data_path   = f"cache_{process_signature}_data.parquet"
        cache_report_path = f"cache_{process_signature}_report.parquet"

        if os.path.exists(cache_data_path) and os.path.exists(cache_report_path):
            st.toast("Loaded from cache instantly! 🚀", icon="⚡")
            st.session_state.all_data_map = pd.read_parquet(cache_data_path)
            st.session_state.final_report = pd.read_parquet(cache_report_path)
            st.session_state.file_mode    = "pre_qc"
            st.session_state.last_processed_files = process_signature
        else:
            try:
                all_dfs        = []
                detected_modes = []

                for uf in uploaded_files:
                    uf.seek(0)
                    if uf.name.endswith(".xlsx"):
                        raw_data = pd.read_excel(uf, engine="openpyxl", dtype=str)
                    else:
                        try:
                            raw_data = pd.read_csv(uf, dtype=str)
                            if len(raw_data.columns) <= 1:
                                uf.seek(0)
                                raw_data = pd.read_csv(uf, sep=";", encoding="ISO-8859-1", dtype=str)
                        except Exception:
                            uf.seek(0)
                            raw_data = pd.read_csv(uf, sep=";", encoding="ISO-8859-1", dtype=str)
                    detected_modes.append(detect_file_type(raw_data))
                    all_dfs.append(raw_data)

                file_mode = detected_modes[0] if detected_modes else "pre_qc"
                st.session_state.file_mode = file_mode

                if file_mode == "post_qc":
                    norm_dfs = [normalize_post_qc(df) for df in all_dfs]
                    merged   = pd.concat(norm_dfs, ignore_index=True).drop_duplicates(
                        subset=["PRODUCT_SET_SID"]
                    )
                    summary_df, results = run_post_qc_checks(merged, support_files)
                    st.session_state.post_qc_summary = summary_df
                    st.session_state.post_qc_results = results
                    st.session_state.post_qc_data    = merged
                    st.session_state.last_processed_files = process_signature
                else:
                    std_dfs     = [standardize_input_data(df) for df in all_dfs]
                    merged_data = pd.concat(std_dfs, ignore_index=True)
                    data_prop   = propagate_metadata(merged_data)
                    is_valid, errors = validate_input_schema(data_prop)

                    if is_valid:
                        data_filtered, det_names = filter_by_country(data_prop, country_validator)
                        if data_filtered.empty:
                            st.error(
                                f"No {country_validator.country} products found. "
                                f"Detected: {', '.join(det_names) if det_names else 'none'}",
                                icon=":material/error:",
                            )
                            st.session_state.last_processed_files = "error"
                            st.stop()

                        actual_counts = data_filtered.groupby("PRODUCT_SET_SID")["PRODUCT_SET_SID"].transform("count")
                        if "COUNT_VARIATIONS" in data_filtered.columns:
                            file_counts = pd.to_numeric(
                                data_filtered["COUNT_VARIATIONS"], errors="coerce"
                            ).fillna(1)
                            data_filtered["COUNT_VARIATIONS"] = actual_counts.combine(file_counts, max)
                        else:
                            data_filtered["COUNT_VARIATIONS"] = actual_counts

                        data = data_filtered.drop_duplicates(subset=["PRODUCT_SET_SID"], keep="first")
                        data_has_warranty = all(
                            c in data.columns for c in ["PRODUCT_WARRANTY", "WARRANTY_DURATION"]
                        )

                        for c in ["NAME", "BRAND", "COLOR", "SELLER_NAME", "CATEGORY_CODE", "LIST_VARIATIONS"]:
                            if c in data.columns:
                                data[c] = data[c].astype(str).fillna("")
                        if "COLOR_FAMILY" not in data.columns:
                            data["COLOR_FAMILY"] = ""

                        data_hash    = df_hash(data) + country_validator.code
                        final_report, _ = cached_validate_products(
                            data_hash, data, support_files,
                            country_validator.code, data_has_warranty
                        )

                        st.session_state.final_report = final_report
                        st.session_state.all_data_map = data

                        # Persist to parquet cache
                        try:
                            data.to_parquet(cache_data_path)
                            final_report.to_parquet(cache_report_path)
                        except Exception as cache_err:
                            logger.warning(f"Parquet cache write failed: {cache_err}")

                        st.session_state.last_processed_files = process_signature
                    else:
                        for e in errors:
                            st.error(e)
                        st.session_state.last_processed_files = "error"

            except Exception as e:
                st.error(f"Processing error: {e}\n\n{traceback.format_exc()}")
                st.session_state.last_processed_files = "error"

# =============================================================================
# SECTION 2 – IMAGE & CATEGORY REVIEW GRID
# =============================================================================
render_image_grid()

# =============================================================================
# SECTION 3 – EXPORTS
# =============================================================================
render_exports_section()
