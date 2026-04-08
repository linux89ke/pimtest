"""
data_utils.py - Data loading, cleaning, transformation and validation helpers
"""

import re
import hashlib
import logging
import pandas as pd
from io import BytesIO
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

from constants import NEW_FILE_MAPPING, COLOR_VARIANT_TO_BASE, MULTI_COUNTRY_VALUES

logger = logging.getLogger(__name__)


# -------------------------------------------------
# TEXT & KEY HELPERS
# -------------------------------------------------

def clean_category_code(code) -> str:
    try:
        if pd.isna(code):
            return ""
        s = str(code).strip()
        if '.' in s:
            s = s.split('.')[0]
        return s
    except:
        return str(code).strip()


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
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
    except Exception as e:
        logger.warning(f"df_hash primary failed, using fallback: {e}")
        fallback_str = str(df.shape) + str(df.columns.tolist())
        return hashlib.md5(fallback_str.encode()).hexdigest()


# -------------------------------------------------
# COLOR EXTRACTION HELPERS
# -------------------------------------------------

def extract_colors(text: str, explicit_color: Optional[str] = None) -> Set[str]:
    colors = set()
    text_lower = str(text).lower() if text else ""
    if explicit_color and pd.notna(explicit_color):
        color_lower = str(explicit_color).lower().strip()
        for variant, base in COLOR_VARIANT_TO_BASE.items():
            if variant in color_lower:
                colors.add(base)
    for variant, base in COLOR_VARIANT_TO_BASE.items():
        if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
            colors.add(base)
    return colors


def remove_attributes(text: str) -> str:
    base = str(text).lower() if text else ""
    for variant in COLOR_VARIANT_TO_BASE.keys():
        base = re.sub(r'\b' + re.escape(variant) + r'\b', '', base)
    base = re.sub(r'\b(?:xxs|xs|small|medium|large|xl|xxl|xxxl)\b', '', base)
    base = re.sub(r'\b\d+\s*(?:gb|tb|inch|inches|"|ram|memory|ddr|pack|piece|pcs)\b', '', base)
    for word in ['new', 'original', 'genuine', 'authentic', 'official', 'premium', 'quality', 'best', 'hot', 'sale', 'promo', 'deal']:
        base = re.sub(r'\b' + word + r'\b', '', base)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', base)).strip()


@dataclass
class ProductAttributes:
    base_name: str
    colors: Set[str]
    sizes: Set[str]
    storage: Set[str]
    memory: Set[str]
    quantities: Set[str]
    raw_name: str


def extract_product_attributes(name: str, explicit_color: Optional[str] = None, brand: Optional[str] = None) -> ProductAttributes:
    name_str = str(name).strip() if pd.notna(name) else ""
    attrs = ProductAttributes(
        base_name="",
        colors=extract_colors(name_str, explicit_color),
        sizes=set(), storage=set(), memory=set(), quantities=set(),
        raw_name=name_str
    )
    base_name = remove_attributes(name_str)
    if brand and pd.notna(brand):
        brand_lower = str(brand).lower().strip()
        if brand_lower not in base_name and brand_lower not in ['generic', 'fashion']:
            base_name = f"{brand_lower} {base_name}"
    attrs.base_name = base_name.strip()
    return attrs


# -------------------------------------------------
# FILE READING HELPERS
# -------------------------------------------------

def _detect_and_read_csv(buf) -> pd.DataFrame:
    _ENCODINGS = ['utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1']
    raw_bytes = buf.read()
    for enc in _ENCODINGS:
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(BytesIO(raw_bytes), sep=sep, encoding=enc, dtype=str)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(BytesIO(raw_bytes), sep=None, engine='python', encoding='utf-8', dtype=str)


def _repair_mojibake(df: pd.DataFrame) -> pd.DataFrame:
    _ILLEGAL_XML = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

    def _fix(val):
        if not isinstance(val, str):
            return val
        for enc in ('cp1252', 'latin-1'):
            try:
                fixed = val.encode(enc).decode('utf-8')
                if fixed != val and '\ufffd' not in fixed:
                    val = fixed
                    break
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue
        return _ILLEGAL_XML.sub('', val)

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(_fix)
    return df


# -------------------------------------------------
# SCHEMA & TRANSFORMATION
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
    if 'MAIN_IMAGE' not in df.columns:
        df['MAIN_IMAGE'] = ''
    return df


def validate_input_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY_CODE', 'ACTIVE_STATUS_COUNTRY']
    errors = [f"Missing: {f}" for f in required if f not in df.columns]
    return len(errors) == 0, errors


def filter_by_country(df: pd.DataFrame, country_validator) -> Tuple[pd.DataFrame, List[str]]:
    if 'ACTIVE_STATUS_COUNTRY' not in df.columns:
        return df, []
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
    if df.empty:
        return df
    for col in ['COLOR_FAMILY', 'PRODUCT_WARRANTY', 'WARRANTY_DURATION', 'WARRANTY_ADDRESS', 'WARRANTY_TYPE', 'COUNT_VARIATIONS', 'LIST_VARIATIONS']:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df.groupby('PRODUCT_SET_SID')[col].transform(lambda x: x.ffill().bfill())
    return df


# -------------------------------------------------
# EXCHANGE RATE & PRICE FORMATTING
# -------------------------------------------------

import streamlit as st

@st.cache_data(ttl=3600)
def fetch_exchange_rate(country: str) -> float:
    from constants import COUNTRY_CURRENCY
    cfg = COUNTRY_CURRENCY.get(country)
    if not cfg:
        return 1.0
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen("https://open.er-api.com/v6/latest/USD", timeout=3) as r:
            data = _json.loads(r.read())
        return float(data["rates"].get(cfg["code"], 1.0))
    except Exception as e:
        logger.warning(f"Exchange rate fetch failed for {country}: {e}")
        fallbacks = {"Kenya": 128.0, "Uganda": 3750.0, "Nigeria": 1550.0, "Ghana": 15.5, "Morocco": 10.1}
        return fallbacks.get(country, 1.0)


def format_local_price(usd_price, country: str) -> str:
    from constants import COUNTRY_CURRENCY
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
