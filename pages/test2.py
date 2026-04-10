"""
main.py - Main Streamlit Application Entry Point
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import st_yled
from io import BytesIO
from datetime import datetime
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import traceback
import json
import zipfile
import os
import shutil
import pickle
import concurrent.futures
from dataclasses import dataclass
import hashlib
import base64
import requests
from PIL import Image

from translations import LANGUAGES, get_translation

# ── NEW MODULAR IMPORTS ───────────────────────────────────────────────────────
from constants import JUMIA_COLORS, PARQUET_CACHE_DIR, FLAG_CACHE_DIR, REASON_MAP, GRID_COLS
from data_utils import (
    clean_category_code, df_hash, standardize_input_data, validate_input_schema, 
    filter_by_country, propagate_metadata, create_match_key, normalize_text,
    _detect_and_read_csv, _repair_mojibake
)
from loaders import load_support_files_lazy, compile_regex_patterns
from ui_components import render_image_grid, render_exports_section, render_flag_expander

from nigeria_rules import (
    load_nigeria_qc_rules, check_nigeria_gift_card, check_nigeria_books,
    check_nigeria_tvs, check_nigeria_hp_toners, check_nigeria_apple,
    check_nigeria_xmas_tree, check_nigeria_rice, check_nigeria_powerbanks
)
from morocco_rules import load_morocco_qc_rules, check_morocco_prohibited_brands
from pricing_rules import check_wrong_price, check_category_max_price, CATEGORY_MAX_PRICES_USD
# ──────────────────────────────────────────────────────────────────────────────

try:
    from postqc import detect_file_type, normalize_post_qc, run_checks as run_post_qc_checks, render_post_qc_section, load_category_map
except ImportError:
    pass

try:
    import _preqc_registry as _reg
except ImportError:
    _reg = None

try:
    from jumia_scraper import enrich_post_qc_df, COUNTRY_BASE_URLS as _SCRAPER_URLS
    _SCRAPER_AVAILABLE = True
except ImportError:
    _SCRAPER_AVAILABLE = False

# ── Category Matcher Engine ───────────────────────────────────────────────────
try:
    from category_matcher_engine import CategoryMatcherEngine, check_wrong_category, get_engine
    _CAT_MATCHER_AVAILABLE = True
except ImportError:
    _CAT_MATCHER_AVAILABLE = False
    def check_wrong_category(data, categories_list=None, cat_path_to_code=None, code_to_path=None, confidence_threshold=0.0):
        if 'CATEGORY' not in data.columns:
            return pd.DataFrame(columns=data.columns)
        flagged = data[data['CATEGORY'].astype(str).str.contains("miscellaneous", case=False, na=False)].copy()
        if not flagged.empty:
            flagged['Comment_Detail'] = "Category contains 'Miscellaneous'"
        return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

@st.cache_resource(show_spinner=False)
def _get_cat_matcher_engine():
    if not _CAT_MATCHER_AVAILABLE:
        return None
    try:
        return get_engine()
    except Exception as e:
        logging.warning("CategoryMatcherEngine init failed: %s", e)
        return None

logger = logging.getLogger(__name__)

# -------------------------------------------------
# CACHE HELPERS
# -------------------------------------------------
os.makedirs(PARQUET_CACHE_DIR, exist_ok=True)
os.makedirs(FLAG_CACHE_DIR, exist_ok=True)

def prune_cache_dir(directory: str, max_files: int = 500):
    try:
        files = sorted(Path(directory).glob("*.pkl"), key=os.path.getmtime)
        stale = files[:-max_files]
        for f in stale: f.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Cache pruning failed for {directory}: {e}")

prune_cache_dir(FLAG_CACHE_DIR)
try:
    for _cf in Path(FLAG_CACHE_DIR).glob("*.pkl"):
        if _cf.stat().st_size < 500: _cf.unlink(missing_ok=True)
except Exception: pass

def save_df_parquet(df, filename):
    try: df.to_parquet(os.path.join(PARQUET_CACHE_DIR, filename))
    except Exception as e: logger.warning(f"Failed to save parquet {filename}: {e}")

def load_df_parquet(filename):
    path = os.path.join(PARQUET_CACHE_DIR, filename)
    if os.path.exists(path):
        try: return pd.read_parquet(path)
        except Exception as e: logger.warning(f"Failed to load parquet {filename}: {e}")
    return None

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

    def should_skip_validation(self, validation_name: str) -> bool: 
        return validation_name in self.skip_validations
    
    def ensure_status_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and 'Status' not in df.columns: 
            df['Status'] = 'Approved'
        return df

# -------------------------------------------------
# CACHE-AWARE VALIDATION CHECKS (STANDARD)
# -------------------------------------------------
FLAG_RELEVANT_COLS = {
    "Wrong Category": ["NAME", "CATEGORY", "CATEGORY_CODE"],
    "Restricted brands": ["NAME", "BRAND", "SELLER_NAME", "CATEGORY_CODE"],
    "Suspected Fake product": ["CATEGORY_CODE", "BRAND", "GLOBAL_SALE_PRICE", "GLOBAL_PRICE"],
    "Seller Not approved to sell Refurb": ["PRODUCT_SET_SID", "CATEGORY_CODE", "SELLER_NAME", "NAME"],
    "Product Warranty": ["PRODUCT_WARRANTY", "WARRANTY_DURATION", "CATEGORY_CODE"],
    "Seller Approve to sell books": ["CATEGORY_CODE", "SELLER_NAME"],
    "Seller Approved to Sell Perfume": ["CATEGORY_CODE", "SELLER_NAME", "BRAND", "NAME"],
    "Counterfeit Sneakers": ["CATEGORY_CODE", "NAME", "BRAND"],
    "Suspected counterfeit Jerseys": ["CATEGORY_CODE", "NAME", "SELLER_NAME"],
    "Unnecessary words in NAME": ["NAME"],
    "Single-word NAME": ["CATEGORY_CODE", "NAME"],
    "Generic BRAND Issues": ["CATEGORY_CODE", "BRAND"],
    "Fashion brand issues": ["CATEGORY_CODE", "BRAND"],
    "BRAND name repeated in NAME": ["BRAND", "NAME"],
    "Wrong Variation": ["COUNT_VARIATIONS", "CATEGORY_CODE"],
    "Generic branded products with genuine brands": ["NAME", "BRAND", "CATEGORY"],
    "Missing COLOR": ["CATEGORY_CODE", "NAME", "COLOR"],
    "Missing Weight/Volume": ["CATEGORY_CODE", "NAME"],
    "Incomplete Smartphone Name": ["CATEGORY_CODE", "NAME"],
    "Duplicate product": ["NAME", "SELLER_NAME", "BRAND", "CATEGORY_CODE"],
    "Perfume Tester": ["CATEGORY_CODE", "NAME"],
    "Wrong Price": ["GLOBAL_PRICE", "GLOBAL_SALE_PRICE"],
    "Category Max Price Exceeded": ["CATEGORY", "GLOBAL_PRICE", "GLOBAL_SALE_PRICE", "CATEGORY_CODE"],
    "Poor images": ["MAIN_IMAGE"],
    "NG - Gift Card Seller":  ["CATEGORY_CODE", "SELLER_NAME"],
    "NG - Books Seller":      ["NAME", "SELLER_NAME"],
    "NG - TV Brand Seller":   ["CATEGORY_CODE", "BRAND", "SELLER_NAME"],
    "NG - HP Toners Seller":  ["CATEGORY_CODE", "BRAND", "SELLER_NAME"],
    "NG - Apple Seller":      ["BRAND", "SELLER_NAME"],
    "NG - Xmas Tree Seller":  ["NAME", "SELLER_NAME"],
    "NG - Rice Brand Seller": ["CATEGORY_CODE", "BRAND", "SELLER_NAME"],
    "Powerbank Not Authorized":["CATEGORY_CODE", "NAME", "BRAND"],
}

def compute_flag_input_hash(data: pd.DataFrame, flag_name: str, kwargs: dict) -> str:
    cols = FLAG_RELEVANT_COLS.get(flag_name, data.columns.tolist())
    available_cols = [c for c in cols if c in data.columns]
    if not available_cols: return "empty"
    df_hash_str = df_hash(data[available_cols])
    kwargs_repr = ""
    _skip_keys = {'categories_list', 'cat_path_to_code', 'code_to_path'}
    for k, v in kwargs.items():
        if k == 'data' or k in _skip_keys: continue
        if isinstance(v, pd.DataFrame): kwargs_repr += df_hash(v)
        else: kwargs_repr += repr(v)
    return hashlib.md5((df_hash_str + kwargs_repr).encode()).hexdigest()

def run_cached_check(func, cache_path, ckwargs):
    if func is check_miscellaneous_category: return func(**ckwargs)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f: return pickle.load(f)
        except: pass
    res = func(**ckwargs)
    try:
        with open(cache_path, 'wb') as f: pickle.dump(res, f)
    except: pass
    return res

# -------------------------------------------------
# STANDARD VALIDATION LOGIC
# -------------------------------------------------

def check_poor_images_aspect_ratio(data: pd.DataFrame) -> pd.DataFrame:
    if 'MAIN_IMAGE' not in data.columns:
        return pd.DataFrame(columns=data.columns)
        
    target = data[data['MAIN_IMAGE'].astype(str).str.startswith('http')].copy()
    if target.empty:
        return pd.DataFrame(columns=data.columns)

    unique_urls = target['MAIN_IMAGE'].unique()
    
    def fetch_image_ratio(url):
        try:
            r = requests.get(url, stream=True, timeout=3)
            if r.status_code == 200:
                img = Image.open(r.raw)
                w, h = img.size
                if w > 0:
                    ratio = h / w
                    if ratio > 1.5:
                        return url, f"Tall Aspect Ratio ({w}x{h})"
                    elif ratio < 0.6:
                        return url, f"Wide Aspect Ratio ({w}x{h})"
        except Exception:
            pass
        return url, None

    url_issues = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_image_ratio, url) for url in unique_urls]
        for future in concurrent.futures.as_completed(futures):
            url, issue = future.result()
            if issue:
                url_issues[url] = issue

    if not url_issues:
        return pd.DataFrame(columns=data.columns)

    mask = target['MAIN_IMAGE'].isin(url_issues.keys())
    flagged = target[mask].copy()
    flagged['Comment_Detail'] = flagged['MAIN_IMAGE'].map(url_issues)
    
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_miscellaneous_category(data: pd.DataFrame, categories_list: list = None, compiled_rules: dict = None, cat_path_to_code: dict = None, code_to_path: dict = None) -> pd.DataFrame:
    if not categories_list or not code_to_path:
        try:
            _sf = st.session_state.get("support_files", {})
            categories_list = categories_list or _sf.get("categories_names_list", [])
            cat_path_to_code = cat_path_to_code or _sf.get("cat_path_to_code", {})
            code_to_path = code_to_path or _sf.get("code_to_path", {})
        except: pass

    if _CAT_MATCHER_AVAILABLE:
        try:
            _engine = _get_cat_matcher_engine()
            if _engine is not None:
                if categories_list and not _engine._tfidf_built:
                    _engine.build_tfidf_index(categories_list)
                return check_wrong_category(
                    data, categories_list, compiled_rules=compiled_rules,
                    cat_path_to_code=cat_path_to_code, code_to_path=code_to_path,
                )
        except Exception as _e:
            logger.warning("check_wrong_category engine error: %s", _e)

    if 'CATEGORY' not in data.columns: return pd.DataFrame(columns=data.columns)
    flagged = data[data['CATEGORY'].astype(str).str.contains("miscellaneous", case=False, na=False)].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Category contains 'Miscellaneous'"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_restricted_brands(data: pd.DataFrame, country_rules: List[Dict]) -> pd.DataFrame:
    if not {'NAME', 'BRAND', 'SELLER_NAME', 'CATEGORY_CODE'}.issubset(data.columns) or not country_rules: return pd.DataFrame(columns=data.columns)
    d = data.copy()
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

def check_suspected_fake_products(data: pd.DataFrame, suspected_fake_df: pd.DataFrame) -> pd.DataFrame:
    if not all(c in data.columns for c in ['CATEGORY_CODE', 'BRAND', 'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE']) or suspected_fake_df.empty:
        return pd.DataFrame(columns=data.columns)
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
        prices = d['price_to_use'].values
        brands = d['_brand_lower'].values
        cats = d['_cat_clean'].values
        d['is_fake'] = [p < brand_cat_price.get((b, c), -1) for p, b, c in zip(prices, brands, cats)]
        return d[d['is_fake'] == True][data.columns].drop_duplicates(subset=['PRODUCT_SET_SID'])
    except Exception as e:
        logger.warning(f"check_suspected_fake_products: {e}")
        return pd.DataFrame(columns=data.columns)

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
    is_phone = d['_cat_clean'].isin(phone_cats)
    is_laptop = d['_cat_clean'].isin(laptop_cats)
    in_scope = is_phone | is_laptop
    has_keyword = d['NAME'].astype(str).str.contains(kw_pattern, na=False)
    approved_phones = sellers.get("Phones", set())
    approved_laptops = sellers.get("Laptops", set())
    not_approved = ((is_phone & ~d['_seller_lower'].isin(approved_phones)) | (is_laptop & ~d['_seller_lower'].isin(approved_laptops)))
    flagged = d[in_scope & has_keyword & not_approved].copy()
    if not flagged.empty:
        def build_comment(row):
            ptype = "Phone" if row['_cat_clean'] in phone_cats else "Laptop"
            match = kw_pattern.search(str(row['NAME']))
            kw_found = match.group(0) if match else "?"
            return f"Unapproved {ptype} refurb seller — keyword '{kw_found}' in name (cat: {row['_cat_clean']})"
        flagged['Comment_Detail'] = flagged.apply(build_comment, axis=1)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_product_warranty(data: pd.DataFrame, warranty_category_codes: List[str]) -> pd.DataFrame:
    d = data.copy()
    for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION']:
        if c not in d.columns: d[c] = ""
        d[c] = d[c].astype(str).fillna('').str.strip()
    if not warranty_category_codes: return pd.DataFrame(columns=d.columns)
    target = d[d['_cat_clean'].isin([clean_category_code(c) for c in warranty_category_codes])]
    if target.empty: return pd.DataFrame(columns=d.columns)
    def is_present(s): return (s != 'nan') & (s != '') & (s != 'none') & (s != 'nat') & (s != 'n/a')
    mask = ~(is_present(target['PRODUCT_WARRANTY']) | is_present(target['WARRANTY_DURATION']))
    return target[mask].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_seller_approved_for_books(data: pd.DataFrame, books_data: Dict, country_code: str, book_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'SELLER_NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    category_codes = books_data.get('category_codes') or set(clean_category_code(c) for c in book_category_codes)
    if not category_codes: return pd.DataFrame(columns=data.columns)
    approved_sellers = books_data.get('sellers', {}).get(country_code, set())
    if not approved_sellers: return pd.DataFrame(columns=data.columns)
    books = data[data['_cat_clean'].isin(category_codes)].copy()
    if books.empty: return pd.DataFrame(columns=data.columns)
    not_approved = ~books['_seller_lower'].isin(approved_sellers)
    flagged = books[not_approved].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Seller not approved to sell books: " + flagged['SELLER_NAME'].astype(str)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_seller_approved_for_perfume(data: pd.DataFrame, perfume_category_codes: List[str], perfume_data: Dict, country_code: str) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'SELLER_NAME', 'BRAND', 'NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    sheet_cat_codes = perfume_data.get('category_codes')
    cat_codes = sheet_cat_codes if sheet_cat_codes else set(clean_category_code(c) for c in perfume_category_codes)
    perfume = data[data['_cat_clean'].isin(cat_codes)].copy()
    if perfume.empty: return pd.DataFrame(columns=data.columns)
    keywords = perfume_data.get('keywords', set())
    approved_sellers = perfume_data.get('sellers', {}).get(country_code, set())
    has_seller_list = bool(approved_sellers)
    GENERIC_PLACEHOLDERS = {'designers collection', 'smart collection', 'generic', 'original', 'fashion'}
    if keywords:
        kw_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r')\b', re.IGNORECASE)
        sneaky_mask = perfume['_brand_lower'].isin(GENERIC_PLACEHOLDERS) & perfume['_name_lower'].apply(lambda x: bool(kw_pattern.search(x)))
    else: sneaky_mask = pd.Series([False] * len(perfume), index=perfume.index)
    if has_seller_list:
        brand_sens_mask = perfume['_brand_lower'].apply(lambda x: bool(kw_pattern.search(x))) if keywords else pd.Series([False]*len(perfume), index=perfume.index)
        needs_approval = sneaky_mask | brand_sens_mask
        not_approved = ~perfume['_seller_lower'].isin(approved_sellers)
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

def check_perfume_tester(data: pd.DataFrame, perfume_category_codes: List[str], perfume_data: Dict) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    sheet_cat_codes = perfume_data.get('category_codes')
    cat_codes = sheet_cat_codes if sheet_cat_codes else set(clean_category_code(c) for c in perfume_category_codes)
    if not cat_codes: return pd.DataFrame(columns=data.columns)
    perfume = data[data['_cat_clean'].isin(cat_codes)].copy()
    if perfume.empty: return pd.DataFrame(columns=data.columns)
    tester_pattern = re.compile(r'\btester\b', re.IGNORECASE)
    flagged = perfume[perfume['_name_lower'].str.contains(tester_pattern, na=False)].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Perfume tester listed for sale: " + flagged['NAME'].astype(str).str[:60]
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_counterfeit_sneakers(data: pd.DataFrame, sneaker_category_codes: List[str], sneaker_sensitive_brands: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME', 'BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    sneakers = data[data['_cat_clean'].isin(set(clean_category_code(c) for c in sneaker_category_codes))].copy()
    if sneakers.empty: return pd.DataFrame(columns=data.columns)
    return sneakers[sneakers['_brand_lower'].isin(['generic', 'fashion']) & sneakers['_name_lower'].apply(lambda x: any(b in x for b in sneaker_sensitive_brands))].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_counterfeit_jerseys(data: pd.DataFrame, jerseys_data: Dict, country_code: str) -> pd.DataFrame:
    if not {"CATEGORY_CODE", "NAME", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    categories = jerseys_data.get("categories", set())
    keywords   = jerseys_data.get("keywords",   {}).get(country_code, set())
    exempted   = jerseys_data.get("exempted",   {}).get(country_code, set())
    if not categories or not keywords: return pd.DataFrame(columns=data.columns)
    kw_pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
    d = data.copy()
    in_scope     = d["_cat_clean"].isin(categories)
    has_keyword  = d["NAME"].astype(str).str.contains(kw_pattern, na=False)
    not_exempted = ~d["_seller_lower"].isin(exempted)
    flagged = d[in_scope & has_keyword & not_exempted].copy()
    if not flagged.empty:
        def build_comment(row):
            match = kw_pattern.search(str(row["NAME"]))
            kw_found = match.group(0) if match else "?"
            return f"Suspected counterfeit jersey — keyword '{kw_found}' (cat: {row['_cat_clean']})"
        flagged["Comment_Detail"] = flagged.apply(build_comment, axis=1)
    return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_unnecessary_words(data: pd.DataFrame, pattern: re.Pattern) -> pd.DataFrame:
    if not {'NAME'}.issubset(data.columns) or pattern is None: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    mask = d['_name_lower'].str.contains(pattern, na=False)
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
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns): 
        return pd.DataFrame(columns=data.columns)
        
    cat_codes = (books_data or {}).get('category_codes') or set(clean_category_code(c) for c in book_category_codes)
    
    d = data.copy()
    
    names = d['NAME'].astype(str).str.strip()
    word_counts = names.str.split().str.len()
    char_counts = names.str.len()
    
    bad_name_mask = (word_counts <= 2) | (char_counts < 15)
    
    if '_cat_clean' in d.columns:
        non_books_mask = ~d['_cat_clean'].isin(cat_codes)
    else:
        non_books_mask = ~d['CATEGORY_CODE'].apply(clean_category_code).isin(cat_codes)
        
    flagged = d[bad_name_mask & non_books_mask].copy()
    
    if not flagged.empty:
        def get_reason(row):
            name_str = str(row['NAME']).strip()
            w_count = len(name_str.split())
            c_count = len(name_str)
            
            if w_count <= 2 and c_count < 15:
                return f"{w_count} words, {c_count} chars"
            elif w_count <= 2:
                return f"{w_count} words"
            else:
                return f"{c_count} chars"
                
        flagged['Comment_Detail'] = flagged.apply(get_reason, axis=1)
        
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_generic_brand_issues(data: pd.DataFrame, valid_category_codes_fas: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    return data[data['_cat_clean'].isin(set(clean_category_code(c) for c in valid_category_codes_fas)) & (data['_brand_lower'] == 'generic')].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_fashion_brand_issues(data: pd.DataFrame, valid_category_codes_fas: List[str], code_to_path: Dict = None) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    if code_to_path is None: code_to_path = {}
    fashion_brand = data[data['_brand_lower'] == 'fashion'].copy()
    if fashion_brand.empty: return pd.DataFrame(columns=data.columns)
    def _in_fashion_domain(cat_code: str) -> bool:
        full_path = code_to_path.get(str(cat_code).strip(), '')
        if full_path: return full_path.strip().lower().startswith('fashion')
        return clean_category_code(cat_code) in fas_codes
    fas_codes = set(clean_category_code(c) for c in valid_category_codes_fas)
    flagged = fashion_brand[~fashion_brand['CATEGORY_CODE'].apply(lambda c: _in_fashion_domain(clean_category_code(c)))].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Brand 'Fashion' used outside Fashion category: " + flagged['CATEGORY_CODE'].astype(str)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_brand_in_name(data: pd.DataFrame) -> pd.DataFrame:
    if not {'BRAND','NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    brands = data['_brand_lower'].values
    names = data['_name_lower'].values
    mask = [b in n if b and b != 'nan' else False for b, n in zip(brands, names)]
    return data[mask].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_wrong_variation(data: pd.DataFrame, allowed_variation_codes: List[str]) -> pd.DataFrame:
    d = data.copy()
    if 'COUNT_VARIATIONS' not in d.columns: d['COUNT_VARIATIONS'] = 1
    if 'CATEGORY_CODE' not in d.columns: return pd.DataFrame(columns=data.columns)
    d['qty_var'] = pd.to_numeric(d['COUNT_VARIATIONS'], errors='coerce').fillna(1).astype(int)
    flagged = d[(d['qty_var'] >= 3) & (~d['_cat_clean'].isin(set(clean_category_code(c) for c in allowed_variation_codes)))].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Variations: " + flagged['qty_var'].astype(str) + ", Category: " + flagged['_cat_clean']
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_generic_with_brand_in_name(data: pd.DataFrame, brands_list: List[str]) -> pd.DataFrame:
    if not {'NAME', 'BRAND'}.issubset(data.columns) or not brands_list: return pd.DataFrame(columns=data.columns)
    _PSEUDO_BRANDS = {'generic', 'fashion', 'unbranded', 'no brand', 'original', 'new'}
    mask = data['_brand_lower'].isin(_PSEUDO_BRANDS)
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
    if not flagged.empty:
        flagged['Comment_Detail'] = (
            "Brand field '" + flagged['_brand_lower'].str.title() +
            "' but name starts with: " + flagged['Detected_Brand']
        )
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

@st.cache_data(show_spinner=False)
def load_valid_colors() -> set:
    valid_set = set()
    try:
        if os.path.exists('colors.txt'):
            with open('colors.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    color = line.strip().lower()
                    if color:
                        valid_set.add(color)
    except Exception as e:
        logger.warning(f"Could not load colors.txt: {e}")
    return valid_set

def check_missing_color(data: pd.DataFrame, pattern: re.Pattern, color_categories: List[str], country_code: str) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns) or pattern is None: 
        return pd.DataFrame(columns=data.columns)
        
    target = data[data['_cat_clean'].isin(set(clean_category_code(c) for c in color_categories))].copy()
    if target.empty: 
        return pd.DataFrame(columns=data.columns)
    
    has_color = 'COLOR' in data.columns
    names = target['NAME'].astype(str).values
    colors = target['COLOR'].astype(str).str.strip().str.lower().values if has_color else [''] * len(target)
    
    valid_colors = load_valid_colors()
    null_like = {'nan', '', 'none', 'null', 'n/a', 'na', '-'}

    # Placeholder/junk values that are NOT real colors — always flag these
    _JUNK_COLORS = {
        'random', 'random color', 'random colour', 'assorted', 'various',
        'as in the picture', 'as in the pictures', 'as the picture',
        'as per image', 'as shown', 'see image', 'see photo',
        'all color available', 'all color availble', 'all colors available',
        'multicolour', 'multicolored', 'multicoloured', 'multi colour',
        'multi color', 'multi-colour', 'multi-color', 'multicolors',
        'mult', 'multic',
    }

    # Color modifier words — "dark brown" → "brown" is the real color token
    # These alone are NOT colors but combined with a base color they are valid
    _MODIFIER_WORDS = {
        'dark', 'light', 'bright', 'deep', 'pale', 'soft', 'matte', 'matt',
        'glossy', 'metallic', 'neon', 'pastel', 'dusty', 'warm', 'cool',
        'royal', 'navy', 'olive', 'mustard', 'burnt', 'forest', 'sky',
        'baby', 'hot', 'ice', 'mint', 'rose', 'coral', 'nude', 'tan',
        'charcoal', 'ash', 'sand', 'cream', 'ivory', 'champagne', 'coffee',
        'chocolate', 'caramel', 'wine', 'burgundy', 'nordic', 'jungle',
        'emerald', 'sapphire', 'ruby', 'amber', 'teal', 'aqua', 'indigo',
        'violet', 'lavender', 'lilac', 'magenta', 'fuchsia', 'maroon',
        'copper', 'bronze', 'gold', 'silver', 'platinum',
        # Descriptive phrases that should NOT count as a base color token
        'dominantly', 'accent', 'accents', 'print', 'stripe', 'striped',
        'check', 'checked', 'pattern', 'bead', 'beaded', 'ring', 'with',
        'and', 'or',
    }

    def _is_valid_color(color_str: str, valid_set: set) -> bool:
        """
        Returns True if color_str represents a real, specific color.

        Strategy:
        1. Reject known junk/placeholder values outright.
        2. Split by all common multi-color separators (comma, slash, ampersand,
           hyphen, pipe, 'and', 'or', 'with').
        3. For each part, check:
           a. Exact match against valid_set (full part).
           b. Word-level token match — any single word in the part that is in
              valid_set and is NOT a pure modifier word.
           This handles "Dark brown" (token "brown"), "nordic blue" (token "blue"),
           "BLACK-RED" (tokens "black", "red"), etc.
        """
        c = color_str.strip().lower()

        # Step 1: reject known junk
        if c in _JUNK_COLORS:
            return False
        # Also catch truncated/symbol-only values
        if re.match(r'^[.\-_*]{1,5}$', c):
            return False

        if not valid_set:
            # No whitelist loaded — accept any non-null, non-junk value
            return True

        # Step 2: split on all separator types
        # Handles: "BLACK-RED", "light grey|dark grey|yellow",
        #          "Black and white", "Black white beige light blue light grey"
        parts = re.split(r'[,/&|\-]|\s+and\s+|\s+or\s+|\s+with\s+', c)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 3a. Exact match on the whole part
            if part in valid_set:
                return True

            # 3b. Word-token match — any token that is a known color (not just modifier)
            tokens = part.split()
            for token in tokens:
                token = token.strip()
                if token in valid_set and token not in _MODIFIER_WORDS:
                    return True

        return False

    mask = []
    for n, c in zip(names, colors):
        # Pass Condition 1: Valid color word in Title
        is_name_valid = bool(pattern.search(n))

        # Pass Condition 2: Valid color in COLOR field
        is_col_valid = False
        if has_color and c not in null_like:
            is_col_valid = _is_valid_color(c, valid_colors)

        if is_col_valid or is_name_valid:
            mask.append(False)
        else:
            mask.append(True)

    flagged = target[mask].copy()

    if not flagged.empty:
        def get_reason(row):
            c_val = str(row.get('COLOR', '')).strip().lower()
            if c_val and c_val not in null_like:
                return f"Invalid color value provided: '{str(row.get('COLOR', '')).strip()}'"
            return "Color missing in both NAME and COLOR attributes"
        flagged['Comment_Detail'] = flagged.apply(get_reason, axis=1)

    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_weight_volume_in_name(data: pd.DataFrame, weight_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns) or not weight_category_codes: return pd.DataFrame(columns=data.columns)
    target = data[data['_cat_clean'].isin(set(clean_category_code(c) for c in weight_category_codes))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    pat = re.compile(
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:kg|kgs|g|gm|gms|grams|mg|mcg|ml|l|ltr|liter|litres|litre|cl|oz|ounces|lb|lbs"
        r"|tablets?|tabs?|capsules?|caps?|sachets?|count|ct|sticks?|iu"
        r"|tea\s*bags?|teabags?|bags?"
        r"|pieces?|pcs|pack|packs"
        r"|dozens?|pairs?|rolls?|sheets?|wipes?|pods?|softgels?|lozenges?|gummies|gummy|units?|serves?|servings?|vegan\s+pieces?)"
        r"|\b\d+[\u0027\u2019]?s\b"
        r"|\b(?:a\s+)?dozen\b"
        r"|\b(?:pack|box|set|bundle|lot)\s+of\s+\d+\b"
        r"|\bper\s+(?:kg|kgs?|g|gm|grams?|mg|mcg|ml|l|ltr|oz|lb)\b"
        r"|\d+\s*(?:\xc2\xb5g|\xce\xbcg|\xb5g|\u00b5g|\u03bcg|mcg|µg|μg)",
        re.IGNORECASE
    )
    return target[~target['_name_lower'].str.contains(pat, na=False)].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_incomplete_smartphone_name(data: pd.DataFrame, smartphone_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns) or not smartphone_category_codes: return pd.DataFrame(columns=data.columns)
    target = data[data['_cat_clean'].isin(set(clean_category_code(c) for c in smartphone_category_codes))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    pat = re.compile(r'\b\d+\s*(gb|tb)\b', re.IGNORECASE)
    flagged = target[~target['_name_lower'].str.contains(pat, na=False)].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Name missing Storage/Memory spec (e.g., 64GB)"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_duplicate_products(data: pd.DataFrame, exempt_categories: List[str] = None, similarity_threshold: float = 0.70, known_colors: List[str] = None, **kwargs) -> pd.DataFrame:
    if not {'NAME', 'SELLER_NAME', 'BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    if exempt_categories and 'CATEGORY_CODE' in d.columns:
        d = d[~d['_cat_clean'].isin(set(clean_category_code(c) for c in exempt_categories))]
    if d.empty: return pd.DataFrame(columns=data.columns)

    d['_norm_name'] = d['NAME'].astype(str).str.lower()
    d['_norm_name'] = d['_norm_name'].str.replace(r'\b(new|sale|original|genuine|authentic|official|premium|quality|best|hot|2024|2025)\b', '', regex=True)
    d['_norm_name'] = d['_norm_name'].str.replace(r'[^\w\s]', '', regex=True)
    d['_norm_name'] = d['_norm_name'].str.replace(r'\s+', '', regex=True)

    _color_pattern = r'\b(' + '|'.join(re.escape(c) for c in sorted(known_colors or [], key=len, reverse=True)) + r')\b' if known_colors else None

    def _extract_color_key(row):
        nl = str(row.get('NAME', '')).lower()
        if _color_pattern and re.search(_color_pattern, nl): return ''
        for col in ('COLOR', 'COLOR_FAMILY'):
            val = str(row.get(col, '')).strip().lower()
            if val and val not in ('nan', 'none', '', 'n/a'): return val
        return ''

    d['_color_key'] = d.apply(_extract_color_key, axis=1)
    d['_dedup_key'] = d['_seller_lower'] + '|' + d['_brand_lower'] + '|' + d['_norm_name'] + '|' + d['_color_key']

    dup_mask = d.duplicated(subset=['_dedup_key'], keep=False)
    if not dup_mask.any(): return pd.DataFrame(columns=data.columns)

    first_occurrence = d.drop_duplicates(subset=['_dedup_key'], keep='first').set_index('_dedup_key')['NAME']
    rdf = d[d.duplicated(subset=['_dedup_key'], keep='first')].copy()
    rdf['Comment_Detail'] = rdf['_dedup_key'].map(lambda k: f"Duplicate: '{str(first_occurrence.get(k, ''))[:40]}'")
    
    base_cols = data.columns.tolist()
    extra_cols = [c for c in ['Comment_Detail'] if c not in base_cols]
    return rdf[base_cols + extra_cols].drop_duplicates(subset=['PRODUCT_SET_SID'])


if _reg is not None:
    _reg.REGISTRY.update({
        'check_restricted_brands':           check_restricted_brands,
        'check_suspected_fake_products':     check_suspected_fake_products,
        'check_refurb_seller_approval':      check_refurb_seller_approval,
        'check_product_warranty':            check_product_warranty,
        'check_seller_approved_for_books':   check_seller_approved_for_books,
        'check_seller_approved_for_perfume': check_seller_approved_for_perfume,
        'check_perfume_tester':              check_perfume_tester,
        'check_counterfeit_sneakers':        check_counterfeit_sneakers,
        'check_counterfeit_jerseys':         check_counterfeit_jerseys,
        'check_prohibited_products':         check_prohibited_products,
        'check_unnecessary_words':           check_unnecessary_words,
        'check_single_word_name':            check_single_word_name,
        'check_generic_brand_issues':        check_generic_brand_issues,
        'check_fashion_brand_issues':        check_fashion_brand_issues,
        'check_brand_in_name':               check_brand_in_name,
        'check_wrong_variation':             check_wrong_variation,
        'check_generic_with_brand_in_name':  check_generic_with_brand_in_name,
        'check_missing_color':               check_missing_color,
        'check_weight_volume_in_name':       check_weight_volume_in_name,
        'check_incomplete_smartphone_name':  check_incomplete_smartphone_name,
        'check_duplicate_products':          check_duplicate_products,
        'check_poor_images_aspect_ratio':    check_poor_images_aspect_ratio,
        'check_miscellaneous_category':      check_miscellaneous_category,
        'check_wrong_price':                 check_wrong_price,
        'check_category_max_price':          check_category_max_price,
        'compile_regex_patterns':            compile_regex_patterns,
        'check_nigeria_gift_card':           check_nigeria_gift_card,
        'check_nigeria_books':               check_nigeria_books,
        'check_nigeria_tvs':                 check_nigeria_tvs,
        'check_nigeria_hp_toners':           check_nigeria_hp_toners,
        'check_nigeria_apple':               check_nigeria_apple,
        'check_nigeria_xmas_tree':           check_nigeria_xmas_tree,
        'check_nigeria_rice':                check_nigeria_rice,
        'check_nigeria_powerbanks':          check_nigeria_powerbanks,
        'load_nigeria_qc_rules':             load_nigeria_qc_rules,
        'check_morocco_prohibited_brands':   check_morocco_prohibited_brands,
        'load_morocco_qc_rules':             load_morocco_qc_rules,
    })

# -------------------------------------------------
# MASTER VALIDATION RUNNER
# -------------------------------------------------
def validate_products(data: pd.DataFrame, support_files: Dict, country_validator: CountryValidator, data_has_warranty_cols: bool, common_sids: Optional[set] = None, skip_validators: Optional[List[str]] = None):
    data['PRODUCT_SET_SID'] = data['PRODUCT_SET_SID'].astype(str).str.strip()
    
    # Pre-calculate optimized lower-cased columns for speed
    data['_name_lower'] = data['NAME'].astype(str).str.lower().fillna('')
    data['_brand_lower'] = data['BRAND'].astype(str).str.lower().str.strip().fillna('')
    data['_seller_lower'] = data['SELLER_NAME'].astype(str).str.lower().str.strip().fillna('')
    data['_cat_clean'] = data['CATEGORY_CODE'].apply(clean_category_code)

    flags_mapping = support_files.get('flags_mapping', {})
    country_restricted_rules = support_files.get('restricted_brands_all', {}).get(country_validator.country, [])
    suspected_fake_df = support_files.get('suspected_fake', {}).get(country_validator.code, pd.DataFrame()) if isinstance(support_files.get('suspected_fake'), dict) else pd.DataFrame()
    country_prohibited_words = support_files.get('prohibited_words_all', {}).get(country_validator.code, [])
    
    validations = [
        ("Wrong Category", check_miscellaneous_category, {
            'categories_list': support_files.get('categories_names_list', []),
            'compiled_rules': st.session_state.get('compiled_json_rules', {}),
            'cat_path_to_code': support_files.get('cat_path_to_code', {}),
            'code_to_path': support_files.get('code_to_path', {}),
        }),
        ("Restricted brands", check_restricted_brands, {'country_rules': country_restricted_rules}),
        ("Suspected Fake product", check_suspected_fake_products, {'suspected_fake_df': suspected_fake_df}),
        ("Seller Not approved to sell Refurb", check_refurb_seller_approval, {'refurb_data': support_files.get('refurb_data', {}), 'country_code': country_validator.code}),
        ("Product Warranty", check_product_warranty, {'warranty_category_codes': support_files.get('warranty_category_codes', [])}),
        ("Seller Approve to sell books", check_seller_approved_for_books, {'books_data': support_files.get('books_data', {}), 'country_code': country_validator.code, 'book_category_codes': support_files.get('book_category_codes', [])}),
        ("Seller Approved to Sell Perfume", check_seller_approved_for_perfume, {'perfume_category_codes': support_files.get('perfume_category_codes', []), 'perfume_data': support_files.get('perfume_data', {}), 'country_code': country_validator.code}),
        ("Perfume Tester", check_perfume_tester, {'perfume_category_codes': support_files.get('perfume_category_codes', []), 'perfume_data': support_files.get('perfume_data', {})}),
        ("Counterfeit Sneakers", check_counterfeit_sneakers, {'sneaker_category_codes': support_files.get('sneaker_category_codes', []), 'sneaker_sensitive_brands': support_files.get('sneaker_sensitive_brands', [])}),
        ("Suspected counterfeit Jerseys", check_counterfeit_jerseys, {'jerseys_data': support_files.get('jerseys_data', {}), 'country_code': country_validator.code}),
        ("Prohibited products", check_prohibited_products, {'prohibited_rules': country_prohibited_words}),
        ("Unnecessary words in NAME", check_unnecessary_words, {'pattern': compile_regex_patterns(support_files.get('unnecessary_words', []))}),
        ("Single-word NAME", check_single_word_name, {'book_category_codes': support_files.get('book_category_codes', []), 'books_data': support_files.get('books_data', {})}),
        ("Generic BRAND Issues", check_generic_brand_issues, {'valid_category_codes_fas': support_files.get('category_fas', [])}),
        ("Fashion brand issues", check_fashion_brand_issues, {'valid_category_codes_fas': support_files.get('category_fas', []), 'code_to_path': support_files.get('code_to_path', {})}),
        ("BRAND name repeated in NAME", check_brand_in_name, {}),
        ("Wrong Variation", check_wrong_variation, {'allowed_variation_codes': list(set(support_files.get('variation_allowed_codes', []) + support_files.get('category_fas', [])))}),
        ("Generic branded products with genuine brands", check_generic_with_brand_in_name, {'brands_list': support_files.get('known_brands', [])}),
        ("Missing COLOR", check_missing_color, {'pattern': compile_regex_patterns(support_files.get('colors', [])), 'color_categories': support_files.get('color_categories', []), 'country_code': country_validator.code}),
        ("Missing Weight/Volume", check_weight_volume_in_name, {'weight_category_codes': support_files.get('weight_category_codes', [])}),
        ("Incomplete Smartphone Name", check_incomplete_smartphone_name, {'smartphone_category_codes': support_files.get('smartphone_category_codes', [])}),
        ("Duplicate product", check_duplicate_products, {'exempt_categories': support_files.get('duplicate_exempt_codes', []), 'known_colors': support_files.get('colors', [])}),
        ("Poor images", check_poor_images_aspect_ratio, {}),
        ("Wrong Price", check_wrong_price, {}),
        ("Category Max Price Exceeded", check_category_max_price, {
            'max_price_map': CATEGORY_MAX_PRICES_USD,
            'code_to_path': support_files.get('code_to_path', {})
        }),
    ]

    if country_validator.code == "NG":
        _ng = support_files.get("ng_qc_rules", {})
        validations += [
            ("NG - Gift Card Seller",  check_nigeria_gift_card,  {"ng_rules": _ng}),
            ("NG - Books Seller",      check_nigeria_books,      {"ng_rules": _ng}),
            ("NG - TV Brand Seller",   check_nigeria_tvs,        {"ng_rules": _ng}),
            ("NG - HP Toners Seller",  check_nigeria_hp_toners,  {"ng_rules": _ng}),
            ("NG - Apple Seller",      check_nigeria_apple,      {"ng_rules": _ng}),
            ("NG - Xmas Tree Seller",  check_nigeria_xmas_tree,  {"ng_rules": _ng}),
            ("NG - Rice Brand Seller", check_nigeria_rice,       {"ng_rules": _ng}),
            ("Powerbank Not Authorized",check_nigeria_powerbanks, {"ng_rules": _ng}),
        ]

    if country_validator.code in ("KE", "UG"):
        _ng = support_files.get("ng_qc_rules", {})
        validations += [
            ("Powerbank Not Authorized", check_nigeria_powerbanks, {"ng_rules": _ng}),
        ]

    if country_validator.code == "MA":
        _ma = load_morocco_qc_rules()

        validations = [v for v in validations if v[0] != "Restricted brands"]
        validations.insert(1, ("Restricted brands", check_restricted_brands, {"country_rules": _ma.get("restricted", [])}))

        ma_prohibited_rules = [{"keyword": kw, "categories": set()} for kw in _ma.get("prohibited_keywords", [])]
        validations = [v for v in validations if v[0] != "Prohibited products"]
        validations.append(("Prohibited products", check_prohibited_products, {"prohibited_rules": ma_prohibited_rules}))

        validations.append(("MA - Marque Interdite", check_morocco_prohibited_brands, {"ma_rules": _ma}))

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
                flag_hash = compute_flag_input_hash(data, name, ckwargs)
                cache_path = os.path.join(FLAG_CACHE_DIR, f"{flag_hash}.pkl")
                future_to_name[executor.submit(run_cached_check, func, cache_path, ckwargs)] = name

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
                    logger.error(f"Validation error in '{name}': {e}")
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

    target_lang = 'fr' if country_validator.country == "Morocco" else 'en'

    rows = []
    processed = set()
    for name, _, _ in validations:
        if name not in results or results[name].empty or 'PRODUCT_SET_SID' not in results[name].columns: continue
        res = results[name]
        rinfo = flags_mapping.get(name, {'reason': "1000007 - Other Reason", 'en': f"Flagged by {name}", 'fr': f"Flagged by {name}", 'ar': f"Flagged by {name}"})
        base_comment = rinfo.get(target_lang, rinfo.get('en'))
        res['PRODUCT_SET_SID'] = res['PRODUCT_SET_SID'].astype(str).str.strip()
        flagged = pd.merge(res[['PRODUCT_SET_SID', 'Comment_Detail']] if 'Comment_Detail' in res.columns else res[['PRODUCT_SET_SID']], data, on='PRODUCT_SET_SID', how='left')
        if 'Comment_Detail' not in flagged.columns and 'Comment_Detail' in res.columns:
            if isinstance(res['Comment_Detail'], pd.DataFrame): flagged['Comment_Detail'] = res['Comment_Detail'].iloc[:, 0]
            else: flagged['Comment_Detail'] = res['Comment_Detail']
        # Merge Reason from result df if the check set it explicitly (e.g. powerbank counterfeit vs wrong-cat)
        if 'Reason' in res.columns:
            reason_map = res.set_index('PRODUCT_SET_SID')['Reason'].to_dict()
        else:
            reason_map = {}

        for _, r in flagged.iterrows():
            sid = str(r['PRODUCT_SET_SID']).strip()
            if sid in processed: continue
            processed.add(sid)
            det = r.get('Comment_Detail', '')
            # Use Comment_Detail directly as the full comment if it looks like a full sentence,
            # otherwise fall back to the standard base_comment + detail pattern
            if pd.notna(det) and det and len(str(det)) > 60:
                comment_str = str(det)
            elif pd.notna(det) and det:
                comment_str = f"{base_comment} ({det})"
            else:
                comment_str = base_comment
            # Honour a Reason override set by the check function itself
            row_reason = reason_map.get(sid, rinfo['reason'])
            rows.append({'ProductSetSid': sid, 'ParentSKU': r.get('PARENTSKU', ''), 'Status': 'Rejected', 'Reason': row_reason, 'Comment': comment_str, 'FLAG': name, 'SellerName': r.get('SELLER_NAME', '')})

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
def cached_validate_products(data_hash: str, _data: pd.DataFrame, _support_files: Dict, country_code: str, data_has_warranty_cols: bool, skip_validators: Optional[List[str]] = None):
    country_name = next((k for k, v in CountryValidator.COUNTRY_CONFIG.items() if v['code'] == country_code), "Kenya")
    cv = CountryValidator(country_name)
    return validate_products(_data, _support_files, cv, data_has_warranty_cols, skip_validators=skip_validators)


# ==========================================
# APP INITIALIZATION & UI
# ==========================================

if 'layout_mode' not in st.session_state: st.session_state.layout_mode = "wide"
if 'ui_lang' not in st.session_state: st.session_state.ui_lang = "en"
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
if 'do_scroll_top' not in st.session_state: st.session_state.do_scroll_top = False
if 'display_df_cache' not in st.session_state: st.session_state.display_df_cache = {}
if 'main_bridge_counter' not in st.session_state: st.session_state.main_bridge_counter = 0
if 'flags_expanded_initialized' not in st.session_state: st.session_state.flags_expanded_initialized = False

try: st.set_page_config(page_title="Product Tool", layout=st.session_state.layout_mode)
except: pass

st_yled.init()

def _t(key):
    return get_translation(st.session_state.ui_lang, key)

rtl_css = """
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {
        direction: rtl !important;
        text-align: right !important;
    }
""" if st.session_state.ui_lang == "ar" else ""

st.markdown(f"""
    <style>
        {rtl_css}
        div[data-testid="stTextInput"]:has(input[placeholder="JTBRIDGE_UNIQUE_DO_NOT_USE"]) {{
            position: absolute !important; width: 1px !important; height: 1px !important;
            padding: 0 !important; margin: -1px !important; overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important; white-space: nowrap !important;
            border: 0 !important; opacity: 0 !important; z-index: -9999 !important;
        }}
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
        div[data-testid="stMetric"] {{
            background: {JUMIA_COLORS['light_gray']}; border-radius: 0 0 8px 8px;
            padding: 12px 16px 16px 16px; text-align: center;
        }}
        div[data-testid="stMetricValue"] {{ color: {JUMIA_COLORS['dark_gray']}; font-weight: 700; font-size: 26px !important; }}
        div[data-testid="stMetricLabel"] {{ color: {JUMIA_COLORS['medium_gray']}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }}
        ::-webkit-scrollbar {{ width: 18px !important; height: 18px !important; }}
        ::-webkit-scrollbar-track {{ background: {JUMIA_COLORS['light_gray']}; border-radius: 8px; }}
        ::-webkit-scrollbar-thumb {{ background: {JUMIA_COLORS['medium_gray']}; border-radius: 8px; border: 3px solid {JUMIA_COLORS['light_gray']}; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {JUMIA_COLORS['primary_orange']}; }}
        * {{ scrollbar-width: auto; scrollbar-color: {JUMIA_COLORS['medium_gray']} {JUMIA_COLORS['light_gray']}; }}
        div[data-testid="stExpander"] {{ border: 1px solid {JUMIA_COLORS['border_gray']}; border-radius: 8px; }}
        div[data-testid="stExpander"] summary {{ background-color: {JUMIA_COLORS['light_gray']}; padding: 12px; border-radius: 8px 8px 0 0; }}
        h1, h2, h3 {{ color: {JUMIA_COLORS['dark_gray']} !important; }}
        div[data-baseweb="segmented-control"] button {{ border-radius: 4px; }}
        div[data-baseweb="segmented-control"] button[aria-pressed="true"] {{ background-color: {JUMIA_COLORS['primary_orange']} !important; color: white !important; }}
    </style>
""", unsafe_allow_html=True)

try:
    support_files = load_support_files_lazy()
    st.session_state.support_files = support_files
    st.session_state['compiled_json_rules'] = support_files.get('compiled_json_rules', {})
except Exception as e:
    st.error(f"Failed to load configs: {e}")
    st.stop()

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
        if isinstance(msg, tuple): st.toast(msg[0], icon=msg[1])
        else: st.toast(msg)
    st.session_state.main_toasts.clear()

def get_image_base64(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as img_file: return base64.b64encode(img_file.read()).decode('utf-8')
        except: pass
    return ""

logo_base64 = get_image_base64("jumia logo.png") or get_image_base64("jumia_logo.png")
logo_html = f"<img src='data:image/png;base64,{logo_base64}' style='height: 42px; margin-right: 15px;'>" if logo_base64 else "<span class='material-symbols-outlined' style='font-size: 42px; margin-right: 15px;'>verified_user</span>"

st.markdown(f"""<div style='background: linear-gradient(135deg, {JUMIA_COLORS['primary_orange']}, {JUMIA_COLORS['secondary_orange']}); padding: 25px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(246, 139, 30, 0.3);'><h1 style='color: white; margin: 0; font-size: 36px; display: flex; align-items: center;'>{logo_html}Product Validation Tool</h1></div>""", unsafe_allow_html=True)

with st.sidebar:
    lang_names = list(LANGUAGES.keys())
    current_lang_code = st.session_state.get('ui_lang', 'en')
    current_lang_name = next((k for k, v in LANGUAGES.items() if v == current_lang_code), "English")
    selected_lang_name = st.selectbox("Language / Langue / اللغة", lang_names, index=lang_names.index(current_lang_name))
    new_lang_code = LANGUAGES[selected_lang_name]
    if new_lang_code != current_lang_code:
        st.session_state.ui_lang = new_lang_code
        st.rerun()
    st.markdown("---")
    st.header(_t("system_status"))
    if st.button(_t("clear_cache"), use_container_width=True, type="secondary"):
        st.cache_data.clear()
        st.session_state.display_df_cache = {}
        if os.path.exists(PARQUET_CACHE_DIR): shutil.rmtree(PARQUET_CACHE_DIR)
        if os.path.exists(FLAG_CACHE_DIR): shutil.rmtree(FLAG_CACHE_DIR)
        st.rerun()
    st.markdown("---")
    st.header(_t("display_settings"))
    new_mode = "wide" if "Wide" in st.radio("Layout Mode", ["Centered", "Wide"], index=1 if st.session_state.layout_mode == "wide" else 0) else "centered"
    if new_mode != st.session_state.layout_mode: st.session_state.layout_mode = new_mode; st.rerun()

# ==========================================
# SECTION 1: UPLOAD & VALIDATION
# ==========================================
st.header(f":material/upload_file: {_t('upload_files')}", anchor=False)

current_country = st.session_state.get('selected_country', get_default_country())
country_choice = st.segmented_control("Country", ["Kenya", "Uganda", "Nigeria", "Ghana", "Morocco"], default=current_country, key="country_selector")

if country_choice and country_choice != current_country:
    st.session_state.selected_country = country_choice
    st.session_state.last_processed_files = None
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.exports_cache = {}
    st.session_state.display_df_cache = {}
    st.session_state.flags_expanded_initialized = False
    st.session_state.ui_lang = "fr" if country_choice == "Morocco" else "en"
    st.toast(f"Switching to {country_choice}…", icon=":material/public:")

country_validator = CountryValidator(st.session_state.selected_country)

uploaded_files = st.file_uploader(
    "Upload files",                                # non-empty label
    type=['csv', 'xlsx'],
    accept_multiple_files=True,
    key="daily_files",
    label_visibility="collapsed"             # hides it visually, same look as before
)

if uploaded_files:
    st.session_state.cached_uploaded_files = [{"name": uf.name, "bytes": uf.read()} for uf in uploaded_files]
elif uploaded_files is not None and len(uploaded_files) == 0:
    st.session_state.cached_uploaded_files = []
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.file_mode = None
    st.session_state.exports_cache = {}
    st.session_state.display_df_cache = {}
    st.session_state.last_processed_files = "empty"

_files_for_processing = st.session_state.get("cached_uploaded_files", [])
process_signature = str(sorted([f["name"] + hashlib.md5(f["bytes"]).hexdigest() for f in _files_for_processing])) + f"_{country_validator.code}" if _files_for_processing else "empty"

if st.session_state.get('last_processed_files') != process_signature:
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.file_mode = None
    st.session_state.intersection_sids = set()
    st.session_state.intersection_count = 0
    st.session_state.grid_page = 0
    st.session_state.exports_cache = {}
    st.session_state.display_df_cache = {}
    st.session_state.flags_expanded_initialized = False
    st.session_state.pop("_grid_review_data_cache", None)
    st.session_state.pop("_grid_warm_urls", None)
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith(("quick_rej_", "grid_chk_", "toast_"))]
    for k in keys_to_delete: del st.session_state[k]

    if process_signature == "empty":
        st.session_state.last_processed_files = "empty"
    else:
        _engine_for_cache = _get_cat_matcher_engine() if _CAT_MATCHER_AVAILABLE else None
        _learning_stamp   = str(len(_engine_for_cache.learning_db)) if _engine_for_cache else "0"
        sig_hash = hashlib.md5((process_signature + _learning_stamp).encode()).hexdigest()
        
        cached_data = load_df_parquet(f"{sig_hash}_data.parquet")
        cached_report = load_df_parquet(f"{sig_hash}_report.parquet")

        if cached_data is not None and cached_report is not None:
            st.session_state.final_report = cached_report
            st.session_state.all_data_map = cached_data
            st.session_state.last_processed_files = process_signature
            st.toast("Loaded from cache", icon=":material/bolt:")
        else:
            try:
                all_dfs = []
                file_sids_sets = []
                detected_modes = []
                for uf in _files_for_processing:
                    _buf = BytesIO(uf["bytes"])
                    if uf["name"].endswith('.xlsx'): raw_data = pd.read_excel(_buf, engine='openpyxl', dtype=str)
                    else: raw_data = _detect_and_read_csv(_buf)
                    raw_data = _repair_mojibake(raw_data)
                    detected_modes.append(detect_file_type(raw_data) if 'detect_file_type' in globals() else 'pre_qc')
                    all_dfs.append(raw_data)

                file_mode = detected_modes[0] if detected_modes else 'pre_qc'
                st.session_state.file_mode = file_mode

                if file_mode == 'post_qc':
                    st.info("Post-QC file detected. Please use the Post-QC page.", icon=":material/fact_check:")
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
                    st.session_state.intersection_sids = set.intersection(*file_sids_sets) if len(file_sids_sets) > 1 else set()
                    st.session_state.intersection_count = len(st.session_state.intersection_sids)
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

                        save_df_parquet(data, f"{sig_hash}_data.parquet")
                        save_df_parquet(final_report, f"{sig_hash}_report.parquet")

                        # ── Pre-warm the visual review grid data while user reads results ──
                        # Build and cache review_data + first-page image URLs now so the
                        # modal opens instantly instead of computing on first click.
                        try:
                            from constants import GRID_COLS
                            _fr = final_report
                            _committed_sids = set()
                            _valid_sids = _fr[_fr["Status"] == "Approved"]["ProductSetSid"].tolist()
                            if "MAIN_IMAGE" not in data.columns:
                                data["MAIN_IMAGE"] = ""
                            _available_cols = [c for c in GRID_COLS if c in data.columns]
                            if "CATEGORY_CODE" in data.columns and "CATEGORY_CODE" not in _available_cols:
                                _available_cols.append("CATEGORY_CODE")
                            _valid_df = _fr[_fr["Status"] == "Approved"][["ProductSetSid"]]
                            _review_data = pd.merge(
                                _valid_df, data[_available_cols],
                                left_on="ProductSetSid", right_on="PRODUCT_SET_SID", how="left",
                            )
                            _code_to_path = support_files.get("code_to_path", {})
                            if _code_to_path and "CATEGORY_CODE" in _review_data.columns:
                                _review_data = _review_data.copy()
                                _review_data["CATEGORY"] = _review_data["CATEGORY_CODE"].apply(
                                    lambda c: _code_to_path.get(str(c).strip(), str(c)) if pd.notna(c) else ""
                                )
                            st.session_state["_grid_review_data_cache"] = _review_data
                            # Pre-fetch URLs for first 2 pages at default 50 ipp
                            _ipp = 50
                            _warm_urls = set()
                            for _url in _review_data.iloc[:_ipp * 2]["MAIN_IMAGE"].astype(str):
                                _url = _url.strip().replace("http://", "https://", 1)
                                if _url.startswith("https"):
                                    _warm_urls.add(_url)
                            st.session_state["_grid_warm_urls"] = list(_warm_urls)
                        except Exception as _pw_err:
                            logger.warning("Grid pre-warm failed: %s", _pw_err)
                    else:
                        for e in errors: st.error(e)
                        st.session_state.last_processed_files = "error"
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.code(traceback.format_exc())
                st.session_state.last_processed_files = "error"

def restore_single_item(sid):
    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status', 'Reason', 'Comment', 'FLAG']] = ['Approved', '', '', 'Approved by User']
    st.session_state.pop(f"quick_rej_{sid}", None)
    st.session_state.pop(f"quick_rej_reason_{sid}", None)
    st.session_state.exports_cache.clear()
    st.session_state.display_df_cache.clear()
    st.session_state.main_toasts.append("Reverted selections.")

# -------------------------------------------------
# JTBRIDGE (HTML GRID MESSAGE HANDLER)
# -------------------------------------------------
_bridge_val = st.text_input(
    "jtbridge", value="",
    placeholder="JTBRIDGE_UNIQUE_DO_NOT_USE",
    key=f"main_bridge_{st.session_state.main_bridge_counter}",
    label_visibility="collapsed",
)

if _bridge_val:
    try:
        _msg = json.loads(_bridge_val)
        if _msg.get("action") == "reject":
            _payload = _msg.get("payload", {})
            if isinstance(_payload, dict) and _payload:
                _rgroups = {}
                for _sid, _rkey in _payload.items(): _rgroups.setdefault(_rkey, []).append(_sid)
                _total = 0
                for _rkey, _sids in _rgroups.items():
                    # ── Handle Custom Comments from the frontend ──
                    if _rkey.startswith("Other Reason (Custom): "):
                        _flag = "Other Reason (Custom)"
                        _code = "1000007 - Other Reason"
                        _cmt = _rkey.split(": ", 1)[1] # Extract the comment they typed
                    else:
                        _flag = REASON_MAP.get(_rkey, "Other Reason (Custom)")
                        _rinfo = support_files["flags_mapping"].get(_flag, {'reason': "1000007 - Other Reason", 'en': "Manual rejection"})
                        _code = _rinfo['reason']
                        _cmt_lang = 'fr' if st.session_state.selected_country == "Morocco" else 'en'
                        _cmt = _rinfo.get(_cmt_lang, _rinfo.get('en'))
                    
                    st.session_state.final_report.loc[
                        st.session_state.final_report["ProductSetSid"].isin(_sids),
                        ["Status", "Reason", "Comment", "FLAG"]
                    ] = ["Rejected", _code, _cmt, _flag]
                    
                    for _s in _sids:
                        st.session_state[f"quick_rej_{_s}"] = True
                        st.session_state[f"quick_rej_reason_{_s}"] = _flag
                    _total += len(_sids)
                    
                st.session_state.exports_cache.clear()
                st.session_state.display_df_cache.clear()
                st.session_state.main_toasts.append(f"Rejected {_total} product(s)")
                st.session_state.main_bridge_counter += 1
                st.session_state.do_scroll_top = False
                st.rerun()

        elif _msg.get("action") == "undo":
            _payload = _msg.get("payload", {})
            _total_restored = 0
            if isinstance(_payload, dict):
                for _sid in _payload.keys():
                    restore_single_item(_sid)
                    _total_restored += 1
            if _total_restored > 0:
                st.session_state.main_bridge_counter += 1
                st.session_state.do_scroll_top = False
                st.rerun()

    except Exception as _e:
        logger.error(f"Bridge parse error: {_e}")

# ==========================================
# RESULTS SECTION
# ==========================================
if _files_for_processing and not st.session_state.final_report.empty and st.session_state.file_mode != 'post_qc':
    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    app_df = fr[fr['Status'] == 'Approved']
    rej_df = fr[fr['Status'] == 'Rejected']

    st.header(f":material/bar_chart: {_t('val_results')}", anchor=False)

    with st.container(border=True):
        cols = st.columns(5 if st.session_state.layout_mode == "wide" else 3)
        is_nigeria = st.session_state.get('selected_country') == 'Nigeria'
        multi_count = int(data['_IS_MULTI_COUNTRY'].sum()) if '_IS_MULTI_COUNTRY' in data.columns else 0

        metrics_config = [
            (_t("total_prod"),  len(data), JUMIA_COLORS['dark_gray']),
            (_t("approved"),    len(app_df), JUMIA_COLORS['success_green']),
            (_t("rejected"),    len(rej_df), JUMIA_COLORS['jumia_red']),
            (_t("rej_rate"),    f"{(len(rej_df)/len(data)*100) if len(data)>0 else 0:.1f}%", JUMIA_COLORS['primary_orange']),
            (_t("multi_skus") if is_nigeria else _t("common_skus"), multi_count if is_nigeria else st.session_state.intersection_count, JUMIA_COLORS['warning_yellow'] if is_nigeria else JUMIA_COLORS['medium_gray']),
        ]
        for i, (label, value, color) in enumerate(metrics_config):
            with cols[i % len(cols)]:
                st.markdown(f"<div style='height:5px;background:{color};border-radius:6px 6px 0 0;'></div>", unsafe_allow_html=True)
                st.metric(label=label, value=value)

    st.subheader(f":material/flag: {_t('flags_breakdown')}", anchor=False)
    if not rej_df.empty:
        if not st.session_state.flags_expanded_initialized and not rej_df.empty:
            top_flag = rej_df['FLAG'].value_counts().index[0]
            st.session_state[f"exp_{top_flag}"] = True
            st.session_state.flags_expanded_initialized = True

        for title in rej_df['FLAG'].unique():
            df_flagged = rej_df[rej_df['FLAG'] == title]
            with st.expander(f"{title} ({len(df_flagged)})", key=f"exp_{title}"):
                render_flag_expander(title, df_flagged, data, all(c in data.columns for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION']), support_files, country_validator, cached_validate_products)
    else:
        st.success("All products passed validation — no rejections found.")


    # ==========================================
    # CALL EXTERNAL RENDERERS
    # ==========================================
    render_image_grid(support_files)
    render_exports_section(support_files, country_validator)
