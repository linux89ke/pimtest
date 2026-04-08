"""
loaders.py - All file loading functions for support/config data
"""

import os
import re
import json
import logging
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional

from data_utils import clean_category_code

logger = logging.getLogger(__name__)

COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
COUNTRY_NAME_TO_TAB = {"Kenya": "KE", "Uganda": "UG", "Nigeria": "NG", "Ghana": "GH", "Morocco": "MA"}


# -------------------------------------------------
# LOW-LEVEL FILE READERS
# -------------------------------------------------

def load_txt_file(filename: str) -> List[str]:
    try:
        if not os.path.exists(os.path.abspath(filename)):
            return []
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.warning(f"load_txt_file({filename}): {e}")
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
    except Exception as e:
        logger.warning(f"load_excel_file({filename}, col={column}): {e}")
        return [] if column else pd.DataFrame()


def safe_excel_read(filename: str, sheet_name, usecols=None) -> pd.DataFrame:
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols=usecols, engine='openpyxl', dtype=str)
        return df.dropna(how='all')
    except Exception as e:
        logger.error(f"safe_excel_read: tab='{sheet_name}' file={filename}: {e}")
        return pd.DataFrame()


# -------------------------------------------------
# SUPPORT DATA LOADERS
# -------------------------------------------------

@st.cache_data(ttl=3600)
def load_prohibited_from_local() -> Dict[str, List[Dict]]:
    FILE_NAME = "Prohibbited.xlsx"
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
                if not keyword or keyword in ('nan', 'keywords'):
                    continue
                categories = set()
                if category_col:
                    cats_raw = str(row.get(category_col, '')).strip()
                    if cats_raw and cats_raw.lower() != 'nan':
                        split_cats = re.split(r'[,\n]+', cats_raw)
                        categories.update([clean_category_code(c.strip()) for c in split_cats if c.strip()])
                country_rules.append({'keyword': keyword, 'categories': categories})
            prohibited_by_country[tab] = country_rules
        except Exception as e:
            logger.warning(f"load_prohibited_from_local tab={tab}: {e}")
            prohibited_by_country[tab] = []
    return prohibited_by_country


@st.cache_data(ttl=3600)
def load_restricted_brands_from_local() -> Dict[str, List[Dict]]:
    FILE_NAME = "Restricted_Brands.xlsx"
    config_by_country = {}
    for country_name, tab_name in COUNTRY_NAME_TO_TAB.items():
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab_name)
            if df.empty:
                config_by_country[country_name] = []
                continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            brand_dict = {}
            for _, row in df.iterrows():
                brand = str(row.get('brand', '')).strip()
                if not brand or brand.lower() == 'nan':
                    continue
                b_lower = brand.lower()
                if b_lower not in brand_dict:
                    brand_dict[b_lower] = {
                        'brand_raw': brand, 'sellers': set(),
                        'categories': set(), 'variations': set(),
                        'has_blank_category': False
                    }
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
                country_rules.append({
                    'brand': b_lower, 'brand_raw': data['brand_raw'],
                    'sellers': data['sellers'], 'categories': data['categories'],
                    'variations': list(data['variations'])
                })
            config_by_country[country_name] = country_rules
        except Exception as e:
            logger.warning(f"load_restricted_brands tab={tab_name}: {e}")
            config_by_country[country_name] = []
    return config_by_country


@st.cache_data(ttl=3600)
def load_refurb_data_from_local() -> dict:
    FILE_NAME = "Refurb.xlsx"
    result = {"sellers": {}, "categories": {"Phones": set(), "Laptops": set()}, "keywords": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab, usecols=[0, 1])
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                phones_set = set(df.iloc[:, 0].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "phones", "phone"}
                laptops_set = set(df.iloc[:, 1].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "laptops", "laptop"}
                result["sellers"][tab] = {"Phones": phones_set, "Laptops": laptops_set}
        except Exception as e:
            logger.warning(f"load_refurb_data tab={tab}: {e}")
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
        logger.warning(f"load_refurb_data categories: {e}")
    try:
        df_names = safe_excel_read(FILE_NAME, sheet_name="Name", usecols=[0])
        if not df_names.empty:
            first_col = df_names.columns[0]
            result["keywords"] = {k for k in df_names[first_col].dropna().astype(str).str.strip().str.lower() if k and k not in ("name", "keyword", "keywords", "words", "nan")}
    except Exception as e:
        logger.warning(f"load_refurb_data keywords: {e}")
        result["keywords"] = {"refurb", "refurbished", "renewed"}
    return result


@st.cache_data(ttl=3600)
def load_perfume_data_from_local() -> Dict:
    FILE_NAME = "Perfume.xlsx"
    result = {"sellers": {}, "keywords": set(), "category_codes": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                seller_col = next((c for c in df.columns if 'seller' in c.lower()), df.columns[0])
                result["sellers"][tab] = set(
                    df[seller_col].dropna().astype(str).str.strip().str.lower()
                    .pipe(lambda s: s[~s.isin(["", "nan", "sellername", "seller name", "seller"])])
                )
        except Exception as e:
            logger.warning(f"load_perfume_data tab={tab}: {e}")
            result["sellers"][tab] = set()
    try:
        df_kw = safe_excel_read(FILE_NAME, sheet_name="Keywords")
        if not df_kw.empty:
            df_kw.columns = [str(c).strip() for c in df_kw.columns]
            kw_col = next((c for c in df_kw.columns if 'brand' in c.lower() or 'keyword' in c.lower()), df_kw.columns[0])
            result["keywords"] = set(
                df_kw[kw_col].dropna().astype(str).str.strip().str.lower()
                .pipe(lambda s: s[~s.isin(["", "nan", "brand", "keyword", "keywords"])])
            )
    except Exception as e:
        logger.warning(f"load_perfume_data keywords: {e}")
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if 'cat' in c.lower()), df_cats.columns[0])
            result["category_codes"] = set(
                df_cats[cat_col].dropna().astype(str).apply(clean_category_code)
                .pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])])
            )
    except Exception as e:
        logger.warning(f"load_perfume_data categories: {e}")
    return result


@st.cache_data(ttl=3600)
def load_books_data_from_local() -> Dict:
    FILE_NAME = "Books_sellers.xlsx"
    result = {"sellers": {}, "category_codes": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                seller_col = next((c for c in df.columns if 'seller' in c.lower()), df.columns[0])
                result["sellers"][tab] = set(
                    df[seller_col].dropna().astype(str).str.strip().str.lower()
                    .pipe(lambda s: s[~s.isin(["", "nan", "sellername", "seller name", "seller"])])
                )
        except Exception as e:
            logger.warning(f"load_books_data tab={tab}: {e}")
            result["sellers"][tab] = set()
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if 'cat' in c.lower()), df_cats.columns[0])
            result["category_codes"] = set(
                df_cats[cat_col].dropna().astype(str).apply(clean_category_code)
                .pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])])
            )
    except Exception as e:
        logger.warning(f"load_books_data categories: {e}")
    return result


@st.cache_data(ttl=3600)
def load_jerseys_from_local() -> Dict:
    FILE_NAME = "Jersey_validation.xlsx"
    result: Dict = {
        "keywords": {tab: set() for tab in COUNTRY_TABS},
        "exempted": {tab: set() for tab in COUNTRY_TABS},
        "categories": set()
    }
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                kw_col = next((c for c in df.columns if "keyword" in c.lower()), df.columns[0])
                result["keywords"][tab] = set(
                    df[kw_col].dropna().astype(str).str.strip().str.lower()
                    .pipe(lambda s: s[~s.isin(["", "nan", "keywords", "keyword"])])
                )
                ex_col = next((c for c in df.columns if "exempt" in c.lower() or "seller" in c.lower()), None)
                if ex_col:
                    result["exempted"][tab] = set(
                        df[ex_col].dropna().astype(str).str.strip().str.lower()
                        .pipe(lambda s: s[~s.isin(["", "nan", "exempted sellers", "seller"])])
                    )
        except Exception as e:
            logger.warning(f"load_jerseys tab={tab}: {e}")
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip().lower() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if "cat" in c), df_cats.columns[0])
            result["categories"] = set(
                df_cats[cat_col].dropna().astype(str).apply(clean_category_code)
                .pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])])
            )
    except Exception as e:
        logger.warning(f"load_jerseys categories: {e}")
    return result


@st.cache_data(ttl=3600)
def load_suspected_fake_from_local() -> Dict:
    if not os.path.exists('suspected_fake.xlsx'):
        logger.warning("suspected_fake.xlsx not found")
        return {code: pd.DataFrame() for code in COUNTRY_TABS}
    result = {}
    for code in COUNTRY_TABS:
        try:
            result[code] = pd.read_excel('suspected_fake.xlsx', sheet_name=code, engine='openpyxl', dtype=str)
        except Exception as e:
            logger.warning(f"load_suspected_fake country={code}: {e}")
            result[code] = pd.DataFrame()
    return result


@st.cache_data(ttl=3600)
def load_flags_mapping(filename="reason.xlsx") -> Dict[str, dict]:
    raw_default = {
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
        'Wrong Variation': ('1000039 - Product Poorly Created. Each Variation Of This Product Should Be Created Uniquely (Not Authorized)', "Create different SKUs instead of variations (variations only for sizes)."),
        'Missing Weight/Volume': ('1000008 - Kindly Improve Product Name Description', "Include weight or volume (e.g., '1kg', '500ml')."),
        'Incomplete Smartphone Name': ('1000008 - Kindly Improve Product Name Description', "Include memory/storage details (e.g., '128GB')."),
        'Wrong Category': ('1000004 - Wrong Category', "Assigned to Wrong Category. Please use correct category."),
        'Poor images': ('1000042 - Kindly follow our product image upload guideline.', "Poor Image Quality"),
        'Perfume Tester': ('1000007 - Other Reason', "Sale of perfume testers is not permitted on Jumia."),
        'NG - Gift Card Seller':  ('1000003 - Restricted Brand', "Seller not authorised to sell Gift Cards in these categories."),
        'NG - Books Seller':      ('1000003 - Restricted Brand', "Seller not authorised to sell this book title, or no seller is approved for it."),
        'NG - TV Brand Seller':   ('1000003 - Restricted Brand', "Seller not authorised to sell this TV brand."),
        'NG - HP Toners Seller':  ('1000003 - Restricted Brand', "Seller not authorised to sell HP Ink/Toners in these categories."),
        'NG - Apple Seller':      ('1000003 - Restricted Brand', "Seller not authorised to sell Apple products."),
        'NG - Xmas Tree Seller':  ('1000003 - Restricted Brand', "Seller not authorised to sell Christmas Tree products."),
        'NG - Rice Brand Seller': ('1000003 - Restricted Brand', "Seller not authorised to sell this rice brand."),
        'NG - Powerbank Capacity':('1000007 - Other Reason', "Only approved brands may list powerbanks with 20,000mAh or above capacity."),
    }

    default_mapping = {
        k: {'reason': v[0], 'en': v[1], 'fr': v[1], 'ar': v[1]}
        for k, v in raw_default.items()
    }

    # Pricing flags
    pricing_reason_code = "1000031 - Kindly Review & Update This Product's Price or Confirm The Price Is Correct By Raising A Claim"
    pricing_en = (
        "The current price of your product differs significantly from the market average.\n"
        "Please review and update the price accordingly, or if you believe the current price is correct, raise a claim with supporting justification.\n\n"
        "Also, keep in mind:\n"
        "- Promotional periods must not exceed 90 days.\n"
        "- Misleading promotions are strictly prohibited.\n"
        "- The original (pre-discount) price must be accurate and should not be inflated before applying a discount."
    )
    pricing_fr = (
        "Le prix actuel de votre produit diffère fortement de la moyenne du marché.\n"
        "Veuillez le revoir et le mettre à jour en conséquence. Si vous estimez que le prix est justifié, vous pouvez soumettre une réclamation accompagnée de preuves.\n\n"
        "À noter également :\n"
        "- Les périodes promotionnelles ne doivent pas dépasser 90 jours.\n"
        "- Les promotions trompeuses sont strictement interdites.\n"
        "- Le prix d'origine (avant remise) doit être exact et ne doit pas être artificiellement gonflé avant l'application de la réduction."
    )
    pricing_ar = (
        "سعر المنتج الحالي يختلف بشكل ملحوظ عن متوسط السوق.\n"
        "يرجى مراجعة السعر وتحديثه، أو في حال كنت ترى أن السعر صحيح، يمكنك تقديم طلب مراجعة (Claim) مع تقديم ما يثبت ذلك."
    )
    for flag_key in ('Wrong Price', 'Category Max Price Exceeded'):
        default_mapping[flag_key] = {'reason': pricing_reason_code, 'en': pricing_en, 'fr': pricing_fr, 'ar': pricing_ar}

    # Morocco flags
    default_mapping['MA - Marque Interdite'] = {
        'reason': '1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim',
        'en': 'Please contact Jumia Seller Support and raise a claim to confirm whether this product is eligible for listing.',
        'fr': (
            "Veuillez contacter le Support Vendeur de Jumia et soumettre une réclamation "
            "afin de confirmer si ce produit est éligible à la mise en ligne."
        ),
        'ar': 'يرجى التواصل مع فريق دعم بائعين جوميا وتقديم طلب مراجعة (Claim) للتأكد من إمكانية عرض هذا المنتج على المنصة.',
    }
    default_mapping['MA - Produit Interdit'] = {
        'reason': '1000033 - Keywords in your content/ Product name / description has been blacklisted',
        'en': 'Your product name or description includes unauthorized or blacklisted keywords.',
        'fr': (
            "Le nom ou la description de votre produit contient des mots-clés non autorisés ou interdits.\n"
            "Veuillez relire attentivement le contenu et supprimer ou remplacer tout mot-clé interdit."
        ),
        'ar': 'اسم المنتج أو وصفه يحتوي على كلمات غير مصرح بها. يرجى مراجعة المحتوى بعناية.',
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
                    comment_en = str(row['comment']).strip()
                    comment_fr = str(row.get('french', comment_en)).strip()
                    comment_ar = str(row.get('arabic', comment_en)).strip()
                    if comment_fr.lower() == 'nan' or not comment_fr:
                        comment_fr = comment_en
                    if comment_ar.lower() == 'nan' or not comment_ar:
                        comment_ar = comment_en
                    if flag and flag.lower() != 'nan':
                        custom_mapping[flag] = {'reason': reason, 'en': comment_en, 'fr': comment_fr, 'ar': comment_ar}
                if custom_mapping:
                    ng_keys = {k: v for k, v in default_mapping.items() if k.startswith('NG - ')}
                    return {**custom_mapping, **ng_keys}
    except Exception as e:
        logger.warning(f"load_flags_mapping({filename}): {e}")

    return default_mapping


@st.cache_data(ttl=3600)
def load_all_support_files() -> Dict:
    """Load all support/config files into a single dictionary."""
    from nigeria_rules import load_nigeria_qc_rules

    def safe_txt(f):
        return load_txt_file(f) if os.path.exists(f) else []

    support = {
        'blacklisted_words':        safe_txt('blacklisted.txt'),
        'book_category_codes':      safe_txt('Books_cat.txt'),
        'books_data':               load_books_data_from_local(),
        'perfume_category_codes':   safe_txt('Perfume_cat.txt'),
        'perfume_data':             load_perfume_data_from_local(),
        'sneaker_category_codes':   safe_txt('Sneakers_Cat.txt'),
        'sneaker_sensitive_brands': [b.lower() for b in safe_txt('Sneakers_Sensitive.txt')],
        'sensitive_words':          [w.lower() for w in safe_txt('sensitive_words.txt')],
        'unnecessary_words':        [w.lower() for w in safe_txt('unnecessary.txt')],
        'colors':                   [c.lower() for c in safe_txt('colors.txt')],
        'color_categories':         safe_txt('color_cats.txt'),
        'category_fas':             safe_txt('Fashion_cat.txt'),
        'reasons':                  load_excel_file('reasons.xlsx'),
        'flags_mapping':            load_flags_mapping(),
        'jerseys_data':             load_jerseys_from_local(),
        'warranty_category_codes':  safe_txt('warranty.txt'),
        'suspected_fake':           load_suspected_fake_from_local(),
        'duplicate_exempt_codes':   safe_txt('duplicate_exempt.txt'),
        'restricted_brands_all':    load_restricted_brands_from_local(),
        'prohibited_words_all':     load_prohibited_from_local(),
        'known_brands':             safe_txt('brands.txt'),
        'variation_allowed_codes':  safe_txt('variation.txt'),
        'weight_category_codes':    safe_txt('weight.txt'),
        'smartphone_category_codes':safe_txt('smartphones.txt'),
        'refurb_data':              load_refurb_data_from_local(),
        'ng_qc_rules':              load_nigeria_qc_rules(),
    }

    # Category map
    _cat_names, _cat_path_to_code, _code_to_path = [], {}, {}
    _cm_path = "category_map.xlsx"
    try:
        if os.path.exists(_cm_path):
            _cm_df = pd.read_excel(_cm_path, engine="openpyxl", dtype=str)
            _cm_df.columns = [c.strip() for c in _cm_df.columns]
            _path_col = next((c for c in _cm_df.columns if c.lower() == "category path"), None) or \
                        next((c for c in _cm_df.columns if "path" in c.lower()), None)
            _code_col = next((c for c in _cm_df.columns if "code" in c.lower()), None)
            if _path_col:
                _valid = _cm_df[_path_col].dropna().astype(str)
                _valid = _valid[_valid.str.strip().ne("")]
                _cat_names = _valid.tolist()
                if _code_col:
                    for _, _row in _cm_df[[_path_col, _code_col]].dropna().iterrows():
                        _p = str(_row[_path_col]).strip()
                        _c = str(_row[_code_col]).strip().split(".")[0]
                        if _p and _c:
                            _cat_path_to_code[_p.lower()] = _c
                            _code_to_path[_c] = _p
        else:
            logger.warning(f"[CategoryMap] {_cm_path} not found.")
    except Exception as _ce:
        logger.error(f"[CategoryMap] Failed to load {_cm_path}: {_ce}")

    support['categories_names_list'] = _cat_names
    support['cat_path_to_code'] = _cat_path_to_code
    support['code_to_path'] = _code_to_path
    support['category_map'] = {}

    # JSON weighted rules
    support['compiled_json_rules'] = load_and_compile_json_rules()
    return support


@st.cache_resource(ttl=3600)
def load_and_compile_json_rules(json_path="category_qc_weighted.json") -> dict:
    import re as _re
    if not os.path.exists(json_path):
        logger.warning(f"{json_path} not found.")
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_rules = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load JSON rules: {e}")
        return {}

    if isinstance(raw_rules, list):
        fixed = {}
        for item in raw_rules:
            if isinstance(item, dict):
                cat = item.get("category") or item.get("Category Path") or item.get("name") or item.get("category_name")
                kws = item.get("keywords") or item.get("weights") or item.get("positive")
                if cat and isinstance(kws, dict):
                    fixed[cat] = kws
        raw_rules = fixed

    if not isinstance(raw_rules, dict):
        logger.warning("JSON rules file has unrecognizable format.")
        return {}

    compiled_rules = {}
    for cat_path, keywords_dict in raw_rules.items():
        if not isinstance(keywords_dict, dict) or not keywords_dict:
            continue
        try:
            safe_kws = {str(k): float(w) for k, w in keywords_dict.items()}
            sorted_kws = sorted(safe_kws.keys(), key=len, reverse=True)
            if not sorted_kws:
                continue
            pattern_str = r'\b(' + '|'.join(_re.escape(k) for k in sorted_kws) + r')\b'
            compiled_rules[str(cat_path)] = {
                'pattern': _re.compile(pattern_str, _re.IGNORECASE),
                'weights': {k.lower(): w for k, w in safe_kws.items()}
            }
        except Exception as e:
            logger.warning(f"Skipping bad JSON rule for {cat_path}: {e}")
    return compiled_rules
