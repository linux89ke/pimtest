import os
import re
import logging
import pandas as pd
import streamlit as st
from typing import Dict

logger = logging.getLogger(__name__)

def _clean_category_code(code) -> str:
    try:
        if pd.isna(code): return ""
        s = str(code).strip()
        if '.' in s: s = s.split('.')[0]
        return s
    except: return str(code).strip()

def _safe_excel_read(filename: str, sheet_name, usecols=None) -> pd.DataFrame:
    if not os.path.exists(filename): return pd.DataFrame()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols=usecols, engine='openpyxl', dtype=str)
        return df.dropna(how='all')
    except Exception as e:
        logger.error(f"safe_excel_read: tab='{sheet_name}' file={filename}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_nigeria_qc_rules() -> Dict:
    FILE_NAME = "Nigeria_QC_Rules.xlsx"
    result: Dict = {
        "gift_card":  {"sellers": set(), "category_codes": set()},
        "books":      {},   
        "tvs":        {"category_codes": set(), "brand_sellers": {}},
        "hp_toners":  {"sellers": set(), "category_codes": set()},
        "apple":      {"sellers": set()},
        "xmas_tree":  {"sellers": set(), "keywords": set()},
        "rice":       {},   
        "powerbanks": {"brands": set(), "category_codes": set()},
    }

    if not os.path.exists(FILE_NAME):
        logger.warning("Nigeria_QC_Rules.xlsx not found — NG-specific checks will be skipped.")
        return result

    try:
        df = _safe_excel_read(FILE_NAME, sheet_name="Gift card")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            seller_col, cat_col = df.columns[0], df.columns[1]
            result["gift_card"]["sellers"] = (
                set(df[seller_col].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "seller"}
            )
            result["gift_card"]["category_codes"] = (
                set(df[cat_col].dropna().astype(str).apply(_clean_category_code)) - {"", "nan"}
            )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules gift_card: {e}")

    try:
        df = _safe_excel_read(FILE_NAME, sheet_name="Books")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            name_col, seller_col = df.columns[0], df.columns[1]
            for _, row in df.iterrows():
                book = str(row[name_col]).strip().lower()
                if not book or book == "nan": continue
                raw_seller = str(row.get(seller_col, "")).strip().lower()
                result["books"][book] = None if (not raw_seller or raw_seller == "nan") else raw_seller
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules books: {e}")

    try:
        df = _safe_excel_read(FILE_NAME, sheet_name="TVs")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            cat_col    = df.columns[0]
            brand_cols = df.columns[1:]
            result["tvs"]["category_codes"] = (
                set(df[cat_col].dropna().astype(str).apply(_clean_category_code)) - {"", "nan"}
            )
            for bc in brand_cols:
                brand_lower = bc.strip().lower()
                result["tvs"]["brand_sellers"][brand_lower] = (
                    set(df[bc].dropna().astype(str).str.strip().str.lower()) - {"", "nan", brand_lower}
                )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules tvs: {e}")

    try:
        df = _safe_excel_read(FILE_NAME, sheet_name="HP ink toners")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            seller_col = df.columns[0]
            cat_col    = df.columns[1] if len(df.columns) > 1 else None
            result["hp_toners"]["sellers"] = (
                set(df[seller_col].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "seller"}
            )
            if cat_col:
                result["hp_toners"]["category_codes"] = (
                    set(df[cat_col].dropna().astype(str).apply(_clean_category_code)) - {"", "nan"}
                )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules hp_toners: {e}")

    try:
        df = _safe_excel_read(FILE_NAME, sheet_name="Apple")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            seller_col = df.columns[0]
            result["apple"]["sellers"] = (
                set(df[seller_col].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "seller"}
            )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules apple: {e}")

    try:
        df = _safe_excel_read(FILE_NAME, sheet_name="Xmas Tree")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            seller_col = df.columns[0]
            kw_col     = df.columns[1] if len(df.columns) > 1 else None
            result["xmas_tree"]["sellers"] = (
                set(df[seller_col].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "seller"}
            )
            if kw_col:
                result["xmas_tree"]["keywords"] = (
                    set(df[kw_col].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "keyword", "keywords"}
                )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules xmas_tree: {e}")

    try:
        df = _safe_excel_read(FILE_NAME, sheet_name="Rice")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            brand_col   = df.columns[0]
            sellers_col = df.columns[1]
            cat_col     = df.columns[2] if len(df.columns) > 2 else None
            for _, row in df.iterrows():
                brand = str(row[brand_col]).strip().lower()
                if not brand or brand == "nan": continue
                raw_sellers = str(row.get(sellers_col, "")).strip()
                sellers = set()
                if raw_sellers and raw_sellers.lower() != "nan":
                    sellers = {s.strip().lower() for s in raw_sellers.split(",") if s.strip()}
                cat_code = ""
                if cat_col:
                    raw_cat = str(row.get(cat_col, "")).strip()
                    if raw_cat and raw_cat.lower() != "nan":
                        cat_code = _clean_category_code(raw_cat)
                if brand not in result["rice"]:
                    result["rice"][brand] = {"sellers": set(), "category_codes": set()}
                result["rice"][brand]["sellers"].update(sellers)
                if cat_code:
                    result["rice"][brand]["category_codes"].add(cat_code)
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules rice: {e}")

    try:
        df = _safe_excel_read(FILE_NAME, sheet_name="20,000mah Powerbanks")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            brand_col = df.columns[0]
            cat_col   = df.columns[1] if len(df.columns) > 1 else None
            result["powerbanks"]["brands"] = (
                set(df[brand_col].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "brand"}
            )
            if cat_col:
                result["powerbanks"]["category_codes"] = (
                    set(df[cat_col].dropna().astype(str).apply(_clean_category_code)) - {"", "nan"}
                )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules powerbanks: {e}")

    return result

def check_nigeria_gift_card(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    rules            = ng_rules.get("gift_card", {})
    cat_codes        = rules.get("category_codes", set())
    approved_sellers = rules.get("sellers", set())
    if not cat_codes or not approved_sellers: return pd.DataFrame(columns=data.columns)
    if not {"CATEGORY_CODE", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    in_scope = data[data["CATEGORY_CODE"].apply(_clean_category_code).isin(cat_codes)].copy()
    if in_scope.empty: return pd.DataFrame(columns=data.columns)
    flagged = in_scope[~in_scope["SELLER_NAME"].astype(str).str.strip().str.lower().isin(approved_sellers)].copy()
    if not flagged.empty: flagged["Comment_Detail"] = "Seller not authorised for Gift Card categories: " + flagged["SELLER_NAME"].astype(str)
    return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_books(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    book_rules = ng_rules.get("books", {})
    if not book_rules or not {"NAME", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d["_name_l"]   = d["NAME"].astype(str).str.strip().str.lower()
    d["_seller_l"] = d["SELLER_NAME"].astype(str).str.strip().str.lower()
    flagged_idx, comment_map = [], {}
    for idx, row in d.iterrows():
        name_l, seller_l = row["_name_l"], row["_seller_l"]
        for book_name, allowed_seller in book_rules.items():
            if book_name in name_l:
                if allowed_seller is None:
                    flagged_idx.append(idx)
                    comment_map[idx] = f"No seller authorised for book: '{book_name[:60]}'"
                elif seller_l != allowed_seller:
                    flagged_idx.append(idx)
                    comment_map[idx] = f"Only '{allowed_seller}' may sell '{book_name[:40]}'"
                break
    if not flagged_idx: return pd.DataFrame(columns=data.columns)
    result = data.loc[flagged_idx].copy()
    result["Comment_Detail"] = result.index.map(comment_map)
    return result.drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_tvs(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    tv_rules      = ng_rules.get("tvs", {})
    cat_codes     = tv_rules.get("category_codes", set())
    brand_sellers = tv_rules.get("brand_sellers", {})
    if not cat_codes or not brand_sellers or not {"CATEGORY_CODE", "BRAND", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    in_scope = data[data["CATEGORY_CODE"].apply(_clean_category_code).isin(cat_codes)].copy()
    if in_scope.empty: return pd.DataFrame(columns=data.columns)
    in_scope["_brand_l"]  = in_scope["BRAND"].astype(str).str.strip().str.lower()
    in_scope["_seller_l"] = in_scope["SELLER_NAME"].astype(str).str.strip().str.lower()
    chunks = []
    for brand_lower, approved in brand_sellers.items():
        if not approved: continue
        brand_rows = in_scope[in_scope["_brand_l"] == brand_lower]
        if brand_rows.empty: continue
        bad = brand_rows[~brand_rows["_seller_l"].isin(approved)].copy()
        if not bad.empty:
            bad["Comment_Detail"] = f"Seller not authorised for {brand_lower.upper()} TVs: " + bad["SELLER_NAME"].astype(str)
            chunks.append(bad)
    if not chunks: return pd.DataFrame(columns=data.columns)
    return pd.concat(chunks).drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_hp_toners(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    rules            = ng_rules.get("hp_toners", {})
    cat_codes        = rules.get("category_codes", set())
    approved_sellers = rules.get("sellers", set())
    if not cat_codes or not approved_sellers or not {"CATEGORY_CODE", "BRAND", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d["_cat"]      = d["CATEGORY_CODE"].apply(_clean_category_code)
    d["_brand_l"]  = d["BRAND"].astype(str).str.strip().str.lower()
    d["_seller_l"] = d["SELLER_NAME"].astype(str).str.strip().str.lower()
    in_scope = d[d["_cat"].isin(cat_codes) & (d["_brand_l"] == "hp")].copy()
    if in_scope.empty: return pd.DataFrame(columns=data.columns)
    flagged = in_scope[~in_scope["_seller_l"].isin(approved_sellers)].copy()
    if not flagged.empty: flagged["Comment_Detail"] = "Seller not authorised for HP Ink/Toners: " + flagged["SELLER_NAME"].astype(str)
    return flagged[[c for c in data.columns if c in flagged.columns] + ["Comment_Detail"]].drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_apple(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    approved_sellers = ng_rules.get("apple", {}).get("sellers", set())
    if not approved_sellers or not {"BRAND", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    apple = data[data["BRAND"].astype(str).str.strip().str.lower() == "apple"].copy()
    if apple.empty: return pd.DataFrame(columns=data.columns)
    flagged = apple[~apple["SELLER_NAME"].astype(str).str.strip().str.lower().isin(approved_sellers)].copy()
    if not flagged.empty: flagged["Comment_Detail"] = "Seller not authorised to sell Apple products: " + flagged["SELLER_NAME"].astype(str)
    return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_xmas_tree(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    rules            = ng_rules.get("xmas_tree", {})
    approved_sellers = rules.get("sellers", set())
    keywords         = rules.get("keywords", set())
    if not approved_sellers or not keywords or not {"NAME", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    kw_pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
    in_scope = data[data["NAME"].astype(str).str.contains(kw_pattern, na=False)].copy()
    if in_scope.empty: return pd.DataFrame(columns=data.columns)
    flagged = in_scope[~in_scope["SELLER_NAME"].astype(str).str.strip().str.lower().isin(approved_sellers)].copy()
    if not flagged.empty:
        def _comment(row):
            m = kw_pattern.search(str(row["NAME"]))
            kw = m.group(0) if m else "?"
            return f"Seller not authorised for Xmas Tree products (keyword '{kw}'): {row['SELLER_NAME']}"
        flagged["Comment_Detail"] = flagged.apply(_comment, axis=1)
    return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_rice(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    rice_rules = ng_rules.get("rice", {})
    if not rice_rules or not {"BRAND", "SELLER_NAME", "CATEGORY_CODE"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d["_brand_l"]  = d["BRAND"].astype(str).str.strip().str.lower()
    d["_seller_l"] = d["SELLER_NAME"].astype(str).str.strip().str.lower()
    d["_cat"]      = d["CATEGORY_CODE"].apply(_clean_category_code)
    chunks = []
    for brand_lower, rules in rice_rules.items():
        approved  = rules.get("sellers", set())
        cat_codes = rules.get("category_codes", set())
        brand_rows = d[d["_brand_l"] == brand_lower].copy()
        if brand_rows.empty: continue
        if cat_codes: brand_rows = brand_rows[brand_rows["_cat"].isin(cat_codes)]
        if brand_rows.empty: continue
        bad = brand_rows[~brand_rows["_seller_l"].isin(approved)].copy()
        if not bad.empty:
            bad["Comment_Detail"] = f"Seller not authorised to sell {brand_lower.title()} rice: " + bad["SELLER_NAME"].astype(str)
            chunks.append(bad)
    if not chunks: return pd.DataFrame(columns=data.columns)
    return pd.concat(chunks).drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_powerbanks(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    pb_rules       = ng_rules.get("powerbanks", {})
    allowed_brands = pb_rules.get("brands", set())
    cat_codes      = pb_rules.get("category_codes", set())
    MIN_MAH        = 20_000

    _COUNTERFEIT_REASON  = "1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)"
    _COUNTERFEIT_COMMENT = (
        "Brand '{brand}' not approved for {mah_str} powerbanks."
        "Your listing has been rejected as Jumia's technical team has confirmed the product is counterfeit."
        "As a result, this item cannot be sold on the platform."
        "Please ensure that all products listed are 100% authentic to comply with Jumia's policies and protect customer trust."
        "If you believe this decision is incorrect or need further clarification, please contact the Seller Support team"
    )
    _WRONG_CAT_REASON  = "1000007 - Wrong Category"
    _WRONG_CAT_COMMENT = (
        "Product name contains 'power bank' / 'powerbank' but is not listed under the correct Powerbank category. "
        "Please relist under the appropriate category."
    )

    _pb_name_pat = re.compile(r'\bpower\s*bank\b', re.IGNORECASE)
    _mah_pat     = re.compile(r'\b(\d[\d,]*)\s*mah\b', re.IGNORECASE)

    if not {"CATEGORY_CODE", "NAME", "BRAND"}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    def _exceeds_threshold(name: str) -> bool:
        for m in _mah_pat.finditer(str(name)):
            try:
                val = int(m.group(1).replace(',', ''))
                if val >= MIN_MAH: return True
            except ValueError: pass
        return False

    d = data.copy()
    d["_cat"]     = d["CATEGORY_CODE"].apply(_clean_category_code)
    d["_name"]    = d["NAME"].astype(str)
    d["_brand_l"] = d["BRAND"].astype(str).str.strip().str.lower()

    chunks = []

    # -- Wrong-category check ------------------------------------------------
    # Products whose NAME mentions "power bank"/"powerbank" but whose
    # category code is NOT in the approved powerbank category codes.
    if cat_codes:
        pb_name_rows = d[d["_name"].str.contains(_pb_name_pat, na=False)].copy()
        wrong_cat = pb_name_rows[~pb_name_rows["_cat"].isin(cat_codes)].copy()
        if not wrong_cat.empty:
            wrong_cat["FLAG"]           = "Powerbank Not Authorized"
            wrong_cat["Reason"]         = _WRONG_CAT_REASON
            wrong_cat["Comment_Detail"] = _WRONG_CAT_COMMENT
            chunks.append(wrong_cat)

    # -- Unapproved-brand check ----------------------------------------------
    # High-capacity powerbanks (>=20,000 mAh in name) in the correct category
    # whose brand is not on the approved list.
    if allowed_brands:
        in_scope = d[d["_cat"].isin(cat_codes)].copy() if cat_codes else d.copy()
        if not in_scope.empty:
            high_cap = in_scope[in_scope["_name"].apply(_exceeds_threshold)].copy()
            if not high_cap.empty:
                flagged = high_cap[~high_cap["_brand_l"].isin(allowed_brands)].copy()
                if not flagged.empty:
                    def _comment(row):
                        m = _mah_pat.search(row["_name"])
                        mah_str = m.group(0) if m else ">=20,000mAh"
                        return _COUNTERFEIT_COMMENT.format(brand=row["BRAND"], mah_str=mah_str)
                    flagged["FLAG"]           = "Powerbank Not Authorized"
                    flagged["Reason"]         = _COUNTERFEIT_REASON
                    flagged["Comment_Detail"] = flagged.apply(_comment, axis=1)
                    chunks.append(flagged)

    if not chunks:
        return pd.DataFrame(columns=data.columns)

    result = pd.concat(chunks)
    keep_cols = [c for c in data.columns if c in result.columns] + ["FLAG", "Reason", "Comment_Detail"]
    return result[keep_cols].drop_duplicates(subset=["PRODUCT_SET_SID"])
