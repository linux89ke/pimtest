import os
import re
import logging
import pandas as pd
import streamlit as st
from typing import Dict

logger = logging.getLogger(__name__)

def _safe_excel_read(filename: str, sheet_name, usecols=None) -> pd.DataFrame:
    if not os.path.exists(filename): return pd.DataFrame()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols=usecols, engine='openpyxl', dtype=str)
        return df.dropna(how='all')
    except Exception as e:
        logger.error(f"safe_excel_read: tab='{sheet_name}' file={filename}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_morocco_qc_rules() -> Dict:
    FILE_NAME = "Morocco_rules.xlsx"
    result: Dict = {
        "restricted":          [],
        "prohibited_brands":   set(),
        "prohibited_keywords": [],
    }

    if not os.path.exists(FILE_NAME):
        logger.warning("Morocco_rules.xlsx not found — MA-specific checks skipped.")
        return result

    # ── brands sheet ──────────────────────────────────────────────────────────
    try:
        df = _safe_excel_read(FILE_NAME, sheet_name="brands")
        if not df.empty:
            df.columns = [str(c).strip().lower() for c in df.columns]
            brand_col  = next((c for c in df.columns if "brand"       in c), df.columns[0])
            vendor_col = next((c for c in df.columns if "vendor"      in c
                                                      or "seller"     in c
                                                      or "authorized" in c), df.columns[1])

            brand_dict: dict = {}
            for _, row in df.iterrows():
                brand = str(row.get(brand_col, "")).strip()
                if not brand or brand.lower() in ("nan", "restricted brand", "brand"):
                    continue
                b_lower = brand.lower()
                if b_lower not in brand_dict:
                    brand_dict[b_lower] = {"brand_raw": brand, "sellers": set()}
                vendor = str(row.get(vendor_col, "")).strip()
                if vendor and vendor.lower() not in ("nan", "none", "authorized vendors", ""):
                    brand_dict[b_lower]["sellers"].add(vendor.strip().lower())

            for b_lower, data in brand_dict.items():
                if data["sellers"]:
                    result["restricted"].append({
                        "brand":      b_lower,
                        "brand_raw":  data["brand_raw"],
                        "sellers":    data["sellers"],
                        "categories": set(),          
                        "variations": [],
                        "has_blank_category": True,   
                    })
                else:
                    result["prohibited_brands"].add(b_lower)
    except Exception as e:
        logger.warning(f"load_morocco_qc_rules brands: {e}")

    # ── keywords sheet ────────────────────────────────────────────────────────
    try:
        df_kw = _safe_excel_read(FILE_NAME, sheet_name="keywords")
        if not df_kw.empty:
            df_kw.columns = [str(c).strip().lower() for c in df_kw.columns]
            kw_col = df_kw.columns[0]
            result["prohibited_keywords"] = [
                str(v).strip().lower()
                for v in df_kw[kw_col].dropna()
                if str(v).strip().lower() not in ("", "nan", "keyword", "keywords")
            ]
    except Exception as e:
        logger.warning(f"load_morocco_qc_rules keywords: {e}")

    return result


def check_morocco_prohibited_brands(data: pd.DataFrame, ma_rules: Dict) -> pd.DataFrame:
    prohibited = ma_rules.get("prohibited_brands", set())
    if not prohibited or not {"BRAND", "NAME"}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    d = data.copy()
    d["_brand_l"] = d["BRAND"].astype(str).str.strip().str.lower()
    d["_name_l"]  = d["NAME"].astype(str).str.lower()

    sorted_brands = sorted(prohibited, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(b) for b in sorted_brands) + r")(?!\w)",
        re.IGNORECASE,
    )

    brand_match = d["_brand_l"].isin(prohibited)
    name_match  = (~brand_match) & d["_name_l"].str.contains(pattern, na=False)
    flagged     = d[brand_match | name_match].copy()

    if not flagged.empty:
        def _comment(row):
            if row["_brand_l"] in prohibited:
                return f"Marque interdite : {row['BRAND']}"
            m = pattern.search(row["_name_l"])
            kw = m.group(0).title() if m else row["BRAND"]
            return f"Marque interdite détectée dans le nom : {kw}"
        flagged["Comment_Detail"] = flagged.apply(_comment, axis=1)

    return (
        flagged
        .drop(columns=["_brand_l", "_name_l"], errors="ignore")
        .drop_duplicates(subset=["PRODUCT_SET_SID"])
    )
