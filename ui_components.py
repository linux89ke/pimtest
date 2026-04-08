"""
ui_components.py - UI rendering components with Browser-Side Image Debugging
"""

import re
import json
import logging
import concurrent.futures
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import base64
from PIL import Image

from constants import JUMIA_COLORS, GRID_COLS
from data_utils import clean_category_code, df_hash, format_local_price
from export_utils import generate_smart_export, prepare_full_data_merged

logger = logging.getLogger(__name__)

# Securely encoded Base64 placeholder
_SVG_RAW = "<svg xmlns='http://www.w3.org/2000/svg' width='150' height='150'><rect width='150' height='150' fill='#f0f0f0'/><text x='75' y='75' text-anchor='middle' dominant-baseline='central' font-size='12' font-family='sans-serif' fill='#999'>No Image</text></svg>"
_NO_IMAGE_SVG = f"data:image/svg+xml;base64,{base64.b64encode(_SVG_RAW.encode('utf-8')).decode('utf-8')}"

def _t(key):
    from translations import get_translation
    return get_translation(st.session_state.get('ui_lang', 'en'), key)

def _clear_flag_df_selection(title: str):
    if f"df_{title}" in st.session_state:
        del st.session_state[f"df_{title}"]

@st.dialog("Confirm Bulk Approval")
def bulk_approve_dialog(sids_to_process, title, subset_data, data_has_warranty_cols_check,
                        support_files, country_validator, validation_runner):
    try:
        from category_matcher_engine import get_engine
        _CAT_MATCHER_AVAILABLE = True
    except ImportError:
        _CAT_MATCHER_AVAILABLE = False

    st.warning(f"You are about to approve **{len(sids_to_process)}** items from `{title}`.")
    if st.button(_t("approve_btn"), type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            data_hash = df_hash(subset_data) + country_validator.code + "_skip_" + title
            new_report, _ = validation_runner(
                data_hash, subset_data, support_files,
                country_validator.code, data_has_warranty_cols_check,
                skip_validators=[title]
            )
            msg_moved, msg_approved = {}, 0
            for sid in sids_to_process:
                new_row = new_report[new_report['ProductSetSid'] == sid]
                if new_row.empty or not str(new_row.iloc[0]['FLAG']):
                    st.session_state.final_report.loc[
                        st.session_state.final_report['ProductSetSid'] == sid,
                        ['Status', 'Reason', 'Comment', 'FLAG']
                    ] = ['Approved', '', '', 'Approved by User']
                    msg_approved += 1
                else:
                    new_flag = str(new_row.iloc[0]['FLAG'])
                    st.session_state.final_report.loc[
                        st.session_state.final_report['ProductSetSid'] == sid,
                        ['Status', 'Reason', 'Comment', 'FLAG']
                    ] = ['Rejected', new_row.iloc[0]['Reason'], new_row.iloc[0]['Comment'], new_flag]
                    msg_moved[new_flag] = msg_moved.get(new_flag, 0) + 1
            st.session_state.exports_cache.clear()
            st.session_state.display_df_cache.clear()
            st.session_state[f"exp_{title}"] = True
            _clear_flag_df_selection(title)
        st.rerun()

def render_flag_expander(title, df_flagged_sids, data, data_has_warranty_cols_check,
                         support_files, country_validator, validation_runner):
    cache_key = f"display_df_{title}"
    if cache_key not in st.session_state.display_df_cache:
        # Standard merging logic...
        df_display = pd.merge(df_flagged_sids[['ProductSetSid']], data, left_on='ProductSetSid', right_on='PRODUCT_SET_SID', how='left')
        st.session_state.display_df_cache[cache_key] = df_display
    else:
        df_display = st.session_state.display_df_cache[cache_key]

    # Render dataframe and buttons...
    # (Existing render_flag_expander logic here)

def build_fast_grid_html(page_data, flags_mapping, country, rejected_state, cols_per_row):
    O = JUMIA_COLORS["primary_orange"]
    G = JUMIA_COLORS["success_green"]
    R = JUMIA_COLORS["jumia_red"]
    
    html_dir = "rtl" if st.session_state.get('ui_lang') == "ar" else "ltr"
    
    labels_dict = {
        "undo": _t("undo"), "clear_sel": _t("clear_sel"),
        "items_pending": _t("items_pending"), "poor_img": _t("poor_img"),
        "more_options": _t("more_options"), "wrong_cat": _t("wrong_cat"),
        "fake_prod": _t("fake_prod"), "restr_brand": _t("restr_brand"),
        "prohibited": _t("prohibited"), "missing_color": _t("missing_color"),
        "wrong_brand": _t("wrong_brand"), "rejected": str(_t('rejected') or 'REJECTED').upper()
    }

    cards_data = []
    for _, row in page_data.iterrows():
        sid = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip().replace("http://", "https://")
        cards_data.append({
            "sid": sid, "img": img_url,
            "name": str(row.get("NAME", "")), "brand": str(row.get("BRAND", "")),
            "cat": str(row.get("CATEGORY", "")), "seller": str(row.get("SELLER_NAME", "")),
        })

    return f"""<!DOCTYPE html>
<html dir="{html_dir}">
<head>
<meta charset="utf-8">
<meta name="referrer" content="no-referrer">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;font-family:sans-serif;}}
  body{{background:#f5f5f5;padding:8px;}}
  .ctrl-bar{{position:-webkit-sticky;position:sticky;top:0;z-index:99999;display:flex;align-items:center;gap:8px;padding:8px 12px;background:rgba(255,255,255,0.95);border-bottom:2px solid {O};}}
  .grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}
  .card{{border:2px solid #e0e0e0;border-radius:8px;padding:10px;background:#fff;position:relative;}}
  .card-img-wrap{{position:relative;height:180px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;overflow:hidden;}}
  .card-img{{width:100%;height:100%;object-fit:contain;}}
  .debug-hud{{position:absolute;inset:0;background:rgba(0,0,0,0.85);color:#0f0;font-family:monospace;font-size:9px;padding:5px;display:none;z-index:100;word-break:break-all;}}
  .warn-badge{{background:{R};color:#fff;font-size:10px;padding:2px 5px;border-radius:4px;margin:2px;display:inline-block;}}
</style>
</head>
<body>
<div class="ctrl-bar">
  <span id="sel-count-bar">0 items</span>
</div>
<div class="grid" id="card-grid"></div>

<script>
var LABELS = {json.dumps(labels_dict)};
var CARDS = {json.dumps(cards_data)};
var COMMITTED = {json.dumps(rejected_state)};
var NO_IMAGE = "{_NO_IMAGE_SVG}";

function onImgLoad(img, sid) {{
    img.closest('.card-img-wrap').style.background = 'white';
}}

function onImgError(img, sid) {{
    img.onerror = null;
    var hud = document.getElementById('debug-' + sid);
    if(hud) {{
        hud.style.display = 'block';
        hud.innerHTML = "<b>IMAGE FAILED:</b><br>" + img.src;
    }}
    img.src = NO_IMAGE;
}}

function renderCard(card) {{
    var sid = card.sid;
    var safeImg = card.img || NO_IMAGE;
    return `<div class="card" id="card-${{sid}}">
        <div class="card-img-wrap">
            <div class="debug-hud" id="debug-${{sid}}"></div>
            <div class="warn-wrap" style="position:absolute;top:0;left:0;z-index:10;"></div>
            <img class="card-img" src="${{safeImg}}" 
                 referrerpolicy="no-referrer" 
                 crossorigin="anonymous"
                 onload="onImgLoad(this,'${{sid}}')" 
                 onerror="onImgError(this,'${{sid}}')">
        </div>
        <div style="font-size:11px;margin-top:5px;"><b>${{card.brand}}</b> - ${{card.name.substring(0,30)}}...</div>
    </div>`;
}}

document.getElementById('card-grid').innerHTML = CARDS.map(renderCard).join('');
</script>
</body></html>"""

@st.fragment
def render_image_grid(support_files):
    if st.session_state.final_report.empty: return
    
    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    
    # Filter only approved items for review
    review_data = pd.merge(fr[fr["Status"]=="Approved"][["ProductSetSid"]], data, left_on="ProductSetSid", right_on="PRODUCT_SET_SID")
    
    page_data = review_data.head(50) # Show first 50
    rejected_state = {{}} # In a real app, populate from session state
    
    grid_html = build_fast_grid_html(page_data, support_files.get("flags_mapping",{{}}), st.session_state.get('selected_country','Kenya'), rejected_state, 4)
    components.html(grid_html, height=800, scrolling=True)
