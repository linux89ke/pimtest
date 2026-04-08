"""
ui_components.py - UI rendering components with Browser-Side Image Debugging
"""

import re
import json
import logging
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import base64

from constants import JUMIA_COLORS, GRID_COLS
from data_utils import clean_category_code, df_hash, format_local_price
from export_utils import generate_smart_export, prepare_full_data_merged

logger = logging.getLogger(__name__)

# Securely encoded Base64 placeholder for broken images - prevents HTML injection errors
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
    st.warning(f"You are about to approve **{len(sids_to_process)}** items from `{title}`.")
    if st.button(_t("approve_btn"), type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            data_hash = df_hash(subset_data) + country_validator.code + "_skip_" + title
            new_report, _ = validation_runner(
                data_hash, subset_data, support_files,
                country_validator.code, data_has_warranty_cols_check,
                skip_validators=[title]
            )
            for sid in sids_to_process:
                new_row = new_report[new_report['ProductSetSid'] == sid]
                if new_row.empty or not str(new_row.iloc[0]['FLAG']):
                    st.session_state.final_report.loc[
                        st.session_state.final_report['ProductSetSid'] == sid,
                        ['Status', 'Reason', 'Comment', 'FLAG']
                    ] = ['Approved', '', '', 'Approved by User']
                else:
                    st.session_state.final_report.loc[
                        st.session_state.final_report['ProductSetSid'] == sid,
                        ['Status', 'Reason', 'Comment', 'FLAG']
                    ] = ['Rejected', new_row.iloc[0]['Reason'], new_row.iloc[0]['Comment'], str(new_row.iloc[0]['FLAG'])]
            
            st.session_state.exports_cache.clear()
            st.session_state.display_df_cache.clear()
            _clear_flag_df_selection(title)
        st.rerun()

def render_flag_expander(title, df_flagged_sids, data, data_has_warranty_cols_check,
                         support_files, country_validator, validation_runner):
    """
    Function #1: Renders the grouped validation rejections.
    """
    cache_key = f"display_df_{title}"
    if cache_key not in st.session_state.display_df_cache:
        df_display = pd.merge(df_flagged_sids[['ProductSetSid']], data, left_on='ProductSetSid', right_on='PRODUCT_SET_SID', how='left')
        st.session_state.display_df_cache[cache_key] = df_display
    else:
        df_display = st.session_state.display_df_cache[cache_key]

    st.write(f"Displaying {len(df_display)} items for {title}")
    
    # Selection and processing logic
    if st.button(f"Bulk Approve {title}", key=f"bulk_{title}"):
        sids = df_display['PRODUCT_SET_SID'].tolist()
        bulk_approve_dialog(sids, title, data[data['PRODUCT_SET_SID'].isin(sids)], data_has_warranty_cols_check, support_files, country_validator, validation_runner)

def build_fast_grid_html(page_data, cols_per_row):
    """
    Constructs the HTML/JS grid with Referrer Policy and Debug HUD.
    """
    cards_data = []
    for _, row in page_data.iterrows():
        sid = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip().replace("http://", "https://")
        cards_data.append({
            "sid": sid, "img": img_url,
            "name": str(row.get("NAME", "")), "brand": str(row.get("BRAND", "")),
        })

    return f"""
    <html>
    <head>
        <meta name="referrer" content="no-referrer">
        <style>
            .grid {{ display: grid; grid-template-columns: repeat({cols_per_row}, 1fr); gap: 10px; font-family: sans-serif; }}
            .card {{ border: 1px solid #ddd; padding: 10px; border-radius: 8px; position: relative; background: white; }}
            .img-wrap {{ height: 150px; display: flex; align-items: center; justify-content: center; background: #f9f9f9; overflow: hidden; position: relative; }}
            img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
            .debug-hud {{ position: absolute; inset: 0; background: rgba(0,0,0,0.85); color: #0f0; font-family: monospace; font-size: 9px; padding: 5px; display: none; word-break: break-all; z-index: 100; }}
        </style>
    </head>
    <body>
        <div class="grid">
            {"".join([f'''
            <div class="card">
                <div class="img-wrap">
                    <div id="debug-{c["sid"]}" class="debug-hud"></div>
                    <img src="{c["img"]}" 
                         referrerpolicy="no-referrer" 
                         crossorigin="anonymous"
                         onerror="this.onerror=null; this.src='{_NO_IMAGE_SVG}'; 
                                  var h=document.getElementById('debug-{c["sid"]}'); 
                                  h.style.display='block'; h.innerText='FAILED: '+this.src;">
                </div>
                <div style="font-size:10px; margin-top:5px;"><b>{c["brand"]}</b><br>{c["name"][:35]}...</div>
            </div>''' for c in cards_data])}
        </div>
    </body>
    </html>
    """

@st.fragment
def render_image_grid(support_files):
    """
    Function #2: Renders the manual visual review grid.
    """
    if st.session_state.final_report.empty:
        return
    
    data = st.session_state.all_data_map
    page_data = data.head(40) # Adjust limit as needed
    
    grid_html = build_fast_grid_html(page_data, 4)
    components.html(grid_html, height=800, scrolling=True)

@st.fragment
def render_exports_section(support_files, country_validator):
    """
    Function #3: Renders report generation and downloads.
    """
    st.subheader(f":material/download: {_t('download_reports')}")
    if st.button("Generate PIM Export", type="primary", use_container_width=True):
        fr = st.session_state.final_report
        # Re-using the logic from export_utils
        data_stream, name, mime = generate_smart_export(fr, "PIM_Export")
        st.download_button("Download Report", data=data_stream, file_name=name, mime=mime, use_container_width=True)
