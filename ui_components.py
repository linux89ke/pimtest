import re
import json
import logging
import concurrent.futures
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from constants import JUMIA_COLORS, GRID_COLS
from data_utils import clean_category_code, df_hash, format_local_price
from export_utils import generate_smart_export, prepare_full_data_merged

logger = logging.getLogger(__name__)

def _t(key):
    from translations import get_translation
    return get_translation(st.session_state.get('ui_lang', 'en'), key)

@st.cache_data(ttl=86400, show_spinner=False)
def analyze_image_quality_cached(url: str):
    if not url or not str(url).startswith("http"): return []
    warnings = []
    try:
        resp = requests.get(url, timeout=1, stream=True)
        if resp.status_code == 200:
            img = Image.open(resp.raw)
            w, h = img.size
            if w < 300 or h < 300: warnings.append("Low Resolution")
            ratio = h / w if w > 0 else 1
            if ratio > 1.5: warnings.append("Tall (Screenshot?)")
            elif ratio < 0.6: warnings.append("Wide Aspect")
    except Exception: pass
    return warnings

def _clear_flag_df_selection(title: str):
    if f"df_{title}" in st.session_state: del st.session_state[f"df_{title}"]

@st.dialog("Confirm Bulk Approval")
def bulk_approve_dialog(sids_to_process, title, subset_data, data_has_warranty_cols_check, support_files, country_validator, validation_runner):
    try:
        from category_matcher_engine import get_engine
        _CAT_MATCHER_AVAILABLE = True
    except ImportError:
        _CAT_MATCHER_AVAILABLE = False

    st.warning(f"You are about to approve **{len(sids_to_process)}** items from `{title}`.")
    if st.button(_t("approve_btn"), type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            data_hash = df_hash(subset_data) + country_validator.code + "_skip_" + title
            
            # Using the passed validation_runner instead of a circular import
            new_report, _ = validation_runner(
                data_hash, subset_data, support_files,
                country_validator.code, data_has_warranty_cols_check,
                skip_validators=[title]
            )
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

            if title == "Wrong Category" and _CAT_MATCHER_AVAILABLE:
                try:
                    engine = get_engine()
                    if engine is not None:
                        learned = 0
                        for sid in sids_to_process:
                            row = subset_data[subset_data['PRODUCT_SET_SID'].astype(str).str.strip() == str(sid)]
                            if row.empty: continue
                            name = str(row.iloc[0].get('NAME', '')).strip()
                            if not name: continue
                            engine.set_compiled_rules(st.session_state.get('compiled_json_rules', {}))
                            predicted = engine.get_category_with_boost(name)
                            if predicted and predicted.lower() not in ('nan', 'none', 'uncategorized', ''):
                                engine.apply_learned_correction(name, predicted, auto_save=False)
                                learned += 1
                        if learned:
                            engine.save_learning_db()
                            st.session_state.main_toasts.append(f"🧠 Engine learned {learned} correction(s) from your approvals.")
                except Exception as _le: logger.warning("Wrong Category approval learning failed: %s", _le)

            if msg_approved > 0: st.session_state.main_toasts.append(f"{msg_approved} items successfully Approved!")
            for flag, count in msg_moved.items(): st.session_state.main_toasts.append(f"{count} items re-flagged as: {flag}")

            st.session_state.exports_cache.clear()
            st.session_state.display_df_cache.clear()
            st.session_state[f"exp_{title}"] = True
            _clear_flag_df_selection(title)
        st.rerun()

def render_flag_expander(title, df_flagged_sids, data, data_has_warranty_cols_check, support_files, country_validator, validation_runner):
    try:
        from category_matcher_engine import get_engine
        _CAT_MATCHER_AVAILABLE = True
    except ImportError: _CAT_MATCHER_AVAILABLE = False

    cache_key = f"display_df_{title}"
    base_display_cols = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'COLOR', 'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE', 'PARENTSKU', 'SELLER_NAME']
    current_display_cols = base_display_cols.copy()
    if title == "Wrong Variation":
        for col in ('COUNT_VARIATIONS', 'LIST_VARIATIONS'):
            if col in data.columns: current_display_cols.append(col)

    if cache_key not in st.session_state.display_df_cache:
        _extra_cols = [c for c in current_display_cols if c in data.columns]
        if 'CATEGORY_CODE' in data.columns and 'CATEGORY_CODE' not in _extra_cols: _extra_cols.append('CATEGORY_CODE')
        df_display = pd.merge(
            df_flagged_sids[['ProductSetSid']], data,
            left_on='ProjectSetSid' if 'ProjectSetSid' in df_flagged_sids.columns else 'ProductSetSid',
            right_on='PRODUCT_SET_SID', how='left'
        )[[c for c in _extra_cols if c in data.columns]]

        _code_to_path = support_files.get('code_to_path', {})
        if _code_to_path and 'CATEGORY_CODE' in df_display.columns:
            df_display['CATEGORY'] = df_display['CATEGORY_CODE'].apply(lambda c: _code_to_path.get(str(c).strip(), '') if pd.notna(c) else '')
            df_display = df_display.drop(columns=['CATEGORY_CODE'])
        df_display = df_display[[c for c in current_display_cols if c in df_display.columns]]
        st.session_state.display_df_cache[cache_key] = df_display
    else: df_display = st.session_state.display_df_cache[cache_key]

    c1, c2 = st.columns(2)
    with c1: search_term = st.text_input(_t("search_grid"), placeholder="Name, Brand...", key=f"s_{title}")
    with c2: seller_filter = st.multiselect("Filter by Seller", sorted(df_display['SELLER_NAME'].astype(str).unique()), key=f"f_{title}")

    df_view = df_display.copy()
    if search_term: df_view = df_view[df_view.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    if seller_filter: df_view = df_view[df_view['SELLER_NAME'].isin(seller_filter)]
    df_view = df_view.reset_index(drop=True)

    if 'NAME' in df_view.columns: df_view['NAME'] = df_view['NAME'].apply(lambda t: re.sub('<[^<]+?>', '', t) if isinstance(t, str) else t)
    if 'GLOBAL_PRICE' in df_view.columns and 'GLOBAL_SALE_PRICE' in df_view.columns:
        def _local_p(row):
            sp, rp = row.get('GLOBAL_SALE_PRICE'), row.get('GLOBAL_PRICE')
            val = sp if pd.notna(sp) and str(sp).strip() != "" else rp
            return format_local_price(val, country_validator.country)
        df_view.insert(df_view.columns.get_loc('GLOBAL_PRICE') + 1 if 'GLOBAL_PRICE' in df_view.columns else len(df_view.columns), 'Local Price', df_view.apply(_local_p, axis=1))

    event = st.dataframe(
        df_view, hide_index=True, use_container_width=True, selection_mode="multi-row", on_select="rerun",
        column_config={
            "PRODUCT_SET_SID": st.column_config.TextColumn(pinned=True),
            "NAME": st.column_config.TextColumn(pinned=True),
            "CATEGORY": st.column_config.TextColumn("Full Category", width="large"),
            "GLOBAL_SALE_PRICE": st.column_config.NumberColumn("Sale Price (USD)", format="$%.2f"),
            "GLOBAL_PRICE": st.column_config.NumberColumn("Price (USD)", format="$%.2f"),
            "Local Price": st.column_config.TextColumn(f"Local Price ({country_validator.country})"),
        }, key=f"df_{title}"
    )
    raw_selected = list(event.selection.rows)
    selected_indices = [i for i in raw_selected if i < len(df_view)]
    st.caption(f"{len(selected_indices)} / {len(df_view)} selected")
    has_selection = len(selected_indices) > 0

    _fm = support_files['flags_mapping']
    _reason_options = [
        "Wrong Category", "Restricted brands", "Suspected Fake product", "Seller Not approved to sell Refurb",
        "Product Warranty", "Seller Approve to sell books", "Seller Approved to Sell Perfume", "Counterfeit Sneakers",
        "Suspected counterfeit Jerseys", "Prohibited products", "Unnecessary words in NAME", "Single-word NAME",
        "Generic BRAND Issues", "Fashion brand issues", "BRAND name repeated in NAME", "Wrong Variation",
        "Generic branded products with genuine brands", "Missing COLOR", "Missing Weight/Volume",
        "Incomplete Smartphone Name", "Duplicate product", "Poor images", "Perfume Tester",
        "NG - Gift Card Seller", "NG - Books Seller", "NG - TV Brand Seller",
        "NG - HP Toners Seller", "NG - Apple Seller", "NG - Xmas Tree Seller",
        "NG - Rice Brand Seller", "NG - Powerbank Capacity", "Wrong Price", "Category Max Price Exceeded",
        "Other Reason (Custom)",
    ]

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button(_t("approve_btn"), key=f"approve_sel_{title}", type="primary", use_container_width=True, disabled=not has_selection):
            sids_to_process = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
            subset = data[data['PRODUCT_SET_SID'].isin(sids_to_process)]
            _clear_flag_df_selection(title)
            bulk_approve_dialog(sids_to_process, title, subset, data_has_warranty_cols_check, support_files, country_validator, validation_runner)

    with btn_col2:
        with st.popover(_t("reject_as"), use_container_width=True, disabled=not has_selection):
            chosen_reason = st.selectbox("Reason", _reason_options, key=f"rej_reason_dd_{title}", label_visibility="collapsed")
            _cmt_lang = 'fr' if st.session_state.get('selected_country') == "Morocco" else 'en'

            if chosen_reason == "Other Reason (Custom)":
                custom_comment = st.text_area("Custom comment", placeholder="Type your rejection reason here...", key=f"custom_comment_{title}", height=80)
                if st.button("Apply", key=f"apply_custom_{title}", type="primary", use_container_width=True, disabled=not has_selection):
                    to_reject = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    final_comment = custom_comment.strip() if custom_comment.strip() else "Other Reason"
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(to_reject), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', '1000007 - Other Reason', final_comment, 'Other Reason (Custom)']
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected with custom reason.")
                    st.session_state.exports_cache.clear()
                    st.session_state.display_df_cache.clear()
                    st.session_state[f"exp_{title}"] = True
                    _clear_flag_df_selection(title)
                    st.rerun()
            else:
                _rinfo = _fm.get(chosen_reason, {'reason': '1000007 - Other Reason', 'en': chosen_reason})
                _rcode = _rinfo['reason']
                _rcmt = _rinfo.get(_cmt_lang, _rinfo.get('en'))
                st.caption(f"Code: {_rcode[:40]}...")
                if st.button("Apply", key=f"apply_dd_{title}", type="primary", use_container_width=True, disabled=not has_selection):
                    to_reject = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(to_reject), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', _rcode, _rcmt, chosen_reason]
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected as '{chosen_reason}'.")

                    if chosen_reason == "Wrong Category" and title != "Wrong Category" and _CAT_MATCHER_AVAILABLE:
                        try:
                            engine = get_engine()
                            _cats = support_files.get('categories_names_list', [])
                            if engine is not None and _cats:
                                if not engine._tfidf_built: engine.build_tfidf_index(_cats)
                                learned = 0
                                for sid in to_reject:
                                    prod_row = data[data['PRODUCT_SET_SID'].astype(str).str.strip() == str(sid)]
                                    if prod_row.empty: continue
                                    name = str(prod_row.iloc[0].get('NAME', '')).strip()
                                    if not name: continue
                                    engine.set_compiled_rules(st.session_state.get('compiled_json_rules', {}))
                                    predicted = engine.get_category_with_boost(name)
                                    if predicted and predicted.lower() not in ('nan', 'none', 'uncategorized', ''):
                                        engine.apply_learned_correction(name, predicted, auto_save=False)
                                        learned += 1
                                if learned:
                                    engine.save_learning_db()
                                    st.session_state.main_toasts.append(f"🧠 Engine noted {learned} missed Wrong Category item(s).")
                        except Exception as _le: logger.warning("Wrong Category manual rejection learning failed: %s", _le)

                    st.session_state.exports_cache.clear()
                    st.session_state.display_df_cache.clear()
                    st.session_state[f"exp_{title}"] = True
                    _clear_flag_df_selection(title)
                    st.rerun()


def build_fast_grid_html(page_data, flags_mapping, country, page_warnings,
                         rejected_state, cols_per_row, prefetch_urls=None):
    """
    Progressive + Branded Ultra-Fast Grid
    • Low-res Jumia-branded SVG placeholder
    • Real image progressive fade-in
    • Animated gradient warning badges
    • Hover tooltips (full name + seller)
    • New smooth zoom modal with backdrop
    """
    O = JUMIA_COLORS["primary_orange"]
    G = JUMIA_COLORS["success_green"]
    R = JUMIA_COLORS["jumia_red"]

    committed_json = json.dumps(rejected_state)
    prefetch_json = json.dumps(prefetch_urls or [])

    html_dir = "rtl" if st.session_state.get('ui_lang') == "ar" else "ltr"
    rejected_label = str(_t('rejected') or 'REJECTED').upper()

    # ── Stylish Jumia-branded low-res placeholder SVG ─────────────────────
    _PLACEHOLDER_SVG = (
        "data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='180' viewBox='0 0 300 180'>"
        "<defs><linearGradient id='g' x1='0%' y1='0%' x2='100%' y2='100%'><stop offset='0%' stop-color='%23FF6600'/><stop offset='100%' stop-color='%23FF8800'/></linearGradient></defs>"
        "<rect width='300' height='180' rx='12' fill='url(%23g)'/>"
        "<text x='150' y='72' text-anchor='middle' font-family='sans-serif' font-size='32' font-weight='700' fill='%23fff' letter-spacing='-1'>JUMIA</text>"
        "<text x='150' y='108' text-anchor='middle' font-family='sans-serif' font-size='13' fill='%23fff' opacity='0.9'>Loading product image...</text>"
        "<circle cx='150' cy='145' r='8' fill='%23fff' opacity='0.3'/>"
        "</svg>"
    )

    cards_data = []
    for _, row in page_data.iterrows():
        sid = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip()
        img_url = img_url.replace("http://", "https://", 1)
        if not img_url.startswith("https"):
            img_url = ""

        sale_p = row.get("GLOBAL_SALE_PRICE")
        reg_p = row.get("GLOBAL_PRICE")
        usd_val = sale_p if pd.notna(sale_p) and str(sale_p).strip() != "" else reg_p
        price_str = format_local_price(usd_val, st.session_state.get('selected_country', 'Kenya')) if pd.notna(usd_val) else ""

        cards_data.append({
            "sid": sid,
            "img": img_url,
            "name": str(row.get("NAME", "")),
            "brand": str(row.get("BRAND", "Unknown Brand")),
            "cat": str(row.get("CATEGORY", "Unknown Category")),
            "seller": str(row.get("SELLER_NAME", "Unknown Seller")),
            "warnings": page_warnings.get(sid, []),
            "price": price_str,
        })

    cards_json = json.dumps(cards_data)

    return f"""<!DOCTYPE html>
<html dir="{html_dir}">
<head>
<meta charset="utf-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;font-family:sans-serif;}}
  body{{background:#f5f5f5;padding:8px;}}
  .ctrl-bar{{position:sticky;top:0;z-index:99999;display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:8px 12px;background:rgba(255,255,255,0.95);backdrop-filter:blur(8px);border-bottom:2px solid {O};border-radius:4px;margin-bottom:12px;box-shadow:0 4px 16px rgba(0,0,0,0.15);}}
  .sel-count{{font-weight:700;color:{O};font-size:13px;min-width:80px;}}
  .reason-sel{{flex:1;min-width:160px;padding:6px 10px;border:1px solid #ccc;border-radius:4px;font-size:12px;background:#fff;cursor:pointer;}}
  .batch-btn{{padding:7px 14px;background:{O};color:#fff;border:none;border-radius:4px;font-weight:700;font-size:12px;cursor:pointer;}}
  .batch-btn:hover{{opacity:.88;}}
  .desel-btn{{padding:7px 12px;background:#fff;color:#555;border:1px solid #ccc;border-radius:4px;font-size:12px;cursor:pointer;}}
  .desel-btn:hover{{background:#f5f5f5;}}
  .grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}
  .card{{border:2px solid #e0e0e0;border-radius:8px;padding:10px;background:#fff;position:relative;transition:border-color .15s,box-shadow .15s;z-index:1;}}
  .card.selected{{border-color:{G};box-shadow:0 0 0 3px rgba(76,175,80,.2);background:rgba(76,175,80,.04);}}
  .card.staged-rej{{border-color:{R};box-shadow:0 0 0 3px rgba(231,60,23,.2);background:rgba(231,60,23,.04);}}
  .card.committed-rej{{border-color:#bbb;opacity:.6;}}
  .card-img-wrap{{position:relative;cursor:pointer;border-radius:12px;background:#f8f8f8;display:flex;align-items:center;justify-content:center;height:180px;overflow:hidden;}}
  .card-img-wrap::before{{content:'';position:absolute;inset:0;background:linear-gradient(90deg,#f0f0f0 25%,#e0e0e0 50%,#f0f0f0 75%);background-size:200% 100%;animation:shimmer 1.4s infinite;z-index:1;border-radius:12px;}}
  .card-img-wrap.img-loaded::before{{display:none;}}
  @keyframes shimmer{{0%{{background-position:200% 0}}100%{{background-position:-200% 0}}}}
  .card-img-placeholder{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;z-index:1;}}
  .card-img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;border-radius:12px;z-index:2;opacity:0;transition:opacity .4s ease;}}
  .card-img.img-loaded{{opacity:1;}}
  .card.committed-rej .card-img{{filter:grayscale(80%);}}
  .warn-wrap{{position:absolute;top:8px;right:8px;display:flex;flex-direction:column;gap:4px;z-index:10;pointer-events:none;}}
  .warn-badge{{background:linear-gradient(90deg,#FFC107,#FF9800);color:#313133;font-size:9px;font-weight:800;padding:3px 8px;border-radius:9999px;box-shadow:0 2px 6px rgba(255,152,0,.3);animation:pulse 2s infinite;}}
  @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.85}}}}
  .price-badge{{position:absolute;top:8px;left:8px;background:rgba(76,175,80,.95);color:#fff;font-size:10px;font-weight:800;padding:3px 8px;border-radius:9999px;z-index:10;pointer-events:none;box-shadow:0 2px 6px rgba(0,0,0,.2);}}
  .meta{{font-size:11px;margin-top:8px;line-height:1.4;}}
  .meta .nm{{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:help;}}
  .meta .br{{color:{O};font-weight:700;margin:2px 0;}}
  .meta .ct{{color:#666;font-size:10px;word-break:break-word;}}
  .meta .sl{{color:#999;font-size:9px;margin-top:4px;border-top:1px dashed #eee;padding-top:4px;cursor:help;}}
  .acts{{display:flex;gap:4px;margin-top:8px;}}
  .act-btn{{flex:1;padding:6px;font-size:11px;border:none;border-radius:4px;cursor:pointer;font-weight:700;color:#fff;background:{O};}}
  .act-more{{flex:1;font-size:11px;border:1px solid #ccc;border-radius:4px;outline:none;cursor:pointer;background:#fff;}}
  #prefetch-status{{font-size:10px;color:#aaa;text-align:right;padding:4px 8px;margin-top:8px;}}
  /* ── Zoom Modal ── */
  #zoom-modal{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.92);z-index:100000;align-items:center;justify-content:center;transition:opacity .3s ease;}}
  #zoom-modal.show{{display:flex;}}
  #zoom-content{{position:relative;max-width:92%;max-height:92vh;border-radius:16px;overflow:hidden;box-shadow:0 30px 80px rgba(0,0,0,.6);}}
  #modal-img{{max-width:100%;max-height:92vh;display:block;}}
  .modal-close{{position:absolute;top:16px;right:16px;width:44px;height:44px;background:#fff;color:#333;border:none;border-radius:50%;font-size:28px;line-height:1;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,.3);z-index:10;}}
</style>
</head>
<body>
<div class="ctrl-bar">
  <span class="sel-count" id="sel-count-bar">0 {_t("items_pending")}</span>
  <select class="reason-sel" id="batch-reason">
    <option value="REJECT_POOR_IMAGE">{_t("poor_img")}</option>
    <option value="REJECT_WRONG_CAT">{_t("wrong_cat")}</option>
    <option value="REJECT_FAKE">{_t("fake_prod")}</option>
    <option value="REJECT_BRAND">{_t("restr_brand")}</option>
    <option value="REJECT_WRONG_BRAND">{_t("wrong_brand")}</option>
    <option value="REJECT_PROHIBITED">{_t("prohibited")}</option>
    <option value="REJECT_COLOR">{_t("missing_color")}</option>
  </select>
  <button class="batch-btn" onclick="doBatchReject()">{_t("batch_reject")}</button>
  <button class="desel-btn" onclick="window.doSelectAll()">{_t("select_all")}</button>
  <button class="desel-btn" onclick="doDeselAll()">{_t("deselect_all")}</button>
</div>
<div class="grid" id="card-grid"></div>
<div id="prefetch-status"></div>
<div id="prefetch-container" style="display:none;position:absolute;width:1px;height:1px;overflow:hidden;"></div>

<!-- Zoom Modal -->
<div id="zoom-modal" onclick="if(event.target.id==='zoom-modal')closeZoom()">
  <div id="zoom-content" onclick="event.stopImmediatePropagation()">
    <img id="modal-img" alt="Zoomed product">
    <button class="modal-close" onclick="closeZoom()">×</button>
  </div>
</div>

<script>
function escapeHtml(u){{return(u||"").toString().replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;");}}
var CARDS = {cards_json};
var COMMITTED = {committed_json};
var PREFETCH_URLS = {prefetch_json};
var PLACEHOLDER = "{_PLACEHOLDER_SVG}";

window._gridSelected = window._gridSelected || {{}};
window._stagedRejections = window._stagedRejections || {{}};
var selected = window._gridSelected;
var staged = window._stagedRejections;

// Bridge to Streamlit (real-time sync ready)
function sendMsg(type, payload) {{
  try {{
    var par = window.parent;
    var inputs = par.document.querySelectorAll('input[type="text"]');
    var bridge = null;
    for (var i = 0; i < inputs.length; i++) {{
      if (inputs[i].getAttribute('aria-label') === 'jtbridge' || inputs[i].placeholder === 'JTBRIDGE_UNIQUE_DO_NOT_USE') {{
        bridge = inputs[i]; break;
      }}
    }}
    if (!bridge) return;
    var msg = JSON.stringify({{action: type, payload: payload}});
    bridge.focus({{preventScroll: true}});
    Object.getOwnPropertyDescriptor(par.HTMLInputElement.prototype, 'value').set.call(bridge, msg);
    bridge.dispatchEvent(new par.Event('input', {{bubbles: true}}));
    setTimeout(() => {{ bridge.blur(); bridge.dispatchEvent(new par.KeyboardEvent('keydown', {{bubbles:true,cancelable:true,key:'Enter',keyCode:13}})); }}, 150);
  }} catch(ex) {{ console.error('jtbridge error:', ex); }}
}}

// Progressive image loading + quality check
function onImgLoad(img, sid) {{
  img.classList.add('img-loaded');
  var wrap = img.closest('.card-img-wrap');
  if (wrap) wrap.classList.add('img-loaded');
  var w = img.naturalWidth, h = img.naturalHeight;
  var warns = [];
  if (w > 0 && h > 0) {{
    if (w < 300 || h < 300) warns.push('Low Resolution');
    var ratio = h / w;
    if (ratio > 1.5) warns.push('Tall (Screenshot?)');
    else if (ratio < 0.6) warns.push('Wide Aspect');
  }}
  if (warns.length) addWarnings(sid, warns);
}}
function onImgError(img, sid) {{
  img.src = PLACEHOLDER;
  img.classList.add('img-loaded');
  addWarnings(sid, ['Broken Image']);
}}
function addWarnings(sid, warns) {{
  var wrap = document.querySelector('#card-' + sid + ' .warn-wrap');
  if (!wrap) return;
  warns.forEach(w => {{
    var badge = document.createElement('span');
    badge.className = 'warn-badge';
    badge.textContent = w;
    wrap.appendChild(badge);
  }});
}}

// Render card with progressive placeholder
function renderCard(card) {{
  var sid = card.sid;
  var isCommitted = sid in COMMITTED;
  var isStaged = sid in staged;
  var isSelected = !isCommitted && !isStaged && (sid in selected);
  var cls = 'card' + (isCommitted ? ' committed-rej' : isStaged ? ' staged-rej' : isSelected ? ' selected' : '');
  var imgSrc = card.img || PLACEHOLDER;
  var shortName = card.name.length > 38 ? escapeHtml(card.name.slice(0,38)) + '…' : escapeHtml(card.name);
  var warnHtml = (card.warnings || []).map(w => `<span class="warn-badge">${{escapeHtml(w)}}</span>`).join('');
  var priceHtml = card.price ? `<div class="price-badge">${{escapeHtml(card.price)}}</div>` : '';
  var zoomHtml = `<div class="zoom-btn" onclick="event.stopPropagation();showZoom('${{sid}}')"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div>`;

  var overlayHtml = '', actHtml = '';
  if (isCommitted) {{
    overlayHtml = `<div class="rej-overlay"><div class="rej-badge">${{rejected_label}}</div><div class="rej-label">${{escapeHtml((COMMITTED[sid]||'').replace(/_/g,' '))}}</div><button class="undo-btn" onclick="event.stopPropagation();window.undoReject('${{sid}}')">{_t('undo')}</button></div>`;
  }} else if (isStaged) {{
    overlayHtml = `<div class="rej-overlay staged"><div class="rej-badge pending">PENDING</div><div class="rej-label">${{escapeHtml((staged[sid]||'').replace(/_/g,' '))}}</div><button class="undo-btn" onclick="event.stopPropagation();window.clearStaged('${{sid}}')">{_t('clear_sel')}</button></div>`;
  }} else {{
    actHtml = `<div class="acts"><button class="act-btn" onclick="event.stopPropagation();window.stageReject('${{sid}}','REJECT_POOR_IMAGE')">{_t('poor_img')}</button><select class="act-more" onchange="if(this.value){{event.stopPropagation();window.stageReject('${{sid}}',this.value);this.value=''}}"><option value="">{_t('more_options')}</option><option value="REJECT_WRONG_CAT">{_t('wrong_cat')}</option><option value="REJECT_FAKE">{_t('fake_prod')}</option><option value="REJECT_BRAND">{_t('restr_brand')}</option><option value="REJECT_PROHIBITED">{_t('prohibited')}</option><option value="REJECT_COLOR">{_t('missing_color')}</option><option value="REJECT_WRONG_BRAND">{_t('wrong_brand')}</option></select></div>`;
  }}

  return `<div class="${{cls}}" id="card-${{sid}}">
    <div class="card-img-wrap" onclick="window.toggleSelect('${{sid}}',event)">
      ${{priceHtml}}
      <div class="warn-wrap">${{warnHtml}}</div>
      <img class="card-img-placeholder" src="${{PLACEHOLDER}}" alt="">
      <img class="card-img" loading="lazy" decoding="async" src="${{escapeHtml(imgSrc)}}" 
           onload="onImgLoad(this,'${{sid}}')" onerror="onImgError(this,'${{sid}}')">
      ${{zoomHtml}}
      ${{overlayHtml}}
      <div class="tick">✔</div>
    </div>
    <div class="meta">
      <div class="nm" title="${{escapeHtml(card.name)}}">${{shortName}}</div>
      <div class="br" title="${{escapeHtml(card.brand)}}">${{escapeHtml(card.brand)}}</div>
      <div class="ct">${{escapeHtml(card.cat)}}</div>
      <div class="sl" title="${{escapeHtml(card.seller)}}">${{escapeHtml(card.seller)}}</div>
    </div>
    ${{actHtml}}
  </div>`;
}}

// Zoom Modal
window.showZoom = function(sid) {{
  var card = CARDS.find(c => c.sid === sid);
  if (!card) return;
  var modal = document.getElementById('zoom-modal');
  var img = document.getElementById('modal-img');
  img.src = card.img || PLACEHOLDER;
  modal.classList.add('show');
}};
window.closeZoom = function() {{
  document.getElementById('zoom-modal').classList.remove('show');
}};

// Rest of your JS functions (unchanged but updated for new modal)
function updateSelCount() {{ document.getElementById('sel-count-bar').textContent = (Object.keys(selected).length + Object.keys(staged).length) + ' {_t("items_pending")}'; }}
function renderAll() {{ document.getElementById('card-grid').innerHTML = CARDS.map(renderCard).join(''); updateSelCount(); }}
function replaceCard(sid) {{ 
  var el = document.getElementById('card-' + sid);
  if (!el) return;
  var card = CARDS.find(c => c.sid === sid);
  if (card) {{ var t = document.createElement('div'); t.innerHTML = renderCard(card); el.replaceWith(t.firstElementChild); }}
}}
window.doSelectAll = function() {{ CARDS.forEach(c => {{ if (!(c.sid in COMMITTED) && !(c.sid in staged)) selected[c.sid] = true; }}); renderAll(); updateSelCount(); }};
window.toggleSelect = function(sid, e) {{ 
  var img = document.querySelector('#card-' + sid + ' .card-img');
  if (img && img.classList.contains('locally-zoomed')) return;
  if (sid in COMMITTED) return;
  if (sid in staged) delete staged[sid];
  else if (sid in selected) delete selected[sid];
  else selected[sid] = true;
  replaceCard(sid); updateSelCount();
}};
window.stageReject = function(sid, r) {{ if (sid in selected) delete selected[sid]; staged[sid] = r; replaceCard(sid); updateSelCount(); }};
window.clearStaged = function(sid) {{ delete staged[sid]; replaceCard(sid); updateSelCount(); }};
window.undoReject = function(sid) {{ sendMsg('undo', {{[sid]: true}}); delete COMMITTED[sid]; replaceCard(sid); updateSelCount(); }};
window.doBatchReject = function() {{
  var br = document.getElementById('batch-reason').value, payload = {{}}, count = 0;
  for (var s in staged) {{ payload[s] = staged[s]; count++; }}
  for (var s in selected) {{ payload[s] = br; count++; }}
  if (count === 0) return;
  for (var s in payload) {{ COMMITTED[s] = payload[s]; delete selected[s]; delete staged[s]; }}
  sendMsg('reject', payload); renderAll(); updateSelCount();
}};
window.doDeselAll = function() {{ for (var k in selected) delete selected[k]; for (var k in staged) delete staged[k]; renderAll(); updateSelCount(); }};

// Background prefetch (next pages)
(function() {{
  if (!PREFETCH_URLS || !PREFETCH_URLS.length) return;
  var container = document.getElementById('prefetch-container');
  var statusEl = document.getElementById('prefetch-status');
  var i = 0, total = PREFETCH_URLS.length, done = 0;
  var runner = window.requestIdleCallback || function(fn){{setTimeout(fn,300);}};
  function prefetchBatch() {{
    var limit = 4;
    var processed = 0;
    while (i < total && processed < limit) {{
      var url = PREFETCH_URLS[i++]; processed++;
      var img = new Image();
      img.onload = () => {{ done++; if (statusEl) statusEl.textContent = `Prefetched ${{done}}/${{total}}`; }};
      img.style.cssText = 'width:1px;height:1px;opacity:0;';
      container.appendChild(img);
      img.src = url;
    }}
    if (i < total) runner(prefetchBatch);
  }}
  setTimeout(prefetchBatch, 800);
}})();

renderAll();
</script>
</body>
</html>"""



@st.fragment
def render_image_grid(support_files):
    if st.session_state.final_report.empty or st.session_state.get('file_mode') == "post_qc": return

    st.markdown("---")
    st.header(f":material/pageview: {_t('manual_review')}", anchor=False)

    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    committed_rej_sids = {k.replace("quick_rej_", "") for k in st.session_state.keys() if k.startswith("quick_rej_") and "reason" not in k}
    valid_grid_df = fr[(fr["Status"] == "Approved") | (fr["ProductSetSid"].isin(committed_rej_sids))]

    c1, c2, c3 = st.columns([1.5, 1.5, 2])
    with c1: search_n = st.text_input("Search by Name", placeholder="Product name…")
    with c2: search_sc = st.text_input("Search by Seller/Category", placeholder="Seller or Category…")
    with c3:
        st.session_state.grid_items_per_page = st.select_slider(
            "Items per page", options=[20, 50, 100, 200],
            value=st.session_state.get('grid_items_per_page', 50),
        )

    if 'MAIN_IMAGE' not in data.columns: data['MAIN_IMAGE'] = ''
    available_cols = [c for c in GRID_COLS if c in data.columns]
    if 'CATEGORY_CODE' in data.columns and 'CATEGORY_CODE' not in available_cols: available_cols.append('CATEGORY_CODE')

    review_data = pd.merge(
        valid_grid_df[["ProductSetSid"]], data[available_cols],
        left_on="ProductSetSid", right_on="PRODUCT_SET_SID", how="left",
    )
    _code_to_path = support_files.get('code_to_path', {})
    if _code_to_path and 'CATEGORY_CODE' in review_data.columns:
        review_data = review_data.copy()
        review_data['CATEGORY'] = review_data['CATEGORY_CODE'].apply(lambda c: _code_to_path.get(str(c).strip(), str(c)) if pd.notna(c) else '')

    if search_n: review_data = review_data[review_data["NAME"].astype(str).str.contains(search_n, case=False, na=False)]
    if search_sc:
        mc = review_data["CATEGORY"].astype(str).str.contains(search_sc, case=False, na=False) if "CATEGORY" in review_data.columns else pd.Series(False, index=review_data.index)
        ms = review_data["SELLER_NAME"].astype(str).str.contains(search_sc, case=False, na=False)
        review_data = review_data[mc | ms]

    ipp = st.session_state.get('grid_items_per_page', 50)
    total_pages = max(1, (len(review_data) + ipp - 1) // ipp)
    if st.session_state.get('grid_page', 0) >= total_pages: st.session_state.grid_page = 0

    pg_cols = st.columns([1, 2, 1], vertical_alignment="center")
    with pg_cols[0]:
        if st.button("◀ Prev Page", use_container_width=True, disabled=st.session_state.grid_page == 0):
            st.session_state.grid_page = max(0, st.session_state.grid_page - 1)
            st.session_state.do_scroll_top = True
            st.rerun(scope="fragment")
    with pg_cols[1]:
        new_page = st.number_input(
            f"Jump to Page (Total: {total_pages} | {len(review_data)} items)",
            min_value=1, max_value=max(1, total_pages),
            value=st.session_state.grid_page + 1, step=1
        )
        if new_page - 1 != st.session_state.grid_page:
            st.session_state.grid_page = new_page - 1
            st.session_state.do_scroll_top = True
            st.rerun(scope="fragment")
    with pg_cols[2]:
        if st.button("Next Page ▶", use_container_width=True, disabled=st.session_state.grid_page >= total_pages - 1):
            st.session_state.grid_page += 1
            st.session_state.do_scroll_top = True
            st.rerun(scope="fragment")

    page_start = st.session_state.grid_page * ipp
    page_data = review_data.iloc[page_start: page_start + ipp]

    page_warnings = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        future_to_sid = {ex.submit(analyze_image_quality_cached, str(r.get("MAIN_IMAGE", "")).strip()): str(r["PRODUCT_SET_SID"]) for _, r in page_data.iterrows()}
        for future in concurrent.futures.as_completed(future_to_sid):
            warns = future.result()
            if warns: page_warnings[future_to_sid[future]] = warns

    rejected_state = {sid: st.session_state[f"quick_rej_reason_{sid}"] for sid in page_data["PRODUCT_SET_SID"].astype(str) if st.session_state.get(f"quick_rej_{sid}")}
    cols_per_row = 3 if st.session_state.get('layout_mode') == "centered" else 4
    grid_html = build_fast_grid_html(page_data, support_files["flags_mapping"], st.session_state.get('selected_country', 'Kenya'), page_warnings, rejected_state, cols_per_row)
    components.html(grid_html, height=800, scrolling=True)

    if st.session_state.get("do_scroll_top", False):
        components.html("<script>window.parent.document.querySelector('.main').scrollTo({top:0,behavior:'smooth'});</script>", height=0)
        st.session_state.do_scroll_top = False

@st.fragment
def render_exports_section(support_files, country_validator):
    if st.session_state.final_report.empty or st.session_state.get('file_mode') == 'post_qc': return

    from datetime import datetime
    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    app_df = fr[fr['Status'] == 'Approved']
    rej_df = fr[fr['Status'] == 'Rejected']
    c_code = st.session_state.get('selected_country', 'Kenya')[:2].upper()
    date_str = datetime.now().strftime('%Y-%m-%d')
    reasons_df = support_files.get('reasons', pd.DataFrame())

    st.markdown("---")
    st.markdown(
        f"<div style='background:linear-gradient(135deg,{JUMIA_COLORS['primary_orange']},{JUMIA_COLORS['secondary_orange']});padding:20px 24px;border-radius:10px;margin-bottom:20px;'>"
        f"<h2 style='color:white;margin:0;font-size:24px;font-weight:700;'>{_t('download_reports')}</h2>"
        f"<p style='color:rgba(255,255,255,0.9);margin:6px 0 0 0;font-size:13px;'>Export validation results in Excel or ZIP format</p></div>",
        unsafe_allow_html=True
    )

    exports_config = [
        ("PIM Export",    fr,     'Complete validation report with all statuses', lambda df: generate_smart_export(df, f"{c_code}_PIM_Export_{date_str}", 'simple', reasons_df)),
        ("Rejected Only", rej_df, 'Products that failed validation',              lambda df: generate_smart_export(df, f"{c_code}_Rejected_{date_str}", 'simple', reasons_df)),
        ("Approved Only", app_df, 'Products that passed validation',              lambda df: generate_smart_export(df, f"{c_code}_Approved_{date_str}", 'simple', reasons_df)),
        ("Full Data",     data,   'Complete dataset with validation flags',       lambda df: generate_smart_export(prepare_full_data_merged(df, fr), f"{c_code}_Full_{date_str}", 'full')),
    ]

    all_cached = all(t in st.session_state.exports_cache for t, _, _, _ in exports_config)
    if all_cached:
        st.success("All reports generated and ready to download.", icon=":material/check_circle:")
    else:
        if st.button("Generate All Reports", type="primary", icon=":material/download:", use_container_width=True):
            with st.spinner("Generating all reports…"):
                for t2, d2, _, f2 in exports_config:
                    if t2 not in st.session_state.exports_cache:
                        res, fname, mime = f2(d2)
                        st.session_state.exports_cache[t2] = {"data": res.getvalue(), "fname": fname, "mime": mime}
            st.rerun()

    cols_count = 4 if st.session_state.get('layout_mode') == "wide" else 2
    for i in range(0, len(exports_config), cols_count):
        cols = st.columns(cols_count)
        for j, col in enumerate(cols):
            if i + j < len(exports_config):
                title, df, desc, func = exports_config[i + j]
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f"<div style='text-align:center;margin-bottom:15px;'>"
                            f"<div style='font-size:18px;font-weight:700;'>{title}</div>"
                            f"<div style='font-size:11px;margin-top:4px;opacity:0.7;'>{desc}</div>"
                            f"<div style='background:{JUMIA_COLORS['light_gray']};color:{JUMIA_COLORS['primary_orange']};padding:8px;border-radius:6px;margin-top:12px;font-weight:600;'>{len(df):,} rows</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        if title not in st.session_state.exports_cache:
                            if st.button("Generate", key=f"gen_{title}", type="primary", use_container_width=True, icon=":material/download:"):
                                with st.spinner("Generating all reports…"):
                                    for t2, d2, _, f2 in exports_config:
                                        if t2 not in st.session_state.exports_cache:
                                            res, fname, mime = f2(d2)
                                            st.session_state.exports_cache[t2] = {"data": res.getvalue(), "fname": fname, "mime": mime}
                                st.rerun()
                        else:
                            cache = st.session_state.exports_cache[title]
                            st.download_button("Download", data=cache["data"], file_name=cache["fname"], mime=cache["mime"],
                                               use_container_width=True, type="primary", icon=":material/file_download:", key=f"dl_{title}")
                            if st.button("Clear", key=f"clr_{title}", use_container_width=True):
                                del st.session_state.exports_cache[title]
                                st.rerun()
