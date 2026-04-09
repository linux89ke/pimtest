"""
ui_components.py - All Streamlit UI rendering components, dialogs, and the image grid
"""

import re
import json
import logging
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import base64

from constants import JUMIA_COLORS, GRID_COLS
from data_utils import df_hash, format_local_price
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

# ===================================================================
# BULK APPROVAL DIALOG (unchanged)
# ===================================================================
@st.dialog("Confirm Bulk Approval", icon=":material/check_circle:")
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
                except Exception as _le:
                    logger.warning("Wrong Category approval learning failed: %s", _le)

            if msg_approved > 0:
                st.session_state.main_toasts.append(f"{msg_approved} items successfully Approved!")
            for flag, count in msg_moved.items():
                st.session_state.main_toasts.append(f"{count} items re-flagged as: {flag}")

            st.session_state.exports_cache.clear()
            st.session_state.display_df_cache.clear()
            st.session_state[f"exp_{title}"] = True
            _clear_flag_df_selection(title)
        st.rerun()

# ===================================================================
# FLAG EXPANDER (unchanged)
# ===================================================================
def render_flag_expander(...):   # ← keep your original function exactly as in the second file
    # (I omitted it here for brevity – just copy it from your "ui_components fix jump.py")
    pass   # ← replace this line with your full render_flag_expander function

# ===================================================================
# FAST GRID HTML – FIXED VERSION (batch reject + no jump)
# ===================================================================
def build_fast_grid_html(page_data, flags_mapping, country, page_warnings,
                         rejected_state, cols_per_row, prefetch_urls=None):
    
    O = JUMIA_COLORS["primary_orange"]
    R = JUMIA_COLORS["jumia_red"]
    committed_json = json.dumps(rejected_state)
    prefetch_json = json.dumps(prefetch_urls or [])
    html_dir = "rtl" if st.session_state.get('ui_lang') == "ar" else "ltr"

    labels_dict = {
        "poor_img": _t("poor_img"), "wrong_cat": _t("wrong_cat"),
        "fake_prod": _t("fake_prod"), "restr_brand": _t("restr_brand"),
        "wrong_brand": _t("wrong_brand"), "prohibited": _t("prohibited"),
        "missing_color": _t("missing_color"), "more_options": _t("more_options"),
        "undo": _t("undo"), "clear_sel": _t("clear_sel"),
        "items_pending": _t("items_pending"), "batch_reject": _t("batch_reject"),
        "select_all": _t("select_all"), "deselect_all": _t("deselect_all"),
        "rejected": str(_t('rejected') or 'REJECTED').upper()
    }
    labels_json = json.dumps(labels_dict)

    _PLACEHOLDER_SVG = (
        "data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='180' viewBox='0 0 300 180'>"
        "<defs><linearGradient id='g' x1='0%' y1='0%' x2='100%' y2='100%'><stop offset='0%' stop-color='%23FFF8F2'/><stop offset='100%' stop-color='%23FFEFE5'/></linearGradient></defs>"
        "<rect width='300' height='180' rx='12' fill='url(%23g)'/>"
        "<text x='150' y='80' text-anchor='middle' font-family='sans-serif' font-size='34' "
        "font-weight='800' fill='%23FF8800' letter-spacing='-1'>JUMIA</text>"
        "<text x='150' y='110' text-anchor='middle' font-family='sans-serif' font-size='14' "
        "font-weight='600' fill='%23FF8800' opacity='0.7'>Loading...</text>"
        "</svg>"
    )

    cards_data = []
    for _, row in page_data.iterrows():
        sid = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip().replace("http://", "https://", 1)
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
<meta name="referrer" content="no-referrer">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;font-family:sans-serif;}}
  body{{background:#ffffff;padding:8px;}}
  .ctrl-bar{{position:-webkit-sticky;position:sticky;top:0;z-index:99999;display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:8px 12px;background:rgba(255,255,255,0.95);backdrop-filter:blur(8px);border-bottom:2px solid {O};border-radius:4px;margin-bottom:12px;box-shadow:0 4px 16px rgba(0,0,0,0.15);}}
  .bottom-bar {{position: relative; border-bottom: none; border-top: 2px solid {O}; margin-top: 16px; margin-bottom: 0; z-index: 10; box-shadow: 0 -4px 16px rgba(0,0,0,0.05);}}
  .sel-count{{font-weight:700;color:{O};font-size:13px;min-width:80px;}}
  .reason-sel{{flex:1;min-width:160px;padding:6px 10px;border:1px solid #ccc;border-radius:4px;font-size:12px;background:#fff;cursor:pointer;}}
  .batch-btn{{padding:7px 14px;background:{O};color:#fff;border:none;border-radius:4px;font-weight:700;font-size:12px;cursor:pointer;}}
  .batch-btn:hover{{opacity:.88;}}
  .desel-btn{{padding:7px 12px;background:#fff;color:#555;border:1px solid #ccc;border-radius:4px;font-size:12px;cursor:pointer;}}
  .desel-btn:hover{{background:#f5f5f5;}}
  .top-btn {{margin-left: auto; background: #313133; color: white; border-color: #313133; font-weight: bold;}}
  .top-btn:hover {{background: #000; color: white;}}
  .grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}
  .card{{border:2px solid #e0e0e0;border-radius:8px;padding:10px;background:#fff;position:relative;transition:border-color .15s,box-shadow .15s;z-index:1;}}
  .card.selected{{border-color:{O};box-shadow:0 0 0 5px rgba(255,136,0,.35);background:rgba(255,136,0,.04);}}
  .card.staged-rej{{border-color:{R};box-shadow:0 0 0 4px rgba(231,60,23,.3);background:rgba(231,60,23,.04);}}
  .card.committed-rej{{border-color:#bbb;opacity:.6;}}
  .card-img-wrap{{position:relative;cursor:pointer;border-radius:8px;background:#fff;display:flex;align-items:center;justify-content:center;height:180px;overflow:hidden;border:1px solid #111;}}
  .card-img-wrap::before{{content:'';position:absolute;inset:0;background:linear-gradient(90deg,#FFF8F2 25%,#FFEFE5 50%,#FFF8F2 75%);background-size:200% 100%;animation:shimmer 1.4s infinite;z-index:1;}}
  .card-img-wrap.img-loaded::before{{display:none;}}
  @keyframes shimmer{{0%{{background-position:200% 0}}100%{{background-position:-200% 0}}}}
  .card-img-placeholder{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;z-index:1;}}
  .card-img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;z-index:2;opacity:0;transition:opacity .4s ease;}}
  .card-img.img-loaded{{opacity:1;}}
  .card.committed-rej .card-img{{filter:grayscale(80%);}}
  .warn-wrap{{position:absolute;top:8px;right:8px;display:flex;flex-direction:column;gap:4px;z-index:10;pointer-events:none;}}
  .warn-badge{{background:linear-gradient(90deg,#FFC107,#FF9800);color:#313133;font-size:9px;font-weight:800;padding:3px 8px;border-radius:9999px;box-shadow:0 2px 6px rgba(255,152,0,.3);animation:pulse 2s infinite;}}
  @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.85}}}}
  .price-badge{{position:absolute;top:8px;left:8px;background:rgba(255,136,0,.95);color:#fff;font-size:10px;font-weight:800;padding:3px 8px;border-radius:9999px;z-index:10;pointer-events:none;box-shadow:0 2px 6px rgba(0,0,0,.2);}}
  .meta{{font-size:11px;margin-top:8px;line-height:1.4;}}
  .meta .nm{{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:help;}}
  .meta .br{{color:{O};font-weight:700;margin:2px 0;}}
  .meta .ct{{color:#666;font-size:10px;word-break:break-word;}}
  .meta .sl{{color:#999;font-size:9px;margin-top:4px;border-top:1px dashed #eee;padding-top:4px;cursor:help;}}
  .acts{{display:flex;gap:4px;margin-top:8px;}}
  .act-btn{{flex:1;padding:6px;font-size:11px;border:none;border-radius:4px;cursor:pointer;font-weight:700;color:#fff;background:{O};}}
  .act-more{{flex:1;font-size:11px;border:1px solid #ccc;border-radius:4px;outline:none;cursor:pointer;background:#fff;}}
  .zoom-btn{{position:absolute;bottom:6px;right:6px;width:22px;height:22px;background:rgba(0,0,0,0.4);color:#fff;border-radius:4px;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:25;border:none;transition:background .2s;}}
  .zoom-btn:hover{{background:rgba(0,0,0,0.7);}}
  .tick{{position:absolute;bottom:6px;left:6px;width:22px;height:22px;border-radius:50%;background:rgba(0,0,0,.18);display:flex;align-items:center;justify-content:center;color:transparent;font-size:13px;font-weight:900;pointer-events:none;z-index:10;}}
  .card.selected .tick{{background:{O};color:#fff;}}
  .rej-overlay{{display:none;position:absolute;inset:0;background:rgba(255,255,255,.90);border-radius:8px;flex-direction:column;align-items:center;justify-content:center;z-index:20;gap:8px;padding:12px;text-align:center;}}
  .card.committed-rej .rej-overlay{{display:flex;}}
  .card.staged-rej .rej-overlay.staged{{display:flex;background:rgba(211,47,47,0.85);}}
  .card.committed-rej .rej-badge{{background:{R};color:#fff;padding:6px 12px;border-radius:6px;font-size:15px;font-weight:800;letter-spacing:0.5px;}}
  .undo-btn{{margin-top:8px;padding:6px 14px;background:#313133;color:#fff;border:none;border-radius:4px;font-size:11px;font-weight:bold;cursor:pointer;}}
  .undo-btn:hover{{background:#000;}}
  #zoom-tooltip {{display: none;position: absolute;z-index: 100000;background: #fff;padding: 10px;border-radius: 8px;box-shadow: 0 10px 40px rgba(0,0,0,0.4);border: 1px solid #ccc;width: 360px;height: 360px;transition: opacity 0.2s ease;}}
  #tooltip-img {{width: 100%;height: 100%;object-fit: contain;display: block;}}
  .tooltip-close {{position: absolute;top: -12px;right: -12px;background: #333;color: #fff;border-radius: 50%;width: 28px;height: 28px;border: 2px solid #fff;cursor: pointer;font-size: 16px;line-height: 1;display: flex;align-items: center;justify-content: center;box-shadow: 0 2px 6px rgba(0,0,0,0.3);}}
  .tooltip-close:hover {{background: #000;}}
  #prefetch-status{{font-size:10px;color:#aaa;text-align:right;padding:4px 8px;margin-top:8px;}}
  .debug-hud{{position:absolute;inset:0;background:rgba(0,0,0,0.85);color:#0f0;font-family:monospace;font-size:9px;padding:5px;display:none;word-break:break-all;z-index:100;}}
</style>
</head>
<body>
<div class="ctrl-bar"> ... (same as before) ... </div>
<div class="grid" id="card-grid"></div>
<div class="ctrl-bar bottom-bar"> ... (same as before) ... </div>

<div id="zoom-tooltip">
  <img id="tooltip-img" alt="Zoomed product" referrerpolicy="no-referrer">
  <button class="tooltip-close" onclick="closeZoom()" title="Close">×</button>
</div>

<div id="prefetch-status"></div>
<div id="prefetch-container" style="display:none;position:absolute;width:1px;height:1px;overflow:hidden;"></div>

<script>
// ... (all helper functions onImgLoad, lazy loading, renderCard, etc. are the same as in your second file) ...

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

    bridge.focus({{preventScroll: true}});           // ← REQUIRED for Streamlit to detect change
    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(par.HTMLInputElement.prototype, 'value').set;
    nativeInputValueSetter.call(bridge, msg);
    bridge.dispatchEvent(new par.Event('input', {{bubbles: true}}));
    bridge.dispatchEvent(new par.KeyboardEvent('keydown', {{bubbles:true,cancelable:true,key:'Enter',keyCode:13}}));
    bridge.dispatchEvent(new par.KeyboardEvent('keyup',   {{bubbles:true,cancelable:true,key:'Enter',keyCode:13}}));
  }} catch(ex) {{ console.error('jtbridge error:', ex); }}
}}

function scrollToTop() {{ /* same as before */ }}

window.undoReject = function(sid) {{
  try {{
    var par = window.parent.document;
    var scrollable = par.querySelector('[data-testid="stModal"] [data-testid="stDialogScrollContent"]') ||
                     par.querySelector('[data-testid="stModal"] > div > div > div:last-child') ||
                     par.querySelector('[role="dialog"]');
    if (scrollable && scrollable.scrollTop > 0) {{
      window.parent.sessionStorage.setItem('__grid_scroll__', scrollable.scrollTop);
    }}
  }} catch(e) {{}}
  delete COMMITTED[sid];
  replaceCard(sid);
  updateSelCount();
  sendMsg('undo', {{[sid]: true}});
}};

window.doBatchReject = function(pos) {{
  // Save scroll position before rerun
  try {{
    var par = window.parent.document;
    var scrollable = par.querySelector('[data-testid="stModal"] [data-testid="stDialogScrollContent"]') ||
                     par.querySelector('[data-testid="stModal"] > div > div > div:last-child') ||
                     par.querySelector('[role="dialog"]');
    if (scrollable && scrollable.scrollTop > 0) {{
      window.parent.sessionStorage.setItem('__grid_scroll__', scrollable.scrollTop);
    }}
  }} catch(e) {{}}

  var selectId = pos === 'top' ? 'batch-reason-top' : 'batch-reason-bottom';
  var br = document.getElementById(selectId).value;
  var payload = {{}}, count = 0;
  for (var s in staged) {{ payload[s] = staged[s]; count++; }}
  for (var s in selected) {{ payload[s] = br; count++; }}
  if (count === 0) return;

  for (var s in payload) {{ COMMITTED[s] = payload[s]; delete selected[s]; delete staged[s]; }}

  renderAll();
  updateSelCount();
  sendMsg('reject', payload);
}};

// ... rest of your original script (renderAll, toggleSelect, etc.) remains exactly the same ...
</script>
</body>
</html>"""
