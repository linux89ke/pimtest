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
    import json
    import pandas as pd
    from constants import JUMIA_COLORS
    from data_utils import format_local_price

    O = JUMIA_COLORS["primary_orange"]
    G = JUMIA_COLORS["success_green"]
    R = JUMIA_COLORS["jumia_red"]

    committed_json = json.dumps(rejected_state)
    prefetch_json = json.dumps(prefetch_urls or [])

    html_dir = "rtl" if st.session_state.get('ui_lang') == "ar" else "ltr"
    rejected_label = str("REJECTED").upper()  # replace with _t if needed

    # Inline SVG placeholder
    _NO_IMAGE_SVG = (
        "data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' width='150' height='150'>"
        "<rect width='150' height='150' fill='%23f0f0f0'/>"
        "<text x='75' y='75' text-anchor='middle' dominant-baseline='central' "
        "font-size='12' font-family='sans-serif' fill='%23999'>No Image</text>"
        "</svg>"
    )

    cards_data = []
    for _, row in page_data.iterrows():
        sid = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip()
        img_url = img_url.replace("http://", "https://", 1)
        if not img_url.startswith("https"):
            img_url = ""  # fallback handled in JS

        # Safe low-res version (we just keep same image for now if no special low-res)
        low_res_url = img_url

        sale_p = row.get("GLOBAL_SALE_PRICE")
        reg_p = row.get("GLOBAL_PRICE")
        usd_val = sale_p if pd.notna(sale_p) and str(sale_p).strip() != "" else reg_p
        price_str = (
            format_local_price(usd_val, country) if pd.notna(usd_val) else ""
        )

        cards_data.append({
            "sid": sid,
            "img": img_url,
            "low_res": low_res_url,
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
.grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}
.card{{border:2px solid #e0e0e0;border-radius:8px;padding:10px;background:#fff;position:relative;transition:border-color .15s,box-shadow .15s;z-index:1;}}
.card.selected{{border-color:{G};box-shadow:0 0 0 3px rgba(76,175,80,.2);background:rgba(76,175,80,.04);}}
.card.staged-rej{{border-color:{R};box-shadow:0 0 0 3px rgba(231,60,23,.2);background:rgba(231,60,23,.04);}}
.card.committed-rej{{border-color:#bbb;opacity:.6;}}
.card-img-wrap{{position:relative;cursor:pointer;border-radius:6px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;height:180px;overflow:hidden;}}
.card-img-wrap::before{{content:'';position:absolute;inset:0;background:linear-gradient(90deg,#f0f0f0 25%,#e0e0e0 50%,#f0f0f0 75%);background-size:200% 100%;animation:shimmer 1.4s infinite;z-index:1;border-radius:6px;}}
.card-img-wrap.img-loaded::before{{display:none;}}
@keyframes shimmer{{0%{{background-position:200% 0}}100%{{background-position:-200% 0}}}}
.card-img{{width:100%;height:180px;object-fit:contain;border-radius:6px;display:block;position:relative;z-index:2;opacity:0;transition:opacity .25s ease,transform .2s ease-out,box-shadow .2s ease-out;}}
.card-img.img-loaded{{opacity:1;}}
.card-img.locally-zoomed{{transform:scale(2.5);box-shadow:0 15px 50px rgba(0,0,0,0.6);border:2px solid {O};position:relative;z-index:9999;border-radius:8px;}}
.tooltip{{position:absolute;bottom:100%;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.75);color:#fff;font-size:10px;padding:4px 6px;border-radius:4px;white-space:nowrap;pointer-events:none;opacity:0;transition:opacity .2s;}}
.card-img-wrap:hover .tooltip{{opacity:1;}}
.warn-badge{{background:rgba(255,193,7,.95);color:#313133;font-size:9px;font-weight:800;padding:3px 7px;border-radius:10px;animation:fadeIn .3s;}}
@keyframes fadeIn{{0%{{opacity:0}}100%{{opacity:1}}}}
.price-badge{{position:absolute;top:6px;left:6px;background:rgba(76,175,80,.95);color:#fff;font-size:10px;font-weight:800;padding:3px 7px;border-radius:10px;z-index:5;pointer-events:none;}}
</style>
</head>
<body>
<div class="grid" id="card-grid"></div>
<script>
const CARDS = {cards_json};
const COMMITTED = {committed_json};
const PREFETCH_URLS = {prefetch_json};
const NO_IMAGE = "{_NO_IMAGE_SVG}";
let selected = {{}};
let staged = {{}};

function escapeHtml(u){{return(u||"").toString().replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;");}}

function loadHighRes(img, hiUrl) {{
    if (!hiUrl) return;
    const hi = new Image();
    hi.src = hiUrl;
    hi.onload = () => {{ img.src = hi.src; img.classList.add('img-loaded'); }};
    hi.onerror = () => {{ img.src = NO_IMAGE; img.classList.add('img-loaded'); }};
}}

function renderCard(card) {{
    const sid = card.sid;
    const isCommitted = sid in COMMITTED;
    const isStaged = sid in staged;
    const isSelected = !isCommitted && !isStaged && (sid in selected);
    const cls = 'card' + (isCommitted?' committed-rej':isStaged?' staged-rej':isSelected?' selected':'');
    const low = card.low_res || card.img || NO_IMAGE;
    const hi = card.img || NO_IMAGE;
    const warnHtml = (card.warnings||[]).map(w=>`<span class="warn-badge">${{escapeHtml(w)}}</span>`).join('');
    const priceHtml = card.price?`<div class="price-badge">${{escapeHtml(card.price)}}</div>`:'';
    const tooltipHtml = `<div class="tooltip">${{escapeHtml(card.name)}} - ${{escapeHtml(card.seller)}}</div>`;
    return `<div class="${{cls}}" id="card-${{sid}}"><div class="card-img-wrap" onclick="toggleSelect('${{sid}}',event)">${{tooltipHtml}}${{priceHtml}}${{warnHtml}}<img class="card-img" src="${{escapeHtml(low)}}" onload="loadHighRes(this,'${{escapeHtml(hi)}}')" onerror="this.src=NO_IMAGE;this.classList.add('img-loaded');" /></div></div>`;
}}

function renderAll() {{
    document.getElementById('card-grid').innerHTML = CARDS.map(renderCard).join('');
}}

function toggleSelect(sid,e) {{
    if(sid in selected) delete selected[sid];
    else selected[sid]=true;
    renderAll();
}}

renderAll();
</script>
</body>
</html>
"""


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
