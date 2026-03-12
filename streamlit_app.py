['PRODUCT_SET_SID'].astype(str).str.strip().isin(processed)].iterrows():
        sid = str(r['PRODUCT_SET_SID']).strip()
        if sid not in processed:
            rows.append({'ProductSetSid': sid, 'ParentSKU': r.get('PARENTSKU', ''), 'Status': 'Approved', 'Reason': "", 'Comment': "", 'FLAG': "", 'SellerName': r.get('SELLER_NAME', '')})
            processed.add(sid)
    final_df = pd.DataFrame(rows)
    for c in["ProductSetSid", "ParentSKU", "Status", "Reason", "Comment", "FLAG", "SellerName"]:
        if c not in final_df.columns: final_df[c] = ""
    return country_validator.ensure_status_column(final_df), results

@st.cache_data(show_spinner=False, ttl=3600)
def cached_validate_products(data_hash: str, _data: pd.DataFrame, _support_files: Dict, country_code: str, data_has_warranty_cols: bool):
    country_name = next((k for k, v in CountryValidator.COUNTRY_CONFIG.items() if v['code'] == country_code), "Kenya")
    cv = CountryValidator(country_name)
    return validate_products(_data, _support_files, cv, data_has_warranty_cols)

# -------------------------------------------------
# EXPORTS UTILITIES
# -------------------------------------------------
def to_excel_base(df, sheet, cols, writer, format_rules=False):
    df_p = df.copy()
    for c in cols:
        if c not in df_p.columns: df_p[c] = pd.NA
    df_to_write = df_p[[c for c in cols if c in df_p.columns]]
    df_to_write.to_excel(writer, index=False, sheet_name=sheet)
    if format_rules and 'Status' in df_to_write.columns:
        wb = writer.book
        ws = writer.sheets[sheet]
        rf = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        gf = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        idx = df_to_write.columns.get_loc('Status')
        ws.conditional_format(1, idx, len(df_to_write), idx, {'type': 'cell', 'criteria': 'equal', 'value': '"Rejected"', 'format': rf})
        ws.conditional_format(1, idx, len(df_to_write), idx, {'type': 'cell', 'criteria': 'equal', 'value': '"Approved"', 'format': gf})

def write_excel_single(df, sheet_name, cols, auxiliary_df=None, aux_sheet_name=None, aux_cols=None, format_status=False, full_data_stats=False):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        to_excel_base(df, sheet_name, cols, writer, format_rules=format_status)
        if auxiliary_df is not None and not auxiliary_df.empty: to_excel_base(auxiliary_df, aux_sheet_name, aux_cols, writer)
        if full_data_stats and 'SELLER_NAME' in df.columns and 'Status' in df.columns:
            ws = writer.book.add_worksheet('Sellers Data')
            fmt = writer.book.add_format({'bold': True, 'bg_color': '#E6F0FA', 'border': 1, 'align': 'center'})
            df['Rejected_Count'] = (df['Status'] == 'Rejected').astype(int)
            df['Approved_Count'] = (df['Status'] == 'Approved').astype(int)
            summ = df.groupby('SELLER_NAME').agg(Rejected=('Rejected_Count', 'sum'), Approved=('Approved_Count', 'sum')).reset_index().sort_values('Rejected', ascending=False)
            summ.insert(0, 'Rank', range(1, len(summ) + 1))
            ws.write(0, 0, "Sellers Summary (This File)", fmt)
            summ.to_excel(writer, sheet_name='Sellers Data', startrow=1, index=False)
    output.seek(0)
    return output

def generate_smart_export(df, filename_prefix, export_type='simple', auxiliary_df=None):
    cols = FULL_DATA_COLS + [c for c in ["Status", "Reason", "Comment", "FLAG", "SellerName"] if c not in FULL_DATA_COLS] if export_type == 'full' else PRODUCTSETS_COLS
    if len(df) <= SPLIT_LIMIT:
        data = write_excel_single(df, "ProductSets", cols, auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS, True, export_type == 'full')
        return data, f"{filename_prefix}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        zb = BytesIO()
        with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(0, len(df), SPLIT_LIMIT):
                chunk = df.iloc[i : i + SPLIT_LIMIT]
                excel_data = write_excel_single(chunk, "ProductSets", cols, auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS, True, export_type == 'full')
                zf.writestr(f"{filename_prefix}_Part_{(i//SPLIT_LIMIT)+1}.xlsx", excel_data.getvalue())
        zb.seek(0)
        return zb, f"{filename_prefix}.zip", "application/zip"

def prepare_full_data_merged(data_df, final_report_df):
    try:
        d_cp, r_cp = data_df.copy(), final_report_df.copy()
        d_cp['PRODUCT_SET_SID'] = d_cp['PRODUCT_SET_SID'].astype(str).str.strip()
        r_cp['ProductSetSid'] = r_cp['ProductSetSid'].astype(str).str.strip()
        merged = pd.merge(d_cp, r_cp[["ProductSetSid", "Status", "Reason", "Comment", "FLAG", "SellerName"]], left_on="PRODUCT_SET_SID", right_on="ProductSetSid", how='left')
        if 'ProductSetSid' in merged.columns: merged.drop(columns=['ProductSetSid'], inplace=True)
        return merged
    except Exception: return pd.DataFrame()

# -------------------------------------------------
# UTILITIES FOR BRIDGE & DATA MUTATION
# -------------------------------------------------
def apply_rejection(sids: list, reason_code: str, comment: str, flag_name: str):
    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(sids),['Status', 'Reason', 'Comment', 'FLAG']] =['Rejected', reason_code, comment, flag_name]
    st.session_state.exports_cache.clear()
    st.session_state.display_df_cache.clear()

def restore_single_item(sid):
    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid,['Status', 'Reason', 'Comment', 'FLAG']] =['Approved', '', '', 'Approved by User']
    st.session_state.pop(f"quick_rej_{sid}", None)
    st.session_state.pop(f"quick_rej_reason_{sid}", None)
    st.session_state.exports_cache.clear()
    st.session_state.display_df_cache.clear()
    st.session_state.main_toasts.append("Restored item to previous state!")

REASON_MAP = {
    "REJECT_POOR_IMAGE": "Poor images",
    "REJECT_WRONG_CAT": "Wrong Category",
    "REJECT_FAKE": "Suspected Fake product",
    "REJECT_BRAND": "Restricted brands",
    "REJECT_PROHIBITED": "Prohibited products",
    "REJECT_COLOR": "Missing COLOR",
    "REJECT_WRONG_BRAND": "Generic branded products with genuine brands",
    "OTHER_CUSTOM": "Other Reason (Custom)"
}

# -------------------------------------------------
# HTML GRID BUILDER
# -------------------------------------------------
def build_fast_grid_html(
    page_data: pd.DataFrame,
    flags_mapping: dict,
    country: str,
    page_warnings: dict,
    rejected_state: dict,
    cols_per_row: int
) -> str:
    O = JUMIA_COLORS["primary_orange"]
    G = JUMIA_COLORS["success_green"]
    R = JUMIA_COLORS["jumia_red"]

    committed_json = json.dumps(rejected_state)

    cards_data =[]
    for _, row in page_data.iterrows():
        sid = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip()
        if not img_url.startswith("http"):
            img_url = "https://via.placeholder.com/150?text=No+Image"
        cards_data.append({
            "sid":      sid,
            "img":      img_url,
            "name":     str(row.get("NAME", "")),
            "brand":    str(row.get("BRAND", "Unknown Brand")),
            "cat":      str(row.get("CATEGORY", "Unknown Category")),
            "seller":   str(row.get("SELLER_NAME", "Unknown Seller")),
            "warnings": page_warnings.get(sid,[]),
        })
    cards_json = json.dumps(cards_data)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;font-family:sans-serif;}}
  body{{background:#f5f5f5;padding:8px;}}
  .info-bar{{
    display:flex; align-items:center; justify-content:space-between;
    gap:12px; padding:10px 14px; background:#fff;
    border:1px solid #e0e0e0; border-radius:8px; margin-bottom:12px;
    font-size:12px; color:#555;
    position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }}
  .sel-count{{font-weight:700;color:{O};}}
  .grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}
  .card{{border:2px solid #e0e0e0;border-radius:8px;padding:10px;background:#fff;
         position:relative;transition:border-color .15s,box-shadow .15s;}}
  .card.selected{{border-color:{G};box-shadow:0 0 0 3px rgba(76,175,80,.2);
                  background:rgba(76,175,80,.04);}}
  .card.committed-rej{{border-color:#bbb;opacity:.5;}}
  .card-img-wrap{{position:relative;cursor:pointer;}}
  .card-img{{width:100%;aspect-ratio:1;object-fit:contain;border-radius:6px;display:block;}}
  .card.committed-rej .card-img{{filter:grayscale(80%);}}
  .tick{{position:absolute;bottom:6px;right:6px;width:22px;height:22px;border-radius:50%;
         background:rgba(0,0,0,.18);display:flex;align-items:center;justify-content:center;
         color:transparent;font-size:13px;font-weight:900;pointer-events:none;}}
  .card.selected .tick{{background:{G};color:#fff;}}
  .warn-wrap{{position:absolute;top:6px;right:6px;display:flex;flex-direction:column;gap:3px;
              z-index:5;pointer-events:none;}}
  .warn-badge{{background:rgba(255,193,7,.95);color:#313133;font-size:9px;font-weight:800;
               padding:3px 7px;border-radius:10px;}}
  .rej-overlay{{display:none;position:absolute;inset:0;background:rgba(255,255,255,.88);
                border-radius:6px;flex-direction:column;align-items:center;
                justify-content:center;z-index:20;gap:5px;padding:8px;text-align:center;}}
  .card.committed-rej .rej-overlay{{display:flex;}}
  .rej-badge{{background:{R};color:#fff;padding:3px 10px;border-radius:10px;
              font-size:11px;font-weight:700;}}
  .rej-label{{font-size:10px;color:{R};font-weight:600;max-width:120px;}}
  .meta{{font-size:11px;margin-top:8px;line-height:1.4;}}
  .meta .nm{{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .meta .br{{color:{O};font-weight:700;margin:2px 0;}}
  .meta .ct{{color:#666;font-size:10px;}}
  .meta .sl{{color:#999;font-size:9px;margin-top:4px;border-top:1px dashed #eee;padding-top:4px;}}
  .acts{{display:flex;gap:4px;margin-top:8px;}}
  .act-btn{{flex:1;padding:6px;font-size:11px;border:none;border-radius:4px;cursor:pointer;
            font-weight:700;color:#fff;background:{O};}}
  .act-more{{flex:1;font-size:11px;border:1px solid #ccc;border-radius:4px;outline:none;
             cursor:pointer;background:#fff;}}
</style>
</head>
<body>
<div class="info-bar">
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="font-size:13px;">Click image to <b>select</b></span>
    <span class="sel-count" id="sel-count-bar">0 selected</span>
  </div>
  <div style="display:flex; gap: 8px;">
    <select id="batch-reason" style="padding: 6px; border-radius: 4px; border: 1px solid #ccc; font-size:12px; outline:none;">
      <option value="REJECT_POOR_IMAGE">Poor Image Quality</option>
      <option value="REJECT_WRONG_CAT">Wrong Category</option>
      <option value="REJECT_FAKE">Suspected Fake</option>
      <option value="REJECT_BRAND">Restricted Brand</option>
      <option value="REJECT_WRONG_BRAND">Wrong Brand</option>
      <option value="REJECT_PROHIBITED">Prohibited Product</option>
    </select>
    <button class="act-btn" onclick="doBatchReject()" style="padding: 6px 12px; font-size:12px; font-weight:bold;">✗ Batch Reject</button>
    <button class="act-more" onclick="doDeselectAll()" style="padding: 6px 12px; font-size:12px;">☐ Deselect All</button>
  </div>
</div>
<div class="grid" id="card-grid"></div>
<script>
const CARDS     = {cards_json};
const COMMITTED = {committed_json};

// Simple local state, reset whenever iframe completely reloads (like on pagination)
let selected = {{}};

function updateSelCount() {{
  const n = Object.keys(selected).length;
  document.getElementById('sel-count-bar').textContent = n + ' selected';
}}

// ── postMessage to parent ─────────────────────────────────────────────────────
function sendMsg(type, payload) {{
  window.parent.postMessage({{type: "jtGrid", action: type, payload: payload}}, "*");
}}

// ── Card rendering ────────────────────────────────────────────────────────────
function renderCard(card) {{
  const {{sid, img, name, brand, cat, seller, warnings}} = card;
  const isCommitted = sid in COMMITTED;
  const isSelected  = (sid in selected) && !isCommitted;
  
  let cls = 'card';
  if (isCommitted)     cls += ' committed-rej';
  else if (isSelected) cls += ' selected';
  
  const shortName = name.length > 38 ? name.slice(0,38)+'…' : name;
  const warnHtml  = warnings.map(w => `<span class="warn-badge">${{w}}</span>`).join('');
  const rejLabel  = isCommitted ? (COMMITTED[sid]||'').replace(/_/g,' ') : '';
  
  const rejOverlay = isCommitted ? `
    <div class="rej-overlay">
      <div class="rej-badge">REJECTED</div>
      <div class="rej-label">${{rejLabel}}</div>
    </div>` : '';
    
  const actHtml = !isCommitted ? `
    <div class="acts">
      <button class="act-btn"
        onclick="event.stopPropagation();quickReject('${{sid}}','REJECT_POOR_IMAGE')">
        Poor Img
      </button>
      <select class="act-more"
        onchange="if(this.value){{event.stopPropagation();quickReject('${{sid}}',this.value);this.value=''}}">
        <option value="">More…</option>
        <option value="REJECT_WRONG_CAT">Wrong Category</option>
        <option value="REJECT_FAKE">Fake Product</option>
        <option value="REJECT_BRAND">Restricted Brand</option>
        <option value="REJECT_PROHIBITED">Prohibited</option>
        <option value="REJECT_COLOR">Wrong Color</option>
        <option value="REJECT_WRONG_BRAND">Wrong Brand</option>
      </select>
    </div>` : '';
    
  return `<div class="${{cls}}" id="card-${{sid}}">
    <div class="card-img-wrap" onclick="toggleSelect('${{sid}}')">
      <div class="warn-wrap">${{warnHtml}}</div>
      <img class="card-img" src="${{img}}" loading="lazy"
           onerror="this.src='https://via.placeholder.com/150?text=No+Image'">
      ${{rejOverlay}}<div class="tick">✓</div>
    </div>
    <div class="meta">
      <div class="nm" title="${{name}}">${{shortName}}</div>
      <div class="br">${{brand}}</div>
      <div class="ct">${{cat}}</div>
      <div class="sl">${{seller}}</div>
    </div>${{actHtml}}</div>`;
}}

function renderAll() {{
  document.getElementById('card-grid').innerHTML = CARDS.map(renderCard).join('');
  updateSelCount();
}}

function replaceCard(sid) {{
  const el = document.getElementById('card-'+sid);
  if (!el) return;
  const card = CARDS.find(c => c.sid === sid);
  if (card) {{ const t=document.createElement('div'); t.innerHTML=renderCard(card); el.replaceWith(t.firstElementChild); }}
}}

function toggleSelect(sid) {{
  if (sid in COMMITTED) return;
  if (sid in selected) delete selected[sid];
  else selected[sid] = 'SELECTED';
  replaceCard(sid);
  updateSelCount();
}}

function doBatchReject() {{
  const reason = document.getElementById('batch-reason').value;
  const toReject = Object.keys(selected);
  if (toReject.length === 0) {{
    sendMsg('toast', {{msg: "No products selected", icon: "⚠️"}});
    return;
  }}
  
  const payload = {{}};
  toReject.forEach(sid => {{
    payload[sid] = reason;
    COMMITTED[sid] = reason;
  }});
  
  // Clear selection instantly to avoid double clicking issues
  selected = {{}};
  updateSelCount();
  toReject.forEach(sid => replaceCard(sid));

  // Send to python wrapper
  sendMsg('reject', payload);
}}

function doDeselectAll() {{
  const keys = Object.keys(selected);
  selected = {{}};
  updateSelCount();
  keys.forEach(sid => replaceCard(sid));
}}

function quickReject(sid, reasonKey) {{
  delete selected[sid];
  COMMITTED[sid] = reasonKey;
  replaceCard(sid);
  updateSelCount();
  sendMsg('reject', {{[sid]: reasonKey}});
}}

renderAll();
</script>
</body>
</html>"""

# -------------------------------------------------
# UI COMPONENTS
# -------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def analyze_image_quality_cached(url: str) -> List[str]:
    if not url or not str(url).startswith("http"): return[]
    warnings =[]
    try:
        resp = requests.get(url, timeout=2, stream=True)
        if resp.status_code == 200:
            img = Image.open(resp.raw)
            w, h = img.size
            if w < 300 or h < 300: warnings.append("Low Resolution")
            ratio = h / w if w > 0 else 1
            if ratio > 1.5: warnings.append("Tall (Screenshot?)")
            elif ratio < 0.6: warnings.append("Wide Aspect")
    except Exception: pass
    return warnings

@st.dialog("Confirm Bulk Approval")
def bulk_approve_dialog(sids_to_process, title, subset_data, data_has_warranty_cols_check, support_files, country_validator):
    st.warning(f"You are about to approve **{len(sids_to_process)}** items from `{title}`.")
    if st.button("Confirm Approval", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            data_hash = df_hash(subset_data) + country_validator.code + "_skip_" + title
            new_report, _ = cached_validate_products(data_hash, subset_data, support_files, country_validator.code, data_has_warranty_cols_check)
            msg_moved, msg_approved = {}, 0
            for sid in sids_to_process:
                new_row = new_report[new_report['ProductSetSid'] == sid]
                if new_row.empty or not str(new_row.iloc[0]['FLAG']):
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid,['Status', 'Reason', 'Comment', 'FLAG']] =['Approved', '', '', 'Approved by User']
                    msg_approved += 1
                else:
                    new_flag = str(new_row.iloc[0]['FLAG'])
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid,['Status', 'Reason', 'Comment', 'FLAG']] =['Rejected', new_row.iloc[0]['Reason'], new_row.iloc[0]['Comment'], new_flag]
                    msg_moved[new_flag] = msg_moved.get(new_flag, 0) + 1
            if msg_approved > 0: st.session_state.main_toasts.append(f"{msg_approved} items successfully Approved!")
            for flag, count in msg_moved.items(): st.session_state.main_toasts.append(f"{count} items re-flagged as: {flag}")
            st.session_state.exports_cache.clear()
            st.session_state.display_df_cache.clear()
        st.rerun()

def render_flag_expander(title, df_flagged_sids, data, data_has_warranty_cols_check, support_files, country_validator):
    cache_key = f"display_df_{title}"
    base_display_cols =['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'COLOR', 'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE', 'PARENTSKU', 'SELLER_NAME']
    current_display_cols = base_display_cols.copy()
    if title == "Wrong Variation":
        if 'COUNT_VARIATIONS' in data.columns: current_display_cols.append('COUNT_VARIATIONS')
        if 'LIST_VARIATIONS' in data.columns: current_display_cols.append('LIST_VARIATIONS')

    if cache_key not in st.session_state.display_df_cache:
        df_display = pd.merge(
            df_flagged_sids[['ProductSetSid']],
            data,
            left_on='ProductSetSid', right_on='PRODUCT_SET_SID', how='left'
        )[[c for c in current_display_cols if c in data.columns]]
        st.session_state.display_df_cache[cache_key] = df_display
    else: df_display = st.session_state.display_df_cache[cache_key]

    c1, c2 = st.columns([1, 1])
    with c1: search_term = st.text_input("Search", placeholder="Name, Brand...", key=f"s_{title}")
    with c2: seller_filter = st.multiselect("Filter by Seller", sorted(df_display['SELLER_NAME'].astype(str).unique()), key=f"f_{title}")

    df_view = df_display.copy()
    if search_term: df_view = df_view[df_view.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    if seller_filter: df_view = df_view[df_view['SELLER_NAME'].isin(seller_filter)]
    df_view = df_view.reset_index(drop=True)
    if 'NAME' in df_view.columns:
        def strip_html(text): return re.sub('<[^<]+?>', '', text) if isinstance(text, str) else text
        df_view['NAME'] = df_view['NAME'].apply(strip_html)

    event = st.dataframe(
        df_view, hide_index=True, use_container_width=True, selection_mode="multi-row", on_select="rerun",
        column_config={
            "PRODUCT_SET_SID": st.column_config.TextColumn(pinned=True),
            "NAME": st.column_config.TextColumn(pinned=True),
            "GLOBAL_SALE_PRICE": st.column_config.NumberColumn("Sale Price (USD)", format="$%.2f"),
            "GLOBAL_PRICE": st.column_config.NumberColumn("Price (USD)", format="$%.2f"),
        }, key=f"df_{title}"
    )
    raw_selected_indices = list(event.selection.rows)
    selected_indices =[i for i in raw_selected_indices if i < len(df_view)]
    st.caption(f"{len(selected_indices)} of {len(df_view)} rows selected")
    has_selection = len(selected_indices) > 0

    _fm = support_files['flags_mapping']
    _reason_options =[
        "Wrong Category", "Restricted brands", "Suspected Fake product", "Seller Not approved to sell Refurb",
        "Product Warranty", "Seller Approve to sell books", "Seller Approved to Sell Perfume", "Counterfeit Sneakers",
        "Suspected counterfeit Jerseys", "Prohibited products", "Unnecessary words in NAME", "Single-word NAME",
        "Generic BRAND Issues", "Fashion brand issues", "BRAND name repeated in NAME", "Wrong Variation",
        "Generic branded products with genuine brands", "Missing COLOR", "Missing Weight/Volume",
        "Incomplete Smartphone Name", "Duplicate product", "Poor images", "Other Reason (Custom)",
    ]

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("✓ Approve Selected", key=f"approve_sel_{title}", type="primary", use_container_width=True, disabled=not has_selection):
            sids_to_process = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
            subset = data[data['PRODUCT_SET_SID'].isin(sids_to_process)]
            bulk_approve_dialog(sids_to_process, title, subset, data_has_warranty_cols_check, support_files, country_validator)
    with btn_col2:
        with st.popover("↓ Reject As...", use_container_width=True, disabled=not has_selection):
            st.markdown("<p style='font-size:12px;font-weight:700;margin:0 0 8px 0;'>Select rejection reason:</p>", unsafe_allow_html=True)
            chosen_reason = st.selectbox("Reason", _reason_options, key=f"rej_reason_dd_{title}", label_visibility="collapsed")
            if chosen_reason == "Other Reason (Custom)":
                custom_comment = st.text_area("Custom comment", placeholder="Type your rejection reason here...", key=f"custom_comment_{title}", height=80)
                if st.button("Apply Custom Rejection", key=f"apply_custom_{title}", type="primary", use_container_width=True, disabled=not has_selection):
                    to_reject = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    final_comment = custom_comment.strip() if custom_comment.strip() else "Other Reason"
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(to_reject),['Status', 'Reason', 'Comment', 'FLAG']] =['Rejected', '1000007 - Other Reason', final_comment, 'Other Reason (Custom)']
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected with custom reason.")
                    st.session_state.exports_cache.clear()
                    st.session_state.display_df_cache.clear()
                    st.rerun()
            else:
                _rcode, _rcmt = _fm.get(chosen_reason, ('1000007 - Other Reason', chosen_reason))
                st.caption(f"Code: {_rcode[:40]}...")
                if st.button("Apply Rejection", key=f"apply_dd_{title}", type="primary", use_container_width=True, disabled=not has_selection):
                    to_reject = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(to_reject), ['Status', 'Reason', 'Comment', 'FLAG']] =['Rejected', _rcode, _rcmt, chosen_reason]
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected as '{chosen_reason}'.")
                    st.session_state.exports_cache.clear()
                    st.session_state.display_df_cache.clear()
                    st.rerun()

# ==========================================
# APP INITIALIZATION
# ==========================================
try: support_files = load_support_files_lazy()
except Exception as e: st.error(f"Failed to load configs: {e}"); st.stop()

def get_image_base64(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as img_file: return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception: return ""
    return ""

logo_base64 = get_image_base64("jumia logo.png") or get_image_base64("jumia_logo.png")
logo_html = f"<img src='data:image/png;base64,{logo_base64}' style='height: 42px; margin-right: 15px;'>" if logo_base64 else "<span class='material-symbols-outlined' style='font-size: 42px; margin-right: 15px;'>verified_user</span>"

st.markdown(f"""<div class="back-to-top" onclick="window.parent.document.querySelector('.main').scrollTo({{top: 0, behavior: 'smooth'}});" title="Back to Top"><span class="material-symbols-outlined">arrow_upward</span></div>""", unsafe_allow_html=True)
st.markdown(f"""<div style='background: linear-gradient(135deg, {JUMIA_COLORS['primary_orange']}, {JUMIA_COLORS['secondary_orange']}); padding: 25px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(246, 139, 30, 0.3);'><h1 style='color: white; margin: 0; font-size: 36px; display: flex; align-items: center;'>{logo_html}Product Validation Tool</h1></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("System Status")
    if st.button("🔄 Clear Cache & Reload Data", use_container_width=True, type="secondary", help="Forces a reload of all support rules from local files."):
        st.cache_data.clear()
        st.session_state.display_df_cache = {}
        st.rerun()
    st.markdown("---")
    st.header("Display Settings")
    new_mode = "wide" if "Wide" in st.radio("Layout Mode", ["Centered", "Wide"], index=1 if st.session_state.layout_mode == "wide" else 0) else "centered"
    if new_mode != st.session_state.layout_mode: st.session_state.layout_mode = new_mode; st.rerun()

# ==========================================
# SECTION 1: UPLOAD & VALIDATION
# ==========================================
st.header(":material/upload_file: Upload Files", anchor=False)

current_country = st.session_state.get('selected_country', 'Kenya')
country_choice = st.segmented_control("Country",["Kenya", "Uganda", "Nigeria", "Ghana", "Morocco"], default=current_country)

if country_choice: st.session_state.selected_country = country_choice
else: country_choice = current_country

country_validator = CountryValidator(st.session_state.selected_country)

uploaded_files = st.file_uploader("Upload CSV or XLSX files", type=['csv', 'xlsx'], accept_multiple_files=True, key="daily_files")

if uploaded_files:
    current_file_signature = sorted([f.name + str(f.size) for f in uploaded_files])
    process_signature = str(current_file_signature) + f"_{country_validator.code}"
else: process_signature = "empty"

if st.session_state.get('last_processed_files') != process_signature:
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.post_qc_summary = pd.DataFrame()
    st.session_state.post_qc_results = {}
    st.session_state.post_qc_data = pd.DataFrame()
    st.session_state.file_mode = None
    st.session_state.intersection_sids = set()
    st.session_state.intersection_count = 0
    st.session_state.grid_page = 0
    st.session_state.exports_cache = {}
    st.session_state.display_df_cache = {}
    # Reset all JS-bridge counters on new file load
    st.session_state.desel_counter = 0
    st.session_state.batch_counter = 0
    st.session_state.clear_counter = 0
    st.session_state.ls_processed_flag = False
    st.session_state.ls_read_trigger = 0
    st.session_state.grid_bridge = ""
    keys_to_delete =[k for k in st.session_state.keys() if k.startswith(("quick_rej_", "grid_chk_", "toast_"))]
    for k in keys_to_delete: del st.session_state[k]

    if process_signature == "empty": st.session_state.last_processed_files = "empty"
    else:
        try:
            all_dfs =[]
            file_sids_sets = []
            detected_modes =[]
            for uf in uploaded_files:
                uf.seek(0)
                if uf.name.endswith('.xlsx'): raw_data = pd.read_excel(uf, engine='openpyxl', dtype=str)
                else:
                    try:
                        raw_data = pd.read_csv(uf, dtype=str)
                        if len(raw_data.columns) <= 1:
                            uf.seek(0)
                            raw_data = pd.read_csv(uf, sep=';', encoding='ISO-8859-1', dtype=str)
                    except:
                        uf.seek(0)
                        raw_data = pd.read_csv(uf, sep=';', encoding='ISO-8859-1', dtype=str)
                detected_modes.append(detect_file_type(raw_data))
                all_dfs.append(raw_data)

            file_mode = detected_modes[0] if detected_modes else 'pre_qc'
            st.session_state.file_mode = file_mode

            if file_mode == 'post_qc':
                norm_dfs =[normalize_post_qc(df) for df in all_dfs]
                merged = pd.concat(norm_dfs, ignore_index=True)
                merged_dedup = merged.drop_duplicates(subset=['PRODUCT_SET_SID'], keep='first')
                with st.spinner("Running Post-QC checks..."):
                    summary_df, results = run_post_qc_checks(merged_dedup, support_files)
                st.session_state.post_qc_summary = summary_df
                st.session_state.post_qc_results = results
                st.session_state.post_qc_data = merged_dedup
                st.session_state.last_processed_files = process_signature
            else:
                std_dfs =[]
                for raw_data in all_dfs:
                    std_data = standardize_input_data(raw_data)
                    if 'PRODUCT_SET_SID' in std_data.columns:
                        std_data['PRODUCT_SET_SID'] = std_data['PRODUCT_SET_SID'].astype(str).str.strip()
                        file_sids_sets.append(set(std_data['PRODUCT_SET_SID'].unique()))
                    std_dfs.append(std_data)
                merged_data = pd.concat(std_dfs, ignore_index=True)
                if len(file_sids_sets) > 1: st.session_state.intersection_sids = set.intersection(*file_sids_sets)
                else: st.session_state.intersection_sids = set()
                st.session_state.intersection_count = len(st.session_state.intersection_sids)
                data_prop = propagate_metadata(merged_data)
                is_valid, errors = validate_input_schema(data_prop)
                if is_valid:
                    data_filtered, det_names = filter_by_country(data_prop, country_validator)
                    if data_filtered.empty:
                        st.error(f"No {country_validator.country} products found. Detected countries: {', '.join(det_names) if det_names else 'None'}", icon=":material/error:")
                        st.stop()
                    actual_counts = data_filtered.groupby('PRODUCT_SET_SID')['PRODUCT_SET_SID'].transform('count')
                    if 'COUNT_VARIATIONS' in data_filtered.columns:
                        file_counts = pd.to_numeric(data_filtered['COUNT_VARIATIONS'], errors='coerce').fillna(1)
                        data_filtered['COUNT_VARIATIONS'] = actual_counts.combine(file_counts, max)
                    else: data_filtered['COUNT_VARIATIONS'] = actual_counts
                    data = data_filtered.drop_duplicates(subset=['PRODUCT_SET_SID'], keep='first')
                    if '_IS_MULTI_COUNTRY' not in data.columns: data['_IS_MULTI_COUNTRY'] = False
                    data_has_warranty = all(c in data.columns for c in['PRODUCT_WARRANTY', 'WARRANTY_DURATION'])
                    for c in['NAME', 'BRAND', 'COLOR', 'SELLER_NAME', 'CATEGORY_CODE', 'LIST_VARIATIONS']:
                        if c in data.columns: data[c] = data[c].astype(str).fillna('')
                    if 'COLOR_FAMILY' not in data.columns: data['COLOR_FAMILY'] = ""

                    data_hash = df_hash(data) + country_validator.code
                    final_report, _ = cached_validate_products(data_hash, data, support_files, country_validator.code, data_has_warranty)

                    st.session_state.final_report = final_report
                    st.session_state.all_data_map = data
                    st.session_state.last_processed_files = process_signature
                else:
                    for e in errors: st.error(e)
                    st.session_state.last_processed_files = "error"
        except Exception as e:
            st.error(f"Processing error: {e}")
            st.code(traceback.format_exc())
            st.session_state.last_processed_files = "error"

# ==========================================
# POST-QC RESULTS SECTION
# ==========================================
if uploaded_files and st.session_state.file_mode == 'post_qc' and not st.session_state.post_qc_summary.empty:
    render_post_qc_section(support_files)

# ==========================================
# RESULTS SECTION
# ==========================================
if uploaded_files and not st.session_state.final_report.empty and st.session_state.file_mode != 'post_qc':
    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    app_df = fr[fr['Status'] == 'Approved']
    rej_df = fr[fr['Status'] == 'Rejected']

    st.header(":material/bar_chart: Validation Results", anchor=False)
    with st.container(border=True):
        cols = st.columns(5 if st.session_state.layout_mode == "wide" else 3)
        is_nigeria = st.session_state.get('selected_country') == 'Nigeria'
        multi_count = int(data['_IS_MULTI_COUNTRY'].sum()) if '_IS_MULTI_COUNTRY' in data.columns else 0

        metrics_config =[
            ("Total Products", len(data), JUMIA_COLORS['dark_gray']),
            ("Approved", len(app_df), JUMIA_COLORS['success_green']),
            ("Rejected", len(rej_df), JUMIA_COLORS['jumia_red']),
            ("Rejection Rate", f"{(len(rej_df)/len(data)*100) if len(data)>0 else 0:.1f}%", JUMIA_COLORS['primary_orange']),
            ("Multi-Country SKUs" if is_nigeria else "Common SKUs", multi_count if is_nigeria else st.session_state.intersection_count, JUMIA_COLORS['warning_yellow'] if is_nigeria else JUMIA_COLORS['medium_gray']),
        ]
        for i, (label, value, color) in enumerate(metrics_config):
            with cols[i % len(cols)]:
                st.markdown(f"""<div class="metric-card-inner" style='text-align: center; padding: 18px 12px; background: {JUMIA_COLORS['light_gray']}; border-radius: 8px; border-left: 4px solid {color};'><div class="metric-card-value" style='font-size: 28px; font-weight: 700; color: {color}; margin-bottom: 4px;'>{value}</div><div class="metric-card-label" style='font-size: 11px; color: {JUMIA_COLORS['medium_gray']}; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;'>{label}</div></div>""", unsafe_allow_html=True)

    st.subheader(":material/flag: Flags Breakdown", anchor=False)
    if not rej_df.empty:
        for title in rej_df['FLAG'].unique():
            df_flagged = rej_df[rej_df['FLAG'] == title]
            with st.expander(f"{title} ({len(df_flagged)})"):
                render_flag_expander(title, df_flagged, data, all(c in data.columns for c in['PRODUCT_WARRANTY', 'WARRANTY_DURATION']), support_files, country_validator)
    else: st.success("All products passed validation — no rejections found.")


# ==========================================
# SECTION 2: MANUAL IMAGE REVIEW
# ==========================================

@st.fragment
def render_image_grid():
    if st.session_state.final_report.empty or st.session_state.file_mode == "post_qc":
        return

    st.markdown("---")
    st.header(":material/pageview: Manual Image & Category Review", anchor=False)

    if "grid_msg_counter" not in st.session_state:
        st.session_state.grid_msg_counter = 0
    if "grid_bridge" not in st.session_state:
        st.session_state.grid_bridge = ""

    # ── Hidden Form Submission Trigger ────────────────────────────────────────
    # We create an invisible Streamlit button. The injected Javascript will 'click' this
    # hidden button after setting the text input value, instantly forcing Streamlit to read it.
    st.markdown("<div style='display: none;'>", unsafe_allow_html=True)
    if st.button("hidden_submit", key=f"bridge_submit_{st.session_state.grid_msg_counter}"):
        pass # The page reruns automatically because a button was clicked.
    st.markdown("</div>", unsafe_allow_html=True)

    # ── postMessage listener injected into PARENT page ────────────────────────
    components.html("""
    <script>
    (function() {
      // Bind to the parent window since that's where the grid posts messages to!
      var target = window.parent || window;
      if (target._jtListenerActive) return;
      target._jtListenerActive = true;
      target.addEventListener("message", function(e) {
        if (!e.data || e.data.type !== "jtGrid") return;
        var action  = e.data.action;
        var payload = e.data.payload;
        
        // Find the hidden Streamlit text input and update it
        var inputs = target.document.querySelectorAll('input[type="text"]');
        var bridge = null;
        inputs.forEach(function(inp) {
          var label = inp.getAttribute("aria-label");
          // Check for exact or partial match to avoid Streamlit label modifications
          if (label && label.indexOf("jtbridge") !== -1) bridge = inp;
        });
        if (!bridge) return;
        
        var msg = JSON.stringify({action: action, payload: payload});
        var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        setter.call(bridge, msg);
        bridge.dispatchEvent(new Event('input', {bubbles: true}));

        // Find the hidden button we created and click it to force Streamlit to process immediately
        var buttons = target.document.querySelectorAll('button');
        buttons.forEach(function(btn) {
          if (btn.innerText && btn.innerText.includes("hidden_submit")) {
            btn.click();
          }
        });

      });
    })();
    </script>
    """, height=0)

    # ── Hidden bridge input — Streamlit reads this natively ───────────────────
    bridge_val = st.text_input(
        "jtbridge",
        value=st.session_state.grid_bridge,
        key=f"grid_bridge_input_{st.session_state.grid_msg_counter}",
        label_visibility="collapsed",
    )

    # ── Process message from iframe ───────────────────────────────────────────
    if bridge_val and bridge_val not in ("", "0"):
        try:
            msg = json.loads(bridge_val)
            action  = msg.get("action", "")
            payload = msg.get("payload", {})
            
            if action == "toast" and isinstance(payload, dict):
                st.toast(payload.get("msg", "Action completed"), icon=payload.get("icon", "ℹ️"))
                st.session_state.grid_msg_counter += 1
                st.rerun(scope="fragment")
                
            elif action == "reject" and isinstance(payload, dict) and payload:
                reason_groups: dict[str, list] = {}
                for sid, reason_key in payload.items():
                    reason_groups.setdefault(reason_key,[]).append(sid)
                
                total = 0
                for reason_key, sids in reason_groups.items():
                    flag_name = REASON_MAP.get(reason_key, "Other Reason (Custom)")
                    code, cmt = support_files["flags_mapping"].get(
                        flag_name, ("1000007 - Other Reason", "Manual rejection")
                    )
                    apply_rejection(sids, code, cmt, flag_name)
                    for s in sids:
                        st.session_state[f"quick_rej_{s}"] = True
                        st.session_state[f"quick_rej_reason_{s}"] = flag_name
                    total += len(sids)
                
                st.session_state.main_toasts.append((f"Rejected {total} product(s)", "✅"))
                st.session_state.exports_cache.clear()
                st.session_state.display_df_cache.clear()
                st.session_state.grid_msg_counter += 1
                st.rerun(scope="app")
        except Exception as e:
            logger.error(f"Grid bridge parse error: {e}")
        finally:
            st.session_state.grid_bridge = ""

    # ── Build UI ──────────────────────────────────────────────────────────────
    fr   = st.session_state.final_report
    data = st.session_state.all_data_map

    committed_rej_sids = {
        k.replace("quick_rej_", "")
        for k in st.session_state.keys()
        if k.startswith("quick_rej_") and "reason" not in k
    }
    mask = (fr["Status"] == "Approved") | (fr["ProductSetSid"].isin(committed_rej_sids))
    valid_grid_df = fr[mask]

    c1, c2, c3 = st.columns([1.5, 1.5, 2])
    with c1: search_n  = st.text_input("Search by Name", placeholder="Product name…")
    with c2: search_sc = st.text_input("Search by Seller/Category", placeholder="Seller or Category…")
    with c3:
        st.session_state.grid_items_per_page = st.select_slider(
            "Items per page", options=[20, 50, 100, 200],
            value=st.session_state.grid_items_per_page,
        )

    available_cols =[c for c in GRID_COLS if c in data.columns]
    review_data = pd.merge(
        valid_grid_df[["ProductSetSid"]],
        data[available_cols],
        left_on="ProductSetSid", right_on="PRODUCT_SET_SID", how="left",
    )
    if search_n:
        review_data = review_data[review_data["NAME"].astype(str).str.contains(search_n, case=False, na=False)]
    if search_sc:
        mc = (review_data["CATEGORY"].astype(str).str.contains(search_sc, case=False, na=False)
              if "CATEGORY" in review_data.columns else pd.Series(False, index=review_data.index))
        ms = review_data["SELLER_NAME"].astype(str).str.contains(search_sc, case=False, na=False)
        review_data = review_data[mc | ms]

    ipp         = st.session_state.grid_items_per_page
    total_pages = max(1, (len(review_data) + ipp - 1) // ipp)
    if st.session_state.grid_page >= total_pages:
        st.session_state.grid_page = 0

    ctrl_cols = st.columns([1, 1, 1, 4])
    with ctrl_cols[0]:
        if st.button("◀ Prev", use_container_width=True, disabled=st.session_state.grid_page == 0):
            st.session_state.grid_page = max(0, st.session_state.grid_page - 1)
            st.session_state.do_scroll_top = True
            st.session_state.grid_msg_counter += 1
            st.rerun(scope="fragment")
    with ctrl_cols[1]:
        st.markdown(
            f"<div style='text-align:center;padding:8px 0;font-weight:700;'>"
            f"Page {st.session_state.grid_page+1} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with ctrl_cols[2]:
        if st.button("Next ▶", use_container_width=True, disabled=st.session_state.grid_page >= total_pages - 1):
            st.session_state.grid_page += 1
            st.session_state.do_scroll_top = True
            st.session_state.grid_msg_counter += 1
            st.rerun(scope="fragment")

    # ── Image quality checks ──────────────────────────────────────────────────
    page_start = st.session_state.grid_page * ipp
    page_data  = review_data.iloc[page_start : page_start + ipp]

    page_warnings: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        future_to_sid = {
            ex.submit(analyze_image_quality_cached, str(r.get("MAIN_IMAGE", "")).strip()): str(r["PRODUCT_SET_SID"])
            for _, r in page_data.iterrows()
        }
        for future in concurrent.futures.as_completed(future_to_sid):
            warns = future.result()
            if warns:
                page_warnings[future_to_sid[future]] = warns

    rejected_state = {
        sid: st.session_state[f"quick_rej_reason_{sid}"]
        for sid in page_data["PRODUCT_SET_SID"].astype(str)
        if st.session_state.get(f"quick_rej_{sid}")
    }

    cols_per_row = 3 if st.session_state.layout_mode == "centered" else 4

    grid_html = build_fast_grid_html(
        page_data,
        support_files["flags_mapping"],
        st.session_state.selected_country,
        page_warnings,
        rejected_state,
        cols_per_row
    )
    
    components.html(grid_html, height=1300, scrolling=True)

    if st.session_state.get("do_scroll_top", False):
        components.html(
            "<script>window.parent.document.querySelector('.main')"
            ".scrollTo({top:0,behavior:'smooth'});</script>",
            height=0,
        )
        st.session_state.do_scroll_top = False


# ==========================================
# SECTION 3: EXPORTS
# ==========================================
@st.fragment
def render_exports_section():
    if st.session_state.final_report.empty or st.session_state.file_mode == 'post_qc':
        return

    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    app_df = fr[fr['Status'] == 'Approved']
    rej_df = fr[fr['Status'] == 'Rejected']
    c_code = st.session_state.selected_country[:2].upper()
    date_str = datetime.now().strftime('%Y-%m-%d')
    reasons_df = support_files.get('reasons', pd.DataFrame())

    st.markdown("---")
    st.markdown(f"""<div style='background: linear-gradient(135deg, {JUMIA_COLORS['primary_orange']}, {JUMIA_COLORS['secondary_orange']}); padding: 20px 24px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(246, 139, 30, 0.25);'><h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>Download Reports</h2><p style='color: rgba(255,255,255,0.9); margin: 6px 0 0 0; font-size: 13px;'>Export validation results in Excel or ZIP format</p></div>""", unsafe_allow_html=True)

    exports_config =[
        ("Final Report",  fr,     'assignment',   'Complete validation report with all statuses', lambda df: generate_smart_export(df, f"{c_code}_Final_{date_str}", 'simple', reasons_df)),
        ("Rejected Only", rej_df, 'block',        'Products that failed validation', lambda df: generate_smart_export(df, f"{c_code}_Rejected_{date_str}", 'simple', reasons_df)),
        ("Approved Only", app_df, 'check_circle', 'Products that passed validation', lambda df: generate_smart_export(df, f"{c_code}_Approved_{date_str}", 'simple', reasons_df)),
        ("Full Data",     data,   'database',     'Complete dataset with validation flags', lambda df: generate_smart_export(prepare_full_data_merged(df, fr), f"{c_code}_Full_{date_str}", 'full')),
    ]

    cols_count = 4 if st.session_state.layout_mode == "wide" else 2
    for i in range(0, len(exports_config), cols_count):
        cols = st.columns(cols_count)
        for j, col in enumerate(cols):
            if i + j < len(exports_config):
                title, df, icon, desc, func = exports_config[i + j]
                with col:
                    with st.container(border=True):
                        st.markdown(f"""<div style='text-align: center; margin-bottom: 15px;'><div style='font-size: 48px; margin-bottom: 8px;' class='material-symbols-outlined'>{icon}</div><div style='font-size: 18px; font-weight: 700;'>{title}</div><div style='font-size: 11px; margin-top: 4px; opacity: 0.7;'>{desc}</div><div style='background: {JUMIA_COLORS['light_gray']}; color: {JUMIA_COLORS['primary_orange']}; padding: 8px; border-radius: 6px; margin-top: 12px; font-weight: 600;'>{len(df):,} rows</div></div>""", unsafe_allow_html=True)
                        export_key = title
                        if export_key not in st.session_state.exports_cache:
                            if st.button("Generate", key=f"gen_{title}", type="primary", use_container_width=True, icon=":material/download:"):
                                with st.spinner(f"Generating {title}..."):
                                    res, fname, mime = func(df)
                                    st.session_state.exports_cache[export_key] = {"data": res.getvalue(), "fname": fname, "mime": mime}
                                st.rerun()
                        else:
                            cache = st.session_state.exports_cache[export_key]
                            st.download_button("Download", data=cache["data"], file_name=cache["fname"], mime=cache["mime"], use_container_width=True, type="primary", icon=":material/file_download:", key=f"dl_{title}")
                            if st.button("Clear", key=f"clr_{title}", use_container_width=True):
                                del st.session_state.exports_cache[export_key]
                                st.rerun()

# ==========================================
# CALL FRAGMENTS
# ==========================================
render_image_grid()
render_exports_section()
