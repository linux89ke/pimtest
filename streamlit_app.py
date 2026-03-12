import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import json
from datetime import datetime
import hashlib
import re
import logging

# ────────────────────────────────────────────────
#  Minimal version for testing the grid + batch fix
# ────────────────────────────────────────────────

st.set_page_config(page_title="Grid Batch Reject Test", layout="wide")

# Fake data generator
@st.cache_data
def generate_test_data(n=120):
    sids = [f"PSID_{i:05d}" for i in range(1, n+1)]
    df = pd.DataFrame({
        "PRODUCT_SET_SID": sids,
        "NAME": [f"Product Amazing Thing {i}" for i in range(1, n+1)],
        "BRAND": ["Samsung", "Apple", "Xiaomi", "Tecno", "Generic"][:n%5+1]* (n//5 + 1),
        "CATEGORY": ["Phones", "Laptops", "Fashion", "Home", "Beauty"][:n%5+1]* (n//5 + 1),
        "SELLER_NAME": ["ShopZ", "MegaDeals", "TrendyHub", "TechWorld"][:n%4+1]* (n//4 + 1),
        "MAIN_IMAGE": ["https://picsum.photos/seed/{}/300".format(i) for i in range(1, n+1)],
        "GLOBAL_PRICE": [99.99 + i*5 for i in range(n)],
    })
    return df

if 'committed' not in st.session_state:
    st.session_state.committed = {}           # sid → reason_key
if 'grid_page' not in st.session_state:
    st.session_state.grid_page = 0
if 'ipp' not in st.session_state:
    st.session_state.ipp = 24
if 'bridge' not in st.session_state:
    st.session_state.bridge = ""

data = generate_test_data(180)

# ── JavaScript Grid ───────────────────────────────────────────────────────────

def build_grid_html(page_df, committed):

    cards = []
    for _, r in page_df.iterrows():
        sid = str(r["PRODUCT_SET_SID"])
        cards.append({
            "sid": sid,
            "img": r["MAIN_IMAGE"],
            "name": r["NAME"][:60] + "…" if len(r["NAME"]) > 60 else r["NAME"],
            "brand": r["BRAND"],
            "cat": r["CATEGORY"],
            "seller": r["SELLER_NAME"],
        })

    cards_json = json.dumps(cards)
    committed_json = json.dumps(committed)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; font-family:sans-serif; }}
  body {{ background:#f8f9fa; padding:16px; }}
  .toolbar {{
    position: sticky; top:0; z-index:10;
    background:white; padding:12px 16px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);
    margin-bottom:16px; display:flex; justify-content:space-between; align-items:center; gap:16px;
    flex-wrap:wrap;
  }}
  .stats {{ font-weight:bold; color:#e65100; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap:16px; }}
  .card {{
    border:2px solid #e0e0e0; border-radius:12px; overflow:hidden; background:white;
    transition: all 0.18s ease; position:relative;
  }}
  .card.selected   {{ border-color:#4caf50; box-shadow:0 0 0 3px rgba(76,175,80,0.2); }}
  .card.committed  {{ opacity:0.5; filter:grayscale(0.4); border-color:#aaa; }}
  .card.pending    {{ border-color:#ff9800; box-shadow:0 0 0 3px rgba(255,152,0,0.3); }}
  .img-wrap {{ position:relative; cursor:pointer; height:240px; background:#f0f0f0; }}
  img {{ width:100%; height:100%; object-fit:contain; }}
  .tick {{
    position:absolute; bottom:8px; right:8px; width:28px; height:28px;
    background:rgba(0,0,0,0.4); border-radius:50%; color:white; font-weight:bold;
    display:flex; align-items:center; justify-content:center; pointer-events:none;
  }}
  .card.selected .tick {{ background:#4caf50; }}
  .meta {{ padding:10px; font-size:13px; }}
  .name {{ font-weight:600; line-height:1.3; max-height:40px; overflow:hidden; }}
  .brand {{ color:#e65100; font-weight:700; margin:4px 0 2px; }}
  .cat, .seller {{ color:#555; font-size:12px; }}
  .status {{ position:absolute; top:8px; right:8px; padding:4px 10px; border-radius:12px; font-size:11px; font-weight:bold; }}
  .status-pending  {{ background:#ff9800; color:white; }}
  .status-rejected {{ background:#d32f2f; color:white; }}
  .actions {{ padding:0 10px 12px; display:flex; gap:8px; }}
  button, select {{ padding:6px 12px; border-radius:6px; border:none; cursor:pointer; font-weight:600; }}
  .quick-btn {{ background:#e65100; color:white; flex:1; }}
  .more {{ background:#eee; color:#333; }}
  #confirm-batch {{ background:#d32f2f; color:white; font-weight:bold; padding:8px 16px; border:none; border-radius:6px; cursor:pointer; }}
  #confirm-batch:disabled {{ opacity:0.5; cursor:not-allowed; background:#ccc; }}
</style>
</head>
<body>

<div class="toolbar">
  <div>
    <span class="stats" id="sel">0 selected</span>
    <span style="margin-left:16px;">Page {st.session_state.grid_page + 1}</span>
  </div>
  <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
    <select id="reason">
      <option value="POOR_IMAGE">Poor Image</option>
      <option value="WRONG_CAT">Wrong Category</option>
      <option value="FAKE">Suspected Fake</option>
      <option value="PROHIBITED">Prohibited</option>
      <option value="BRAND">Restricted Brand</option>
    </select>
    <button id="batch-btn">Batch Reject →</button>
    <span id="pending">Pending: 0</span>
    <button id="confirm-batch" disabled>CONFIRM REJECT</button>
    <button onclick="deselectAll()">Deselect All</button>
  </div>
</div>

<div class="grid" id="grid"></div>

<script>
const CARDS = {cards_json};
const COMMITTED = {committed_json};

let selected = {{}};
let pending = {{}};

function $(id) {{ return document.getElementById(id); }}

function updateUI() {{
  $('sel').textContent = Object.keys(selected).length + " selected";
  $('pending').textContent = "Pending: " + Object.keys(pending).length;
  $('confirm-batch').disabled = Object.keys(pending).length === 0;
}}

function render() {{
  const html = CARDS.map(c => {{
    const sid = c.sid;
    let cls = 'card';
    let status = '';
    if (sid in COMMITTED) {{
      cls += ' committed';
      status = `<div class="status status-rejected">REJECTED</div>`;
    }} else if (sid in pending) {{
      cls += ' pending';
      status = `<div class="status status-pending">PENDING</div>`;
    }} else if (sid in selected) {{
      cls += ' selected';
    }}
    return `
      <div class="${{cls}}" id="c${{sid}}">
        ${{status}}
        <div class="img-wrap" onclick="toggle('${{sid}}')">
          <img src="${{c.img}}" loading="lazy" onerror="this.src='https://via.placeholder.com/240?text=×'">
          <div class="tick">✓</div>
        </div>
        <div class="meta">
          <div class="name">${{c.name}}</div>
          <div class="brand">${{c.brand}}</div>
          <div class="cat">${{c.cat}}</div>
          <div class="seller">${{c.seller}}</div>
        </div>
        <div class="actions">
          <button class="quick-btn" onclick="event.stopPropagation(); quickReject('${{sid}}','POOR_IMAGE')">Poor Img</button>
          <select class="more" onchange="if(this.value) {{event.stopPropagation(); quickReject('${{sid}}',this.value); this.value='';}}">
            <option value="">More…</option>
            <option value="WRONG_CAT">Wrong Cat</option>
            <option value="FAKE">Fake</option>
            <option value="PROHIBITED">Prohibited</option>
            <option value="BRAND">Brand</option>
          </select>
        </div>
      </div>`;
  }}).join('');
  $('grid').innerHTML = html;
  updateUI();
}}

function toggle(sid) {{
  if (sid in COMMITTED || sid in pending) return;
  if (sid in selected) delete selected[sid];
  else selected[sid] = true;
  updateCard(sid);
  updateUI();
}}

function updateCard(sid) {{
  const el = $('c' + sid);
  if (!el) return;
  const c = CARDS.find(x => x.sid === sid);
  if (!c) return;
  const tmp = document.createElement('div');
  tmp.innerHTML = renderCard(c);  // we'll define renderCard separately if needed
  el.replaceWith(tmp.firstChild);
}}

function quickReject(sid, reason) {{
  if (sid in COMMITTED) return;
  delete selected[sid];
  pending[sid] = reason;
  updateCard(sid);
  updateUI();
  send('pending_change', {{sid, reason}});
}}

function batchStart() {{
  const reason = $('reason').value;
  if (!reason) return;
  Object.keys(selected).forEach(sid => {{
    if (!(sid in COMMITTED)) pending[sid] = reason;
  }});
  selected = {{}};
  render();
  updateUI();
}}

function confirmBatch() {{
  if (Object.keys(pending).length === 0) return;
  send('batch_confirm', pending);
  // Move pending → committed locally (optimistic)
  Object.assign(COMMITTED, pending);
  pending = {{}};
  render();
  updateUI();
}}

function deselectAll() {{
  selected = {{}};
  render();
}}

function send(type, payload) {{
  window.parent.postMessage({{type:"grid", action:type, payload}}, "*");
}}

$('batch-btn').onclick = batchStart;
$('confirm-batch').onclick = confirmBatch;

render();
</script>
</body>
</html>
"""

# ── Bridge ───────────────────────────────────────────────────────────────────

components.html("""
<script>
window.addEventListener("message", function(e) {
  if (!e.data || e.data.type !== "grid") return;
  const el = [...document.querySelectorAll('input[type="text"]')]
    .find(inp => inp.getAttribute("aria-label")?.includes("bridge"));
  if (!el) return;
  el.value = JSON.stringify(e.data);
  el.dispatchEvent(new Event("input", {bubbles:true}));
});
</script>
""", height=0)

bridge = st.text_input("bridge", value=st.session_state.bridge, label_visibility="collapsed", key="bridge_input")

if bridge and bridge != st.session_state.bridge:
    try:
        msg = json.loads(bridge)
        act = msg.get("action")
        pay = msg.get("payload", {})

        if act == "pending_change":
            st.toast(f"Pending change: {pay.get('sid')} → {pay.get('reason')}", icon="🟠")

        elif act == "batch_confirm":
            count = len(pay)
            for sid, reason in pay.items():
                st.session_state.committed[sid] = reason
            st.toast(f"✅ Confirmed batch rejection of {count} items", icon="🟢")
            st.session_state.bridge = ""
            st.rerun()

    except Exception as e:
        st.error(f"Bridge parse error: {e}")
    finally:
        st.session_state.bridge = bridge

# ── UI ───────────────────────────────────────────────────────────────────────

st.title("Batch Reject Grid – Race Condition Test")

c1, c2, c3 = st.columns([1,1,2])
with c1:
    st.session_state.ipp = st.select_slider("Items/page", [12,24,36,48,72], value=st.session_state.ipp)
with c2:
    if st.button("Reset committed"):
        st.session_state.committed.clear()
        st.rerun()

page_start = st.session_state.grid_page * st.session_state.ipp
page_end   = page_start + st.session_state.ipp
page_df    = data.iloc[page_start:page_end]

st.caption(f"Showing {len(page_df)} items  |  committed = {len(st.session_state.committed)}")

html = build_grid_html(page_df, st.session_state.committed)
components.html(html, height=1400, scrolling=True)

# Pagination
pc, nc = st.columns(2)
with pc:
    if st.button("← Prev", disabled=st.session_state.grid_page <= 0):
        st.session_state.grid_page -= 1
        st.rerun()
with nc:
    if st.button("Next →", disabled=page_end >= len(data)):
        st.session_state.grid_page += 1
        st.rerun()

st.markdown("---")
st.write("**How to test the fix:**")
st.markdown("""
1. Click several images quickly to select
2. Choose reason → **Batch Reject →**
3. See orange **Pending** counter rise
4. Click **CONFIRM REJECT** → should reject reliably
5. Try very fast clicking → pending should still collect everything
""")
