import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import json
import random

# ────────────────────────────────────────────────
#   Batch Rejection Grid Test – Fully Fixed Version
#   (March 2025 style – safe against fast clicking)
# ────────────────────────────────────────────────

st.set_page_config(page_title="Batch Reject Grid – Fixed", layout="wide")

# ── Test data ────────────────────────────────────────────────────────────────

@st.cache_data
def generate_test_data(n=180):
    sids = [f"PSID_{i:05d}" for i in range(1, n+1)]

    brand_pool  = ["Samsung", "Apple", "Xiaomi", "Tecno", "Generic", "Huawei", "Oppo", "Vivo", "Nokia", "Infinix"]
    cat_pool    = ["Phones", "Laptops", "Fashion", "Home", "Beauty", "Accessories", "Electronics"]
    seller_pool = ["ShopZ", "MegaDeals", "TrendyHub", "TechWorld", "FashionNova", "GadgetPro", "HomeEssentials"]

    brands  = [random.choice(brand_pool)  for _ in range(n)]
    cats    = [random.choice(cat_pool)    for _ in range(n)]
    sellers = [random.choice(seller_pool) for _ in range(n)]

    df = pd.DataFrame({
        "PRODUCT_SET_SID": sids,
        "NAME":           [f"Product Amazing Thing {i}" for i in range(1, n+1)],
        "BRAND":          brands,
        "CATEGORY":       cats,
        "SELLER_NAME":    sellers,
        "MAIN_IMAGE":     [f"https://picsum.photos/seed/{i}/300/300" for i in range(1, n+1)],
        "GLOBAL_PRICE":   [round(49.99 + i * 4.5, 2) for i in range(n)],
    })
    return df

data = generate_test_data(180)

# ── Session state ────────────────────────────────────────────────────────────

if 'committed' not in st.session_state:
    st.session_state.committed = {}     # sid → reason (final)
if 'pending' not in st.session_state:
    st.session_state.pending = {}       # sid → reason (waiting confirm)
if 'grid_page' not in st.session_state:
    st.session_state.grid_page = 0
if 'ipp' not in st.session_state:
    st.session_state.ipp = 24
if 'bridge' not in st.session_state:
    st.session_state.bridge = ""

# ── Grid HTML + JS ───────────────────────────────────────────────────────────

def build_grid_html(page_df, committed, pending):

    cards = []
    for _, r in page_df.iterrows():
        sid = str(r["PRODUCT_SET_SID"])
        cards.append({
            "sid": sid,
            "img": r["MAIN_IMAGE"],
            "name": r["NAME"][:58] + "…" if len(r["NAME"]) > 58 else r["NAME"],
            "brand": r["BRAND"],
            "cat": r["CATEGORY"],
            "seller": r["SELLER_NAME"],
        })

    cards_json    = json.dumps(cards)
    committed_json = json.dumps(committed)
    pending_json   = json.dumps(pending)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{margin:0;padding:0;box-sizing:border-box;font-family:system-ui,sans-serif;}}
  body {{background:#f9fafb;padding:16px;}}
  .toolbar {{
    position:sticky;top:0;z-index:10;
    background:white;padding:12px 20px;border-radius:10px;
    box-shadow:0 2px 10px rgba(0,0,0,0.08);margin-bottom:20px;
    display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;
  }}
  .stats {{font-weight:600;color:#c2410c;}}
  .grid {{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;}}
  .card {{
    border:2px solid #e5e7eb;border-radius:12px;overflow:hidden;background:white;
    transition:all .18s;position:relative;
  }}
  .card.selected   {{border-color:#16a34a;box-shadow:0 0 0 3px rgba(22,163,74,.15);}}
  .card.committed  {{opacity:.54;filter:grayscale(.5);border-color:#9ca3af;}}
  .card.pending    {{border-color:#ea580c;box-shadow:0 0 0 3px rgba(234,88,12,.2);}}
  .img-wrap {{position:relative;cursor:pointer;height:260px;background:#f3f4f6;}}
  img {{width:100%;height:100%;object-fit:contain;}}
  .tick {{
    position:absolute;bottom:10px;right:10px;width:32px;height:32px;
    background:rgba(0,0,0,.45);border-radius:50%;color:white;font-weight:bold;
    display:flex;align-items:center;justify-content:center;pointer-events:none;font-size:18px;
  }}
  .card.selected .tick {{background:#16a34a;}}
  .meta {{padding:12px;font-size:13.5px;line-height:1.4;}}
  .name {{font-weight:600;max-height:44px;overflow:hidden;}}
  .brand {{color:#c2410c;font-weight:700;margin:4px 0 3px;}}
  .cat,.seller {{color:#4b5563;font-size:12.5px;}}
  .status {{
    position:absolute;top:10px;right:10px;padding:5px 12px;
    border-radius:999px;font-size:11px;font-weight:700;color:white;
  }}
  .status-pending  {{background:#ea580c;}}
  .status-rejected {{background:#dc2626;}}
  .actions {{padding:0 12px 14px;display:flex;gap:10px;}}
  button,select {{padding:7px 14px;border-radius:8px;border:none;cursor:pointer;font-weight:600;font-size:13px;}}
  .quick-btn {{background:#c2410c;color:white;flex:1;}}
  .more      {{background:#e5e7eb;color:#1f2937;}}
  #batch-btn       {{background:#c2410c;color:white;}}
  #confirm-batch   {{background:#dc2626;color:white;font-weight:bold;padding:9px 18px;}}
  #confirm-batch:disabled {{opacity:.5;cursor:not-allowed;background:#9ca3af;}}
</style>
</head>
<body>

<div class="toolbar">
  <div>
    <span class="stats" id="sel">0 selected</span>
    <span style="margin-left:20px;color:#6b7280;">Page {st.session_state.grid_page + 1}</span>
  </div>
  <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;">
    <select id="reason">
      <option value="POOR_IMAGE">Poor Image</option>
      <option value="WRONG_CAT">Wrong Category</option>
      <option value="FAKE">Suspected Fake</option>
      <option value="PROHIBITED">Prohibited</option>
      <option value="BRAND">Restricted Brand</option>
    </select>
    <button id="batch-btn">Batch → Pending</button>
    <span id="pending-count" style="font-weight:600;color:#ea580c;">Pending: 0</span>
    <button id="confirm-batch" disabled>CONFIRM REJECT</button>
    <button onclick="deselectAll()">Deselect All</button>
  </div>
</div>

<div class="grid" id="grid"></div>

<script>
const CARDS     = {cards_json};
const COMMITTED = {committed_json};
const PENDING   = {pending_json};

let selected = {{}};
let pending  = {{...PENDING}};

function $(id){{return document.getElementById(id);}}

function updateUI(){{
  $('sel').textContent = Object.keys(selected).length + " selected";
  $('pending-count').textContent = "Pending: " + Object.keys(pending).length;
  $('confirm-batch').disabled = Object.keys(pending).length === 0;
}}

function render(){{
  const html = CARDS.map(c => {{
    const sid = c.sid;
    let cls = 'card';
    let statusHtml = '';
    if (sid in COMMITTED) {{
      cls += ' committed';
      statusHtml = `<div class="status status-rejected">REJECTED</div>`;
    }} else if (sid in pending) {{
      cls += ' pending';
      statusHtml = `<div class="status status-pending">PENDING</div>`;
    }} else if (sid in selected) {{
      cls += ' selected';
    }}
    return `
      <div class="${{cls}}" id="c${{sid}}">
        ${{statusHtml}}
        <div class="img-wrap" onclick="toggle('${{sid}}')">
          <img src="${{c.img}}" loading="lazy" onerror="this.src='https://via.placeholder.com/260?text=×'">
          <div class="tick">✓</div>
        </div>
        <div class="meta">
          <div class="name">${{c.name}}</div>
          <div class="brand">${{c.brand}}</div>
          <div class="cat">${{c.cat}}</div>
          <div class="seller">${{c.seller}}</div>
        </div>
        <div class="actions">
          <button class="quick-btn" onclick="event.stopPropagation();quickReject('${{sid}}','POOR_IMAGE')">Poor Img</button>
          <select class="more" onchange="if(this.value){{event.stopPropagation();quickReject('${{sid}}',this.value);this.value='';}}">
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

function toggle(sid){{
  if (sid in COMMITTED || sid in pending) return;
  if (sid in selected) delete selected[sid];
  else selected[sid] = true;
  render();
  updateUI();
}}

function quickReject(sid, reason){{
  if (sid in COMMITTED) return;
  delete selected[sid];
  pending[sid] = reason;
  render();
  updateUI();
  send('pending_change', {{sid, reason}});
}}

function batchToPending(){{
  const reason = $('reason').value;
  if (!reason) return alert("Select a reason first");
  Object.keys(selected).forEach(sid => {{
    if (!(sid in COMMITTED)) pending[sid] = reason;
  }});
  selected = {{}};
  render();
  updateUI();
}}

function confirmBatch(){{
  if (Object.keys(pending).length === 0) return;
  send('batch_confirm', pending);
  Object.assign(COMMITTED, pending);
  pending = {{}};
  render();
  updateUI();
}}

function deselectAll(){{
  selected = {{}};
  render();
}}

function send(type, payload){{
  window.parent.postMessage({{type:"grid_msg", action:type, payload}}, "*");
}}

$('batch-btn').onclick = batchToPending;
$('confirm-batch').onclick = confirmBatch;

render();
</script>
</body>
</html>
"""

# ── Bridge ───────────────────────────────────────────────────────────────────

components.html("""
<script>
window.addEventListener("message", e => {
  if (!e.data || e.data.type !== "grid_msg") return;
  const inputs = document.querySelectorAll('input[type="text"]');
  let bridge = null;
  for (const inp of inputs) {
    if (inp.getAttribute("aria-label")?.includes("bridge")) {
      bridge = inp; break;
    }
  }
  if (!bridge) return;
  bridge.value = JSON.stringify(e.data);
  bridge.dispatchEvent(new Event("input", {bubbles:true}));
});
</script>
""", height=0)

bridge = st.text_input(
    "bridge",
    value=st.session_state.bridge,
    label_visibility="collapsed",
    key="bridge_input"
)

if bridge and bridge != st.session_state.bridge:
    try:
        msg = json.loads(bridge)
        act = msg.get("action")
        pay = msg.get("payload", {})

        if act == "pending_change":
            st.toast(f"→ Pending: {pay.get('sid')} – {pay.get('reason')}", icon="🟠")

        elif act == "batch_confirm":
            count = len(pay)
            for sid, reason in pay.items():
                st.session_state.committed[sid] = reason
            st.session_state.pending.clear()
            st.toast(f"✅ Committed rejection of {count} items", icon="🟢")
            st.rerun()

    except Exception as e:
        st.error(f"Bridge parse error: {e}")
    finally:
        st.session_state.bridge = bridge

# ── UI ───────────────────────────────────────────────────────────────────────

st.title("Batch Rejection Grid – Fast Click Safe")
st.caption("Select many cards quickly → Batch → Pending → Confirm")

c1, c2, c3 = st.columns([1,1,2])
with c1: st.session_state.ipp = st.select_slider("Items/page", [12,24,36,48,72], value=st.session_state.ipp)
with c2:
    if st.button("Reset everything"):
        st.session_state.committed.clear()
        st.session_state.pending.clear()
        st.rerun()

page_start = st.session_state.grid_page * st.session_state.ipp
page_end   = page_start + st.session_state.ipp
page_df    = data.iloc[page_start:page_end]

st.caption(f"Showing {len(page_df)} items  |  committed: {len(st.session_state.committed)}  |  pending: {len(st.session_state.pending)}")

components.html(
    build_grid_html(page_df, st.session_state.committed, st.session_state.pending),
    height=1450,
    scrolling=True
)

# Pagination
pc, nc = st.columns(2)
with pc:
    if st.button("← Prev", disabled=st.session_state.grid_page <= 0):
        st.session_state.grid_page = max(0, st.session_state.grid_page - 1)
        st.rerun()
with nc:
    if st.button("Next →", disabled=page_end >= len(data)):
        st.session_state.grid_page += 1
        st.rerun()

st.markdown("---")
st.markdown("""
### Test instructions

1. Click many images **very quickly** → they should stay selected
2. Pick a reason → **Batch → Pending**
3. See orange **Pending** count rise
4. Click **CONFIRM REJECT** → all pending items become rejected
5. Fast multi-clicking should no longer lose items

The batch is now collected locally in JS → only one message is sent on confirm.
""")
