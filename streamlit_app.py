
import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")

# -----------------------------
# FAST IMAGE LOADER (10x faster)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_image(url):
    try:
        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content))
        return img
    except:
        return None

# -----------------------------
# SESSION STATE
# -----------------------------
if "final_report" not in st.session_state:
    st.session_state.final_report = pd.DataFrame()

if "selected_sids" not in st.session_state:
    st.session_state.selected_sids = set()

# -----------------------------
# REJECTION REASONS
# -----------------------------
REJECTION_FLAGS = {
    "POOR_IMAGE": ("1000001 - Poor Image", "Image quality too low"),
    "WRONG_CATEGORY": ("1000002 - Wrong Category", "Incorrect category"),
    "WRONG_BRAND": ("1000003 - Wrong Brand", "Brand mismatch"),
    "DUPLICATE": ("1000004 - Duplicate Listing", "Duplicate product"),
    "OTHER": ("1000007 - Other Reason", "Manual rejection")
}

# -----------------------------
# APPLY REJECTION
# -----------------------------
def apply_rejection(sids, code, comment, flag):

    df = st.session_state.final_report

    if df.empty:
        return

    mask = df["PRODUCT_SET_SID"].isin(sids)

    df.loc[mask, "Status"] = "REJECTED"
    df.loc[mask, "Reason"] = code
    df.loc[mask, "Comment"] = comment
    df.loc[mask, "FLAG"] = flag

    st.session_state.final_report = df


# -----------------------------
# FILE UPLOAD
# -----------------------------
st.title("Product QC Tool")

file = st.file_uploader("Upload Product File", type=["csv","xlsx"])

if file:

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.session_state.final_report = df

df = st.session_state.final_report

if df.empty:
    st.stop()


# -----------------------------
# PAGINATION
# -----------------------------
items_per_page = 20

total_items = len(df)

page = st.number_input("Page", min_value=1, max_value=(total_items//items_per_page)+1, value=1)

start = (page-1)*items_per_page
end = start + items_per_page

grid = df.iloc[start:end]


# -----------------------------
# PRODUCT GRID
# -----------------------------
for i,row in grid.iterrows():

    sid = row["PRODUCT_SET_SID"]

    with st.container():

        col1,col2,col3 = st.columns([1,3,1])

        with col1:

            img = load_image(str(row.get("MAIN_IMAGE","")))

            if img:
                st.image(img,width=150)

        with col2:

            st.write("**",row.get("NAME",""),"**")
            st.write("Brand:",row.get("BRAND",""))
            st.write("Category:",row.get("CATEGORY",""))
            st.write("Seller:",row.get("SELLER_NAME",""))

        with col3:

            selected = st.checkbox("Select", key=f"select_{sid}")

            if selected:
                st.session_state.selected_sids.add(sid)
            else:
                st.session_state.selected_sids.discard(sid)

        r1,r2,r3,r4,r5 = st.columns(5)

        if r1.button("Poor Img", key=f"poor_{sid}"):
            code,comment = REJECTION_FLAGS["POOR_IMAGE"]
            apply_rejection([sid],code,comment,"POOR_IMAGE")
            st.rerun()

        if r2.button("Wrong Cat", key=f"cat_{sid}"):
            code,comment = REJECTION_FLAGS["WRONG_CATEGORY"]
            apply_rejection([sid],code,comment,"WRONG_CATEGORY")
            st.rerun()

        if r3.button("Wrong Brand", key=f"brand_{sid}"):
            code,comment = REJECTION_FLAGS["WRONG_BRAND"]
            apply_rejection([sid],code,comment,"WRONG_BRAND")
            st.rerun()

        if r4.button("Duplicate", key=f"dup_{sid}"):
            code,comment = REJECTION_FLAGS["DUPLICATE"]
            apply_rejection([sid],code,comment,"DUPLICATE")
            st.rerun()

        if r5.button("Other", key=f"other_{sid}"):
            code,comment = REJECTION_FLAGS["OTHER"]
            apply_rejection([sid],code,comment,"OTHER")
            st.rerun()

        st.divider()


# -----------------------------
# BATCH REJECT
# -----------------------------
st.subheader("Batch Actions")

col1,col2 = st.columns(2)

with col1:

    if st.button("Reject Selected Batch"):

        if st.session_state.selected_sids:

            code,comment = REJECTION_FLAGS["OTHER"]

            apply_rejection(
                list(st.session_state.selected_sids),
                code,
                comment,
                "BATCH_REJECT"
            )

            st.session_state.selected_sids.clear()

            st.success("Batch rejected")

            st.rerun()

with col2:

    if st.button("Clear Selection"):
        st.session_state.selected_sids.clear()
        st.rerun()


# -----------------------------
# EXPORT RESULTS
# -----------------------------
st.subheader("Export")

csv = st.session_state.final_report.to_csv(index=False).encode()

st.download_button(
    "Download QC Report",
    csv,
    "qc_results.csv",
    "text/csv"
)
