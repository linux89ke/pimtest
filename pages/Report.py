import streamlit as st
import pandas as pd
import datetime
import re
from io import BytesIO

st.set_page_config(page_title="PIM Weekly Analysis Dashboard", layout="wide")

# Country mapping based on standard prefixes
COUNTRY_MAP = {
    "KE": "Kenya",
    "UG": "Uganda",
    "NG": "Nigeria",
    "GH": "Ghana",
    "MA": "Morocco",
    "MO": "Morocco", 
    "EG": "Egypt",
    "CI": "Ivory Coast",
    "SN": "Senegal",
    "ZA": "South Africa"
}

def parse_file_metadata(filename):
    """Extracts country, date, and week number from the filename."""
    prefix = filename[:2].upper()
    country = COUNTRY_MAP.get(prefix, "Unknown_Country")
    
    date_obj = None
    week_num = None
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        date_obj = datetime.datetime.strptime(match.group(), '%Y-%m-%d')
        week_num = date_obj.isocalendar()[1]
        
    return country, date_obj, week_num

def get_col(df, possible_names):
    """Helper to safely find a column name ignoring exact case."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def generate_excel_report(daily_summary, top_sellers, top_reasons, top_categories):
    """Creates an in-memory Excel file for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        daily_summary.to_excel(writer, sheet_name='Daily & Weekly Summary')
        top_sellers.to_excel(writer, sheet_name='Top 5 Rejected Sellers')
        top_reasons.to_excel(writer, sheet_name='Top 5 Rejection Reasons')
        top_categories.to_excel(writer, sheet_name='Top 5 Rejected Categories')
    return output.getvalue()

st.title("📊 PIM Weekly Export Analyzer")
st.markdown("Upload your `ProductSets` files (CSV or Excel) to generate a weekly performance report.")

# Accept both CSV and Excel formats
uploaded_files = st.file_uploader("Upload ProductSets files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

if uploaded_files:
    all_data = []
    
    primary_country, _, primary_week = parse_file_metadata(uploaded_files[0].name)
    
    for file in uploaded_files:
        # Read based on file extension
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file) 
            
        country, file_date, week_num = parse_file_metadata(file.name)
        
        if file_date:
            df['Date'] = file_date
            df['Day'] = file_date.strftime('%A')
            df['Country'] = country
            all_data.append(df)
    
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        
        # Safely identify column names based on the uploaded format
        status_col = get_col(master_df, ['Status', 'STATUS', 'status'])
        seller_col = get_col(master_df, ['SellerName', 'SELLER_NAME', 'seller_name'])
        flag_col = get_col(master_df, ['FLAG', 'Flag', 'flag', 'Reason'])
        cat_col = get_col(master_df, ['CATEGORY', 'Category', 'category'])
        
        st.success(f"✅ Data loaded successfully for **{primary_country}** (Week {primary_week})")
        
        if status_col:
            # 1. Daily & Weekly Totals
            daily_summary = master_df.groupby(['Day', status_col]).size().unstack(fill_value=0)
            
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            # Only keep days that exist in the data, but ordered correctly
            available_days = [d for d in days_order if d in daily_summary.index]
            daily_summary = daily_summary.reindex(days_order).fillna(0).astype(int)
            
            daily_summary['Total'] = daily_summary.sum(axis=1)
            
            weekly_approved = daily_summary['Approved'].sum() if 'Approved' in daily_summary.columns else 0
            weekly_rejected = daily_summary['Rejected'].sum() if 'Rejected' in daily_summary.columns else 0
            weekly_total = daily_summary['Total'].sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Weekly Approved", weekly_approved)
            col2.metric("Weekly Rejected", weekly_rejected, delta=f"{(weekly_rejected/weekly_total)*100:.1f}% Rate" if weekly_total > 0 else "0%", delta_color="inverse")
            col3.metric("Total Processed", weekly_total)

            st.subheader("📅 Daily Breakdown")
            st.table(daily_summary)

            # Filter rejected products
            rejected_df = master_df[master_df[status_col] == 'Rejected']

            colA, colB, colC = st.columns(3)

            # 2. Top 5 Rejected Sellers
            with colA:
                st.subheader("🚩 Top 5 Rejected Sellers")
                if seller_col:
                    top_sellers = rejected_df[seller_col].value_counts().head(5)
                else:
                    top_sellers = pd.Series(["Seller column not found"], index=["N/A"])
                st.dataframe(top_sellers, use_container_width=True)

            # 3. Top 5 Rejection Reasons (Using FLAG column)
            with colB:
                st.subheader("🔍 Top 5 Rejection Reasons")
                if flag_col:
                    top_reasons = rejected_df[flag_col].value_counts().head(5)
                else:
                    top_reasons = pd.Series(["FLAG column not found"], index=["N/A"])
                st.dataframe(top_reasons, use_container_width=True)

            # 4. Top 5 Rejected Categories (Using CATEGORY column)
            with colC:
                st.subheader("📁 Top 5 Rejected Categories")
                if cat_col:
                    top_categories = rejected_df[cat_col].value_counts().head(5)
                else:
                    top_categories = pd.Series(["CATEGORY column not found"], index=["N/A"])
                st.dataframe(top_categories, use_container_width=True)

            # 5. Downloadable Report
            st.divider()
            report_data = generate_excel_report(daily_summary, top_sellers, top_reasons, top_categories)
            
            download_filename = f"{primary_country}_Week{primary_week}_Report.xlsx"
            
            st.download_button(
                label=f"📥 Download {primary_country} Report (Excel)",
                data=report_data,
                file_name=download_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Could not find the 'Status' column in the uploaded files to calculate approvals/rejections.")
    else:
        st.error("Could not find dates in the filenames. Please ensure filenames include YYYY-MM-DD.")
