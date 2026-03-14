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
    "MO": "Morocco",  # Included since your uploaded file uses 'MO'
    "EG": "Egypt",
    "CI": "Ivory Coast",
    "SN": "Senegal",
    "ZA": "South Africa"
}

def parse_file_metadata(filename):
    """Extracts country, date, and week number from the filename."""
    # Extract Country Code (first two letters)
    prefix = filename[:2].upper()
    country = COUNTRY_MAP.get(prefix, "Unknown_Country")
    
    # Extract Date
    date_obj = None
    week_num = None
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        date_obj = datetime.datetime.strptime(match.group(), '%Y-%m-%d')
        # Get ISO week number
        week_num = date_obj.isocalendar()[1]
        
    return country, date_obj, week_num

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
st.markdown("Upload your `ProductSets` CSV files to generate a weekly performance report.")

uploaded_files = st.file_uploader("Upload ProductSets CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    all_data = []
    
    # Assume the batch belongs to the country and week of the first uploaded file
    primary_country, _, primary_week = parse_file_metadata(uploaded_files[0].name)
    
    for file in uploaded_files:
        df = pd.read_csv(file)
        country, file_date, week_num = parse_file_metadata(file.name)
        
        if file_date:
            df['Date'] = file_date
            df['Day'] = file_date.strftime('%A')
            df['Country'] = country
            all_data.append(df)
    
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        
        # Dashboard Header
        st.success(f"✅ Data loaded successfully for **{primary_country}** (Week {primary_week})")
        
        # 1. Daily & Weekly Totals
        daily_summary = master_df.groupby(['Day', 'Status']).size().unstack(fill_value=0)
        
        # Ensure all days are represented in order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_summary = daily_summary.reindex(days_order).fillna(0).astype(int)
        
        # Add totals
        daily_summary['Total'] = daily_summary.sum(axis=1)
        
        # Weekly Totals
        weekly_approved = daily_summary['Approved'].sum() if 'Approved' in daily_summary.columns else 0
        weekly_rejected = daily_summary['Rejected'].sum() if 'Rejected' in daily_summary.columns else 0
        weekly_total = daily_summary['Total'].sum()

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Weekly Approved", weekly_approved)
        col2.metric("Weekly Rejected", weekly_rejected, delta=f"{(weekly_rejected/weekly_total)*100:.1f}% Rate" if weekly_total > 0 else "0%", delta_color="inverse")
        col3.metric("Total Processed", weekly_total)

        st.subheader("📅 Daily Breakdown")
        st.table(daily_summary)

        # Filter out rejected products for top lists
        rejected_df = master_df[master_df['Status'] == 'Rejected']

        colA, colB, colC = st.columns(3)

        # 2. Top 5 Rejected Sellers
        with colA:
            st.subheader("🚩 Top 5 Rejected Sellers")
            top_sellers = rejected_df['SellerName'].value_counts().head(5)
            st.dataframe(top_sellers, use_container_width=True)

        # 3. Top 5 Rejection Reasons
        with colB:
            st.subheader("🔍 Top 5 Rejection Reasons")
            # Extract just the code and short reason if it's too long
            top_reasons = rejected_df['Reason'].str.split('-').str[0:2].str.join('-').value_counts().head(5)
            st.dataframe(top_reasons, use_container_width=True)

        # 4. Top 5 Rejected Categories (Based on FLAG)
        with colC:
            st.subheader("📁 Top 5 Rejected Categories")
            top_categories = rejected_df['FLAG'].value_counts().head(5)
            st.dataframe(top_categories, use_container_width=True)

        # 5. Downloadable Report
        st.divider()
        report_data = generate_excel_report(daily_summary, top_sellers, top_reasons, top_categories)
        
        # Format filename: Country_WeekNumber_Report.xlsx
        download_filename = f"{primary_country}_Week{primary_week}_Report.xlsx"
        
        st.download_button(
            label=f"📥 Download {primary_country} Report (Excel)",
            data=report_data,
            file_name=download_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("Could not find dates in the filenames. Please ensure filenames include YYYY-MM-DD.")
