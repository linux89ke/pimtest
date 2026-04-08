"""
export_utils.py - Excel export, ZIP generation, and download helpers
"""

import logging
import zipfile
import pandas as pd
from io import BytesIO
from typing import Tuple

from constants import FULL_DATA_COLS, PRODUCTSETS_COLS, REJECTION_REASONS_COLS, SPLIT_LIMIT

logger = logging.getLogger(__name__)


def _repair_mojibake(df: pd.DataFrame) -> pd.DataFrame:
    """Re-imported locally to avoid circular imports — fixes encoding artifacts."""
    import re
    _ILLEGAL_XML = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

    def _fix(val):
        if not isinstance(val, str):
            return val
        for enc in ('cp1252', 'latin-1'):
            try:
                fixed = val.encode(enc).decode('utf-8')
                if fixed != val and '\ufffd' not in fixed:
                    val = fixed
                    break
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue
        return _ILLEGAL_XML.sub('', val)

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(_fix)
    return df


def to_excel_base(df: pd.DataFrame, sheet: str, cols: list, writer, format_rules: bool = False):
    """Write a DataFrame to an Excel sheet with optional conditional formatting."""
    from constants import JUMIA_COLORS
    df_p = df.copy()
    for c in cols:
        if c not in df_p.columns:
            df_p[c] = pd.NA
    df_to_write = df_p[[c for c in cols if c in df_p.columns]]
    df_to_write = _repair_mojibake(df_to_write.copy())
    df_to_write.to_excel(writer, index=False, sheet_name=sheet)
    if format_rules and 'Status' in df_to_write.columns:
        wb = writer.book
        ws = writer.sheets[sheet]
        rf = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        gf = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        idx = df_to_write.columns.get_loc('Status')
        ws.conditional_format(1, idx, len(df_to_write), idx, {'type': 'cell', 'criteria': 'equal', 'value': '"Rejected"', 'format': rf})
        ws.conditional_format(1, idx, len(df_to_write), idx, {'type': 'cell', 'criteria': 'equal', 'value': '"Approved"', 'format': gf})


def write_excel_single(
    df: pd.DataFrame,
    sheet_name: str,
    cols: list,
    auxiliary_df: pd.DataFrame = None,
    aux_sheet_name: str = None,
    aux_cols: list = None,
    format_status: bool = False,
    full_data_stats: bool = False
) -> BytesIO:
    """Write one or two DataFrames into a single Excel workbook."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        to_excel_base(df, sheet_name, cols, writer, format_rules=format_status)
        if auxiliary_df is not None and not auxiliary_df.empty:
            to_excel_base(auxiliary_df, aux_sheet_name, aux_cols, writer)
        if full_data_stats and 'SELLER_NAME' in df.columns and 'Status' in df.columns:
            ws = writer.book.add_worksheet('Sellers Data')
            fmt = writer.book.add_format({'bold': True, 'bg_color': '#E6F0FA', 'border': 1, 'align': 'center'})
            df_copy = df.copy()
            df_copy['Rejected_Count'] = (df_copy['Status'] == 'Rejected').astype(int)
            df_copy['Approved_Count'] = (df_copy['Status'] == 'Approved').astype(int)
            summ = (
                df_copy.groupby('SELLER_NAME')
                .agg(Rejected=('Rejected_Count', 'sum'), Approved=('Approved_Count', 'sum'))
                .reset_index()
                .sort_values('Rejected', ascending=False)
            )
            summ.insert(0, 'Rank', range(1, len(summ) + 1))
            ws.write(0, 0, "Sellers Summary (This File)", fmt)
            summ.to_excel(writer, sheet_name='Sellers Data', startrow=1, index=False)
    output.seek(0)
    return output


def generate_smart_export(
    df: pd.DataFrame,
    filename_prefix: str,
    export_type: str = 'simple',
    auxiliary_df: pd.DataFrame = None
) -> Tuple[BytesIO, str, str]:
    """
    Generate an Excel or ZIP export depending on row count.
    Returns (data, filename, mimetype).
    """
    cols = (
        FULL_DATA_COLS + [c for c in ["Status", "Reason", "Comment", "FLAG", "SellerName"] if c not in FULL_DATA_COLS]
        if export_type == 'full'
        else PRODUCTSETS_COLS
    )
    if len(df) <= SPLIT_LIMIT:
        data = write_excel_single(
            df, "ProductSets", cols,
            auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS,
            True, export_type == 'full'
        )
        return data, f"{filename_prefix}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        zb = BytesIO()
        with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(0, len(df), SPLIT_LIMIT):
                chunk = df.iloc[i: i + SPLIT_LIMIT]
                excel_data = write_excel_single(
                    chunk, "ProductSets", cols,
                    auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS,
                    True, export_type == 'full'
                )
                zf.writestr(f"{filename_prefix}_Part_{(i // SPLIT_LIMIT) + 1}.xlsx", excel_data.getvalue())
        zb.seek(0)
        return zb, f"{filename_prefix}.zip", "application/zip"


def prepare_full_data_merged(data_df: pd.DataFrame, final_report_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the raw product data with the final validation report."""
    import streamlit as st
    try:
        d_cp = data_df.copy()
        r_cp = final_report_df.copy()
        d_cp['PRODUCT_SET_SID'] = d_cp['PRODUCT_SET_SID'].astype(str).str.strip()
        r_cp['ProductSetSid'] = r_cp['ProductSetSid'].astype(str).str.strip()

        _code_to_path = st.session_state.get('support_files', {}).get('code_to_path', {})
        if _code_to_path and 'CATEGORY_CODE' in d_cp.columns:
            d_cp['FULL_CATEGORY_PATH'] = d_cp['CATEGORY_CODE'].apply(
                lambda c: _code_to_path.get(str(c).strip(), '') if pd.notna(c) else ''
            )
        else:
            d_cp['FULL_CATEGORY_PATH'] = ''

        merged = pd.merge(
            d_cp,
            r_cp[["ProductSetSid", "Status", "Reason", "Comment", "FLAG", "SellerName"]],
            left_on="PRODUCT_SET_SID", right_on="ProductSetSid", how='left'
        )
        if 'ProductSetSid' in merged.columns:
            merged.drop(columns=['ProductSetSid'], inplace=True)
        return merged
    except Exception as e:
        logger.error(f"prepare_full_data_merged: {e}")
        return pd.DataFrame()
