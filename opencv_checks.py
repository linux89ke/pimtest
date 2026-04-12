"""
opencv_checks.py
================
Drop-in OpenCV-powered image validation checks for the Jumia Product
Validation Tool.  Every public function follows the same contract as the
existing check_image_* helpers in main.py:

    check_*(data: pd.DataFrame, **kwargs) -> pd.DataFrame

The returned DataFrame contains flagged rows from `data` plus a
'Comment_Detail' column and, importantly, a new 'CV_FLAG' column set to
True so the UI can highlight these rows with a camera badge.

Usage
-----
1.  pip install opencv-python-headless   (add to requirements.txt)
2.  from opencv_checks import (
        check_image_product_coverage,
        check_image_duplicate_visual,
        check_image_color_mismatch,
        check_image_exposure,
        check_image_blurry_cv,          # upgraded replacement
        CV_FLAG_COL,
        render_cv_badge,
    )
3.  Register each function in _reg.REGISTRY and the validations list
    inside validate_products() — see the WIRING GUIDE comment at the
    bottom of this file.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ── Public constants ──────────────────────────────────────────────────────────
CV_FLAG_COL = "CV_Detected"          # bool column added to flagged rows
CV_BADGE_HTML = (
    "<span title='Flagged by OpenCV image analysis' style='"
    "display:inline-flex;align-items:center;gap:3px;"
    "background:#fff3cd;color:#856404;border:1px solid #ffc107;"
    "border-radius:4px;padding:1px 6px;font-size:11px;font-weight:600;"
    "white-space:nowrap;'>📷 CV</span>"
)


def render_cv_badge() -> str:
    """Return the HTML badge string for use in Streamlit markdown cells."""
    return CV_BADGE_HTML


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_cv2_image(url: str, timeout: int = 5) -> Optional[np.ndarray]:
    """
    Download *url* and return a cv2 BGR ndarray, or None on any failure.
    Uses stream=True so we don't download the full body before decoding.
    """
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.content
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img          # shape (H, W, 3)  BGR
    except Exception:
        return None


def _mark_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp CV_FLAG_COL = True on every row of *df*."""
    df = df.copy()
    df[CV_FLAG_COL] = True
    return df


def _url_series(data: pd.DataFrame) -> pd.Series:
    return data["MAIN_IMAGE"].astype(str)


# ── HSV colour range table ────────────────────────────────────────────────────
# Each entry: list of (lower, upper) tuples in OpenCV HSV space
# H: 0-180   S: 0-255   V: 0-255
# Red wraps around 0/180 so it needs two ranges.
_HSV_COLOR_RANGES: dict[str, list[tuple]] = {
    "red":    [((0,   70,  50), (10,  255, 255)),
               ((170, 70,  50), (180, 255, 255))],
    "orange": [((11,  70,  50), (25,  255, 255))],
    "yellow": [((26,  70,  50), (34,  255, 255))],
    "green":  [((35,  40,  40), (85,  255, 255))],
    "blue":   [((100, 60,  50), (130, 255, 255))],
    "purple": [((131, 50,  50), (160, 255, 255))],
    "pink":   [((161, 50,  50), (169, 255, 255))],
    "white":  [((0,   0,  200), (180,  30, 255))],
    "black":  [((0,   0,    0), (180, 255,  40))],
    "grey":   [((0,   0,   41), (180,  25, 200))],
    "gray":   [((0,   0,   41), (180,  25, 200))],
    "brown":  [((10,  60,  20), (20,  220, 160))],
    "silver": [((0,   0,  150), (180,  25, 220))],
    "gold":   [((22,  80,  80), (32,  255, 255))],
}

_BG_LOWER = np.array([0,   0, 200])   # HSV lower bound for white bg
_BG_UPPER = np.array([180, 30, 255])  # HSV upper bound for white bg


def _bg_mask(hsv: np.ndarray) -> np.ndarray:
    """Return a mask that is 255 where pixels are background (white/near-white)."""
    return cv2.inRange(hsv, _BG_LOWER, _BG_UPPER)


def _colour_key(raw: str) -> Optional[str]:
    """Map a raw color string to the closest key in _HSV_COLOR_RANGES."""
    c = raw.strip().lower()
    for key in _HSV_COLOR_RANGES:
        if key in c:
            return key
    return None


# =============================================================================
# 1.  Product coverage — bounding box vs canvas
# =============================================================================

def check_image_product_coverage(
    data: pd.DataFrame,
    min_coverage: float = 0.55,   # below this → rejected
    warn_coverage: float = 0.75,  # below this → advisory only
    **kwargs,
) -> pd.DataFrame:
    """
    Uses Canny edge detection + contour bounding rect to estimate what
    fraction of the image canvas the product occupies.

    *min_coverage*   – products below this are flagged as rejected.
    *warn_coverage*  – products between min and warn are stored as advisory
                       commentary in st.session_state['_coverage_commentary']
                       (same pattern as check_image_blurry).
    """
    if "MAIN_IMAGE" not in data.columns:
        return pd.DataFrame(columns=data.columns)

    target = data[_url_series(data).str.startswith("http")].copy()
    if target.empty:
        return pd.DataFrame(columns=data.columns)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    def analyse(url: str):
        img = _fetch_cv2_image(url)
        if img is None:
            return url, None
        h, w = img.shape[:2]
        canvas = h * w
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bg = _bg_mask(hsv)
        product = cv2.bitwise_not(bg)
        closed = cv2.morphologyEx(product, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return url, 0.0
        all_pts = np.concatenate(contours)
        x, y, bw, bh = cv2.boundingRect(all_pts)
        coverage = (bw * bh) / canvas
        return url, round(coverage, 3)

    url_cov: dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(analyse, url): url
            for url in target["MAIN_IMAGE"].unique()
        }
        for f in concurrent.futures.as_completed(futures):
            url, cov = f.result()
            if cov is not None:
                url_cov[url] = cov

    if not url_cov:
        return pd.DataFrame(columns=data.columns)

    reject_map: dict[str, str] = {}
    commentary: dict[str, str] = {}
    for url, cov in url_cov.items():
        pct = f"{cov * 100:.0f}%"
        if cov < min_coverage:
            reject_map[url] = (
                f"Product covers only {pct} of image "
                f"(minimum {min_coverage * 100:.0f}% required)"
            )
        elif cov < warn_coverage:
            commentary[url] = (
                f"Product coverage low ({pct}) — consider reframing the photo"
            )

    # Store advisory (non-rejected) items in session state
    try:
        existing = st.session_state.get("_coverage_commentary", {})
        for row in target.itertuples(index=False):
            url = str(getattr(row, "MAIN_IMAGE", ""))
            sid = str(getattr(row, "PRODUCT_SET_SID", ""))
            if url in commentary:
                existing[sid] = commentary[url]
        st.session_state["_coverage_commentary"] = existing
    except Exception:
        pass

    if not reject_map:
        return pd.DataFrame(columns=data.columns)

    flagged = target[target["MAIN_IMAGE"].isin(reject_map)].copy()
    flagged["Comment_Detail"] = flagged["MAIN_IMAGE"].map(reject_map)
    return _mark_cv(flagged).drop_duplicates(subset=["PRODUCT_SET_SID"])


# =============================================================================
# 2.  Visual duplicate detection — ORB feature matching
# =============================================================================

def check_image_duplicate_visual(
    data: pd.DataFrame,
    match_ratio_threshold: float = 0.80,
    max_keypoints: int = 500,
    max_per_seller: int = 40,          # skip sellers with huge catalogs inline
    **kwargs,
) -> pd.DataFrame:
    """
    Uses ORB keypoints + BFMatcher to detect visually identical images
    uploaded by the SAME SELLER under different product titles.

    Cross-seller duplicates are intentionally ignored (legitimate competing
    listings).  Sellers with > max_per_seller images are skipped to keep
    runtime bounded — run them in a background job.
    """
    required = {"MAIN_IMAGE", "SELLER_NAME", "PRODUCT_SET_SID"}
    if not required.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    target = data[_url_series(data).str.startswith("http")].copy()
    if target.empty:
        return pd.DataFrame(columns=data.columns)

    orb = cv2.ORB_create(nfeatures=max_keypoints)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_descriptor(url: str):
        img = _fetch_cv2_image(url)
        if img is None:
            return url, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = orb.detectAndCompute(gray, None)
        return url, des

    url_to_des: dict[str, np.ndarray] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(get_descriptor, url): url
            for url in target["MAIN_IMAGE"].unique()
        }
        for f in concurrent.futures.as_completed(futures):
            url, des = f.result()
            if des is not None and len(des) >= 10:
                url_to_des[url] = des

    if not url_to_des:
        return pd.DataFrame(columns=data.columns)

    flagged_sids: set[str] = set()
    comment_map: dict[str, str] = {}

    seller_col = "_seller_lower" if "_seller_lower" in target.columns else "SELLER_NAME"

    for seller, grp in target.groupby(seller_col):
        rows = (
            grp[grp["MAIN_IMAGE"].isin(url_to_des)]
            .reset_index(drop=True)
        )
        if len(rows) < 2 or len(rows) > max_per_seller:
            continue

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                url_i = rows.loc[i, "MAIN_IMAGE"]
                url_j = rows.loc[j, "MAIN_IMAGE"]
                if url_i == url_j:
                    continue
                des_i = url_to_des[url_i]
                des_j = url_to_des[url_j]
                matches = bf.match(des_i, des_j)
                n_total = min(len(des_i), len(des_j))
                if n_total == 0:
                    continue
                ratio = len(matches) / n_total
                if ratio >= match_ratio_threshold:
                    sid_i = str(rows.loc[i, "PRODUCT_SET_SID"])
                    sid_j = str(rows.loc[j, "PRODUCT_SET_SID"])
                    flagged_sids.add(sid_i)
                    flagged_sids.add(sid_j)
                    pct = f"{ratio * 100:.0f}%"
                    comment_map.setdefault(
                        sid_i,
                        f"Visual duplicate detected ({pct} keypoint match) — same image as SID {sid_j}",
                    )
                    comment_map.setdefault(
                        sid_j,
                        f"Visual duplicate detected ({pct} keypoint match) — same image as SID {sid_i}",
                    )

    if not flagged_sids:
        return pd.DataFrame(columns=data.columns)

    flagged = data[
        data["PRODUCT_SET_SID"].astype(str).isin(flagged_sids)
    ].copy()
    flagged["Comment_Detail"] = (
        flagged["PRODUCT_SET_SID"].astype(str).map(comment_map)
    )
    return _mark_cv(flagged).drop_duplicates(subset=["PRODUCT_SET_SID"])


# =============================================================================
# 3.  Color mismatch — HSV masking vs declared COLOR field
# =============================================================================

_MIN_COLOUR_FRACTION = 0.05   # declared colour must cover ≥5 % of product pixels

def check_image_color_mismatch(
    data: pd.DataFrame,
    color_categories: Optional[list] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    For products with a declared COLOR value, checks whether a meaningful
    fraction of the product's pixels (non-background) actually matches that
    hue using HSV range masking.

    Only runs on products whose category is in *color_categories* when that
    list is provided.
    """
    if "MAIN_IMAGE" not in data.columns or "COLOR" not in data.columns:
        return pd.DataFrame(columns=data.columns)

    null_like = {"nan", "", "none", "null", "n/a", "na", "-"}

    target = data[
        _url_series(data).str.startswith("http")
        & ~data["COLOR"].astype(str).str.lower().str.strip().isin(null_like)
    ].copy()

    if color_categories and "_cat_clean" in target.columns:
        try:
            from data_utils import clean_category_code
            target = target[
                target["_cat_clean"].isin(
                    {clean_category_code(c) for c in color_categories}
                )
            ]
        except ImportError:
            pass

    if target.empty:
        return pd.DataFrame(columns=data.columns)

    # Build url → declared colour key map
    url_key: dict[str, str] = {}
    for _, row in target.iterrows():
        url = str(row["MAIN_IMAGE"])
        key = _colour_key(str(row.get("COLOR", "")))
        if key:
            url_key[url] = key

    if not url_key:
        return pd.DataFrame(columns=data.columns)

    def analyse(url: str, declared_key: str):
        img = _fetch_cv2_image(url)
        if img is None or declared_key not in _HSV_COLOR_RANGES:
            return url, None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bg    = _bg_mask(hsv)
        fg    = cv2.bitwise_not(bg)
        product_px = int(cv2.countNonZero(fg))
        if product_px < 200:
            return url, None

        ranges = _HSV_COLOR_RANGES[declared_key]
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            combined = cv2.bitwise_or(
                combined,
                cv2.inRange(hsv, np.array(lo), np.array(hi)),
            )
        colour_on_product = cv2.bitwise_and(combined, fg)
        frac = int(cv2.countNonZero(colour_on_product)) / product_px
        return url, round(frac, 4)

    url_frac: dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(analyse, url, key): url
            for url, key in url_key.items()
        }
        for f in concurrent.futures.as_completed(futures):
            url, frac = f.result()
            if frac is not None:
                url_frac[url] = frac

    if not url_frac:
        return pd.DataFrame(columns=data.columns)

    flagged_rows = []
    seen_sids: set[str] = set()
    for _, row in target.iterrows():
        url = str(row["MAIN_IMAGE"])
        if url not in url_frac:
            continue
        sid = str(row["PRODUCT_SET_SID"])
        if sid in seen_sids:
            continue
        frac = url_frac[url]
        if frac < _MIN_COLOUR_FRACTION:
            r = row.copy()
            r["Comment_Detail"] = (
                f"Color mismatch: declared '{row['COLOR']}' but only "
                f"{frac * 100:.1f}% of product pixels match that hue"
            )
            r[CV_FLAG_COL] = True
            flagged_rows.append(r)
            seen_sids.add(sid)

    if not flagged_rows:
        return pd.DataFrame(columns=data.columns)
    return pd.DataFrame(flagged_rows).drop_duplicates(subset=["PRODUCT_SET_SID"])


# =============================================================================
# 4.  Exposure check — grayscale histogram analysis
# =============================================================================

def check_image_exposure(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Flags images that are:
    - Overexposed : ≥ 88 % of pixels in the top-10 % brightness range (230-255)
    - Underexposed: ≥ 88 % of pixels in the bottom-10 % brightness range (0-25)

    Both indicate the product is not visible enough for a buying decision.
    """
    if "MAIN_IMAGE" not in data.columns:
        return pd.DataFrame(columns=data.columns)

    target = data[_url_series(data).str.startswith("http")].copy()
    if target.empty:
        return pd.DataFrame(columns=data.columns)

    OVER_THRESHOLD  = 0.88
    UNDER_THRESHOLD = 0.88

    def analyse(url: str):
        img = _fetch_cv2_image(url)
        if img is None:
            return url, None
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist  = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total = hist.sum()
        if total == 0:
            return url, None
        bright = hist[230:].sum() / total
        dark   = hist[:26].sum()  / total
        if bright >= OVER_THRESHOLD:
            return url, f"Overexposed — {bright * 100:.0f}% of pixels at max brightness (washed out)"
        if dark >= UNDER_THRESHOLD:
            return url, f"Underexposed — {dark * 100:.0f}% of pixels at min brightness (too dark)"
        return url, None

    url_issues: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(analyse, url): url
            for url in target["MAIN_IMAGE"].unique()
        }
        for f in concurrent.futures.as_completed(futures):
            url, issue = f.result()
            if issue:
                url_issues[url] = issue

    if not url_issues:
        return pd.DataFrame(columns=data.columns)

    flagged = target[target["MAIN_IMAGE"].isin(url_issues)].copy()
    flagged["Comment_Detail"] = flagged["MAIN_IMAGE"].map(url_issues)
    return _mark_cv(flagged).drop_duplicates(subset=["PRODUCT_SET_SID"])


# =============================================================================
# 5.  Blurry image — Laplacian variance (upgrades the PIL resolution check)
# =============================================================================

def check_image_blurry_cv(
    data: pd.DataFrame,
    blur_threshold: float = 80.0,
    **kwargs,
) -> pd.DataFrame:
    """
    OpenCV Laplacian variance sharpness check.
    Replaces the old PIL pixel-count heuristic (check_image_blurry in main.py).

    variance < blur_threshold            → rejected as blurry
    blur_threshold ≤ variance < 2×       → advisory only (session state)
    """
    if "MAIN_IMAGE" not in data.columns:
        return pd.DataFrame(columns=data.columns)

    target = data[_url_series(data).str.startswith("http")].copy()
    if target.empty:
        return pd.DataFrame(columns=data.columns)

    def analyse(url: str):
        img = _fetch_cv2_image(url)
        if img is None:
            return url, None
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return url, round(float(lap_var), 2)

    url_var: dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(analyse, url): url
            for url in target["MAIN_IMAGE"].unique()
        }
        for f in concurrent.futures.as_completed(futures):
            url, var = f.result()
            if var is not None:
                url_var[url] = var

    if not url_var:
        return pd.DataFrame(columns=data.columns)

    reject_map: dict[str, str] = {}
    commentary: dict[str, str] = {}
    for url, var in url_var.items():
        if var < blur_threshold:
            reject_map[url] = (
                f"Image blurry — sharpness score {var:.1f} "
                f"(threshold {blur_threshold:.0f})"
            )
        elif var < blur_threshold * 2:
            commentary[url] = (
                f"Image sharpness marginal (score {var:.1f}) — consider re-shooting"
            )

    # Advisory rows → session state (same pattern as existing check_image_blurry)
    try:
        existing = st.session_state.get("_image_blurry_commentary", {})
        for row in target.itertuples(index=False):
            url = str(getattr(row, "MAIN_IMAGE", ""))
            sid = str(getattr(row, "PRODUCT_SET_SID", ""))
            if url in commentary:
                existing[sid] = commentary[url]
        st.session_state["_image_blurry_commentary"] = existing
    except Exception:
        pass

    if not reject_map:
        return pd.DataFrame(columns=data.columns)

    flagged = target[target["MAIN_IMAGE"].isin(reject_map)].copy()
    flagged["Comment_Detail"] = flagged["MAIN_IMAGE"].map(reject_map)
    return _mark_cv(flagged).drop_duplicates(subset=["PRODUCT_SET_SID"])


# =============================================================================
# WIRING GUIDE — paste these snippets into main.py
# =============================================================================
"""
──────────────────────────────────────────────────────────────────────────────
A)  IMPORT  (top of main.py)
──────────────────────────────────────────────────────────────────────────────

from opencv_checks import (
    check_image_product_coverage,
    check_image_duplicate_visual,
    check_image_color_mismatch,
    check_image_exposure,
    check_image_blurry_cv,
    CV_FLAG_COL,
    render_cv_badge,
)

──────────────────────────────────────────────────────────────────────────────
B)  FLAG_RELEVANT_COLS  (add these 4 entries)
──────────────────────────────────────────────────────────────────────────────

    "Image Product Coverage":  ["MAIN_IMAGE"],
    "Image Visual Duplicate":  ["MAIN_IMAGE", "SELLER_NAME"],
    "Image Color Mismatch":    ["MAIN_IMAGE", "COLOR", "CATEGORY_CODE"],
    "Image Exposure":          ["MAIN_IMAGE"],

──────────────────────────────────────────────────────────────────────────────
C)  validations list inside validate_products()
    Add after the existing image check lines:
──────────────────────────────────────────────────────────────────────────────

    ("Image Product Coverage", check_image_product_coverage, {}),
    ("Image Visual Duplicate", check_image_duplicate_visual, {}),
    ("Image Color Mismatch",   check_image_color_mismatch,
        {"color_categories": support_files.get("color_categories", [])}),
    ("Image Exposure",         check_image_exposure, {}),

    # Optional: replace old PIL-based blurry check entirely
    # ("Image Blurry",          check_image_blurry_cv, {}),

──────────────────────────────────────────────────────────────────────────────
D)  _reg.REGISTRY  (add inside the if _reg is not None block)
──────────────────────────────────────────────────────────────────────────────

    'check_image_product_coverage': check_image_product_coverage,
    'check_image_duplicate_visual': check_image_duplicate_visual,
    'check_image_color_mismatch':   check_image_color_mismatch,
    'check_image_exposure':         check_image_exposure,
    'check_image_blurry_cv':        check_image_blurry_cv,

──────────────────────────────────────────────────────────────────────────────
E)  flags_mapping JSON  (add an entry for each new flag)
──────────────────────────────────────────────────────────────────────────────

    "Image Product Coverage": {
        "reason": "1000007 - Poor Image",
        "en": "Product does not fill enough of the image canvas",
        "fr": "Le produit ne remplit pas suffisamment le cadre de l'image",
        "ar": "المنتج لا يملأ مساحة كافية من الصورة"
    },
    "Image Visual Duplicate": {
        "reason": "1000007 - Duplicate Image",
        "en": "Image is visually identical to another listing by this seller",
        "fr": "L'image est identique à une autre annonce de ce vendeur",
        "ar": "الصورة مطابقة بصريًا لإدراج آخر من هذا البائع"
    },
    "Image Color Mismatch": {
        "reason": "1000007 - Wrong Variation",
        "en": "Declared color does not match the product image",
        "fr": "La couleur déclarée ne correspond pas à l'image du produit",
        "ar": "اللون المُعلن لا يتطابق مع صورة المنتج"
    },
    "Image Exposure": {
        "reason": "1000007 - Poor Image",
        "en": "Image is overexposed or underexposed",
        "fr": "L'image est surexposée ou sous-exposée",
        "ar": "الصورة مضاءة بشكل مفرط أو ناقص"
    },

──────────────────────────────────────────────────────────────────────────────
F)  HIGHLIGHT BADGE in the results table
    Inside render_flag_expander (ui_components.py) or wherever you build
    the display DataFrame, add this column to flag OpenCV rows:
──────────────────────────────────────────────────────────────────────────────

    from opencv_checks import CV_FLAG_COL, render_cv_badge

    # After merging flagged rows into display_df:
    if CV_FLAG_COL in display_df.columns:
        display_df["Source"] = display_df[CV_FLAG_COL].apply(
            lambda v: render_cv_badge() if v is True else ""
        )

    # Then include "Source" as a Streamlit column_config:
    st.dataframe(
        display_df,
        column_config={
            "Source": st.column_config.Column(
                label="Detection",
                help="📷 CV = flagged by OpenCV image analysis",
                width="small",
            )
        },
        ...
    )

──────────────────────────────────────────────────────────────────────────────
G)  METRICS BADGE in the summary section
    In the metrics block after validation, add a CV-specific counter:
──────────────────────────────────────────────────────────────────────────────

    _cv_flags = [
        "Image Product Coverage", "Image Visual Duplicate",
        "Image Color Mismatch",   "Image Exposure", "Image Blurry",
    ]
    _cv_count = int(fr[fr["FLAG"].isin(_cv_flags)].shape[0])
    # Then render as an extra metric tile:
    st.metric("CV Detected", _cv_count, help="Products flagged by OpenCV image analysis")

──────────────────────────────────────────────────────────────────────────────
H)  ADVISORY COVERAGE commentary  (add to results section alongside the
    existing _blurry_commentary block in main.py)
──────────────────────────────────────────────────────────────────────────────

    _coverage_commentary = st.session_state.get("_coverage_commentary", {})
    _cov_in_scope = {
        sid: comment for sid, comment in _coverage_commentary.items()
        if fr[fr["ProductSetSid"] == sid]["Status"].eq("Approved").any()
    }
    if _cov_in_scope:
        with st.expander(
            f":material/photo_camera: Low Coverage Advisory — "
            f"{len(_cov_in_scope)} product(s) (not rejected)",
            expanded=False,
        ):
            st.info(
                "These products passed validation but the product occupies "
                "less than 75% of the image canvas. Not rejected — advisory only.",
                icon=":material/info:",
            )
            _rows = []
            for _sid, _comment in _cov_in_scope.items():
                _row = data[data["PRODUCT_SET_SID"] == _sid]
                if not _row.empty:
                    _rows.append({
                        "PRODUCT_SET_SID": _sid,
                        "NAME": _row.iloc[0].get("NAME", ""),
                        "SELLER_NAME": _row.iloc[0].get("SELLER_NAME", ""),
                        "Coverage Note": _comment,
                    })
            if _rows:
                st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)
"""
