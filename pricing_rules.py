import pandas as pd
import re


CATEGORY_MAX_PRICES_USD = {
    "Automobile": 2000.0,            
    "Computing": 5000.0,             
    "Electronics": 5000.0,           
    "Home & Office": 3000.0,         
    "Industrial & Scientific": 2500.0, 
    "Garden & Outdoors": 2000.0,     
    "Musical Instruments": 2000.0,   
    "Sporting Goods": 2000.0,        
    "Gaming": 1200.0,                
    "Baby Products": 600.0,          
    "Toys & Games": 500.0,           
    "Fashion": 500.0,                
    "Health & Beauty": 500.0,        
    "Grocery": 300.0,                
    "Books, Movies & Music": 250.0,  
    "Pet Supplies": 200.0            
}

def check_wrong_price(data: pd.DataFrame) -> pd.DataFrame:
    """
    Flags products with missing/zero prices or extreme, unrealistic discounts (>95%).
    """
    if not {'GLOBAL_PRICE', 'GLOBAL_SALE_PRICE'}.issubset(data.columns): 
        return pd.DataFrame(columns=data.columns)

    d = data.copy()
    
    # Convert to numeric, coercing errors to NaN
    d['price'] = pd.to_numeric(d['GLOBAL_PRICE'], errors='coerce')
    d['sale_price'] = pd.to_numeric(d['GLOBAL_SALE_PRICE'], errors='coerce')

    # Condition 1: Base price is <= 0 OR Sale price is present and <= 0
    invalid_base = d['price'].notna() & (d['price'] <= 0)
    invalid_sale = d['sale_price'].notna() & (d['sale_price'] <= 0)
    invalid_price = invalid_base | invalid_sale

    # Condition 2: Discount is > 95%
    valid_prices = (d['price'] > 0) & d['sale_price'].notna() & (d['sale_price'] > 0)
    discount_pct = 1 - (d['sale_price'] / d['price'])
    extreme_discount = valid_prices & (discount_pct > 0.95)

    flagged = d[invalid_price | extreme_discount].copy()

    if not flagged.empty:
        def build_comment(row):
            p = row['price']
            sp = row['sale_price']
            if pd.isna(p) or p <= 0 or (pd.notna(sp) and sp <= 0):
                return f"Invalid price detected (Price: {p}, Sale: {sp})"
            return f"Extreme discount > 95% (Price: {p}, Sale: {sp})"
        
        flagged['Comment_Detail'] = flagged.apply(build_comment, axis=1)

    return (
        flagged
        .drop(columns=['price', 'sale_price'], errors='ignore')
        .drop_duplicates(subset=['PRODUCT_SET_SID'])
    )

def check_category_max_price(data: pd.DataFrame, max_price_map: dict, code_to_path: dict = None) -> pd.DataFrame:
    """
    Flags products where the price exceeds the realistic maximum for its category.
    Resolves the category code to the full path to apply rules to all subcategories.
    """
    if not {'CATEGORY_CODE', 'GLOBAL_PRICE', 'GLOBAL_SALE_PRICE'}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    if code_to_path is None:
        code_to_path = {}

    d = data.copy()
    
    # Ensure prices are numeric (defaulting missing to 0)
    d['price'] = pd.to_numeric(d['GLOBAL_PRICE'], errors='coerce').fillna(0)
    d['sale_price'] = pd.to_numeric(d['GLOBAL_SALE_PRICE'], errors='coerce').fillna(0)
    
    # Get the highest price the seller is trying to charge
    d['max_listed_price'] = d[['price', 'sale_price']].max(axis=1)

    flagged_indices = []
    comment_map = {}

    for idx, row in d.iterrows():
        listed_price = row['max_listed_price']
        
        # Skip items with zero/negative price (handled by check_wrong_price)
        if listed_price <= 0:
            continue

        # 1. Get the raw category code
        cat_code = str(row.get('CATEGORY_CODE', '')).strip()
        
        # 2. Look up the full path (e.g., "Fashion > Men > Shoes") 
        # Fallback to the text in 'CATEGORY' column if code isn't found
        full_path = code_to_path.get(cat_code, str(row.get('CATEGORY', '')))
        
        # 3. Extract the Level 1 root category (e.g., "Fashion")
        level_1_cat = full_path.split('>')[0].strip() if '>' in full_path else full_path.strip()

        # 4. Look up the price cap (defaulting to $1,000 if unknown)
        cap = max_price_map.get(level_1_cat, 1000.0)

        if listed_price > cap:
            flagged_indices.append(idx)
            comment_map[idx] = f"Price (${listed_price:,.2f}) exceeds realistic max for '{level_1_cat}' (${cap:,.2f})"

    if not flagged_indices:
        return pd.DataFrame(columns=data.columns)

    result = d.loc[flagged_indices].copy()
    result['Comment_Detail'] = result.index.map(comment_map)
    
    return (
        result
        .drop(columns=['price', 'sale_price', 'max_listed_price'], errors='ignore')
        .drop_duplicates(subset=['PRODUCT_SET_SID'])
    )
