import pandas as pd

def _map_ordinal_series(s: pd.Series) -> pd.Series:
    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)
    
    # Year mapping
    if valset == {'1st', '2nd', '3rd', '4th'}:
        return s.map({'1st': 1, '2nd': 2, '3rd': 3, '4th': 4}).astype("Int64")
        
    if valset == {'Low', 'Medium', 'High'}:
        return s.map({'Low': 0, 'Medium': 1, 'High': 2}).astype("Int64")
    
    if valset == {'Poor', 'Average', 'Good'}:
        return s.map({'Poor': 0, 'Average': 1, 'Good': 2}).astype("Int64")

    mapping = set()
    sorted_vals = sorted(vals)
    for i in range(len(vals)):
        mapping[sorted_vals[i]] = i

    return s.astype(str).map(mapping).astype("Int64")

def build_features(df: pd.DataFrame, target_col: str = "burnout_level") -> pd.DataFrame:
    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    # === STEP 1: Identify Feature Types ===
    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"   Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    # === STEP 2: Split Categorical by Cardinality ===
    nomial_cols = ['gender', 'course']
    ordinal_cols = [c for c in obj_cols if c not in nomial_cols]
    
    print(f"   Nomial features: {len(nomial_cols)} | Ordinal features: {len(ordinal_cols)}")
    if nomial_cols:
        print(f"      Binary: {nomial_cols}")
    if ordinal_cols:
        print(f"      Multi-category: {ordinal_cols}")

    # === STEP 3: Apply Ordinal Encoding ===
    for c in ordinal_cols:
        df[c] = _map_ordinal_series(df[c].astype(str))
        print(f'Ordinal encoding for {c}')

   
    # === STEP 4: One-Hot Encoding for  ===
    # CRITICAL: drop_first=True prevents multicollinearity
    if nomial_cols:
        print(f"   Applying one-hot encoding to {len(nomial_cols)} columns...")
        original_shape = df.shape
        
        # Apply one-hot encoding with drop_first=True (same as serving)
        df = pd.get_dummies(df, columns=nomial_cols, drop_first=True)
        
        new_features = df.shape[1] - original_shape[1] + len(ordinal_cols)
        print(f"      Created {new_features} new features from {len(ordinal_cols)} categorical columns")

    # === STEP 5: Convert Boolean Columns ===
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   Converted {len(bool_cols)} boolean columns to int: {bool_cols}")
    
    # === STEP 6: Data Type Cleanup ===
    # Convert nullable integers (Int64) to standard integers for XGBoost
    for c in ordinal_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)

    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df