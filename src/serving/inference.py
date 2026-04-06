import os
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
MODEL_DIR = "/app/model"

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development (OPTIONAL)
    try:
        # Try loading from local MLflow tracking
        import glob
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        mlruns_pattern = os.path.join(project_root, "mlruns", "*", "*", "artifacts", "model")
        local_model_paths = glob.glob(mlruns_pattern)
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE SCHEMA LOADING ===
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    feature_file = os.path.join(project_root, "artifacts", "feature_columns.json")
    import json
    with open(feature_file) as f:
        FEATURE_COLS = json.load(f)
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns from {feature_file}: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===

# Deterministic binary feature mappings (consistent with training)
ORDINAL_MAP = {
    'year': {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4},
    'stress_level': {'Low': 0, 'Medium': 1, 'High': 2},
    'sleep_quality': {'Poor': 0, 'Average': 1, 'Good': 2},
    'internet_quality': {'Poor': 0, 'Average': 1, 'Good': 2},
}

NOMINAL_COLS = ['gender', 'course']

# Numeric columns that need type coercion
NUMERIC_COLS = [
    "age", 
    "daily_study_hours", 
    "daily_sleep_hours", 
    "screen_time_hours", 
    "physical_activity_hours", 
    "anxiety_score", 
    "depression_score", 
    "academic_pressure_score", 
    "financial_stress_score", 
    "social_support_score", 
    "cgpa", 
    "attendance_percentage"
]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in ORDINAL_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Nominal Features ===
    df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert multiclass prediction (0/1/2) to actionable language
    if result == 2:
        return "High Burnout Risk"
    elif result == 1:
        return "Medium Burnout Risk"
    else:
        return "Low Burnout Risk"