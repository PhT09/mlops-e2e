"""ONNX-based inference pipeline for LightGBM student burnout prediction.

This module provides a production-ready inference API that:
1. Loads a pre-trained LightGBM model exported to ONNX format
2. Applies the same feature engineering transformations used during training
3. Returns human-readable burnout risk predictions

Key architectural decision:
-----------------------
We use ONNX Runtime (not the LightGBM Python API) for inference because:
- **Framework independence**: onnxruntime is model-agnostic and can load ONNX
  models from any source framework (LightGBM, XGBoost, PyTorch, TensorFlow, etc.)
  without requiring the original framework to be installed.
- **Performance**: onnxruntime's C++ inference engine is optimized for production
  serving and typically faster than Python-based model APIs.
- **Deployment flexibility**: The same ONNX model can be deployed to edge devices,
  mobile apps, or cloud services without Python dependencies.
- **Version stability**: ONNX format provides a stable contract between training
  and serving, reducing risks from library version mismatches.

The inference code here is completely framework-agnostic — changing from LightGBM
to another model type only requires re-exporting the ONNX file; this inference
code remains unchanged.
"""

import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Module-level constants (resolved relative to this file's location)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")

# ---------------------------------------------------------------------------
# SINGLETON: Load ONNX model at import time (once, not per-request)
# ---------------------------------------------------------------------------
_ONNX_PATH = os.path.join(_ARTIFACTS_DIR, "model.onnx")

try:
    # CPUExecutionProvider: explicit provider list ensures deterministic behavior
    # and avoids onnxruntime falling through to a GPU/TensorRT provider that may
    # not be available on the serving machine.
    _SESSION = ort.InferenceSession(
        _ONNX_PATH,
        providers=["CPUExecutionProvider"],
    )
    _INPUT_NAME = _SESSION.get_inputs()[0].name   # typically "float_input"
    print(f"[ONNX] InferenceSession ready — {_ONNX_PATH}")
except Exception as _e:
    raise RuntimeError(
        f"[ONNX] Failed to load model from '{_ONNX_PATH}'. "
        "Run 'python scripts/run_pipeline.py' first to generate the file.\n"
        f"Underlying error: {_e}"
    )

# ---------------------------------------------------------------------------
# SINGLETON: Load feature schema (exact column order from training)
# ---------------------------------------------------------------------------
_FEATURE_FILE = os.path.join(_ARTIFACTS_DIR, "feature_columns.json")

try:
    with open(_FEATURE_FILE) as _f:
        FEATURE_COLS = json.load(_f)
    print(f"[ONNX] Feature schema loaded — {len(FEATURE_COLS)} columns")
except Exception as _e:
    raise RuntimeError(
        f"[ONNX] Failed to load feature columns from '{_FEATURE_FILE}': {_e}"
    )

# ---------------------------------------------------------------------------
# FEATURE TRANSFORMATION CONSTANTS
# (kept exactly in sync with build_features.py / training pipeline)
# ---------------------------------------------------------------------------

# Ordinal features: deterministic integer mappings that mirror build_features.py
ORDINAL_MAP = {
    "year":             {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4},
    "stress_level":     {"Low": 0, "Medium": 1, "High": 2},
    "sleep_quality":    {"Poor": 0, "Average": 1, "Good": 2},
    "internet_quality": {"Poor": 0, "Average": 1, "Good": 2},
}

# Nominal (one-hot encoded) features — must match pd.get_dummies(drop_first=True)
NOMINAL_COLS = ["gender", "course"]

# Numeric features that may arrive as strings from external JSON payloads
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
    "attendance_percentage",
]

# ---------------------------------------------------------------------------
# Label mapping: ONNX returns int64 class index → human-readable string
# ---------------------------------------------------------------------------
_LABEL_MAP = {0: "Low Burnout Risk", 1: "Medium Burnout Risk", 2: "High Burnout Risk"}


# ---------------------------------------------------------------------------
# Internal transformation (mirrors training pipeline — do not change lightly)
# ---------------------------------------------------------------------------
def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering steps used during training.

    This function MUST stay in sync with ``src/features/build_features.py``
    and the preprocessing done in ``scripts/run_pipeline.py``.  Any drift
    between the two will silently degrade model accuracy.
    """
    df = df.copy()

    # Strip accidental whitespace from column names (defensive)
    df.columns = df.columns.str.strip()

    # --- STEP 1: Numeric coercion ---
    # Inputs arrive as JSON strings from FastAPI; coerce to float and fill NaN.
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # --- STEP 2: Ordinal encoding ---
    # Apply deterministic integer mappings (identical to build_features.py)
    for c, mapping in ORDINAL_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")    # nullable int so .map() NaNs are preserved
                .fillna(0)
                .astype(int)
            )

    # --- STEP 3: One-hot encoding for nominal features ---
    # drop_first=True mirrors training; column names must match FEATURE_COLS
    df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True)

    # --- STEP 4: Boolean → integer ---
    # ONNX models expect numeric arrays; boolean columns must be cast to int.
    # This applies to one-hot encoded dummy columns created by pd.get_dummies().
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # --- STEP 5: Align with training feature schema ---
    # Missing OHE columns (unseen categories) are filled with 0.
    # Extra unexpected columns are dropped silently.
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


# ---------------------------------------------------------------------------
# Public API — must NOT change (src/app/main.py depends on this signature)
# ---------------------------------------------------------------------------
def predict(input_dict: dict) -> str:
    """
    Transform raw input and return a burnout risk label.

    Parameters
    ----------
    input_dict : dict
        Raw key-value pairs from the FastAPI request (Pydantic .dict()).

    Returns
    -------
    str
        One of: "Low Burnout Risk", "Medium Burnout Risk", "High Burnout Risk".
    """
    # -- 1. Wrap in a single-row DataFrame --
    df = pd.DataFrame([input_dict])

    # -- 2. Apply feature engineering (ordinal + OHE + alignment) --
    df_enc = _serve_transform(df)

    # -- 3. Cast to float32 BEFORE handing to ONNX --
    # WHY float32:  The ONNX graph was compiled with FloatTensorType (= float32).
    # onnxruntime enforces strict type matching and will raise an InvalidArgument
    # error if the input dtype is float64 (numpy default) or int.  Explicit cast
    # here prevents silent type promotion inside the C++ runtime.
    X = df_enc.values.astype(np.float32)

    # -- 4. Run ONNX inference --
    # session.run() returns [labels, probabilities] because zipmap=False was set
    # in export_onnx.py.  labels has shape (n_rows,) with dtype int64.
    outputs = _SESSION.run(None, {_INPUT_NAME: X})
    class_idx = int(outputs[0][0])   # scalar int for the single input row

    # -- 5. Map class index to business label --
    return _LABEL_MAP.get(class_idx, "Unknown Burnout Risk")