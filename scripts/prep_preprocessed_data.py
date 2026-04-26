import os, sys
import pandas as pd

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(project_root, "data", "raw", "student_mental_health_burnout_relabeled.csv")
OUT = os.path.join(project_root, "data", "processed", "student_mental_health_burnout_processed.csv")

# 1) load raw
df = pd.read_csv(RAW)

# 2) preprocess (drops id, fixes TotalCharges, etc.)
df = preprocess_data(df, target_col="burnout_level")

# 3) ensure target is 0/1/2 only if still object
if "burnout_level" in df.columns and df["burnout_level"].dtype == "object":
    df["burnout_level"] = df["burnout_level"].str.strip().map({'Low': 0, 'Medium': 1, 'High': 2}).astype("Int64")

# sanity checks                                                                 
assert df["burnout_level"].isna().sum() == 0, "burnout_level has NaNs after preprocess"
assert set(df["burnout_level"].unique()) <= {0, 1, 2}, "burnout_level not 0/1/2 after preprocess"

# 4) features
df_processed = build_features(df, target_col="burnout_level")

# 5) save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)
print(f"Processed dataset saved to {OUT} | Shape: {df_processed.shape}")