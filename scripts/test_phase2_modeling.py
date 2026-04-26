"""Phase 2: Modeling with LightGBM and Optuna hyperparameter tuning.

This script demonstrates LightGBM model training with automated hyperparameter
search using Optuna on the processed student burnout dataset.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import optuna

print("=== Phase 2: Modeling with LightGBM ===")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(project_root, "data", "processed", "student_mental_health_burnout_processed.csv")
TARGET_COL = "burnout_level"

df = pd.read_csv(DATA_PATH)

# Ensure target is numeric (0: Low, 1: Medium, 2: High)
if df[TARGET_COL].dtype == "object":
    df[TARGET_COL] = df[TARGET_COL].str.strip().map({'Low': 0, 'Medium': 1, 'High': 2})

assert df[TARGET_COL].isna().sum() == 0, "burnout_level has NaNs"
assert set(df[TARGET_COL].unique()) <= {0, 1, 2}, "burnout_level not 0/1/2"

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def objective(trial):
    """
    Optuna objective function for LightGBM hyperparameter optimization.
    
    LightGBM-specific parameters:
    - num_leaves: max number of leaves in one tree (replaces XGBoost's max_depth somewhat)
    - min_child_samples: minimum number of data points in a leaf (similar to XGBoost's min_child_weight)
    - subsample: fraction of training data to use per tree (same as XGBoost)
    - colsample_bytree: fraction of features to use per tree (same as XGBoost)
    - reg_alpha: L1 regularization (same as XGBoost)
    - reg_lambda: L2 regularization (same as XGBoost)
    - class_weight: 'balanced' handles class imbalance automatically (replaces XGBoost's scale_pos_weight)
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight": "balanced",  # Handle class imbalance
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import recall_score
    return recall_score(y_test, y_pred, average="weighted")


# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\n" + "="*60)
print("Optuna Hyperparameter Tuning Results")
print("="*60)
print(f"Best Recall: {study.best_value:.4f}")
print(f"\nBest Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("="*60)