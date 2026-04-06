import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

print("=== Phase 2: Modeling with XGBoost ===")

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(project_root, "data", "processed", "student_mental_health_burnout_processed.csv")
TARGET_COL = "burnout_level"

df = pd.read_csv(DATA_PATH)

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
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import recall_score
    return recall_score(y_test, y_pred, average="weighted")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best Params:", study.best_params)
print("Best Recall:", study.best_value)