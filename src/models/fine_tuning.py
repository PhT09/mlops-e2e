import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score

def tune_model(X, y):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall_weighted")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Params:", study.best_params)
    return study.best_params

def tune_pretrained_model(base_model, X, y, n_trials=10):
    print(f"Starting Optuna hyperparameter tuning for fine-tuning on {len(X)} samples...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        
        # Mô hình khởi tạo với tham số mới
        model = XGBClassifier(**params)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        for train_idx, val_idx in cv.split(X_np, y_np):
            # Split
            X_train, y_train = X_np[train_idx], y_np[train_idx]
            X_val, y_val = X_np[val_idx], y_np[val_idx]
            
            # FIT VỚI CHUẨN MÔ HÌNH CŨ: xgb_model=base_model
            model.fit(X_train, y_train, xgb_model=base_model)
            
            # Evaluate
            preds = model.predict(X_val)
            scores.append(recall_score(y_val, preds, average="weighted", zero_division=0))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Best Fine-Tuning Params:", study.best_params)
    return study.best_params

import time
def fine_tune_xgboost(base_model, X_train, y_train, **params):
    if params:
        base_model.set_params(**params)
        
    t0 = time.time()
    # Tham số xgb_model=base_model báo hiệu cho XGBoost biết là cần train mọc cây nối tiếp
    base_model.fit(X_train, y_train, xgb_model=base_model)
    train_time = time.time() - t0
    
    return base_model, train_time