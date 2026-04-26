import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
import time

def tune_model(X, y):
    """Hyperparameter tuning for LightGBM using Optuna.
    
    Search space is designed for multi-class classification with
    imbalanced classes (student burnout prediction).
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
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
        scores = cross_val_score(model, X, y, cv=3, scoring="recall_weighted")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Params:", study.best_params)
    return study.best_params

def tune_pretrained_model(base_model, X, y, n_trials=10):
    """Fine-tune a pretrained LightGBM model using Optuna.
    
    LightGBM supports incremental training via the init_model parameter,
    which allows adding more trees to an existing booster.
    
    Parameters
    ----------
    base_model : LGBMClassifier
        Previously trained LightGBM model to fine-tune.
    X : array-like
        Training features.
    y : array-like
        Training labels.
    n_trials : int
        Number of Optuna trials to run.
    
    Returns
    -------
    dict
        Best hyperparameters found during tuning.
    """
    print(f"Starting Optuna hyperparameter tuning for fine-tuning on {len(X)} samples...")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        
        # Initialize model with new parameters
        model = LGBMClassifier(**params)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        for train_idx, val_idx in cv.split(X_np, y_np):
            # Split
            X_train, y_train = X_np[train_idx], y_np[train_idx]
            X_val, y_val = X_np[val_idx], y_np[val_idx]
            
            # Fine-tune: init_model parameter allows incremental training
            # This adds new trees to the existing booster from base_model
            model.fit(
                X_train, y_train,
                init_model=base_model  # Start from pretrained model
            )
            
            # Evaluate
            preds = model.predict(X_val)
            scores.append(recall_score(y_val, preds, average="weighted", zero_division=0))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Best Fine-Tuning Params:", study.best_params)
    return study.best_params

def fine_tune_lightgbm(base_model, X_train, y_train, **params):
    """Fine-tune a LightGBM model with new hyperparameters.
    
    This function uses LightGBM's init_model parameter to perform
    incremental training, adding new trees to the existing booster.
    
    Parameters
    ----------
    base_model : LGBMClassifier
        Previously trained model to fine-tune.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    **params : dict
        New hyperparameters to apply.
    
    Returns
    -------
    tuple[LGBMClassifier, float]
        Fine-tuned model and training time in seconds.
    """
    if params:
        base_model.set_params(**params)
        
    t0 = time.time()
    # init_model parameter signals LightGBM to continue training from base_model
    base_model.fit(X_train, y_train, init_model=base_model)
    train_time = time.time() - t0
    
    return base_model, train_time