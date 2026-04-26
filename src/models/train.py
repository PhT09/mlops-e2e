import mlflow
import pandas as pd
import mlflow.lightgbm
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_lightgbm(X_train, y_train):
    """Train a LightGBM classifier with optimized hyperparameters.
    
    These parameters were found via Optuna hyperparameter tuning on the
    relabeled student burnout dataset, achieving ROC AUC 0.7355.
    
    Returns
    -------
    tuple[LGBMClassifier, float]
        Fitted model and training time in seconds.
    """
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    return model, train_time

def train_model(df: pd.DataFrame, target_col: str):
    """End-to-end training function with MLflow logging.
    
    This is an alternative entry point that handles train/test split
    and MLflow logging internally.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Train model
        model, train_time = train_lightgbm(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, average="weighted")

        # Log params, metrics, and model
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("train_time", train_time)
        mlflow.lightgbm.log_model(model, "model")

        # Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")