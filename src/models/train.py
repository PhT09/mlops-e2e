import mlflow
import pandas as pd
import mlflow.xgboost
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=301,
        learning_rate=0.034,
        max_depth=7,
        subsample=0.95,
        colsample_bytree=0.98,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss"
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    return model, train_time

def train_model(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Train model
        model, train_time = train_xgboost(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, average="weighted")

        # Log params, metrics, and model
        mlflow.log_param("n_estimators", 301)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("train_time", train_time)
        mlflow.xgboost.log_model(model, "model")

        # Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")