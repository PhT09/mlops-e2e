"""Fine-tuning script for LightGBM models using MLflow.

This script loads a previously trained LightGBM model from MLflow and fine-tunes
it on new data. LightGBM supports incremental training via the init_model parameter,
which allows adding new trees to an existing booster without starting from scratch.

Usage:
------
# Using fixed learning rate:
python scripts/fine_tune.py --input data/raw/student_data.csv --target burnout_level --learning_rate 0.01

# Using Optuna to find hyperparameters:
python scripts/fine_tune.py --input data/raw/student_data.csv --target burnout_level --tune --n_trials 20
"""

import os
import sys
import time
import argparse
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)

# === Fix import path for local modules ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local modules - Core pipeline components
from src.data.load_data import load_data
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features
from src.models.fine_tuning import tune_pretrained_model, fine_tune_lightgbm

def get_latest_run_id(experiment_name, client):
    """Retrieve the latest run ID from the specified MLflow experiment."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    runs = client.search_runs(
        [experiment.experiment_id], 
        order_by=["start_time desc"], 
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")
        
    return runs[0].info.run_id

def evaluate_and_log(model, X_test, y_test, prefix=""):
    """Evaluates the model and logs metrics to MLflow."""
    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0
    
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    try:
        # For binary classification or if predict_proba is applicable
        y_prob = model.predict_proba(X_test)
        if len(set(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except:
        roc_auc = 0.0

    metrics = {
        f"{prefix}precision": precision,
        f"{prefix}recall": recall,
        f"{prefix}f1": f1,
        f"{prefix}roc_auc": roc_auc,
        f"{prefix}pred_time": pred_time
    }
    
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
        
    print(f"\nModel Performance ({prefix.strip('_')}):")
    print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
    print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")
    print(f"Detailed Classification Report:\n{classification_report(y_test, y_pred, digits=3)}")
    
    return metrics

def main(args):    
    # === MLflow Setup ===
    mlruns_path = args.mlflow_uri or f"file:///{project_root.replace(chr(92), '/')}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)
    client = mlflow.tracking.MlflowClient()

    # === Fetch the base model ===
    if args.run_id:
        base_run_id = args.run_id
    else:
        print(f"Fetching latest run for experiment: {args.experiment}")
        base_run_id = get_latest_run_id(args.experiment, client)
        
    model_uri = f"runs:/{base_run_id}/model"
    print(f"Loading base LightGBM model from: {model_uri}")
    
    try:
        # Load the LightGBM model from MLflow
        base_model = mlflow.lightgbm.load_model(model_uri)
    except Exception as e:
        print(f"Could not load LightGBM model. Ensure the path is correct and model exists. Error: {e}")
        return

    # === Data Loading & Processing ===
    print(f"Loading fine-tuning data from {args.input}...")
    df = load_data(args.input)
    df = preprocess_data(df)
    
    target = args.target
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
        
    df_enc = build_features(df, target_col=target)
    for c in df_enc.select_dtypes(include=["bool"]).columns:
        df_enc[c] = df_enc[c].astype(int)
        
    X = df_enc.drop(columns=[target])
    y = df_enc[target]
    
    # Check if features match the model's expected features
    # LightGBM stores feature names in feature_name_ attribute
    model_features = base_model.feature_name_ if hasattr(base_model, 'feature_name_') else None
    if model_features is not None:
        missing_cols = set(model_features) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Features missing in new data: {missing_cols}")
        X = X[model_features]  # Align column order

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # === Start Fine-tuning Run ===
    with mlflow.start_run(run_name="fine_tuning"):
        mlflow.log_param("base_run_id", base_run_id)
        mlflow.log_param("is_finetune", True)
        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_param("learning_rate_finetune", args.learning_rate)

        # Baseline evaluation (before fine-tuning)
        print("Evaluating base model on new test set...")
        evaluate_and_log(base_model, X_test, y_test, prefix="base_")

        # Use Optuna to find best parameters if requested
        if args.tune:
            print("\n--- Tuning Hyperparameters for Fine-Tuning ---")
            best_params = tune_pretrained_model(base_model, X_train, y_train, n_trials=args.n_trials)
            mlflow.log_params(best_params)
        else:
            best_params = {"learning_rate": args.learning_rate}
            mlflow.log_param("learning_rate", args.learning_rate)
        
        print("Fine-tuning LightGBM model...")
        finetuned_model, train_time = fine_tune_lightgbm(base_model, X_train, y_train, **best_params)
        
        mlflow.log_metric("finetune_train_time", train_time)
        print(f"Model fine-tuned in {train_time:.2f} seconds")

        # Evaluation after fine-tuning
        print("Evaluating fine-tuned model...")
        evaluate_and_log(finetuned_model, X_test, y_test, prefix="finetuned_")

        # Save the fine-tuned model
        print("Saving fine-tuned LightGBM model to MLflow...")
        mlflow.lightgbm.log_model(finetuned_model, artifact_path="model")
        print("Fine-tuning complete. Model saved to new MLflow run.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune a LightGBM MLflow model.")
    p.add_argument("--input", type=str, required=True, help="path to new data CSV")
    p.add_argument("--target", type=str, default="burnout_level", help="Target column. Defaults to 'burnout_level'.")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Student Burnout Prediction", help="MLflow Experiment Name")
    p.add_argument("--run_id", type=str, default=None, help="Specific Run ID to load. If empty, loads the latest run.")
    p.add_argument("--mlflow_uri", type=str, default=None, help="MLflow Tracking URI (optional).")
    p.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for fine-tuning. Typically smaller than base.")
    p.add_argument("--tune", action="store_true", help="Use Optuna to search for best fine-tuning hyperparameters")
    p.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials if --tune is set")

    args = p.parse_args()
    main(args)