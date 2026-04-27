"""Baseline model training pipeline with MLflow tracking.

This script trains a LightGBM baseline model and logs it to MLflow.
It does NOT export to ONNX - use scripts/export_model.py for that.

Workflow:
---------
1. Training Phase (THIS SCRIPT):
   - Load and preprocess data
   - Feature engineering
   - Train baseline LightGBM model
   - Log to MLflow with runID: YYYYmmddHHMM_lightgbm_baseline
   - Save feature metadata for serving

2. Export Phase (scripts/export_model.py):
   - Load model from MLflow run_id
   - Export to ONNX
   - Benchmark throughput
   - Log throughput metric

Usage:
------
python scripts/run_pipeline.py \
    --input data/raw/student_mental_health_burnout_relabeled.csv \
    --target burnout_level
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
project_root = "/Workspace/Users/trannguyentoanphat1592005@gmail.com/mlops-e2e"
sys.path.append(project_root)

# Local modules - Core pipeline components
from src.data.load_data import load_data                   # Data loading with error handling
from src.data.preprocess_data import preprocess_data       # Basic data cleaning
from src.features.build_features import build_features     # Feature engineering (CRITICAL for model performance)

# Optional: Data validation (requires great_expectations)
try:
    from src.utils.validate_data import validate_student_data
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    print("Warning: great_expectations not installed. Skipping data validation.")

from src.models.train import train_lightgbm                # LightGBM model training
from src.models.evalutate import evaluate_model            # Model evaluation
from src.models.export_onnx import get_run_name            # Run naming utility

def is_databricks_environment():
    """Check if running in Databricks environment."""
    return 'DATABRICKS_RUNTIME_VERSION' in os.environ

def setup_mlflow_tracking(args):
    """Configure MLflow tracking based on environment."""
    if args.mlflow_uri:
        # User explicitly provided tracking URI
        mlflow.set_tracking_uri(args.mlflow_uri)
        print(f"✓ MLflow Tracking URI: {args.mlflow_uri}")
    elif is_databricks_environment():
        # In Databricks, use managed MLflow (no explicit URI needed)
        # MLflow is pre-configured and integrated with workspace
        print("✓ Using Databricks Managed MLflow")
        # No need to set tracking URI - Databricks handles it automatically
    else:
        # Local development: use file-based tracking
        mlruns_dir = os.path.join(project_root, "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        tracking_uri = f"file://{mlruns_dir}"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✓ MLflow Tracking URI: {tracking_uri}")

def get_experiment_name(base_name):
    """Get proper experiment name based on environment."""
    if is_databricks_environment():
        # Databricks requires absolute paths for experiments
        # Extract user email from project_root path
        # project_root = "/Workspace/Users/<email>/mlops-e2e"
        import re
        match = re.search(r'/Users/([^/]+)/', project_root)
        if match:
            user_email = match.group(1)
        else:
            user_email = "default_user"
        
        # Return absolute path
        return f"/Users/{user_email}/{base_name}"
    else:
        # Local: use simple name
        return base_name

def main(args):    
    # === MLflow Setup ===
    setup_mlflow_tracking(args)
    # Get proper experiment name (absolute path for Databricks)
    experiment_name = get_experiment_name(args.experiment)
    mlflow.set_experiment(experiment_name)  # Creates experiment if doesn't exist
    print(f"✓ Experiment: {experiment_name}")

    # Generate standardized run name: YYYYmmddHHMM_lightgbm_baseline
    run_name = get_run_name(model_name="lightgbm", suffix="baseline")
    
    # Start MLflow run with standardized naming
    with mlflow.start_run(run_name=run_name):
        print(f"\n{'='*60}")
        print(f"BASELINE MODEL TRAINING")
        print(f"{'='*60}")
        print(f"MLflow Run: {run_name}")
        print(f"Experiment: {args.experiment}")
        print(f"{'='*60}\n")
        
        # === Log hyperparameters and configuration ===
        mlflow.log_param("model", "lightgbm")           # Model type for comparison
        mlflow.log_param("model_stage", "baseline")     # Distinguish baseline from tuned
        mlflow.log_param("test_size", args.test_size)   # Train/test split ratio

        # === STAGE 1: Data Loading & Validation ===
        print("Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === Data Quality Validation ===
        if HAS_VALIDATION:
            print("Validating data quality with Great Expectations...")
            is_valid, failed = validate_student_data(df)
            mlflow.log_metric("data_quality_pass", int(is_valid))  # Track data quality over time

            if not is_valid:
                # Log validation failures for debugging
                import json
                failed_path = os.path.join(project_root, "artifacts", "failed_expectations.json")
                with open(failed_path, "w") as f:
                    json.dump(failed, f, indent=2)
                raise ValueError(f"Data quality check failed. Issues: {failed}")
            else:
                print("Data validation passed. Logged to MLflow.")
        else:
            print("Skipping data validation (great_expectations not available)")
            mlflow.log_metric("data_quality_pass", -1)  # -1 indicates validation was skipped

        # === STAGE 2: Data Preprocessing ===
        print("Preprocessing data...")
        df = preprocess_data(df)  # Basic cleaning (handle missing values, fix data types)

        # Save processed dataset for reproducibility and debugging
        processed_path = os.path.join(project_root, "data", "processed", "student_mental_health_burnout_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 3: Feature Engineering ===
        print("Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        # Apply feature engineering transformations
        df_enc = build_features(df, target_col=target) 
        
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"Feature engineering completed: {df_enc.shape[1]} features")

        # === Save Feature Metadata for Serving Consistency ===
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Get feature columns (exclude target)
        feature_cols = list(df_enc.drop(columns=[target]).columns)
        
        # Save locally for development serving (inference.py reads from here)
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        print(f"✓ Saved feature_columns.json with {len(feature_cols)} features")

        # Save feature columns as text file for reference
        feature_cols_txt_path = os.path.join(artifacts_dir, "feature_columns.txt")
        with open(feature_cols_txt_path, "w") as f:
            f.write("\n".join(feature_cols))
        print(f"✓ Saved feature_columns.txt")

        # Save preprocessing artifacts for serving pipeline
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target                  # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"✓ Saved preprocessing.pkl")

        # === STAGE 4: Train/Test Split ===
        print("Splitting data...")
        X = df_enc.drop(columns=[target])  # Feature matrix
        y = df_enc[target]                 # Target vector
        
        # Stratified split to maintain class distribution in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,    # Default: 20% for testing
            stratify=y,                  # Maintain class balance
            random_state=42              # Reproducible splits
        )
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

       
        # === STAGE 5: Model Training with Optimized Hyperparameters ===
        print("Training LightGBM model...")
        
        model, train_time = train_lightgbm(X_train, y_train)

        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Model Evaluation ===
        print("Evaluating model performance...")
        
        # Generate predictions and metrics
        y_pred, metrics = evaluate_model(model, X_test, y_test)
        
        pred_time = metrics["pred_time"]
        mlflow.log_metric("pred_time", pred_time)  # Track inference performance

        # === Log Evaluation Metrics to MLflow ===
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        roc_auc = metrics["roc_auc"]
        
        # Log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # === STAGE 7: Log LightGBM model to MLflow ===
        print("Saving LightGBM model to MLflow...")
        try:
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model",  # Creates a 'model/' folder in MLflow run artifacts
                registered_model_name=None  # Don't auto-register to avoid permission issues
            )
            print("✓ LightGBM model logged to MLflow")
        except Exception as e:
            print(f"⚠ Warning: Could not log model to MLflow: {e}")
            print("  Training metrics are still logged. You can export model manually.")
            
            # Save model locally as fallback
            import joblib
            local_model_path = os.path.join(artifacts_dir, "lightgbm_model.pkl")
            joblib.dump(model, local_model_path)
            print(f"  ✓ Model saved locally: {local_model_path}")

        # === Get current run ID ===
        current_run = mlflow.active_run()
        run_id = current_run.info.run_id

        # === Final Performance Summary ===
        print(f"\n{'='*60}")
        print(f"BASELINE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"Run Name: {run_name}")
        print(f"Training time: {train_time:.2f}s")
        print(f"Inference time: {pred_time:.4f}s")
        print(f"Samples per second: {len(X_test)/pred_time:.0f}")
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))
        print(f"\nArtifacts saved to: {artifacts_dir}")
        print(f"   - feature_columns.json (feature schema)")
        print(f"   - preprocessing.pkl (preprocessing metadata)")
        if os.path.exists(os.path.join(artifacts_dir, "lightgbm_model.pkl")):
            print(f"   - lightgbm_model.pkl (model backup)")
        print(f"\n{'='*60}")
        print(f"NEXT STEPS:")
        print(f"{'='*60}")
        print(f"1. Fine-tune this model:")
        print(f"   python scripts/fine_tune.py --run_id {run_id} --input <data>")
        print(f"\n2. Export to ONNX for serving:")
        print(f"   python scripts/export_model.py --run_id {run_id}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train baseline LightGBM model with MLflow tracking")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/student_mental_health_burnout.csv)")
    p.add_argument("--target", type=str, default="burnout_level")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Student Burnout Prediction")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI (not needed in Databricks)")

    args = p.parse_args()
    main(args)
