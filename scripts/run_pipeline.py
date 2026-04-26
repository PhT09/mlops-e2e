import os
import sys
import time
import argparse
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)

# === Fix import path for local modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.load_data import load_data                   # Data loading with error handling
from src.data.preprocess_data import preprocess_data            # Basic data cleaning
from src.features.build_features import build_features     # Feature engineering (CRITICAL for model performance)
from src.utils.validate_data import validate_student_data    # Data quality validation
from src.models.train import train_xgboost                 # Model training
from src.models.evalutate import evaluate_model            # Model evaluation
from src.models.export_onnx import export_to_onnx          # ONNX export for serving

def main(args):    
    # === MLflow Setup ===
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file:///{project_root.replace(chr(92), '/')}/mlruns"  # Local file-based tracking
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # Creates experiment if doesn't exist

    # Start MLflow run 
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        mlflow.log_param("model", "xgboost")            # Model type for comparison
        mlflow.log_param("test_size", args.test_size)   # Train/test split ratio

        # === STAGE 1: Data Loading & Validation ===
        print("Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === Data Quality Validation ===
        print("Validating data quality with Great Expectations...")
        is_valid, failed = validate_student_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))  # Track data quality over time

        if not is_valid:
            # Log validation failures for debugging
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow.")

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
        
        # Save locally for development serving
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        # Log to MLflow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # Save preprocessing artifacts for serving pipeline
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target                  # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

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
        print("Training XGBoost model...")
        
        model, train_time = train_xgboost(X_train, y_train)

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

        # === STAGE 7: Log sklearn model to MLflow (kept for experiment UI) ===
        # We still log the native sklearn/XGBoost model here so that:
        #   - MLflow UI shows a fully-runnable model artefact for ad-hoc inference
        #   - Experiment comparisons can use mlflow.sklearn.load_model() in notebooks
        # The model is NOT used by the FastAPI layer (ONNX handles serving instead).
        print("Saving sklearn model to MLflow (experiment tracking)...")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"  # Creates a 'model/' folder in MLflow run artifacts
        )
        print("Sklearn model logged to MLflow for experiment UI")

        # === STAGE 8: Export model to ONNX (used by FastAPI serving layer) ===
        # WHY a separate ONNX export?
        #   The FastAPI inference layer (src/serving/inference.py) loads model.onnx
        #   via onnxruntime, which is a lightweight C++ runtime that requires no
        #   MLflow, sklearn, or XGBoost at serve time.  This reduces the container
        #   image footprint and gives ~2–5x faster cold-start than pyfunc.load_model.
        print("Exporting model to ONNX format for FastAPI serving...")
        onnx_path = export_to_onnx(
            model=model,
            n_features=X_train.shape[1],   # Pin input width in the ONNX graph
            artifacts_dir=artifacts_dir,
            mlflow_artifact_path="onnx_model",
        )
        print(f"ONNX model ready for serving at: {onnx_path}")

        # === Final Performance Summary ===
        print(f"\n⏱Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)

r"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \
    --input data/raw/student_mental_health_burnout.csv \
    --target burnout_level

"""