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
    print("⚠ Warning: great_expectations not installed. Skipping data validation.")

from src.models.train import train_lightgbm                # LightGBM model training
from src.models.evalutate import evaluate_model            # Model evaluation
from src.models.export_onnx import export_to_onnx          # ONNX export for serving

def main(args):    
    # === MLflow Setup ===
    # Configure MLflow to use local file-based tracking
    
    # Set up local tracking directory
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    else:
        # Use local mlruns directory in project
        mlruns_dir = os.path.join(project_root, "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    mlflow.set_experiment(args.experiment)  # Creates experiment if doesn't exist

    # Start MLflow run 
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        mlflow.log_param("model", "lightgbm")           # Model type for comparison
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
                    json.dump(failed, indent=2)
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

        # === STAGE 7: Log LightGBM model to MLflow (kept for experiment UI) ===
        print("Saving LightGBM model to MLflow (experiment tracking)...")
        try:
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model"  # Creates a 'model/' folder in MLflow run artifacts
            )
            print("✓ LightGBM model logged to MLflow")
        except Exception as e:
            print(f"⚠ Warning: Could not log model to MLflow: {e}")
            print("  Continuing with local ONNX export...")

        # === STAGE 8: Export model to ONNX (used by FastAPI serving layer) ===
        # The FastAPI inference layer (src/serving/inference.py) loads model.onnx
        # from the artifacts directory for serving
        print("Exporting model to ONNX format for FastAPI serving...")
        onnx_path = export_to_onnx(
            model=model,
            n_features=X_train.shape[1],   # Pin input width in the ONNX graph
            artifacts_dir=artifacts_dir,
            mlflow_artifact_path=None,  # Skip MLflow logging, just save locally
        )
        print(f"✓ ONNX model ready for serving at: {onnx_path}")

        # === Final Performance Summary ===
        print(f"\n⏱ Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))
        
        print(f"\n✅ Pipeline complete! Artifacts saved to: {artifacts_dir}")
        print(f"   - model.onnx (ONNX model for serving)")
        print(f"   - feature_columns.json (feature schema)")
        print(f"   - preprocessing.pkl (preprocessing metadata)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run student burnout prediction pipeline with LightGBM + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/student_mental_health_burnout.csv)")
    p.add_argument("--target", type=str, default="burnout_level")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Student Burnout Prediction")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)

r"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \
    --input data/raw/student_mental_health_burnout_relabeled.csv \
    --target burnout_level

"""