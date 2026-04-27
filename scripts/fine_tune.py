"""
Fine-tune a trained LightGBM model using Optuna hyperparameter search.

This script:
1. Loads a baseline model from MLflow (by run_id OR run_name)
2. Performs optional Optuna hyperparameter tuning
3. Logs the tuned model and comparative metrics to MLflow
"""

import os
import sys
import argparse
from pathlib import Path

project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
import mlflow.lightgbm
import time
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

from src.models.export_onnx import get_run_name
from src.features.build_features import build_features
from src.utils.mlflow_helpers import resolve_run_id, print_run_info


def is_databricks_environment():
    """Check if running in Databricks."""
    return 'DATABRICKS_RUNTIME_VERSION' in os.environ


def get_experiment_name(base_name):
    """Get proper experiment name for current environment."""
    if is_databricks_environment():
        import re
        match = re.search(r'/Users/([^/]+)/', project_root)
        user_email = match.group(1) if match else "default_user"
        return f"/Users/{user_email}/{base_name}"
    else:
        return base_name


def setup_mlflow_tracking(mlflow_uri):
    """Configure MLflow tracking for environment."""
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    elif not is_databricks_environment():
        mlflow.set_tracking_uri(f"file://{project_root}/mlruns")


def load_training_data(test_size=0.2, random_state=42):
    """Load, apply feature engineering, and split training data."""
    # Load processed data
    data_path = os.path.join(project_root, 'data', 'processed', 'student_mental_health_burnout_processed.csv')
    df = pd.read_csv(data_path)
    
    # Apply feature engineering (same as baseline training)
    df_features = build_features(df, target_col='burnout_level')
    
    # Split features and target
    X = df_features.drop(columns=['burnout_level'])
    y = df_features['burnout_level']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test, model_params=None):
    """Train LightGBM model and evaluate."""
    # Default params if not provided
    if model_params is None:
        model_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.1,
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
    
    # Ensure required params
    model_params.setdefault('random_state', 42)
    model_params.setdefault('verbose', -1)
    
    # Train
    t0 = time.time()
    model = lgb.LGBMClassifier(**model_params)
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    metrics = {
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
        'train_time': train_time
    }
    
    return model, metrics


def fine_tune_model(
    base_run_identifier,
    experiment_name,
    tune=False,
    n_trials=10,
    mlflow_uri=None
):
    """
    Fine-tune a trained model with optional hyperparameter search.
    
    Parameters
    ----------
    base_run_identifier : str
        Either run_id (UUID) or run_name of the baseline model
    experiment_name : str
        Name of the MLflow experiment
    tune : bool
        If True, perform Optuna hyperparameter search
    n_trials : int
        Number of Optuna trials
    mlflow_uri : str, optional
        MLflow tracking URI
    
    Returns
    -------
    dict
        Results with run_id, run_name, and performance metrics
    """
    # Setup MLflow
    setup_mlflow_tracking(mlflow_uri)
    
    experiment_name_full = get_experiment_name(experiment_name)
    mlflow.set_experiment(experiment_name_full)
    
    # Resolve run_id from run_id or run_name
    print(f"\n{'='*60}")
    print("RESOLVING BASELINE MODEL")
    print(f"{'='*60}")
    print(f"Input: {base_run_identifier}")
    print(f"Experiment: {experiment_name_full}")
    
    try:
        base_run_id = resolve_run_id(base_run_identifier, experiment_name_full)
        print(f"Resolved to run_id: {base_run_id}")
        print_run_info(base_run_id, experiment_name_full)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    
    # Load baseline model
    print(f"\n{'='*60}")
    print("LOADING BASELINE MODEL")
    print(f"{'='*60}")
    
    client = mlflow.tracking.MlflowClient()
    base_run = client.get_run(base_run_id)
    base_run_name = base_run.data.tags.get("mlflow.runName", "unknown")
    
    model_uri = f"runs:/{base_run_id}/model"
    base_model = mlflow.lightgbm.load_model(model_uri)
    print(f"Loaded model from: {base_run_name}")
    
    # Load baseline metrics
    base_metrics = {k: v for k, v in base_run.data.metrics.items()}
    print(f"Baseline metrics: F1={base_metrics.get('f1', 0):.3f}, "
          f"ROC AUC={base_metrics.get('roc_auc', 0):.3f}")
    
    # Load data
    print(f"\n{'='*60}")
    print("LOADING & PREPROCESSING DATA")
    print(f"{'='*60}")
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Hyperparameter tuning
    best_params = None
    if tune:
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING ({n_trials} trials)")
        print(f"{'='*60}")
        
        import optuna
        from sklearn.model_selection import cross_val_score
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            
            model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted').mean()
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        print(f"\nBest parameters found:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"Best CV F1: {study.best_value:.3f}")
    
    # Train fine-tuned model
    print(f"\n{'='*60}")
    print("TRAINING FINE-TUNED MODEL")
    print(f"{'='*60}")
    
    model_params = best_params if best_params else base_model.get_params()
    run_name = get_run_name('lightgbm', 'tuned')
    
    with mlflow.start_run(run_name=run_name) as run:
        # Train model
        tuned_model, metrics = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            model_params=model_params
        )
        
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param('model_stage', 'tuned')
        mlflow.log_param('base_run_id', base_run_id)
        mlflow.log_param('base_run_name', base_run_name)
        mlflow.log_param('tuning_enabled', tune)
        
        if best_params:
            for k, v in best_params.items():
                mlflow.log_param(f'best_{k}', v)
        
        # Log comparative metrics
        mlflow.log_metrics({
            'finetuned_precision': metrics['precision'],
            'finetuned_recall': metrics['recall'],
            'finetuned_f1': metrics['f1'],
            'finetuned_roc_auc': metrics['roc_auc'],
            'base_precision': base_metrics.get('precision', 0),
            'base_recall': base_metrics.get('recall', 0),
            'base_f1': base_metrics.get('f1', 0),
            'base_roc_auc': base_metrics.get('roc_auc', 0),
            'f1_improvement': metrics['f1'] - base_metrics.get('f1', 0),
        })
        
        # Log model
        mlflow.lightgbm.log_model(tuned_model, "model")
        
        run_id = run.info.run_id
        
        print(f"\n{'='*60}")
        print("FINE-TUNING COMPLETE")
        print(f"{'='*60}")
        print(f"Run ID:   {run_id}")
        print(f"Run Name: {run_name}")
        print(f"\nBaseline Performance:")
        print(f"  F1:      {base_metrics.get('f1', 0):.3f}")
        print(f"  ROC AUC: {base_metrics.get('roc_auc', 0):.3f}")
        print(f"\nFine-tuned Performance:")
        print(f"  F1:      {metrics['f1']:.3f} ({metrics['f1'] - base_metrics.get('f1', 0):+.3f})")
        print(f"  ROC AUC: {metrics['roc_auc']:.3f} ({metrics['roc_auc'] - base_metrics.get('roc_auc', 0):+.3f})")
        
        if tune:
            print(f"\nHyperparameter tuning completed ({n_trials} trials)")
        else:
            print(f"\nNo hyperparameter tuning (use --tune to enable)")
        
        print(f"\n{'='*60}")
        print("NEXT STEPS")
        print(f"{'='*60}")
        print(f"1. Export to ONNX:")
        print(f"   python scripts/export_model.py --run_identifier {run_id}")
        print(f"   OR")
        print(f"   python scripts/export_model.py --run_identifier {run_name}")
        print(f"\n2. Compare models in MLflow UI:")
        print(f"   Experiment: {experiment_name_full}")
        print(f"{'='*60}\n")
        
        return {
            'run_id': run_id,
            'run_name': run_name,
            'metrics': metrics,
            'base_metrics': base_metrics
        }


def main(args):
    """Main execution function."""
    results = fine_tune_model(
        base_run_identifier=args.base_run_identifier,
        experiment_name=args.experiment_name,
        tune=args.tune,
        n_trials=args.n_trials,
        mlflow_uri=args.mlflow_uri
    )
    
    return results


if __name__ == '____':
    parser = argparse.ArgumentParser(description='Fine-tune LightGBM model')
    parser.add_argument(
        '--base_run_identifier',
        type=str,
        required=True,
        help='Baseline model run_id (UUID) or run_name (e.g., 202604271449_lightgbm_baseline)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='Student Burnout Prediction',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable Optuna hyperparameter tuning'
    )
    parser.add_argument(
        '--n_trials',
        type=str,
        default=10,
        help='Number of Optuna trials (if --tune enabled)'
    )
    parser.add_argument(
        '--mlflow_uri',
        type=str,
        default=None,
        help='MLflow tracking URI (auto-detected if not provided)'
    )
    
    args = parser.parse_args()
    main(args)
