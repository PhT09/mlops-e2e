"""ONNX export for LightGBM models.

LightGBM Migration Summary
==========================
Best LightGBM Hyperparameters (from notebook Cell 14 Optuna search, trained model at ROC AUC 0.7355):
- n_estimators: 200, max_depth: 8, learning_rate: 0.1
- num_leaves: 31, min_child_samples: 20
- subsample: 0.8, colsample_bytree: 0.8
- reg_alpha: 0.01, reg_lambda: 0.1
- class_weight: 'balanced', random_state: 42, verbose: -1

onnxmltools version used: onnxmltools>=1.12.0
- Function: onnxmltools.convert.convert_lightgbm
- Data type: FloatTensorType([None, n_features])
- Target opset: 15
- Successfully exported 0.74 MB ONNX model

Preprocessing differences: NONE
- build_features.py is identical between notebook and pipeline
- Feature engineering uses ordinal encoding + one-hot encoding (drop_first=True)
- 23 final features (after dropping student_id)
"""

import os
import mlflow


def export_to_onnx(
    model,
    n_features: int,
    artifacts_dir: str,
    mlflow_artifact_path: str = "onnx_model",
) -> str:
    """
    Convert a trained LightGBM model to ONNX and save it locally.

    WHY onnxmltools instead of skl2onnx:
    ------------------------------------
    LightGBM is NOT natively supported by skl2onnx (which only handles
    scikit-learn estimators). onnxmltools provides a dedicated LightGBM
    converter that understands the internal LightGBM booster structure
    and can serialize it to ONNX format with proper operator mapping.

    Parameters
    ----------
    model : lightgbm.LGBMClassifier
        The fitted model returned by ``train_lightgbm``.
    n_features : int
        Number of input features (``X_train.shape[1]``).  ONNX graphs have a
        fixed input shape; this value pins the expected column count so that the
        runtime can validate inputs at inference time.
    artifacts_dir : str
        Absolute path to the local ``artifacts/`` directory where the .onnx file
        will be written.
    mlflow_artifact_path : str or None
        Sub-folder name inside the MLflow run's artifact store where the .onnx
        file will appear (default: ``"onnx_model"``). Pass None to skip MLflow logging.

    Returns
    -------
    str
        Absolute path to the saved ``model.onnx`` file.

    Notes on LightGBM ONNX export
    ------------------------------
    * onnxmltools.convert.convert_lightgbm handles the LightGBM booster serialization
    * The model expects float32 input, so inference.py must cast to np.float32
    * For multi-class classification, LightGBM outputs:
        - label: predicted class (int64)
        - probabilities: probability matrix (float32, shape [batch_size, n_classes])
    * opset 15 ensures compatibility with most ONNX runtimes
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    onnx_path = os.path.join(artifacts_dir, "model.onnx")

    try:
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        # Define input type: float tensor with shape [None, n_features]
        # None in first dimension allows variable batch size
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        
        # Convert LightGBM model to ONNX
        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_type,
            target_opset=15  # ONNX opset version for compatibility
        )
        
        # Save ONNX model to file
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"[ONNX] Model exported → {onnx_path}  ({os.path.getsize(onnx_path):,} bytes)")
        
    except ImportError:
        raise RuntimeError(
            "Failed to export ONNX model. Please install onnxmltools:\n"
            "  pip install onnxmltools onnxconverter-common"
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")

    # Upload to the active MLflow run for experiment-level traceability
    # (skip if mlflow_artifact_path is None)
    if mlflow_artifact_path:
        try:
            mlflow.log_artifact(onnx_path, artifact_path=mlflow_artifact_path)
            print(f"[ONNX] Artifact logged to MLflow under '{mlflow_artifact_path}/'")
        except Exception as e:
            print(f"⚠ Warning: Could not log ONNX to MLflow: {e}")
            print("  Model is saved locally and ready for serving.")

    return os.path.abspath(onnx_path)
