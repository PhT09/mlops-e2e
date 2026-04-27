"""ONNX export for trained LightGBM models from MLflow.

This module provides production-ready ONNX export with throughput benchmarking:
1. Loads a pre-trained LightGBM model from MLflow using run_id
2. Converts to ONNX format for framework-independent serving
3. Benchmarks inference throughput and logs metrics back to MLflow
4. Saves ONNX model locally for deployment

Architecture Overview:
---------------------
Training Phase (run_pipeline.py):
  → Train baseline model
  → Log to MLflow with runID: YYYYmmddHHMM_lightgbm_baseline

Fine-tuning Phase (fine_tune.py):
  → Load model from MLflow run_id
  → Fine-tune with new hyperparameters
  → Log to MLflow with runID: YYYYmmddHHMM_lightgbm_tuned
  → Log best_params to MLflow

Export Phase (THIS MODULE):
  → Load any model version from MLflow run_id
  → Export to ONNX format
  → Benchmark throughput (samples/sec)
  → Log throughput_onnx metric to original MLflow run
  → Save to artifacts/ for serving

Why separate export from training?
----------------------------------
- Different models can be exported without retraining
- Throughput benchmarking done once at deployment time
- Clean separation: training metrics vs. serving metrics
- Easier rollback: switch ONNX export without touching training runs
"""

import os
import time
import json
import mlflow
import mlflow.lightgbm
import numpy as np


def export_to_onnx(
    run_id: str,
    artifacts_dir: str,
    mlflow_tracking_uri: str = None,
    benchmark_samples: int = 1000,
) -> str:
    """
    Load a trained LightGBM model from MLflow and export to ONNX format.

    This function:
    1. Loads model from specified MLflow run_id
    2. Converts to ONNX using onnxmltools
    3. Benchmarks inference throughput
    4. Logs throughput_onnx metric back to the MLflow run
    5. Saves ONNX model to local artifacts directory

    Parameters
    ----------
    run_id : str
        MLflow run ID to load the model from. Can be from baseline or tuned runs.
        Format: UUID string (e.g., "4c1a3a8f0dbb4f98a77a44b38afa2d55")
    artifacts_dir : str
        Absolute path to local artifacts/ directory where model.onnx will be saved.
    mlflow_tracking_uri : str, optional
        MLflow tracking URI. If None, uses current active tracking URI.
    benchmark_samples : int, default=1000
        Number of synthetic samples to use for throughput benchmarking.
        Larger values give more stable throughput estimates but take longer.

    Returns
    -------
    str
        Absolute path to the saved model.onnx file.

    Raises
    ------
    ValueError
        If run_id is not found in MLflow or model artifact doesn't exist.
    RuntimeError
        If ONNX export fails (missing dependencies or conversion errors).

    Examples
    --------
    # Export baseline model
    export_to_onnx(
        run_id="4c1a3a8f0dbb4f98a77a44b38afa2d55",
        artifacts_dir="/workspace/mlops-e2e/artifacts"
    )

    # Export fine-tuned model
    export_to_onnx(
        run_id="8899655389b84316af2fbca645924a8f",
        artifacts_dir="/workspace/mlops-e2e/artifacts"
    )

    Notes
    -----
    * Requires onnxmltools>=1.12.0 and onnxruntime>=1.16.0
    * LightGBM outputs for multi-class classification:
        - label: predicted class (int64)
        - probabilities: probability matrix (float32, [batch_size, n_classes])
    * Throughput is measured in samples/second on ONNX Runtime with CPU
    * Original training code uses float64, but ONNX requires float32 input
    """
    # === MLflow Setup ===
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    client = mlflow.tracking.MlflowClient()

    # === Step 1: Load Model from MLflow ===
    print(f"[ONNX Export] Loading model from MLflow run: {run_id}")
    
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.lightgbm.load_model(model_uri)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        raise ValueError(
            f"Failed to load model from run_id '{run_id}'. "
            f"Ensure the run exists and contains a 'model' artifact.\n"
            f"Error: {e}"
        )

    # === Step 2: Get Model Metadata ===
    # Extract number of features from the trained model
    if hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
    elif hasattr(model, 'feature_name_'):
        n_features = len(model.feature_name_)
    else:
        raise RuntimeError(
            "Cannot determine number of features from model. "
            "Model must have 'n_features_in_' or 'feature_name_' attribute."
        )
    
    print(f"✓ Model expects {n_features} input features")

    # === Step 3: Convert to ONNX ===
    os.makedirs(artifacts_dir, exist_ok=True)
    onnx_path = os.path.join(artifacts_dir, "model.onnx")

    try:
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        print("[ONNX Export] Converting LightGBM → ONNX...")
        
        # Define input type: float32 tensor with shape [None, n_features]
        # None in first dimension allows variable batch size
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        
        # Convert LightGBM model to ONNX
        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_type,
            target_opset=15  # ONNX opset 15 for broad compatibility
        )
        
        # Save ONNX model to file
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"✓ ONNX model exported → {onnx_path} ({file_size_mb:.2f} MB)")
        
    except ImportError:
        raise RuntimeError(
            "Failed to export ONNX model. Please install required packages:\n"
            "  pip install onnxmltools>=1.12.0 onnxconverter-common>=1.13.0"
        )
    except Exception as e:
        raise RuntimeError(f"ONNX conversion failed: {e}")

    # === Step 4: Benchmark Throughput ===
    print(f"[ONNX Export] Benchmarking throughput with {benchmark_samples} samples...")
    
    try:
        import onnxruntime as ort
        
        # Load ONNX model into runtime
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        
        # Generate synthetic test data
        X_bench = np.random.randn(benchmark_samples, n_features).astype(np.float32)
        
        # Warmup run (JIT compilation, cache warming)
        _ = session.run(None, {input_name: X_bench[:10]})
        
        # Actual benchmark
        start_time = time.time()
        _ = session.run(None, {input_name: X_bench})
        elapsed_time = time.time() - start_time
        
        throughput = benchmark_samples / elapsed_time  # samples/second
        
        print(f"✓ Throughput: {throughput:.2f} samples/sec")
        print(f"  Latency: {elapsed_time*1000:.2f} ms for {benchmark_samples} samples")
        
    except ImportError:
        print("⚠ Warning: onnxruntime not installed. Skipping throughput benchmark.")
        print("  Install with: pip install onnxruntime>=1.16.0")
        throughput = None
    except Exception as e:
        print(f"⚠ Warning: Throughput benchmark failed: {e}")
        throughput = None

    # === Step 5: Log Throughput to MLflow ===
    if throughput is not None:
        try:
            # Log metric to the ORIGINAL training/tuning run
            client.log_metric(run_id, "throughput_onnx", throughput)
            print(f"✓ Throughput metric logged to MLflow run {run_id}")
        except Exception as e:
            print(f"⚠ Warning: Could not log throughput to MLflow: {e}")

    # === Step 6: Save Model Metadata ===
    # Save feature count for inference pipeline validation
    metadata = {
        "run_id": run_id,
        "n_features": n_features,
        "onnx_opset": 15,
        "throughput_samples_per_sec": throughput,
        "model_size_mb": file_size_mb
    }
    
    metadata_path = os.path.join(artifacts_dir, "onnx_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved → {metadata_path}")

    # === Final Summary ===
    print("\n" + "="*60)
    print("ONNX EXPORT COMPLETE")
    print("="*60)
    print(f"Source MLflow Run: {run_id}")
    print(f"ONNX Model: {onnx_path}")
    print(f"Model Size: {file_size_mb:.2f} MB")
    print(f"Input Shape: (?, {n_features})")
    if throughput:
        print(f"Throughput: {throughput:.2f} samples/sec")
    print("="*60)

    return os.path.abspath(onnx_path)


def get_run_name(model_name: str = "lightgbm", suffix: str = "baseline") -> str:
    """
    Generate standardized MLflow run name.

    Parameters
    ----------
    model_name : str
        Model type (e.g., "lightgbm", "xgboost")
    suffix : str
        Run type: "baseline" or "tuned"

    Returns
    -------
    str
        Run name in format: YYYYmmddHHMM_<model_name>_<suffix>
        Example: "202501151430_lightgbm_baseline"
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return f"{timestamp}_{model_name}_{suffix}"
