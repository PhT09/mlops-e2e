import os
import mlflow
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_to_onnx(
    model,
    n_features: int,
    artifacts_dir: str,
    mlflow_artifact_path: str = "onnx_model",
) -> str:
    """
    Convert a trained sklearn-compatible XGBoost pipeline to ONNX and log it.

    Parameters
    ----------
    model : sklearn-compatible estimator
        The fitted model returned by ``train_xgboost``.
    n_features : int
        Number of input features (``X_train.shape[1]``).  ONNX graphs have a
        fixed input shape; this value pins the expected column count so that the
        runtime can validate inputs at inference time.
    artifacts_dir : str
        Absolute path to the local ``artifacts/`` directory where the .onnx file
        will be written before being uploaded to MLflow.
    mlflow_artifact_path : str
        Sub-folder name inside the MLflow run's artifact store where the .onnx
        file will appear (default: ``"onnx_model"``).

    Returns
    -------
    str
        Absolute path to the saved ``model.onnx`` file.

    Notes on conversion options
    ---------------------------
    * ``FloatTensorType([None, n_features])``:
        The leading ``None`` means the batch dimension is dynamic (any number of
        rows).  The second dimension is fixed to enforce the correct feature count.
        We use *Float* (float32) because ONNX's canonical numeric type is float32
        and onnxruntime will raise a type-mismatch error if you pass float64 data
        to a float32 graph — which is why inference.py casts the numpy array with
        ``.astype(np.float32)`` before calling the session.

    * ``target_opset=17``:
        ONNX operator set 17 is broadly supported across onnxruntime ≥ 1.15 and
        covers all operators emitted by XGBoost / sklearn pipelines.

    * ``options={"zipmap": False}``:
        By default skl2onnx wraps classifier output in a ZipMap node that returns
        a list-of-dicts ``[{"0": p0, "1": p1, "2": p2}]``.  Setting zipmap=False
        makes the graph output a plain int64 label array and a float32 probability
        matrix — much easier to index in the serving layer.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    onnx_path = os.path.join(artifacts_dir, "model.onnx")

    # Build the ONNX graph
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=17,
        options={type(model): {"zipmap": False}},
    )

    # Persist locally so FastAPI can load it without calling MLflow at runtime
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"[ONNX] Model exported → {onnx_path}  ({os.path.getsize(onnx_path):,} bytes)")

    # Upload to the active MLflow run for experiment-level traceability
    # (mlflow.log_artifact expects the LOCAL path; it copies the file into the
    #  run's artifact store under the given subfolder)
    mlflow.log_artifact(onnx_path, artifact_path=mlflow_artifact_path)
    print(f"[ONNX] Artifact logged to MLflow under '{mlflow_artifact_path}/'")

    return os.path.abspath(onnx_path)
