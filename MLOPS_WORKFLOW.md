# MLOps Workflow Guide

## Overview

This project follows a 3-stage MLOps workflow with clear separation between training, fine-tuning, and deployment:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  1. BASELINE    │      │  2. FINE-TUNE   │      │  3. EXPORT      │
│    TRAINING     │─────▶│   (Optional)    │─────▶│   TO ONNX       │
│                 │      │                 │      │                 │
│ run_pipeline.py │      │  fine_tune.py   │      │ export_model.py │
└─────────────────┘      └─────────────────┘      └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
    MLflow Run              MLflow Run              MLflow Run
  YYYYmmddHHMM_           YYYYmmddHHMM_           (updates original)
  lightgbm_baseline       lightgbm_tuned          + throughput_onnx
  + model artifact        + model artifact        + model.onnx file
  + training metrics      + best_params
```

---

## Stage 1: Baseline Training

**Script:** `scripts/run_pipeline.py`

**Purpose:** Train a LightGBM baseline model with default hyperparameters and log to MLflow.

**What it does:**
- Load and preprocess data
- Feature engineering (ordinal encoding + one-hot encoding)
- Train LightGBM classifier with optimized hyperparameters
- Evaluate on test set (precision, recall, F1, ROC AUC)
- Log model + metrics to MLflow
- Save feature metadata for serving
- **Does NOT export to ONNX**

**Run Naming Convention:**
```
YYYYmmddHHMM_lightgbm_baseline
Example: 202501151430_lightgbm_baseline
```

**Usage:**
```bash
python scripts/run_pipeline.py \
    --input data/raw/student_mental_health_burnout_relabeled.csv \
    --target burnout_level \
    --test_size 0.2 \
    --experiment "Student Burnout Prediction"
```

**Outputs:**
- MLflow run with baseline model
- `artifacts/feature_columns.json` - feature schema
- `artifacts/preprocessing.pkl` - preprocessing metadata
- `data/processed/*.csv` - processed dataset

**Example Output:**
```
============================================================
BASELINE TRAINING COMPLETE
============================================================
Run ID: 4c1a3a8f0dbb4f98a77a44b38afa2d55
Run Name: 202501151430_lightgbm_baseline
Training time: 2.34s
F1 Score: 0.735 | ROC AUC: 0.756

NEXT STEPS:
1. Fine-tune this model:
   python scripts/fine_tune.py --run_id 4c1a3a8f0dbb4f98a77a44b38afa2d55 --input <data>

2. Export to ONNX for serving:
   python scripts/export_model.py --run_id 4c1a3a8f0dbb4f98a77a44b38afa2d55
```

---

## Stage 2: Fine-Tuning (Optional)

**Script:** `scripts/fine_tune.py`

**Purpose:** Load a baseline model from MLflow and fine-tune it with incremental learning.

**What it does:**
- Load baseline (or previous tuned) model from MLflow run_id
- Load new training data
- Optionally use Optuna to search for best hyperparameters (`--tune`)
- Fine-tune model using LightGBM's `init_model` (incremental learning)
- Evaluate on new test set
- Log fine-tuned model + metrics to MLflow
- Log best hyperparameters if using Optuna
- **Does NOT export to ONNX**

**Run Naming Convention:**
```
YYYYmmddHHMM_lightgbm_tuned
Example: 202501151445_lightgbm_tuned
```

**Usage:**

**Option A: Fixed Learning Rate**
```bash
python scripts/fine_tune.py \
    --run_id 4c1a3a8f0dbb4f98a77a44b38afa2d55 \
    --input data/raw/student_data_new.csv \
    --target burnout_level \
    --learning_rate 0.01
```

**Option B: Hyperparameter Tuning with Optuna**
```bash
python scripts/fine_tune.py \
    --run_id 4c1a3a8f0dbb4f98a77a44b38afa2d55 \
    --input data/raw/student_data_new.csv \
    --target burnout_level \
    --tune \
    --n_trials 20
```

**MLflow Tracking:**
- `base_run_id`: Reference to the baseline model
- `model_stage`: "tuned"
- `best_*`: Best hyperparameters from Optuna (if `--tune` used)
- `base_*`: Metrics of baseline model on new test set
- `finetuned_*`: Metrics of fine-tuned model on new test set

**Example Output:**
```
============================================================
FINE-TUNING COMPLETE
============================================================
Run ID: 8899655389b84316af2fbca645924a8f
Run Name: 202501151445_lightgbm_tuned
Base Model: 4c1a3a8f0dbb4f98a77a44b38afa2d55
Training time: 1.87s

Best Hyperparameters:
  learning_rate: 0.008
  n_estimators: 50
  max_depth: 6

NEXT STEPS:
Export to ONNX for serving:
   python scripts/export_model.py --run_id 8899655389b84316af2fbca645924a8f
```

---

## Stage 3: ONNX Export

**Script:** `scripts/export_model.py`

**Purpose:** Load any trained model (baseline or tuned) from MLflow and export to ONNX format.

**What it does:**
- Load model from MLflow run_id (baseline or tuned)
- Convert LightGBM model to ONNX format using `onnxmltools`
- Benchmark inference throughput (samples/second)
- Log `throughput_onnx` metric back to the original MLflow run
- Save ONNX model to `artifacts/model.onnx`
- Save metadata (`onnx_metadata.json`)

**Why Separate Export?**
- Flexibility: Export any model version without retraining
- Different models can be benchmarked independently
- Serving metrics separate from training metrics
- Easier rollback: switch ONNX exports without touching training

**Usage:**
```bash
# Export baseline model
python scripts/export_model.py \
    --run_id 4c1a3a8f0dbb4f98a77a44b38afa2d55

# Export fine-tuned model
python scripts/export_model.py \
    --run_id 8899655389b84316af2fbca645924a8f

# Custom output directory
python scripts/export_model.py \
    --run_id 8899655389b84316af2fbca645924a8f \
    --output_dir ./deployment/artifacts

# Custom MLflow URI
python scripts/export_model.py \
    --run_id abc123 \
    --mlflow_uri http://localhost:5000
```

**Outputs:**
- `artifacts/model.onnx` - ONNX model for serving
- `artifacts/onnx_metadata.json` - Model metadata
- MLflow metric: `throughput_onnx` (samples/second)

**Example Output:**
```
============================================================
ONNX EXPORT COMPLETE
============================================================
Source MLflow Run: 8899655389b84316af2fbca645924a8f
ONNX Model: /workspace/mlops-e2e/artifacts/model.onnx
Model Size: 0.74 MB
Input Shape: (?, 23)
Throughput: 8234.56 samples/sec
============================================================

Export successful!
ONNX model ready for deployment: /workspace/mlops-e2e/artifacts/model.onnx
```

---

## Complete Workflow Example

### Scenario: Train baseline → Fine-tune → Deploy

**Step 1: Train Baseline**
```bash
python scripts/run_pipeline.py \
    --input data/raw/student_mental_health_burnout_relabeled.csv \
    --target burnout_level
```
→ Output: Run ID `4c1a3a8f0dbb4f98a77a44b38afa2d55`

**Step 2: Fine-Tune on New Data**
```bash
python scripts/fine_tune.py \
    --run_id 4c1a3a8f0dbb4f98a77a44b38afa2d55 \
    --input data/raw/student_data_2024_Q2.csv \
    --target burnout_level \
    --tune \
    --n_trials 20
```
→ Output: Run ID `8899655389b84316af2fbca645924a8f`

**Step 3: Export Fine-Tuned Model to ONNX**
```bash
python scripts/export_model.py \
    --run_id 8899655389b84316af2fbca645924a8f
```
→ Output: `artifacts/model.onnx` ready for serving

**Step 4: Serve via FastAPI**
```bash
python -m uvicorn src.app.main:app --reload
```
→ API endpoint: `http://localhost:8000/predict`

---

## MLflow Tracking Summary

### Baseline Run
- **Run Name:** `YYYYmmddHHMM_lightgbm_baseline`
- **Parameters:**
  - `model`: "lightgbm"
  - `model_stage`: "baseline"
  - `test_size`: 0.2
  - (all LightGBM hyperparameters)
- **Metrics:**
  - `train_time`, `pred_time`
  - `precision`, `recall`, `f1`, `roc_auc`
  - `data_quality_pass`
- **Artifacts:**
  - `model/` - LightGBM model

### Fine-Tuned Run
- **Run Name:** `YYYYmmddHHMM_lightgbm_tuned`
- **Parameters:**
  - `model_stage`: "tuned"
  - `base_run_id`: <baseline_run_id>
  - `is_finetune`: True
  - `best_*`: Best hyperparameters (if --tune used)
- **Metrics:**
  - `base_*`: Baseline performance on new data
  - `finetuned_*`: Fine-tuned performance
  - `finetune_train_time`
- **Artifacts:**
  - `model/` - Fine-tuned LightGBM model

### After ONNX Export
- **Updated Metric (on original run):**
  - `throughput_onnx`: Samples/second on ONNX Runtime

---

## Dependencies

**Required:**
```bash
pip install mlflow lightgbm scikit-learn pandas numpy
pip install onnxmltools>=1.12.0 onnxconverter-common>=1.13.0
pip install onnxruntime>=1.16.0
```

**Optional (for hyperparameter tuning):**
```bash
pip install optuna
```

**Optional (for data validation):**
```bash
pip install great-expectations
```

---

## Architecture Decisions

### Why NOT export ONNX during training?
1. **Separation of Concerns:** Training focuses on model quality, export focuses on serving performance
2. **Flexibility:** Export different models without retraining
3. **Benchmarking:** Throughput is measured once at deployment time, not training time
4. **Experimentation:** Try multiple ONNX configurations (opset versions, optimizations) without retraining

### Why separate fine-tuning script?
1. **Incremental Learning:** LightGBM supports adding trees to existing models
2. **Hyperparameter Search:** Optuna integration for finding best fine-tuning params
3. **Lineage Tracking:** Clear parent-child relationship in MLflow (`base_run_id`)
4. **Data Drift Handling:** Retrain on new data without starting from scratch

### Why standardized run naming?
1. **Chronological Ordering:** Easy to identify when models were trained
2. **Model Type Tagging:** Distinguish baseline from tuned models
3. **Automation:** Scripts can parse run names for deployment pipelines
4. **Human Readability:** Clear context without opening MLflow UI

---

## Troubleshooting

### "Failed to load model from run_id"
- Verify run_id exists: `mlflow runs describe --run-id <run_id>`
- Check if model artifact exists in the run
- Ensure MLflow tracking URI is correct

### "Features missing in new data"
- Fine-tuning data must have same features as baseline training
- Check `artifacts/feature_columns.json` for expected features
- Re-engineer features to match baseline schema

### "ONNX export failed"
- Install dependencies: `pip install onnxmltools onnxruntime`
- Verify LightGBM model is properly logged to MLflow
- Check model compatibility (some custom models may not convert)

### "Throughput benchmark failed"
- Install `onnxruntime`: `pip install onnxruntime>=1.16.0`
- Model will still be exported, only benchmark is skipped
- You can benchmark manually later

---

## Best Practices

1. **Always train baseline first** before fine-tuning
2. **Use descriptive experiment names** in MLflow
3. **Save run_id** from each stage for reproducibility
4. **Version your data** alongside model runs
5. **Monitor throughput** after each ONNX export
6. **Compare metrics** between baseline and tuned models
7. **Document hyperparameter choices** in MLflow tags/notes
8. **Test inference** before deploying to production

---

## Next Steps

After exporting ONNX model:
1. Test inference with `src/serving/inference.py`
2. Deploy FastAPI service: `uvicorn src.app.main:app`
3. Run integration tests
4. Monitor model performance in production
5. Retrain or fine-tune when performance degrades
