# Cross-Environment MLflow Integration Guide

## Overview

This guide explains how the MLOps pipeline works seamlessly across **Local** and **Databricks** environments. Both environments use **MLflow run IDs (UUIDs)** internally, but you can reference runs using **either run_id OR run_name** in both environments.

---

## MLflow Run Identification

### Key Concept: Run ID vs Run Name

MLflow **always** generates a unique run_id (UUID) for every run, regardless of environment:

```
run_id: "6ca55f540219470381434c7ae622c89d"  ← 32-character UUID
run_name: "202604271449_lightgbm_baseline"  ← Human-readable timestamp
```

**Both identifiers work in both environments!**

---

## Environment Comparison

 Feature | Local | Databricks |
---------|-------|------------|
 **MLflow Tracking** | File-based (`mlruns/`) | Managed service |
 **Run ID** | UUID generated | UUID generated |
 **Run Name** | Custom name supported | Custom name supported |
 **Experiment Name** | Simple string | Workspace path (`/Users/...`) |
 **API Access** | Same Python API | Same Python API |
 **UI Access** | Local web server | Integrated workspace UI |

---

## Using the Pipeline in Both Environments

### 1. Training (run_pipeline.py)

**Local:**
```bash
cd /path/to/mlops-e2e
python scripts/run_pipeline.py \
    --input data/raw/student_mental_health_burnout_relabeled.csv \
    --experiment "Student Burnout Prediction"
```

**Databricks Notebook:**
```python
from scripts.run_pipeline import main

class Args:
    input = 'data/raw/student_mental_health_burnout_relabeled.csv'
    experiment_name = 'Student Burnout Prediction'
    mlflow_uri = None  # Auto-detected

main(Args())
```

**Output (same in both):**
```
Training completed!
Run ID:   6ca55f540219470381434c7ae622c89d
Run Name: 202604271449_lightgbm_baseline
```

---

### 2. Fine-tuning (fine_tune.py)

**You can use EITHER run_id OR run_name in both environments!**

#### Option A: Using Run ID (UUID)

**Local:**
```bash
python scripts/fine_tune.py \
    --base_run_identifier "6ca55f540219470381434c7ae622c89d" \
    --experiment_name "Student Burnout Prediction" \
    --tune --n_trials 10
```

**Databricks:**
```python
from scripts.fine_tune import main

class Args:
    base_run_identifier = "6ca55f540219470381434c7ae622c89d"
    experiment_name = 'Student Burnout Prediction'
    tune = True
    n_trials = 10

main(Args())
```

#### Option B: Using Run Name

**Local:**
```bash
python scripts/fine_tune.py \
    --base_run_identifier "202604271449_lightgbm_baseline" \
    --experiment_name "Student Burnout Prediction" \
    --tune --n_trials 10
```

**Databricks:**
```python
from scripts.fine_tune import main

class Args:
    base_run_identifier = "202604271449_lightgbm_baseline"  # Run name works!
    experiment_name = 'Student Burnout Prediction'
    tune = True
    n_trials = 10

main(Args())
```

---

### 3. ONNX Export (export_model.py)

**Again, use EITHER run_id OR run_name in both environments!**

#### Using Run ID

**Local:**
```bash
python scripts/export_model.py \
    --run_identifier "6ca55f540219470381434c7ae622c89d" \
    --output_dir artifacts
```

**Databricks:**
```python
from scripts.export_model import main

class Args:
    run_identifier = "6ca55f540219470381434c7ae622c89d"
    experiment_name = 'Student Burnout Prediction'
    output_dir = 'artifacts'

main(Args())
```

#### Using Run Name

**Local:**
```bash
python scripts/export_model.py \
    --run_identifier "202604271449_lightgbm_baseline" \
    --output_dir artifacts
```

**Databricks:**
```python
from scripts.export_model import main

class Args:
    run_identifier = "202604271449_lightgbm_baseline"  # Run name works!
    experiment_name = 'Student Burnout Prediction'
    output_dir = 'artifacts'

main(Args())
```

#### Using "latest" (Auto-find Most Recent)

**Both environments:**
```python
class Args:
    run_identifier = "latest"  # Automatically finds latest run
    experiment_name = 'Student Burnout Prediction'
```

---

## How It Works: resolve_run_id()

The `src/utils/mlflow_helpers.py` provides environment-agnostic run resolution:

```python
from src.utils.mlflow_helpers import resolve_run_id

# Case 1: Input is run_id (32 hex chars)
run_id = resolve_run_id(
    "6ca55f540219470381434c7ae622c89d",
    "Student Burnout Prediction"
)
# → Returns: "6ca55f540219470381434c7ae622c89d" (same)

# Case 2: Input is run_name
run_id = resolve_run_id(
    "202604271449_lightgbm_baseline",
    "Student Burnout Prediction"
)
# → Searches MLflow, returns: "6ca55f540219470381434c7ae622c89d"
```

**Logic:**
1. Check if input is 32-character hex → Treat as run_id, verify it exists
2. Otherwise → Search for run with matching `mlflow.runName` tag
3. Return the resolved run_id

---

## Experiment Name Handling

### Local Environment

Simple string:
```python
experiment_name = "Student Burnout Prediction"
```

### Databricks Environment

Requires workspace path:
```python
experiment_name = "/Users/user@example.com/Student Burnout Prediction"
```

**Auto-detection:**
```python
def get_experiment_name(base_name):
    if is_databricks_environment():
        # Extract user from project path
        import re
        match = re.search(r'/Users/([^/]+)/', project_root)
        user_email = match.group(1)
        return f"/Users/{user_email}/{base_name}"
    else:
        return base_name
```

---

## Finding Run Identifiers

### In MLflow UI (Both Environments)

1. Navigate to MLflow Experiments
2. Click on your run
3. Copy **either**:
   * **Run ID** (under "Run ID" field)
   * **Run Name** (under Tags → `mlflow.runName`)

### Programmatically

```python
from src.utils.mlflow_helpers import get_latest_run

# Get latest run from experiment
run_id, run_name = get_latest_run("Student Burnout Prediction")
print(f"Latest: {run_name} (ID: {run_id})")

# Get latest baseline
run_id, run_name = get_latest_run("Student Burnout Prediction", stage="baseline")

# Get latest tuned
run_id, run_name = get_latest_run("Student Burnout Prediction", stage="tuned")
```

---

## Common Patterns

### Pattern 1: CLI with Run Name (Easier to Remember)

```bash
# Train baseline
python scripts/run_pipeline.py --input data/raw/data.csv
# Output: 202604271449_lightgbm_baseline

# Fine-tune using name
python scripts/fine_tune.py --base_run_identifier "202604271449_lightgbm_baseline"

# Export using name
python scripts/export_model.py --run_identifier "202604271449_lightgbm_baseline"
```

### Pattern 2: Notebook with Run ID (More Reliable)

```python
# Cell 1: Train
results = train_pipeline()
baseline_run_id = results['run_id']  # Store UUID

# Cell 2: Fine-tune
fine_tune(base_run_identifier=baseline_run_id)

# Cell 3: Export
export_onnx(run_identifier=baseline_run_id)
```

### Pattern 3: Mixed (Best of Both)

```python
# Use name for readability in config
BASE_MODEL = "202604271449_lightgbm_baseline"

# Resolve to run_id internally (automatic)
run_id = resolve_run_id(BASE_MODEL, experiment_name)

# Use run_id for critical operations
export_onnx(run_identifier=run_id)
```

---

## Troubleshooting

### Error: Run Not Found

**Symptom:**
```
ValueError: Run with name '202604271449_lightgbm_baseline' not found
```

**Solutions:**
1. Verify run_name is exact (case-sensitive, check MLflow UI)
2. Verify experiment_name is correct
3. Try using run_id (UUID) instead
4. Check you're in the right workspace (Databricks)

### Error: Invalid Run ID

**Symptom:**
```
ValueError: Run ID '202604271449_lightgbm_baseline' not found
```

**Cause:** Passed run_name where run_id expected (old code)

**Solution:** Update to latest scripts that support both

### Error: Experiment Not Found

**Databricks-specific:**
```
Experiment 'Student Burnout Prediction' not found
```

**Solution:** Use full workspace path
```python
experiment_name = "/Users/your.email@databricks.com/Student Burnout Prediction"
```

Or use `get_experiment_name()` helper (auto-detects)

---

## Migration Checklist

If migrating from old code that only accepted run_id:

- [x] Update `fine_tune.py` to use `resolve_run_id()`
- [x] Update `export_model.py` to use `resolve_run_id()`
- [x] Update notebook cells with examples for both identifiers
- [x] Update CLI argument names: `--run_id` → `--run_identifier`
- [x] Add helper functions in `src/utils/mlflow_helpers.py`
- [x] Test in both Local and Databricks environments
- [x] Update documentation

---

## Best Practices

### For Development (Notebooks)

**Use run_id (UUID)**: More reliable, no name collisions
```python
BASE_RUN_ID = "6ca55f540219470381434c7ae622c89d"
```

### For Production (CLI/Scripts)

**Use run_name**: Easier to track, audit, debug
```bash
python fine_tune.py --base_run_identifier "202604271449_lightgbm_baseline"
```

### For Automation

**Use "latest"**: Auto-find most recent run
```python
export_model(run_identifier="latest")
```

---

## Summary

 Aspect | Local | Databricks | Notes |
--------|-------|------------|-------|
 **Run ID format** | UUID | UUID | Always 32-char hex |
 **Run name format** | Custom | Custom | Our convention: `YYYYmmddHHMM_model_stage` |
 **Accepting run_id** | ✅ | ✅ | Works in all scripts |
 **Accepting run_name** | ✅ | ✅ | Works in all scripts |
 **Auto-resolution** | ✅ | ✅ | `resolve_run_id()` handles both |
 **MLflow API** | Same | Same | No code changes needed |
 **Experiment naming** | Simple string | Workspace path | Auto-detected |

**Key Insight:** MLflow generates run_ids in both environments. The pipeline now accepts **both run_id and run_name** everywhere, making it truly cross-environment compatible.

---

## Quick Reference

**Get identifiers from a run:**
```python
from src.utils.mlflow_helpers import print_run_info

print_run_info("6ca55f540219470381434c7ae622c89d", "Student Burnout Prediction")
# or
print_run_info("202604271449_lightgbm_baseline", "Student Burnout Prediction")
```

**Find latest run:**
```python
from src.utils.mlflow_helpers import get_latest_run

run_id, run_name = get_latest_run("Student Burnout Prediction")
```

**Resolve any identifier:**
```python
from src.utils.mlflow_helpers import resolve_run_id

run_id = resolve_run_id("202604271449_lightgbm_baseline", "Student Burnout Prediction")
```

---