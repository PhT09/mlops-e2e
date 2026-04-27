"""MLflow utility functions for cross-environment compatibility.

This module provides helpers to work with MLflow runs across different environments:
- Local: Uses run_id (UUID)
- Databricks: Uses run_name (YYYYmmddHHMM_model_stage format)
"""

import mlflow
from typing import Optional


def resolve_run_id(run_identifier: str, experiment_name: str) -> str:
    """
    Resolve run_id from either run_id (UUID) or run_name.
    
    This allows code to work in both:
    - Local: pass run_id directly (e.g., "6ca55f540219470381434c7ae622c89d")
    - Databricks: pass run_name (e.g., "202604271449_lightgbm_baseline")
    
    Parameters
    ----------
    run_identifier : str
        Either a run_id (32-char hex UUID) or run_name (timestamp-based name)
    experiment_name : str
        Name of the MLflow experiment (used for run_name lookup)
    
    Returns
    -------
    str
        The run_id (UUID)
    
    Raises
    ------
    ValueError
        If run_identifier cannot be resolved to a valid run
    
    Examples
    --------
    # Using run_id (local)
    run_id = resolve_run_id("6ca55f540219470381434c7ae622c89d", "My Experiment")
    
    # Using run_name (Databricks)
    run_id = resolve_run_id("202604271449_lightgbm_baseline", "My Experiment")
    """
    client = mlflow.tracking.MlflowClient()
    
    # Check if it's already a run_id (32 hex characters)
    if len(run_identifier) == 32 and all(c in '0123456789abcdef' for c in run_identifier):
        # Verify it exists
        try:
            client.get_run(run_identifier)
            return run_identifier
        except:
            raise ValueError(f"Run ID '{run_identifier}' not found")
    
    # Otherwise, treat as run_name and search for it
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Search for run by name
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_identifier}'",
        max_results=1
    )
    
    if not runs:
        raise ValueError(
            f"Run with name '{run_identifier}' not found in experiment '{experiment_name}'. "
            f"\nProvide either:"
            f"\n  - run_id (UUID): e.g., '6ca55f540219470381434c7ae622c89d'"
            f"\n  - run_name: e.g., '202604271449_lightgbm_baseline'"
        )
    
    return runs[0].info.run_id


def get_latest_run(experiment_name: str, stage: Optional[str] = None) -> tuple[str, str]:
    """
    Get the latest run from an experiment.
    
    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment
    stage : str, optional
        Filter by model stage ('baseline' or 'tuned')
    
    Returns
    -------
    tuple[str, str]
        (run_id, run_name) of the latest run
    
    Examples
    --------
    run_id, run_name = get_latest_run("Student Burnout Prediction", stage="baseline")
    print(f"Latest baseline: {run_name} (ID: {run_id})")
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    filter_string = f"params.model_stage = '{stage}'" if stage else ""
    
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time desc"],
        max_results=1
    )
    
    if not runs:
        stage_msg = f" with stage='{stage}'" if stage else ""
        raise ValueError(f"No runs found in experiment '{experiment_name}'{stage_msg}")
    
    run = runs[0]
    return run.info.run_id, run.data.tags.get("mlflow.runName", "unknown")


def print_run_info(run_identifier: str, experiment_name: str):
    """
    Print information about a run.
    
    Parameters
    ----------
    run_identifier : str
        Either run_id or run_name
    experiment_name : str
        Name of the MLflow experiment
    """
    run_id = resolve_run_id(run_identifier, experiment_name)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    print("="*60)
    print("RUN INFORMATION")
    print("="*60)
    print(f"Run ID:   {run.info.run_id}")
    print(f"Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"Status:   {run.info.status}")
    print(f"Stage:    {run.data.params.get('model_stage', 'N/A')}")
    print(f"Started:  {run.info.start_time}")
    print("="*60)
