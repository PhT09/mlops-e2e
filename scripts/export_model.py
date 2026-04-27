"""
Export a trained LightGBM model to ONNX format.

This script:
1. Loads a model from MLflow (by run_id OR run_name)
2. Converts to ONNX format
3. Benchmarks throughput
4. Logs ONNX metrics back to MLflow run
"""

import os
import sys
import argparse
from pathlib import Path

project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

from src.models.export_onnx import export_to_onnx
from src.utils.mlflow_helpers import resolve_run_id, print_run_info, get_latest_run


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


def main(args):
    """Main execution function."""
    import mlflow
    import json
    
    # Setup MLflow
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    elif not is_databricks_environment():
        mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
    
    experiment_name_full = get_experiment_name(args.experiment_name)
    
    if args.run_identifier.lower() == 'latest':
        print(f"\n{'='*60}")
        print("FINDING LATEST RUN")
        print(f"{'='*60}")
        print(f"Experiment: {experiment_name_full}")
        
        try:
            run_id, run_name = get_latest_run(experiment_name_full)
            print(f"Latest run: {run_name}")
            print(f"   Run ID: {run_id}")
        except ValueError as e:
            print(f"Error: {e}")
            return None
    else:
        # Resolve run_id from run_id or run_name
        print(f"\n{'='*60}")
        print("RESOLVING MODEL RUN")
        print(f"{'='*60}")
        print(f"Input: {args.run_identifier}")
        print(f"Experiment: {experiment_name_full}")
        
        try:
            run_id = resolve_run_id(args.run_identifier, experiment_name_full)
            print(f"Resolved to run_id: {run_id}")
            print_run_info(run_id, experiment_name_full)
        except ValueError as e:
            print(f"Error: {e}")
            return None
    
    # Export to ONNX
    print(f"\n{'='*60}")
    print("EXPORTING TO ONNX")
    print(f"{'='*60}")
    
    try:
        onnx_path = export_to_onnx(
            run_id=run_id,
            artifacts_dir=args.output_dir,
            mlflow_tracking_uri=args.mlflow_uri,
            benchmark_samples=args.benchmark_samples
        )
        
        # Load metadata
        metadata_path = os.path.join(args.output_dir, "onnx_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n{'='*60}")
        print("EXPORT COMPLETE")
        print(f"{'='*60}")
        print(f"ONNX model saved to: {onnx_path}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Throughput: {metadata.get('throughput_samples_per_sec', 0):.0f} samples/sec")
        print(f"MLflow updated with throughput metric")
        print(f"{'='*60}\n")
        
        return {
            'onnx_path': onnx_path,
            'metadata_path': metadata_path,
            'throughput': metadata.get('throughput_samples_per_sec', 0)
        }
        
    except Exception as e:
        print(f"\nExport failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model to ONNX format')
    parser.add_argument(
        '--run_identifier',
        type=str,
        required=True,
        help='Model run_id (UUID), run_name (e.g., 202604271449_lightgbm_baseline), or "latest"'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='Student Burnout Prediction',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for ONNX model (default: {project_root}/artifacts)'
    )
    parser.add_argument(
        '--mlflow_uri',
        type=str,
        default=None,
        help='MLflow tracking URI (auto-detected if not provided)'
    )
    parser.add_argument(
        '--benchmark_samples',
        type=int,
        default=10000,
        help='Number of samples for throughput benchmark'
    )
    
    args = parser.parse_args()
    
    # Set default output_dir if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'artifacts')
    
    main(args)
