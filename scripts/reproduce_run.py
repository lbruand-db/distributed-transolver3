"""Reproduce a past training run from its MLflow run ID.

Fetches all parameters from the MLflow run and prints the exact command
to re-run training with the same configuration.

Usage:
    python scripts/reproduce_run.py --run_id <mlflow_run_id>
    python scripts/reproduce_run.py --run_id <mlflow_run_id> --execute
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Reproduce a training run from MLflow")
    parser.add_argument("--run_id", required=True, help="MLflow run ID to reproduce")
    parser.add_argument(
        "--experiment",
        default="/Shared/transolver3-experiments",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the reproduced command (default: print only)",
    )
    args = parser.parse_args()

    try:
        import mlflow
    except ImportError:
        print("mlflow is required: pip install mlflow>=2.10")
        sys.exit(1)

    mlflow.set_experiment(args.experiment)
    run = mlflow.get_run(args.run_id)
    params = run.data.params

    print(f"Reproducing run: {args.run_id}")
    print(f"Run name: {run.info.run_name}")
    print(f"Status: {run.info.status}")
    print("\nLogged parameters:")
    for k, v in sorted(params.items()):
        print(f"  {k} = {v}")

    # Map MLflow params to CLI args
    param_to_flag = {
        "field": "--field",
        "epochs": "--epochs",
        "lr": "--lr",
        "weight_decay": "--weight_decay",
        "subset_size": "--subset_size",
        "n_layers": "--n_layers",
        "n_hidden": "--n_hidden",
        "n_head": "--n_head",
        "slice_num": "--slice_num",
        "num_tiles": "--num_tiles",
    }

    cmd_parts = [
        "python",
        "Industrial-Scale-Benchmarks/exp_drivaer_ml_distributed.py",
        "--data_dir", "<DATA_DIR>",
    ]

    for param_name, flag in param_to_flag.items():
        if param_name in params:
            cmd_parts.extend([flag, str(params[param_name])])

    # Add world_size as num-gpus if distributed
    world_size = params.get("world_size", "1")
    if int(world_size) > 1:
        cmd_parts.extend(["--use-distributor", "--num-gpus", world_size])

    cmd = " \\\n    ".join(cmd_parts)
    print(f"\nReproduction command:\n{cmd}")
    print("\nReplace <DATA_DIR> with your data path.")

    if args.execute:
        import subprocess

        # Replace placeholder
        data_dir = input("Enter data_dir path: ").strip()
        final_cmd = [p.replace("<DATA_DIR>", data_dir) for p in cmd_parts]
        print(f"\nExecuting: {' '.join(final_cmd)}")
        subprocess.run(final_cmd, check=True)


if __name__ == "__main__":
    main()
