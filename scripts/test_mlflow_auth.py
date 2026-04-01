"""Quick smoke test for MLflow auth propagation in TorchDistributor child processes.

Run via DAB:
    databricks bundle deploy -t a10g --profile DEFAULT
    databricks bundle run test_mlflow_auth -t a10g --profile DEFAULT

Or directly on a GPU cluster notebook:
    %run /Workspace/.../scripts/test_mlflow_auth.py
"""

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(_this_dir, ".."))

EXPERIMENT_NAME = "/Shared/transolver3-experiments"


def child_process_test():
    """Runs inside each torchrun child process."""
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print(f"[rank {rank}] DATABRICKS_HOST = {os.environ.get('DATABRICKS_HOST', '<NOT SET>')}")
    print(f"[rank {rank}] DATABRICKS_TOKEN = {'***' + os.environ.get('DATABRICKS_TOKEN', '')[-4:] if os.environ.get('DATABRICKS_TOKEN') else '<NOT SET>'}")

    # Only rank 0 tests MLflow
    if rank == 0:
        import mlflow

        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="auth-smoke-test") as run:
            mlflow.log_param("world_size", world_size)
            mlflow.log_param("test_type", "auth_propagation")
            mlflow.log_metric("dummy_loss", 0.42, step=0)
            mlflow.log_metric("dummy_loss", 0.21, step=1)
            print(f"[rank {rank}] MLflow run logged: {run.info.run_id}")
            print(f"[rank {rank}] Experiment: {EXPERIMENT_NAME}")

    if dist.is_initialized():
        dist.barrier()

    print(f"[rank {rank}] SUCCESS")


if __name__ == "__main__":
    if "--use-distributor" in sys.argv:
        from transolver3.databricks_training import launch_distributed_training

        idx = sys.argv.index("--num-gpus") if "--num-gpus" in sys.argv else None
        num_gpus = int(sys.argv[idx + 1]) if idx else 2

        # Clean argv for child processes
        clean_argv = []
        skip_next = False
        for a in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if a == "--use-distributor":
                continue
            if a == "--num-gpus":
                skip_next = True
                continue
            clean_argv.append(a)
        sys.argv = [sys.argv[0]] + clean_argv

        launch_distributed_training(child_process_test, num_gpus, cli_args=clean_argv)
    else:
        # Direct torchrun or single-process execution
        child_process_test()
