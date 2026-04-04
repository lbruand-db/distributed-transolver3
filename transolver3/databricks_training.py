# Copyright 2026 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Databricks training integration for Transolver-3.

Provides:
  - Environment detection (Databricks vs local)
  - TorchDistributor launcher (auto-fallback to torchrun locally)
  - Spark-based mesh preprocessing for dataset statistics

All Databricks-specific imports are guarded. Install with:
  pip install transolver3[databricks]
"""

import json
import os
import subprocess
import sys


def is_on_databricks():
    """Check if running on a Databricks cluster.

    Returns:
        True if DATABRICKS_RUNTIME_VERSION environment variable is set.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _propagate_databricks_auth_env():
    """Ensure Databricks auth env vars are set for child processes.

    On Databricks, the driver has implicit auth via internal APIs.
    But torchrun child processes don't inherit it. This function
    extracts credentials from MLflow's own working auth (which uses
    the driver's internal APIs) and sets them as env vars so child
    processes can authenticate independently.

    Must be called in the driver process before TorchDistributor launches.
    """
    if os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        return  # already set

    # Extract host + token from MLflow's Databricks credential provider.
    # This works because MLflow CAN auth in the driver — we just need to
    # capture the same credentials as env vars for child processes.
    try:
        from mlflow.utils.databricks_utils import get_databricks_host_creds

        creds = get_databricks_host_creds()
        if creds.host and creds.token:
            os.environ["DATABRICKS_HOST"] = creds.host
            os.environ["DATABRICKS_TOKEN"] = creds.token
            return
    except Exception:
        pass

    # Fallback: Databricks SDK
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        config = w.config
        if config.host and config.token:
            os.environ["DATABRICKS_HOST"] = config.host
            os.environ["DATABRICKS_TOKEN"] = config.token
            return
    except Exception:
        pass


def launch_distributed_training(train_fn, num_gpus, cli_args=None, **kwargs):
    """Launch distributed training using TorchDistributor on Databricks,
    or torchrun locally.

    Args:
        train_fn: callable that runs the training logic, or a string script path.
            On Databricks, the script path is passed to TorchDistributor.run()
            so that child processes inherit PYTHONPATH (avoids cloudpickle
            module resolution issues).
            Locally, a torchrun subprocess is spawned.
        num_gpus: number of GPU processes to launch
        cli_args: list of CLI arguments to forward to the script (used on Databricks)
        **kwargs: additional arguments passed as CLI flags (local fallback)

    Returns:
        Result from TorchDistributor.run() on Databricks, or
        subprocess.CompletedProcess locally.
    """
    script = _resolve_script_path(train_fn)

    if is_on_databricks():
        from pyspark.ml.torch.distributor import TorchDistributor

        # Ensure Databricks auth env vars are set so torchrun child processes
        # can reach MLflow. The driver has these from the notebook context but
        # they may not propagate to children automatically.
        _propagate_databricks_auth_env()

        distributor = TorchDistributor(
            num_processes=num_gpus,
            local_mode=True,
            use_gpu=True,
        )
        # Pass script path (not function) so torchrun runs it directly.
        # This avoids cloudpickle serialization issues where child processes
        # cannot find transolver3 module.
        # Forward CLI args so child processes get --data_dir, --epochs, etc.
        if cli_args:
            return distributor.run(script, *cli_args)
        return distributor.run(script)
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            script,
        ]
        # Pass kwargs as CLI arguments
        for k, v in kwargs.items():
            cmd.append(f"--{k}")
            cmd.append(str(v))

        return subprocess.run(cmd, check=True)


def _resolve_script_path(train_fn):
    """Resolve the file path of a training function or module.

    Args:
        train_fn: a callable, a module, or a string path

    Returns:
        Absolute path to the script file
    """
    if isinstance(train_fn, str):
        return os.path.abspath(train_fn)

    # If it's a function, find its module file
    module = getattr(train_fn, "__module__", None)
    if module and module != "__main__":
        import importlib

        mod = importlib.import_module(module)
        if hasattr(mod, "__file__") and mod.__file__:
            return os.path.abspath(mod.__file__)

    # If it has __file__ directly (e.g., a module object)
    if hasattr(train_fn, "__file__") and train_fn.__file__:
        return os.path.abspath(train_fn.__file__)

    # If it's a function executed via exec(compile(src, filename, 'exec')),
    # the code object's co_filename has the original file path.
    code = getattr(train_fn, "__code__", None)
    if code and getattr(code, "co_filename", None):
        co_file = code.co_filename
        if os.path.isfile(co_file):
            return os.path.abspath(co_file)

    raise ValueError(
        f"Cannot resolve script path for {train_fn}. "
        "Pass a string path, a function from an importable module, or a module object."
    )


def preprocess_with_spark(spark, data_dir, catalog, schema, table):
    """Compute per-sample mesh statistics using Spark and write to Delta.

    Parallelizes statistics computation across Spark workers. Each worker
    reads one .npz file and extracts coordinate/target statistics.

    Args:
        spark: SparkSession
        data_dir: path to directory containing .npz mesh files
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name for stats output (e.g., "mesh_stats")

    Returns:
        PySpark DataFrame with per-sample statistics
    """
    try:
        from pyspark.sql import Row
        from pyspark.sql.types import StructType, StructField, StringType
    except ImportError:
        raise ImportError(
            "pyspark is required for preprocess_with_spark. Install with: pip install transolver3[databricks]"
        )
    import numpy as np

    # Collect file paths
    files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith(".npz")]

    # Create a DataFrame of file paths to distribute work
    file_schema = StructType([StructField("file_path", StringType(), False)])
    files_df = spark.createDataFrame([(f,) for f in files], schema=file_schema)

    def compute_stats(file_path):
        """UDF-like function to compute stats for a single .npz file."""
        data = np.load(file_path, mmap_mode="r")
        sample_id = os.path.splitext(os.path.basename(file_path))[0]

        stats = {"sample_id": sample_id, "file_path": file_path}

        for key in data.keys():
            arr = np.array(data[key])  # materialize from mmap
            stats[f"{key}_shape"] = json.dumps(list(arr.shape))
            if arr.ndim >= 1 and arr.dtype in (np.float32, np.float64):
                stats[f"{key}_min"] = float(arr.min())
                stats[f"{key}_max"] = float(arr.max())
                stats[f"{key}_mean"] = float(arr.mean())
                stats[f"{key}_std"] = float(arr.std())

        return Row(**{k: str(v) for k, v in stats.items()})

    # Use RDD map for flexible schema (different .npz files may have different keys)
    stats_rdd = files_df.rdd.map(lambda row: compute_stats(row.file_path))
    stats_df = spark.createDataFrame(stats_rdd)

    full_table = f"{catalog}.{schema}.{table}"
    stats_df.write.format("delta").mode("overwrite").saveAsTable(full_table)
    return spark.table(full_table)
