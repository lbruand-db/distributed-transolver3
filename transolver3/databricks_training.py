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


def launch_distributed_training(train_fn, num_gpus, **kwargs):
    """Launch distributed training using TorchDistributor on Databricks,
    or torchrun locally.

    Args:
        train_fn: callable that runs the training logic.
            On Databricks, this is passed to TorchDistributor.run().
            Locally, a torchrun subprocess is spawned.
        num_gpus: number of GPU processes to launch
        **kwargs: additional arguments passed to train_fn

    Returns:
        Result from TorchDistributor.run() on Databricks, or
        subprocess.CompletedProcess locally.
    """
    if is_on_databricks():
        from pyspark.ml.torch.distributor import TorchDistributor

        distributor = TorchDistributor(
            num_processes=num_gpus,
            local_mode=True,
            use_gpu=True,
        )
        return distributor.run(train_fn, **kwargs)
    else:
        # Local fallback: use torchrun
        # train_fn must be importable as a script with if __name__ == '__main__'
        script = _resolve_script_path(train_fn)
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
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
            "pyspark is required for preprocess_with_spark. "
            "Install with: pip install transolver3[databricks]"
        )
    import numpy as np

    # Collect file paths
    files = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.endswith(".npz")
    ]

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
