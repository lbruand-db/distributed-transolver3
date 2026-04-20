"""Preprocess mesh data: register metadata and compute statistics in Delta.

Usage (DAB workflow task):
  python preprocess.py --data_dir /Volumes/ml/transolver3/meshes \
                       --catalog ml --schema transolver3
"""

import argparse
import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(_this_dir, ".."))

from pyspark.sql import SparkSession  # noqa: E402

from transolver3.data_catalog import register_mesh_metadata  # noqa: E402
from transolver3.databricks_training import preprocess_with_spark  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Preprocess mesh data into Delta tables")
    parser.add_argument("--data_dir", required=True, help="Path to .npz mesh files")
    parser.add_argument("--catalog", default="ml")
    parser.add_argument("--schema", default="transolver3")
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    import time

    print(f"[1/2] Registering mesh metadata from {args.data_dir}...", flush=True)
    t0 = time.time()
    meta_df = register_mesh_metadata(spark, args.catalog, args.schema, "mesh_metadata", args.data_dir)
    t1 = time.time()
    n_meta = meta_df.count()
    print(f"  Registered {n_meta} samples ({t1 - t0:.1f}s)", flush=True)

    print("[2/2] Computing per-sample statistics (Spark distributed)...", flush=True)
    t0 = time.time()
    stats_df = preprocess_with_spark(spark, args.data_dir, args.catalog, args.schema, "mesh_stats")
    t1 = time.time()
    n_stats = stats_df.count()
    print(f"  Computed stats for {n_stats} samples ({t1 - t0:.1f}s)", flush=True)

    print("Preprocessing complete.", flush=True)


if __name__ == "__main__":
    main()
