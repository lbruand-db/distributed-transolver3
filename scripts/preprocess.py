"""Preprocess mesh data: register metadata and compute statistics in Delta.

Usage (DAB workflow task):
  python preprocess.py --data_dir /Volumes/ml/transolver3/meshes \
                       --catalog ml --schema transolver3
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

    print(f"Registering mesh metadata from {args.data_dir}...")
    meta_df = register_mesh_metadata(
        spark, args.catalog, args.schema, "mesh_metadata", args.data_dir
    )
    print(f"Registered {meta_df.count()} samples in {args.catalog}.{args.schema}.mesh_metadata")

    print("Computing per-sample statistics...")
    stats_df = preprocess_with_spark(
        spark, args.data_dir, args.catalog, args.schema, "mesh_stats"
    )
    print(f"Computed stats for {stats_df.count()} samples in {args.catalog}.{args.schema}.mesh_stats")


if __name__ == "__main__":
    main()
