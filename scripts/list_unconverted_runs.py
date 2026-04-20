"""
Discover raw VTP runs that have not yet been converted to NPZ.

Outputs a JSON array of run IDs via dbutils.jobs.taskValues (on Databricks)
or stdout (locally). Designed as the first task in a for_each_task pipeline.

Usage:
    python scripts/list_unconverted_runs.py \
        --data_dir /Volumes/lucasbruand_catalog/cfd/data
"""

import argparse
import json
import os
import re


def main():
    parser = argparse.ArgumentParser(description="List unconverted DrivAerML runs")
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw")
    if not os.path.exists(raw_dir):
        print(f"ERROR: raw directory does not exist: {raw_dir}", flush=True)
        raise SystemExit(1)

    # Discover available raw runs
    available = set()
    for entry in os.listdir(raw_dir):
        m = re.match(r"run_(\d+)", entry)
        if m:
            run_id = int(m.group(1))
            vtp = os.path.join(raw_dir, entry, f"boundary_{run_id}.vtp")
            if os.path.exists(vtp):
                available.add(run_id)
    print(f"{len(available)} raw VTP files found", flush=True)

    # Discover already-converted
    existing = set()
    for entry in os.listdir(args.data_dir):
        m = re.match(r"drivaer_(\d+)\.npz", entry)
        if m:
            existing.add(int(m.group(1)))
    print(f"{len(existing)} NPZ files already exist", flush=True)

    to_convert = sorted(available - existing)
    print(f"{len(to_convert)} runs to convert", flush=True)

    # Output as task value for for_each_task.
    # Pass the Python list directly — taskValues.set handles JSON serialization.
    # Passing json.dumps(list) would double-encode it as a JSON string.
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        dbutils.jobs.taskValues.set(key="run_ids", value=to_convert)
        print(f"Set task value 'run_ids': {len(to_convert)} run IDs", flush=True)
    except Exception:
        # Running locally — just print
        print(f"run_ids={json.dumps(to_convert)}", flush=True)


if __name__ == "__main__":
    main()
