"""
Download missing DrivAerML VTP files from HuggingFace and convert to .npz.

Designed to run on Databricks as a job task. Downloads one run at a time from
HuggingFace (neashton/drivaerml), converts to NPZ using pyvista, writes
directly to the UC Volume, then deletes the temp VTP to conserve disk.

Usage (Databricks job or local):
    python scripts/download_convert_vtp.py \
        --output_dir /Volumes/lucasbruand_catalog/cfd/data \
        [--runs 56 58 60]          # specific runs, or omit for all missing
        [--dry-run]                # list what would be converted
"""

import argparse
import csv
import io
import os
import shutil
import sys
import tempfile

import numpy as np


HF_REPO = "neashton/drivaerml"
HF_REPO_TYPE = "dataset"


def get_available_run_ids():
    """Query HuggingFace for all available run IDs with boundary VTP files."""
    import re

    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type=HF_REPO_TYPE)
    vtp_files = [f for f in files if "boundary" in f and f.endswith(".vtp")]
    return sorted(int(re.search(r"run_(\d+)", f).group(1)) for f in vtp_files)


def get_existing_run_ids(output_dir):
    """List run IDs that already have .npz files in the output directory.

    Tries dbutils.fs.ls first (Databricks), falls back to subprocess ls,
    then os.listdir. UC Volumes can be finicky with Python's os module.
    """
    import re

    fnames = []

    # Method 1: dbutils (Databricks runtime)
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        entries = dbutils.fs.ls(output_dir)
        fnames = [e.name for e in entries]
        print(f"  (listed {len(fnames)} entries via dbutils.fs.ls)", flush=True)
    except Exception:
        pass

    # Method 2: subprocess ls
    if not fnames:
        try:
            import subprocess

            result = subprocess.run(
                ["ls", output_dir], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                fnames = result.stdout.strip().split("\n")
                print(f"  (listed {len(fnames)} entries via ls)", flush=True)
        except Exception:
            pass

    # Method 3: os.listdir
    if not fnames:
        try:
            fnames = os.listdir(output_dir)
            print(f"  (listed {len(fnames)} entries via os.listdir)", flush=True)
        except Exception as e:
            print(f"  WARNING: cannot list {output_dir}: {e}", flush=True)
            return []

    existing = []
    for fname in fnames:
        m = re.match(r"drivaer_(\d+)\.npz", fname)
        if m:
            existing.append(int(m.group(1)))
    return sorted(existing)


def download_and_convert_one(run_id, output_dir, tmp_dir, geo_params_all):
    """Download VTP + CSV from HF, convert to NPZ, write to output_dir."""
    import pyvista as pv
    from huggingface_hub import hf_hub_download

    out_path = os.path.join(output_dir, f"drivaer_{run_id:03d}.npz")
    if os.path.exists(out_path):
        print(f"  SKIP run {run_id}: {out_path} already exists", flush=True)
        return True

    # --- Download VTP ---
    print(f"  Downloading run_{run_id}/boundary_{run_id}.vtp ...", flush=True)
    try:
        vtp_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"run_{run_id}/boundary_{run_id}.vtp",
            repo_type=HF_REPO_TYPE,
            cache_dir=tmp_dir,
        )
    except Exception as e:
        print(f"  ERROR downloading VTP for run {run_id}: {e}", flush=True)
        return False

    # --- Download geo_parameters CSV ---
    params = None
    try:
        csv_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"run_{run_id}/geo_parameters_{run_id}.csv",
            repo_type=HF_REPO_TYPE,
            cache_dir=tmp_dir,
        )
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            params = np.array(
                [float(v) for k, v in row.items() if k.strip().lower() != "run"],
                dtype=np.float32,
            )
        print(f"    Params: {len(params)} values from per-run CSV", flush=True)
    except Exception:
        # Fall back to geo_parameters_all.csv
        if geo_params_all is not None and run_id in geo_params_all:
            params = geo_params_all[run_id]
            print(f"    Params: {len(params)} values from geo_parameters_all.csv", flush=True)
        else:
            params = np.zeros(16, dtype=np.float32)
            print("    WARNING: no params found, using zeros", flush=True)

    # --- Read mesh ---
    print(f"    Loading mesh ...", flush=True)
    mesh = pv.read(vtp_path)
    print(f"    {mesh.n_cells:,} cells, {mesh.n_points:,} points", flush=True)

    # Cell-centered coordinates
    coords = np.array(mesh.cell_centers().points, dtype=np.float32)

    # Cell normals
    mesh_n = mesh.compute_normals(
        point_normals=False, cell_normals=True, auto_orient_normals=True
    )
    normals = np.array(mesh_n.cell_data["Normals"], dtype=np.float32)

    # Pressure
    pressure = None
    for name in ["CpMeanTrim", "Cp", "pMeanTrim", "p", "pressure"]:
        if name in mesh.cell_data:
            pressure = np.array(mesh.cell_data[name], dtype=np.float32)
            print(
                f"    Pressure: '{name}' range=[{pressure.min():.4f}, {pressure.max():.4f}]",
                flush=True,
            )
            break
    if pressure is None:
        raise ValueError(
            f"No pressure field for run {run_id}. Available: {list(mesh.cell_data.keys())}"
        )
    if pressure.ndim == 1:
        pressure = pressure[:, None]

    # Wall shear stress
    wall_shear = None
    for name in ["wallShearStressMeanTrim", "wallShearStress", "WallShearStress"]:
        if name in mesh.cell_data:
            wall_shear = np.array(mesh.cell_data[name], dtype=np.float32)
            print(f"    Wall shear: '{name}' shape={wall_shear.shape}", flush=True)
            break
    if wall_shear is None:
        print("    WARNING: No wall shear field, using zeros", flush=True)
        wall_shear = np.zeros((coords.shape[0], 3), dtype=np.float32)
    if wall_shear.ndim == 1:
        wall_shear = wall_shear[:, None]

    # --- Save NPZ ---
    np.savez(
        out_path,
        params=params,
        surface_coords=coords,
        surface_normals=normals,
        surface_pressure=pressure,
        surface_wall_shear=wall_shear,
    )
    size_mb = os.path.getsize(out_path) / 1024**2
    print(
        f"    Saved {out_path} ({size_mb:.1f} MB, {coords.shape[0]:,} cells)",
        flush=True,
    )

    # --- Cleanup HF cache for this file to save disk ---
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
    except Exception:
        pass

    return True


def load_geo_params_all(tmp_dir):
    """Download and parse geo_parameters_all.csv as fallback for per-run CSVs."""
    from huggingface_hub import hf_hub_download

    try:
        csv_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="geo_parameters_all.csv",
            repo_type=HF_REPO_TYPE,
            cache_dir=tmp_dir,
        )
        result = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = int(row.get("run", row.get("Run", -1)))
                if run_id < 0:
                    continue
                vals = [float(v) for k, v in row.items() if k.strip().lower() != "run"]
                result[run_id] = np.array(vals, dtype=np.float32)
        print(f"Loaded geo_parameters_all.csv: {len(result)} runs", flush=True)
        return result
    except Exception as e:
        print(f"WARNING: Could not load geo_parameters_all.csv: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Download & convert DrivAerML VTP to NPZ")
    parser.add_argument(
        "--output_dir",
        default="/Volumes/lucasbruand_catalog/cfd/data",
        help="UC Volume path for output .npz files",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=None,
        help="Specific run IDs to convert (default: all missing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be converted without downloading",
    )
    args = parser.parse_args()

    # Determine which runs to convert
    print("Querying HuggingFace for available runs...", flush=True)
    available = set(get_available_run_ids())
    print(f"  {len(available)} runs available on HF", flush=True)

    existing = set(get_existing_run_ids(args.output_dir))
    print(f"  {len(existing)} runs already in {args.output_dir}", flush=True)

    if args.runs:
        to_convert = sorted(r for r in args.runs if r in available)
        skipped = sorted(r for r in args.runs if r not in available)
        if skipped:
            print(f"  WARNING: runs not on HF (skipped): {skipped}", flush=True)
    else:
        to_convert = sorted(available - existing)

    print(f"\n  {len(to_convert)} runs to convert", flush=True)

    if args.dry_run:
        print(f"  Dry run — would convert: {to_convert}", flush=True)
        return

    if not to_convert:
        print("  Nothing to do.", flush=True)
        return

    # Download fallback geo params
    tmp_dir = tempfile.mkdtemp(prefix="hf_drivaer_")
    geo_params_all = load_geo_params_all(tmp_dir)

    # Process each run
    success = 0
    failed = []
    for i, run_id in enumerate(to_convert):
        print(f"\n[{i + 1}/{len(to_convert)}] Run {run_id}", flush=True)
        try:
            ok = download_and_convert_one(run_id, args.output_dir, tmp_dir, geo_params_all)
            if ok:
                success += 1
            else:
                failed.append(run_id)
        except Exception as e:
            print(f"  FAILED run {run_id}: {e}", flush=True)
            failed.append(run_id)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Done. {success} converted, {len(failed)} failed.", flush=True)
    if failed:
        print(f"Failed runs: {failed}", flush=True)


if __name__ == "__main__":
    main()
