"""
Convert a single DrivAerML VTP file to NPZ format.

Reads a raw VTP + CSV from data_dir/raw/run_NNN/ and writes an NPZ file
to data_dir/drivaer_NNN.npz. Designed to be called by a for_each_task
in Databricks, where each serverless task converts one run.

Output NPZ schema:
    params:             (d_params,)  float32 — geometric deformation parameters
    surface_coords:     (N, 3)       float32 — cell-centered coordinates
    surface_normals:    (N, 3)       float32 — cell normals
    surface_pressure:   (N, 1)       float32 — Cp (pressure coefficient)
    surface_wall_shear: (N, 3)       float32 — wall shear stress vector

Usage:
    python scripts/convert_vtp_to_npz_batch.py \
        --data_dir /Volumes/lucasbruand_catalog/cfd/data \
        --run 56
"""

import argparse
import csv
import os
import shutil
import tempfile

import numpy as np


def load_run_params(raw_dir, run_id):
    """Load geometric parameters for a run, trying per-run CSV then global."""
    # Per-run CSV
    csv_path = os.path.join(raw_dir, f"run_{run_id:03d}", f"geo_parameters_{run_id}.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            return np.array(
                [float(v) for k, v in row.items() if k.strip().lower() != "run"],
                dtype=np.float32,
            )

    # Global fallback
    global_csv = os.path.join(raw_dir, "geo_parameters_all.csv")
    if os.path.exists(global_csv):
        with open(global_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = int(row.get("run", row.get("Run", -1)))
                if rid == run_id:
                    vals = [float(v) for k, v in row.items() if k.strip().lower() != "run"]
                    return np.array(vals, dtype=np.float32)

    print(f"  WARNING: no params found for run {run_id}, using zeros", flush=True)
    return np.zeros(16, dtype=np.float32)


def convert_one(run_id, data_dir):
    """Convert a single run's VTP to NPZ."""
    import pyvista as pv

    raw_dir = os.path.join(data_dir, "raw")
    vtp_path = os.path.join(raw_dir, f"run_{run_id:03d}", f"boundary_{run_id}.vtp")
    out_path = os.path.join(data_dir, f"drivaer_{run_id:03d}.npz")

    if os.path.exists(out_path):
        print(f"Run {run_id}: already exists, skipping", flush=True)
        return

    if not os.path.exists(vtp_path):
        print(f"Run {run_id}: FAILED — VTP not found: {vtp_path}", flush=True)
        raise SystemExit(1)

    print(f"Run {run_id}: loading {vtp_path} ...", flush=True)

    # Load params
    params = load_run_params(raw_dir, run_id)

    # Read mesh
    mesh = pv.read(vtp_path)
    print(f"  {mesh.n_cells:,} cells, {mesh.n_points:,} points", flush=True)

    # Cell-centered coordinates
    coords = np.array(mesh.cell_centers().points, dtype=np.float32)

    # Cell normals
    mesh_n = mesh.compute_normals(
        point_normals=False, cell_normals=True, auto_orient_normals=True
    )
    normals = np.array(mesh_n.cell_data["Normals"], dtype=np.float32)

    # Pressure — try known field names in priority order
    pressure = None
    for name in ["CpMeanTrim", "Cp", "pMeanTrim", "p", "pressure"]:
        if name in mesh.cell_data:
            pressure = np.array(mesh.cell_data[name], dtype=np.float32)
            print(f"  Pressure: '{name}' range=[{pressure.min():.4f}, {pressure.max():.4f}]", flush=True)
            break
    if pressure is None:
        print(f"Run {run_id}: FAILED — no pressure field. Available: {list(mesh.cell_data.keys())}", flush=True)
        raise SystemExit(1)
    if pressure.ndim == 1:
        pressure = pressure[:, None]

    # Wall shear stress
    wall_shear = None
    for name in ["wallShearStressMeanTrim", "wallShearStress", "WallShearStress"]:
        if name in mesh.cell_data:
            wall_shear = np.array(mesh.cell_data[name], dtype=np.float32)
            print(f"  Wall shear: '{name}' shape={wall_shear.shape}", flush=True)
            break
    if wall_shear is None:
        print("  WARNING: No wall shear field, using zeros", flush=True)
        wall_shear = np.zeros((coords.shape[0], 3), dtype=np.float32)
    if wall_shear.ndim == 1:
        wall_shear = wall_shear[:, None]

    # Save NPZ — write to local temp file first, then copy to UC Volume.
    # Direct np.savez to FUSE-mounted UC Volumes can fail with IOError.
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.savez(
            tmp_path,
            params=params,
            surface_coords=coords,
            surface_normals=normals,
            surface_pressure=pressure,
            surface_wall_shear=wall_shear,
        )
        shutil.copy2(tmp_path, out_path)
    finally:
        os.unlink(tmp_path)
    size_mb = os.path.getsize(out_path) / 1024**2
    print(f"  Saved {out_path} ({size_mb:.1f} MB, {coords.shape[0]:,} cells)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Convert a single DrivAerML VTP to NPZ")
    parser.add_argument("--data_dir", required=True, help="Base directory with raw/ subfolder")
    parser.add_argument("--run", type=int, required=True, help="Run ID to convert")
    args = parser.parse_args()

    convert_one(args.run, args.data_dir)


if __name__ == "__main__":
    main()
