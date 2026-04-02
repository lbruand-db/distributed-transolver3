"""
Convert DrivAerML VTP boundary files to .npz format for Databricks training.

Reads boundary_i.vtp + geo_parameters_i.csv from the raw HuggingFace download
and produces one .npz per run with the keys expected by DrivAerMLDataset:

    params              (d_params,)   float32  — geometry deformation parameters
    surface_coords      (N_s, 3)      float32  — cell-center coordinates
    surface_normals     (N_s, 3)      float32  — cell normals
    surface_pressure    (N_s, 1)      float32  — pressure coefficient (CpMeanTrim)
    surface_wall_shear  (N_s, 3)      float32  — wall shear stress vector

Usage:
    uv run python scripts/convert_vtp_to_npz.py \
        --input_dir data/drivaerml \
        --output_dir data/drivaerml_npz \
        --runs 1 2 3

    # Then create train/test splits:
    #   data/drivaerml_npz/train.txt  (one filename per line)
    #   data/drivaerml_npz/test.txt
"""

import argparse
import csv
import os
import sys

import numpy as np


def parse_geo_parameters(csv_path):
    """Read geometry parameters from DrivAerML CSV."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
        vals = [float(v) for k, v in row.items() if k.strip().lower() != "run"]
    return np.array(vals, dtype=np.float32)


def convert_one_run(input_dir, output_dir, run_id):
    """Convert a single DrivAerML run from VTP to NPZ."""
    import pyvista as pv

    vtp_path = os.path.join(input_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
    csv_path = os.path.join(input_dir, f"run_{run_id}", f"geo_parameters_{run_id}.csv")

    if not os.path.exists(vtp_path):
        print(f"  SKIP run {run_id}: {vtp_path} not found")
        return None

    print(f"  Loading {vtp_path} ...")
    mesh = pv.read(vtp_path)
    print(f"    {mesh.n_cells:,} cells, {mesh.n_points:,} points")

    # --- Cell-centered coordinates ---
    coords = np.array(mesh.cell_centers().points, dtype=np.float32)

    # --- Cell normals ---
    mesh_n = mesh.compute_normals(
        point_normals=False, cell_normals=True, auto_orient_normals=True
    )
    normals = np.array(mesh_n.cell_data["Normals"], dtype=np.float32)

    # --- Pressure (prefer CpMeanTrim) ---
    pressure = None
    for name in ["CpMeanTrim", "Cp", "pMeanTrim", "p", "pressure"]:
        if name in mesh.cell_data:
            pressure = np.array(mesh.cell_data[name], dtype=np.float32)
            print(f"    Pressure field: '{name}' range=[{pressure.min():.4f}, {pressure.max():.4f}]")
            break
    if pressure is None:
        raise ValueError(f"No pressure field found. Available: {list(mesh.cell_data.keys())}")
    if pressure.ndim == 1:
        pressure = pressure[:, None]

    # --- Wall shear stress ---
    wall_shear = None
    for name in ["wallShearStressMeanTrim", "wallShearStress", "WallShearStress"]:
        if name in mesh.cell_data:
            wall_shear = np.array(mesh.cell_data[name], dtype=np.float32)
            print(f"    Wall shear field: '{name}' shape={wall_shear.shape}")
            break
    if wall_shear is None:
        print("    WARNING: No wall shear field, using zeros")
        wall_shear = np.zeros((coords.shape[0], 3), dtype=np.float32)
    if wall_shear.ndim == 1:
        wall_shear = wall_shear[:, None]

    # --- Geometry parameters ---
    if os.path.exists(csv_path):
        params = parse_geo_parameters(csv_path)
        print(f"    Params: {len(params)} values from {csv_path}")
    else:
        params = np.zeros(16, dtype=np.float32)
        print(f"    WARNING: {csv_path} not found, using zero params")

    # --- Save NPZ ---
    out_name = f"drivaer_{run_id:03d}.npz"
    out_path = os.path.join(output_dir, out_name)
    np.savez(
        out_path,
        params=params,
        surface_coords=coords,
        surface_normals=normals,
        surface_pressure=pressure,
        surface_wall_shear=wall_shear,
    )
    size_mb = os.path.getsize(out_path) / 1024**2
    print(f"    Saved {out_path} ({size_mb:.1f} MB, {coords.shape[0]:,} cells)")
    return out_name


def write_split_files(output_dir, all_names, test_fraction=0.2, seed=42):
    """Write train.txt and test.txt split files."""
    rng = np.random.default_rng(seed)
    names = list(all_names)
    rng.shuffle(names)

    n_test = max(1, int(len(names) * test_fraction))
    test_names = names[:n_test]
    train_names = names[n_test:]

    # If only 1-2 samples, use all for both train and test
    if len(names) <= 2:
        train_names = names
        test_names = names[-1:]

    for split, split_names in [("train", train_names), ("test", test_names)]:
        path = os.path.join(output_dir, f"{split}.txt")
        with open(path, "w") as f:
            for name in sorted(split_names):
                f.write(name + "\n")
        print(f"  {split}.txt: {len(split_names)} samples")


def main():
    parser = argparse.ArgumentParser(description="Convert DrivAerML VTP to NPZ")
    parser.add_argument("--input_dir", required=True, help="Directory with run_*/boundary_*.vtp")
    parser.add_argument("--output_dir", required=True, help="Output directory for .npz files")
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3], help="Run IDs to convert")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Converting {len(args.runs)} runs: {args.runs}")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}\n")

    converted = []
    for run_id in args.runs:
        name = convert_one_run(args.input_dir, args.output_dir, run_id)
        if name:
            converted.append(name)
        print()

    if not converted:
        print("ERROR: No runs converted.")
        sys.exit(1)

    print(f"Converted {len(converted)} runs. Writing split files...")
    write_split_files(args.output_dir, converted)
    print("\nDone. Upload to UC Volumes with:")
    print(f"  databricks fs cp -r {args.output_dir} /Volumes/ml/transolver3/data/")


if __name__ == "__main__":
    main()
