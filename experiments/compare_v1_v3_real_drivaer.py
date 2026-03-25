"""
Transolver v1 vs v3 comparison on REAL DrivAerML surface data.

Reads actual DrivAerML boundary VTP files from HuggingFace, extracts surface
pressure and wall shear stress, trains both models, and renders pressure
heatmaps on the real car geometry.

Usage:
  .venv/bin/python experiments/compare_v1_v3_real_drivaer.py \
      --data_dir data/drivaerml --epochs 60
"""

import sys
import os
import argparse
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compare_v1_v3_drivaer import TransolverV1
from transolver3.model import Transolver3
from transolver3.amortized_training import (
    AmortizedMeshSampler, relative_l2_loss,
    create_optimizer, create_scheduler,
)


# ============================================================================
# Real DrivAerML VTP loader
# ============================================================================

def load_drivaer_vtp(vtp_path, params_csv=None, subsample=None, seed=42):
    """Load a DrivAerML boundary VTP file and extract surface fields.

    Args:
        vtp_path: path to boundary_X.vtp
        params_csv: path to geo_parameters_X.csv (optional)
        subsample: if set, randomly subsample to this many points
        seed: random seed for subsampling

    Returns:
        dict with coords, normals, pressure, wall_shear, params
    """
    mesh = pv.read(vtp_path)
    print(f"  Loaded {vtp_path}: {mesh.n_points:,} points, {mesh.n_cells:,} cells")
    print(f"  Point data: {list(mesh.point_data.keys())}")
    print(f"  Cell data:  {list(mesh.cell_data.keys())}")

    # DrivAerML stores fields as cell data. Use cell centers as coordinates
    # and cell normals for the geometric features.
    has_cell_data = len(mesh.cell_data) > 0 and len(mesh.point_data) == 0

    if has_cell_data:
        print("  Using cell-centered data (DrivAerML default)")
        # Cell centers as coordinates
        coords = np.array(mesh.cell_centers().points, dtype=np.float32)  # (N_cells, 3)

        # Cell normals
        mesh_with_normals = mesh.compute_normals(point_normals=False, cell_normals=True,
                                                  auto_orient_normals=True)
        normals = np.array(mesh_with_normals.cell_data['Normals'], dtype=np.float32)

        data_dict = mesh.cell_data
    else:
        print("  Using point data")
        coords = np.array(mesh.points, dtype=np.float32)
        mesh_with_normals = mesh.compute_normals(point_normals=True, cell_normals=False,
                                                  auto_orient_normals=True)
        normals = np.array(mesh_with_normals.point_data['Normals'], dtype=np.float32)
        data_dict = mesh.point_data

    # Extract pressure — prefer Cp (pressure coefficient, normalized) over raw p
    pressure = None
    for name in ['CpMeanTrim', 'Cp', 'pMeanTrim', 'p', 'pressure']:
        if name in data_dict:
            pressure = np.array(data_dict[name], dtype=np.float32)
            print(f"  Using pressure field: '{name}' range=[{pressure.min():.2f}, {pressure.max():.2f}]")
            break
    if pressure is None:
        raise ValueError(f"No pressure field found. Available: {list(data_dict.keys())}")
    if pressure.ndim == 1:
        pressure = pressure[:, None]  # (N, 1)

    # Extract wall shear stress
    wall_shear = None
    for name in ['wallShearStressMeanTrim', 'wallShearStress', 'WallShearStress']:
        if name in data_dict:
            wall_shear = np.array(data_dict[name], dtype=np.float32)
            print(f"  Using wall shear field: '{name}' shape={wall_shear.shape}")
            break
    if wall_shear is None:
        print("  WARNING: No wall shear field found, using zeros")
        wall_shear = np.zeros((coords.shape[0], 3), dtype=np.float32)
    if wall_shear.ndim == 1:
        wall_shear = wall_shear[:, None]

    # Load geometry parameters
    params = np.zeros(16, dtype=np.float32)  # 16 parameters per DrivAerML spec
    if params_csv is not None and os.path.exists(params_csv):
        import csv
        with open(params_csv) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            # Skip 'Run' column
            param_vals = [float(v) for k, v in row.items() if k.strip().lower() != 'run']
            params = np.array(param_vals[:16], dtype=np.float32)
            print(f"  Loaded {len(param_vals)} geometry parameters")

    # Subsample if needed
    if subsample is not None and coords.shape[0] > subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(coords.shape[0], subsample, replace=False)
        coords = coords[idx]
        normals = normals[idx]
        pressure = pressure[idx]
        wall_shear = wall_shear[idx]
        print(f"  Subsampled to {subsample:,} points")

    return {
        'coords': coords,
        'normals': normals,
        'pressure': pressure,
        'wall_shear': wall_shear,
        'params': params,
    }


def compute_target_stats(samples):
    """Compute mean and std of targets across all samples for z-score normalization."""
    all_pressure = np.concatenate([s['pressure'] for s in samples], axis=0)
    all_shear = np.concatenate([s['wall_shear'] for s in samples], axis=0)
    all_targets = np.concatenate([all_pressure, all_shear], axis=1)

    mean = all_targets.mean(axis=0)
    std = all_targets.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # prevent division by zero

    # Also compute param stats for normalization
    all_params = np.stack([s['params'] for s in samples], axis=0)
    param_mean = all_params.mean(axis=0)
    param_std = all_params.std(axis=0)
    param_std = np.where(param_std < 1e-8, 1.0, param_std)

    stats = {
        'target_mean': mean, 'target_std': std,
        'param_mean': param_mean, 'param_std': param_std,
    }
    print(f"  Target stats: mean={mean}, std={std}")
    print(f"  Param stats:  mean_range=[{param_mean.min():.1f},{param_mean.max():.1f}], "
          f"std_range=[{param_std.min():.1f},{param_std.max():.1f}]")
    return stats


class RealDrivAerDataset(Dataset):
    """Dataset wrapping loaded DrivAerML samples with z-score normalization."""

    def __init__(self, samples, stats=None, subset_size=None, coord_scale=1.0):
        self.samples = samples
        self.subset_size = subset_size
        self.coord_scale = coord_scale
        self.stats = stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        coords = torch.from_numpy(s['coords'].copy())
        normals = torch.from_numpy(s['normals'].copy())
        params = torch.from_numpy(s['params'].copy())
        pressure = torch.from_numpy(s['pressure'].copy())
        wall_shear = torch.from_numpy(s['wall_shear'].copy())

        # Subsample for training
        N = coords.shape[0]
        if self.subset_size is not None and N > self.subset_size:
            idx_sub = torch.randperm(N)[:self.subset_size]
            coords = coords[idx_sub]
            normals = normals[idx_sub]
            pressure = pressure[idx_sub]
            wall_shear = wall_shear[idx_sub]
            N = self.subset_size

        # Normalize coords (min-max -> [0, coord_scale])
        cmin = coords.min(dim=0).values
        cmax = coords.max(dim=0).values
        coords_norm = (coords - cmin) / (cmax - cmin + 1e-8) * self.coord_scale

        # Normalize params (z-score)
        if self.stats is not None:
            params = (params - torch.from_numpy(self.stats['param_mean'])) / \
                     torch.from_numpy(self.stats['param_std'])

        # Input: [coords_norm(3), normals(3), params(16)] = 22
        p_broadcast = params.unsqueeze(0).expand(N, -1)
        x = torch.cat([coords_norm, normals, p_broadcast], dim=-1)

        # Target: [pressure(1), wall_shear(3)] = 4
        target = torch.cat([pressure, wall_shear], dim=-1)

        # Normalize target (z-score)
        if self.stats is not None:
            t_mean = torch.from_numpy(self.stats['target_mean'])
            t_std = torch.from_numpy(self.stats['target_std'])
            target = (target - t_mean) / t_std

        return {'x': x, 'target': target}


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, args, device, label):
    """Train a model and return history."""
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = create_scheduler(optimizer, total_steps)
    sampler = AmortizedMeshSampler(args.train_subset)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {label} | Parameters: {n_params:,}")
    print(f"{'='*60}")

    history = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        t0 = time.time()

        for batch in train_loader:
            x = batch['x'].to(device)
            target = batch['target'].to(device)
            optimizer.zero_grad()

            if args.train_subset > 0 and x.shape[1] > args.train_subset:
                N = x.shape[1]
                indices = sampler.sample(N).to(device)
                x_sub = x[:, indices]
                target_sub = target[:, indices]
            else:
                x_sub = x
                target_sub = target

            pred = model(x_sub, num_tiles=args.num_tiles)
            loss = relative_l2_loss(pred, target_sub)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            count += 1

        train_loss = epoch_loss / max(count, 1)
        epoch_time = time.time() - t0
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'time_s': epoch_time})

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"loss={train_loss:.6f} | time={epoch_time:.2f}s")

    return history


@torch.no_grad()
def predict_full(model, x, num_tiles=0):
    model.eval()
    return model(x.unsqueeze(0), num_tiles=num_tiles).squeeze(0).cpu().numpy()


# ============================================================================
# Visualization
# ============================================================================

def render_car_pressure(coords, pressure, title, ax, vmin, vmax, view='top',
                        cmap='RdBu_r', point_size=0.15):
    """Render pressure on real car geometry."""
    norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)

    if view == 'top':
        x_plot, y_plot = coords[:, 0], coords[:, 1]
        xlabel, ylabel = 'x (streamwise, m)', 'y (lateral, m)'
    elif view == 'side':
        x_plot, y_plot = coords[:, 0], coords[:, 2]
        xlabel, ylabel = 'x (streamwise, m)', 'z (height, m)'
    elif view == 'front':
        x_plot, y_plot = coords[:, 1], coords[:, 2]
        xlabel, ylabel = 'y (lateral, m)', 'z (height, m)'
    else:
        raise ValueError(f"Unknown view: {view}")

    sc = ax.scatter(x_plot, y_plot, c=pressure, cmap=cmap, norm=norm,
                    s=point_size, alpha=0.9, edgecolors='none', rasterized=True)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=8)
    return sc


def render_error(coords, error, title, ax, vmax, view='top', point_size=0.15):
    if view == 'top':
        x_plot, y_plot = coords[:, 0], coords[:, 1]
    elif view == 'side':
        x_plot, y_plot = coords[:, 0], coords[:, 2]
    else:
        x_plot, y_plot = coords[:, 1], coords[:, 2]

    sc = ax.scatter(x_plot, y_plot, c=error, cmap='hot_r', vmin=0, vmax=vmax,
                    s=point_size, alpha=0.9, edgecolors='none', rasterized=True)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=8)
    return sc


def create_visualizations(coords_3d, gt_pressure, v1_pressure, v3_pressure,
                          out_dir, n_render=50000):
    """Create all pressure heatmap PNGs."""
    os.makedirs(out_dir, exist_ok=True)

    # Subsample for rendering
    N = coords_3d.shape[0]
    if N > n_render:
        idx = np.random.RandomState(0).choice(N, n_render, replace=False)
    else:
        idx = np.arange(N)

    c = coords_3d[idx]
    gt = gt_pressure[idx]
    v1 = v1_pressure[idx]
    v3 = v3_pressure[idx]
    v1_err = np.abs(v1 - gt)
    v3_err = np.abs(v3 - gt)

    # Color range from ground truth
    vmin, vmax = np.percentile(gt, [2, 98])

    err_vmax = np.percentile(np.concatenate([v1_err, v3_err]), 98)

    # ---- Figure 1: Combined 3-view pressure comparison (3x3) ----
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('DrivAerML Surface Pressure: Ground Truth vs Transolver v1 vs v3',
                 fontsize=16, fontweight='bold', y=0.98)

    views = ['top', 'side', 'front']
    view_labels = ['Top View (x, y)', 'Side View (x, z)', 'Front View (y, z)']

    for row, (view, vlabel) in enumerate(zip(views, view_labels)):
        sc0 = render_car_pressure(c, gt, f'Ground Truth — {vlabel}',
                                   axes[row, 0], vmin, vmax, view=view)
        sc1 = render_car_pressure(c, v1, f'v1 Prediction — {vlabel}',
                                   axes[row, 1], vmin, vmax, view=view)
        sc2 = render_car_pressure(c, v3, f'v3 Prediction — {vlabel}',
                                   axes[row, 2], vmin, vmax, view=view)

    cbar = fig.colorbar(sc0, ax=axes, orientation='vertical', fraction=0.015, pad=0.03)
    cbar.set_label('Cp (pressure coefficient)', fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    path = os.path.join(out_dir, 'drivaer_pressure_3view.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 2: Side-view comparison (the money shot for a car) ----
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('DrivAerML — Side View Pressure Comparison',
                 fontsize=15, fontweight='bold', y=1.02)

    render_car_pressure(c, gt, 'Ground Truth', axes[0], vmin, vmax, view='side', point_size=0.3)
    render_car_pressure(c, v1, 'Transolver v1', axes[1], vmin, vmax, view='side', point_size=0.3)
    sc = render_car_pressure(c, v3, 'Transolver v3', axes[2], vmin, vmax, view='side', point_size=0.3)

    cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Cp (pressure coefficient)', fontsize=12)

    plt.tight_layout()
    path = os.path.join(out_dir, 'drivaer_pressure_sideview.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 3: Top-view comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle('DrivAerML — Top View Pressure Comparison',
                 fontsize=15, fontweight='bold', y=1.02)

    render_car_pressure(c, gt, 'Ground Truth', axes[0], vmin, vmax, view='top', point_size=0.3)
    render_car_pressure(c, v1, 'Transolver v1', axes[1], vmin, vmax, view='top', point_size=0.3)
    sc = render_car_pressure(c, v3, 'Transolver v3', axes[2], vmin, vmax, view='top', point_size=0.3)

    cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Cp (pressure coefficient)', fontsize=12)

    plt.tight_layout()
    path = os.path.join(out_dir, 'drivaer_pressure_topview.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 4: Error maps (side view) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('DrivAerML — Absolute Pressure Error (Side View)',
                 fontsize=15, fontweight='bold', y=1.02)

    sc1 = render_error(c, v1_err,
                        f'v1 Error (mean={np.abs(v1_pressure - gt_pressure).mean():.4f})',
                        axes[0], err_vmax, view='side', point_size=0.3)
    sc2 = render_error(c, v3_err,
                        f'v3 Error (mean={np.abs(v3_pressure - gt_pressure).mean():.4f})',
                        axes[1], err_vmax, view='side', point_size=0.3)

    cbar = fig.colorbar(sc2, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('|Cp Error|', fontsize=12)

    plt.tight_layout()
    path = os.path.join(out_dir, 'drivaer_pressure_errors.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 5: Profiles + error histogram ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DrivAerML — Pressure Profiles & Error Distribution',
                 fontsize=14, fontweight='bold', y=1.02)

    # Centerline slice (y near 0)
    y_mid = np.median(c[:, 1])
    y_tol = (c[:, 1].max() - c[:, 1].min()) * 0.03
    mask = np.abs(c[:, 1] - y_mid) < y_tol
    if mask.sum() > 50:
        x_sl = c[mask, 0]
        sort_i = np.argsort(x_sl)
        axes[0].plot(x_sl[sort_i], gt[mask][sort_i], 'k-', lw=1.5, label='Ground Truth', alpha=0.8)
        axes[0].plot(x_sl[sort_i], v1[mask][sort_i], 'b--', lw=1.0, label='v1', alpha=0.7)
        axes[0].plot(x_sl[sort_i], v3[mask][sort_i], 'r--', lw=1.0, label='v3', alpha=0.7)
        axes[0].set_xlabel('x (streamwise, m)')
        axes[0].set_ylabel('Cp')
        axes[0].set_title('Centerline Pressure Profile')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

    # Error histogram
    axes[1].hist(v1_err, bins=100, alpha=0.6, density=True,
                  label=f'v1 (mean={np.abs(v1_pressure-gt_pressure).mean():.2f})', color='blue')
    axes[1].hist(v3_err, bins=100, alpha=0.6, density=True,
                  label=f'v3 (mean={np.abs(v3_pressure-gt_pressure).mean():.2f})', color='red')
    axes[1].set_xlabel('|Cp Error|')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Error Distribution')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'drivaer_pressure_profiles.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='v1 vs v3 on real DrivAerML')
    p.add_argument('--data_dir', required=True, help='Path to drivaerml/ directory')
    p.add_argument('--train_runs', type=int, nargs='+', default=[1, 2],
                   help='Run indices for training')
    p.add_argument('--test_run', type=int, default=3,
                   help='Run index for testing/visualization')
    p.add_argument('--load_subsample', type=int, default=200000,
                   help='Subsample VTP to this many points on load')
    p.add_argument('--train_subset', type=int, default=20000,
                   help='Points per amortized training step')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--n_hidden', type=int, default=128)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--slice_num', type=int, default=32)
    p.add_argument('--num_tiles', type=int, default=4)
    p.add_argument('--eval_interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--out_dir', default='./experiments/results')
    p.add_argument('--render_points', type=int, default=80000,
                   help='Max points for rendering')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load real DrivAerML data ---
    print("\n--- Loading DrivAerML surface data ---")
    train_samples = []
    for run_id in args.train_runs:
        vtp = os.path.join(args.data_dir, f'run_{run_id}', f'boundary_{run_id}.vtp')
        csv = os.path.join(args.data_dir, f'run_{run_id}', f'geo_parameters_{run_id}.csv')
        if not os.path.exists(vtp):
            print(f"  SKIP run {run_id}: {vtp} not found")
            continue
        sample = load_drivaer_vtp(vtp, csv, subsample=args.load_subsample, seed=args.seed + run_id)
        train_samples.append(sample)

    test_vtp = os.path.join(args.data_dir, f'run_{args.test_run}', f'boundary_{args.test_run}.vtp')
    test_csv = os.path.join(args.data_dir, f'run_{args.test_run}', f'geo_parameters_{args.test_run}.csv')
    test_sample = load_drivaer_vtp(test_vtp, test_csv, subsample=args.load_subsample,
                                    seed=args.seed + args.test_run)

    if not train_samples:
        print("ERROR: No training data found. Download VTP files first.")
        return

    # Determine input dim from actual data
    d_params = train_samples[0]['params'].shape[0]
    space_dim = 3 + 3 + d_params  # coords + normals + params
    out_dim = 1 + train_samples[0]['wall_shear'].shape[1]  # pressure + wall_shear
    print(f"\nInput dim: {space_dim} (3 coords + 3 normals + {d_params} params)")
    print(f"Output dim: {out_dim} (1 pressure + {out_dim-1} wall_shear)")

    # Compute normalization stats from training data
    print("\n--- Computing normalization stats ---")
    stats = compute_target_stats(train_samples)

    # Create datasets with normalization
    train_ds = RealDrivAerDataset(train_samples, stats=stats, subset_size=args.load_subsample)
    test_ds = RealDrivAerDataset([test_sample], stats=stats, subset_size=None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # --- Build models ---
    torch.manual_seed(args.seed)
    model_v1 = TransolverV1(
        space_dim=space_dim, n_layers=args.n_layers, n_hidden=args.n_hidden,
        n_head=args.n_head, fun_dim=0, out_dim=out_dim,
        slice_num=args.slice_num, mlp_ratio=1, dropout=0.0,
    ).to(device)

    torch.manual_seed(args.seed)
    model_v3 = Transolver3(
        space_dim=space_dim, n_layers=args.n_layers, n_hidden=args.n_hidden,
        n_head=args.n_head, fun_dim=0, out_dim=out_dim,
        slice_num=args.slice_num, mlp_ratio=1, dropout=0.0,
        num_tiles=args.num_tiles,
    ).to(device)

    # --- Train ---
    torch.manual_seed(args.seed)
    history_v1 = train_model(model_v1, train_loader, args, device, "Transolver v1")

    torch.manual_seed(args.seed)
    history_v3 = train_model(model_v3, train_loader, args, device, "Transolver v3")

    # --- Evaluate on test sample ---
    print("\n--- Evaluating on test sample ---")
    test_batch = test_ds[0]
    x_test = test_batch['x'].to(device)
    target_norm = test_batch['target'].numpy()  # normalized targets

    pred_v1_norm = predict_full(model_v1, x_test, num_tiles=0)
    pred_v3_norm = predict_full(model_v3, x_test, num_tiles=args.num_tiles)

    # Relative L2 in normalized space
    def rel_l2(pred, gt):
        return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-8))

    v1_l2 = rel_l2(pred_v1_norm, target_norm)
    v3_l2 = rel_l2(pred_v3_norm, target_norm)
    print(f"  v1 test relative L2: {v1_l2:.4f} ({v1_l2*100:.2f}%)")
    print(f"  v3 test relative L2: {v3_l2:.4f} ({v3_l2*100:.2f}%)")

    # Denormalize back to physical units for visualization
    t_mean = stats['target_mean']
    t_std = stats['target_std']
    target_phys = target_norm * t_std + t_mean
    pred_v1_phys = pred_v1_norm * t_std + t_mean
    pred_v3_phys = pred_v3_norm * t_std + t_mean

    # --- Render pressure heatmaps on REAL car geometry ---
    print("\n--- Rendering pressure heatmaps ---")
    coords_3d = test_sample['coords']  # raw 3D coords (meters)

    gt_pressure = target_phys[:, 0]
    v1_pressure = pred_v1_phys[:, 0]
    v3_pressure = pred_v3_phys[:, 0]

    create_visualizations(
        coords_3d, gt_pressure, v1_pressure, v3_pressure,
        args.out_dir, n_render=args.render_points,
    )

    # --- Summary ---
    v1_params = sum(p.numel() for p in model_v1.parameters())
    v3_params = sum(p.numel() for p in model_v3.parameters())
    print(f"\n{'='*60}")
    print(f"  SUMMARY (Real DrivAerML)")
    print(f"{'='*60}")
    print(f"  {'':.<25} {'v1':>12} {'v3':>12}")
    print(f"  {'Parameters':.<25} {v1_params:>12,} {v3_params:>12,}")
    print(f"  {'Test L2 (%)':.<25} {v1_l2*100:>11.2f}% {v3_l2*100:>11.2f}%")
    print(f"  {'Final train loss':.<25} {history_v1[-1]['train_loss']:>12.6f} {history_v3[-1]['train_loss']:>12.6f}")

    # Save results (convert numpy types for JSON)
    def jsonify(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        'args': {k: jsonify(v) for k, v in vars(args).items()},
        'v1_test_l2': float(v1_l2),
        'v3_test_l2': float(v3_l2),
        'v1_params': v1_params,
        'v3_params': v3_params,
        'v1_history': history_v1,
        'v3_history': history_v3,
    }
    path = os.path.join(args.out_dir, 'v1_v3_real_drivaer.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=jsonify)
    print(f"\nResults saved to {path}")
    print(f"Figures saved to {args.out_dir}/drivaer_*.png")


if __name__ == '__main__':
    main()
