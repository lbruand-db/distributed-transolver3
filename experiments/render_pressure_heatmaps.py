"""
Render pressure heatmaps comparing ground truth, Transolver v1, and v3 predictions.

Produces PNG images showing pressure fields on the synthetic DrivAerML car surface:
  - Ground truth pressure
  - v1 predicted pressure
  - v3 predicted pressure
  - Absolute error maps for both

Usage:
  .venv/bin/python experiments/render_pressure_heatmaps.py [--epochs 80]
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Reuse models and dataset from comparison script
from compare_v1_v3_drivaer import (
    TransolverV1, SyntheticDrivAerML, train_and_evaluate
)
from transolver3.model import Transolver3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_points', type=int, default=20000)
    p.add_argument('--subset_size', type=int, default=10000)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--n_hidden', type=int, default=128)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--slice_num', type=int, default=32)
    p.add_argument('--num_tiles', type=int, default=2)
    p.add_argument('--n_train', type=int, default=15)
    p.add_argument('--n_test', type=int, default=5)
    p.add_argument('--eval_interval', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--out_dir', default='./experiments/results')
    return p.parse_args()


@torch.no_grad()
def predict_full(model, x, num_tiles=0):
    """Run full inference on a single sample."""
    model.eval()
    return model(x.unsqueeze(0), num_tiles=num_tiles).squeeze(0).cpu().numpy()


def render_surface_pressure(coords_3d, pressure, title, ax, vmin, vmax, cmap='RdBu_r'):
    """Render a pressure scalar field on 2D projection of a 3D car surface.

    Uses the (x, z) plane (top-down view) which shows the car's streamwise
    profile and lateral extent.
    """
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    sc = ax.scatter(
        coords_3d[:, 0], coords_3d[:, 2],
        c=pressure, cmap=cmap, norm=norm,
        s=0.3, alpha=0.8, edgecolors='none', rasterized=True,
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x (streamwise)')
    ax.set_ylabel('z (lateral)')
    ax.set_aspect('equal')
    return sc


def render_error_map(coords_3d, error, title, ax, vmax):
    """Render absolute error as a heatmap."""
    sc = ax.scatter(
        coords_3d[:, 0], coords_3d[:, 2],
        c=error, cmap='hot_r', vmin=0, vmax=vmax,
        s=0.3, alpha=0.8, edgecolors='none', rasterized=True,
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x (streamwise)')
    ax.set_ylabel('z (lateral)')
    ax.set_aspect('equal')
    return sc


def render_side_view(coords_3d, pressure, title, ax, vmin, vmax, cmap='RdBu_r'):
    """Side view (x, y) showing height profile of the car."""
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    sc = ax.scatter(
        coords_3d[:, 0], coords_3d[:, 1],
        c=pressure, cmap=cmap, norm=norm,
        s=0.3, alpha=0.8, edgecolors='none', rasterized=True,
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x (streamwise)')
    ax.set_ylabel('y (height)')
    ax.set_aspect('equal')
    return sc


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    space_dim = 12
    out_dim = 4

    # Create datasets
    train_ds = SyntheticDrivAerML(n_samples=args.n_train, n_points=args.n_points, seed=args.seed)
    test_ds = SyntheticDrivAerML(n_samples=args.n_test, n_points=args.n_points, seed=args.seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Train v1
    print("Training Transolver v1...")
    torch.manual_seed(args.seed)
    model_v1 = TransolverV1(
        space_dim=space_dim, n_layers=args.n_layers, n_hidden=args.n_hidden,
        n_head=args.n_head, fun_dim=0, out_dim=out_dim,
        slice_num=args.slice_num, mlp_ratio=1, dropout=0.0,
    ).to(device)
    torch.manual_seed(args.seed)
    train_and_evaluate(model_v1, train_loader, test_loader, args, device, "Transolver v1")

    # Train v3
    print("\nTraining Transolver v3...")
    torch.manual_seed(args.seed)
    model_v3 = Transolver3(
        space_dim=space_dim, n_layers=args.n_layers, n_hidden=args.n_hidden,
        n_head=args.n_head, fun_dim=0, out_dim=out_dim,
        slice_num=args.slice_num, mlp_ratio=1, dropout=0.0,
        num_tiles=args.num_tiles,
    ).to(device)
    torch.manual_seed(args.seed)
    train_and_evaluate(model_v3, train_loader, test_loader, args, device, "Transolver v3")

    # --- Render on a test sample ---
    sample = test_ds[0]
    x = sample['x'].to(device)
    target = sample['target'].numpy()

    # Raw 3D coords (first 3 dims of input, before normalization — reconstruct)
    # The input x has normalized coords; use the raw coords from the dataset
    raw_data = test_ds.data[0]
    coords_3d = raw_data['coords'].numpy()

    # Predictions
    pred_v1 = predict_full(model_v1, x, num_tiles=0)
    pred_v3 = predict_full(model_v3, x, num_tiles=args.num_tiles)

    # Extract pressure (channel 0) and shear magnitude
    gt_pressure = target[:, 0]
    v1_pressure = pred_v1[:, 0]
    v3_pressure = pred_v3[:, 0]

    v1_error = np.abs(v1_pressure - gt_pressure)
    v3_error = np.abs(v3_pressure - gt_pressure)

    # Global pressure range for consistent colormaps
    p_all = np.concatenate([gt_pressure, v1_pressure, v3_pressure])
    vmin, vmax = np.percentile(p_all, [2, 98])
    # Ensure symmetric around zero
    vlim = max(abs(vmin), abs(vmax))
    vmin, vmax = -vlim, vlim

    err_vmax = np.percentile(np.concatenate([v1_error, v3_error]), 98)

    # Subsample for faster rendering if too many points
    n_render = min(args.n_points, 20000)
    if args.n_points > n_render:
        idx = np.random.RandomState(0).choice(args.n_points, n_render, replace=False)
    else:
        idx = np.arange(args.n_points)

    c3d = coords_3d[idx]
    gt_p = gt_pressure[idx]
    v1_p = v1_pressure[idx]
    v3_p = v3_pressure[idx]
    v1_e = v1_error[idx]
    v3_e = v3_error[idx]

    # =========================================================================
    # Figure 1: Top-down pressure comparison (3 panels)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Surface Pressure — Top-Down View (x, z)', fontsize=14, fontweight='bold', y=1.02)

    sc0 = render_surface_pressure(c3d, gt_p, 'Ground Truth', axes[0], vmin, vmax)
    render_surface_pressure(c3d, v1_p, 'Transolver v1', axes[1], vmin, vmax)
    render_surface_pressure(c3d, v3_p, 'Transolver v3', axes[2], vmin, vmax)

    cbar = fig.colorbar(sc0, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Pressure', fontsize=11)

    plt.tight_layout()
    path1 = os.path.join(args.out_dir, 'pressure_topdown.png')
    fig.savefig(path1, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path1}")

    # =========================================================================
    # Figure 2: Side-view pressure comparison (3 panels)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Surface Pressure — Side View (x, y)', fontsize=14, fontweight='bold', y=1.02)

    render_side_view(c3d, gt_p, 'Ground Truth', axes[0], vmin, vmax)
    render_side_view(c3d, v1_p, 'Transolver v1', axes[1], vmin, vmax)
    sc2 = render_side_view(c3d, v3_p, 'Transolver v3', axes[2], vmin, vmax)

    cbar = fig.colorbar(sc0, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Pressure', fontsize=11)

    plt.tight_layout()
    path2 = os.path.join(args.out_dir, 'pressure_sideview.png')
    fig.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path2}")

    # =========================================================================
    # Figure 3: Error maps (v1 vs v3)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Absolute Pressure Error — Top-Down View', fontsize=14, fontweight='bold', y=1.02)

    render_error_map(c3d, v1_e, f'v1 Error (mean={v1_error.mean():.4f})', axes[0], err_vmax)
    sc2 = render_error_map(c3d, v3_e, f'v3 Error (mean={v3_error.mean():.4f})', axes[1], err_vmax)

    cbar = fig.colorbar(sc2, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('|Error|', fontsize=11)

    plt.tight_layout()
    path3 = os.path.join(args.out_dir, 'pressure_error_maps.png')
    fig.savefig(path3, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path3}")

    # =========================================================================
    # Figure 4: Combined 2x3 panel (GT / v1 / v3, top-down + side)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Pressure Heatmap Comparison: Ground Truth vs v1 vs v3',
                 fontsize=15, fontweight='bold', y=1.01)

    # Top row: top-down
    render_surface_pressure(c3d, gt_p, 'Ground Truth (top-down)', axes[0, 0], vmin, vmax)
    render_surface_pressure(c3d, v1_p, 'v1 Prediction (top-down)', axes[0, 1], vmin, vmax)
    sc = render_surface_pressure(c3d, v3_p, 'v3 Prediction (top-down)', axes[0, 2], vmin, vmax)

    # Bottom row: side view
    render_side_view(c3d, gt_p, 'Ground Truth (side)', axes[1, 0], vmin, vmax)
    render_side_view(c3d, v1_p, 'v1 Prediction (side)', axes[1, 1], vmin, vmax)
    sc2 = render_side_view(c3d, v3_p, 'v3 Prediction (side)', axes[1, 2], vmin, vmax)

    cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.015, pad=0.03)
    cbar.set_label('Pressure', fontsize=12)

    plt.tight_layout()
    path4 = os.path.join(args.out_dir, 'pressure_combined.png')
    fig.savefig(path4, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path4}")

    # =========================================================================
    # Figure 5: Convergence plot
    # =========================================================================
    # Re-extract from the train logs (we don't have the history objects here,
    # so recompute quick eval at saved checkpoints — or just plot from the
    # training printout). Instead, let's do a 1D pressure profile comparison.

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Pressure Profile Slices', fontsize=14, fontweight='bold', y=1.02)

    # Slice along streamwise axis (z near 0)
    z_mask = np.abs(c3d[:, 2]) < 0.3
    x_slice = c3d[z_mask, 0]
    sort_idx = np.argsort(x_slice)
    x_sorted = x_slice[sort_idx]

    axes[0].plot(x_sorted, gt_p[z_mask][sort_idx], 'k-', lw=1.5, label='Ground Truth', alpha=0.8)
    axes[0].plot(x_sorted, v1_p[z_mask][sort_idx], 'b--', lw=1.0, label='v1', alpha=0.7)
    axes[0].plot(x_sorted, v3_p[z_mask][sort_idx], 'r--', lw=1.0, label='v3', alpha=0.7)
    axes[0].set_xlabel('x (streamwise)')
    axes[0].set_ylabel('Pressure')
    axes[0].set_title('Centerline Slice (|z| < 0.3)')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Histogram of errors
    axes[1].hist(v1_e, bins=80, alpha=0.6, label=f'v1 (mean={v1_error.mean():.4f})', color='blue', density=True)
    axes[1].hist(v3_e, bins=80, alpha=0.6, label=f'v3 (mean={v3_error.mean():.4f})', color='red', density=True)
    axes[1].set_xlabel('|Pressure Error|')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Error Distribution')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path5 = os.path.join(args.out_dir, 'pressure_profiles.png')
    fig.savefig(path5, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path5}")

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
