"""
Transolver v1 vs v3 comparison on DrivAerML-like synthetic data.

Generates synthetic data mimicking the DrivAerML surface benchmark:
  - Coords (N, 3) + normals (N, 3) + params (N, d_params) -> input (N, 3+3+6=12)
  - Target: pressure (N, 1) + wall shear (N, 3) -> output (N, 4)

Both models are trained with identical config, data, and loss. We compare:
  - Convergence (relative L2 error over epochs)
  - Wall-clock time per epoch
  - Peak GPU memory (if CUDA available)

Usage:
  python experiments/compare_v1_v3_drivaer.py [--n_points 50000] [--epochs 50]
"""

import sys
import os
import argparse
import time
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from timm.layers import trunc_normal_

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transolver3.common import MLP, ACTIVATION
from transolver3.amortized_training import relative_l2_loss, AmortizedMeshSampler


# ============================================================================
# Transolver v1 (recovered from git history, adapted for DrivAerML interface)
# ============================================================================

class Physics_Attention_V1(nn.Module):
    """Original Physics-Attention from Transolver v1 (irregular mesh variant)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # V1: two N-domain linear projections (O(N * inner_dim))
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        # V1: N-domain output projection (O(N * inner_dim))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape

        # (1) Slice — projects on N domain
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        # (2) Attention among slice tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v)  # B H G D

        # (3) Deslice — einsum on N domain
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class TransolverV1Block(nn.Module):
    """Transolver v1 encoder block."""

    def __init__(self, num_heads, hidden_dim, dropout, act='gelu',
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_V1(
            hidden_dim, heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class TransolverV1(nn.Module):
    """Transolver v1 model adapted for DrivAerML-style input/output."""

    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act='gelu', mlp_ratio=1, fun_dim=0, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False):
        super().__init__()
        self.__name__ = 'TransolverV1'
        self.ref = ref
        self.unified_pos = unified_pos
        self.n_hidden = n_hidden
        self.space_dim = space_dim

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref * ref, n_hidden * 2,
                                  n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2,
                                  n_hidden, n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList([
            TransolverV1Block(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx=None, T=None, **kwargs):
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx


# ============================================================================
# Synthetic DrivAerML-like dataset
# ============================================================================

class SyntheticDrivAerML(Dataset):
    """Synthetic dataset mimicking DrivAerML surface fields.

    Generates a simple but non-trivial target: a function of coordinates
    that both models should learn to approximate. The target function
    combines spatial variation (pressure ~ f(x,y,z)) with parametric
    dependence, mimicking real CFD physics.
    """

    def __init__(self, n_samples=20, n_points=50000, d_params=6, seed=42):
        super().__init__()
        self.n_samples = n_samples
        self.n_points = n_points
        self.d_params = d_params
        rng = np.random.RandomState(seed)

        self.data = []
        for i in range(n_samples):
            # Surface-like coords on a car-shaped geometry
            coords = rng.randn(n_points, 3).astype(np.float32)
            coords[:, 0] *= 2.0  # elongate in x (streamwise)
            coords[:, 1] = np.abs(coords[:, 1]) * 0.5  # half-body (y>0)
            coords[:, 2] *= 0.8  # lateral

            # Normals (roughly unit)
            normals = rng.randn(n_points, 3).astype(np.float32)
            normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

            # Parametric deformation (6 geometric params)
            params = rng.randn(d_params).astype(np.float32) * 0.1

            # --- Generate physics-inspired target ---
            # Pressure: depends on streamwise position + param[0]*curvature
            pressure = (
                -0.5 * np.exp(-coords[:, 0]**2)  # stagnation
                + 0.3 * coords[:, 0]  # recovery
                + params[0] * np.sin(coords[:, 1] * np.pi)
                + 0.05 * rng.randn(n_points)  # noise
            ).astype(np.float32).reshape(-1, 1)

            # Wall shear: tangential, depends on normals + coords
            shear = (
                0.1 * normals * np.abs(coords[:, 0:1])
                + params[1] * coords * 0.02
                + 0.01 * rng.randn(n_points, 3)
            ).astype(np.float32)

            self.data.append({
                'coords': torch.from_numpy(coords),
                'normals': torch.from_numpy(normals),
                'params': torch.from_numpy(params),
                'pressure': torch.from_numpy(pressure),
                'shear': torch.from_numpy(shear),
            })

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        d = self.data[idx]
        coords = d['coords']
        normals = d['normals']
        params = d['params']

        # Normalize coords (min-max, scale=1000 like DrivAerML loader)
        cmin = coords.min(dim=0).values
        cmax = coords.max(dim=0).values
        coords_norm = (coords - cmin) / (cmax - cmin + 1e-8) * 1000.0

        # Input: [coords_norm(3), normals(3), params_broadcast(6)] = 12
        N = coords_norm.shape[0]
        p_broadcast = params.unsqueeze(0).expand(N, -1)
        x = torch.cat([coords_norm, normals, p_broadcast], dim=-1)

        # Target: [pressure(1), shear(3)] = 4
        target = torch.cat([d['pressure'], d['shear']], dim=-1)

        return {'x': x, 'target': target}


# ============================================================================
# Training loop
# ============================================================================

def train_and_evaluate(model, train_loader, test_loader, args, device, label):
    """Train a model and return per-epoch metrics."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    sampler = AmortizedMeshSampler(args.subset_size)

    history = []
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()

        for batch in train_loader:
            x = batch['x'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            N = x.shape[1]
            indices = sampler.sample(N).to(device)
            x_sub = x[:, indices]
            target_sub = target[:, indices]

            # V3 uses num_tiles kwarg; V1 ignores it via **kwargs
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

        # Peak memory
        peak_mem_mb = 0
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

        # Evaluate every eval_interval epochs or at the end
        test_error = None
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            test_error = evaluate(model, test_loader, args, device)

        record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_l2': test_error,
            'time_s': epoch_time,
            'peak_mem_mb': peak_mem_mb,
        }
        history.append(record)

        if test_error is not None:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train_loss={train_loss:.6f} | "
                  f"test_L2={test_error:.4f} ({test_error*100:.2f}%) | "
                  f"time={epoch_time:.2f}s | "
                  f"mem={peak_mem_mb:.0f}MB")
        elif (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train_loss={train_loss:.6f} | "
                  f"time={epoch_time:.2f}s")

    return history


@torch.no_grad()
def evaluate(model, test_loader, args, device):
    model.eval()
    all_errors = []
    for batch in test_loader:
        x = batch['x'].to(device)
        target = batch['target'].to(device)
        # No subsampling at eval — use full points
        pred = model(x, num_tiles=args.num_tiles)
        diff_norm = torch.norm(pred - target, p=2, dim=(1, 2))
        target_norm = torch.norm(target, p=2, dim=(1, 2))
        error = diff_norm / (target_norm + 1e-8)
        all_errors.append(error.cpu())
    return torch.cat(all_errors).mean().item()


# ============================================================================
# Main comparison
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Transolver v1 vs v3 comparison')
    p.add_argument('--n_train', type=int, default=20, help='Training samples')
    p.add_argument('--n_test', type=int, default=5, help='Test samples')
    p.add_argument('--n_points', type=int, default=50000,
                   help='Points per sample (DrivAerML surface has ~8.8M)')
    p.add_argument('--subset_size', type=int, default=10000,
                   help='Amortized training subset size')
    p.add_argument('--epochs', type=int, default=50, help='Training epochs')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--n_layers', type=int, default=8,
                   help='Number of layers (paper uses 24; 8 for quick test)')
    p.add_argument('--n_hidden', type=int, default=128,
                   help='Hidden dim (paper uses 256; 128 for quick test)')
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--slice_num', type=int, default=32)
    p.add_argument('--num_tiles', type=int, default=4,
                   help='Tiles for v3 (v1 ignores this)')
    p.add_argument('--eval_interval', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--save_dir', default='./experiments/results')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # DrivAerML surface: input_dim=12 (coords3 + normals3 + params6), output_dim=4
    space_dim = 12  # 3 + 3 + 6
    out_dim = 4     # pressure(1) + wall_shear(3)

    print(f"\nSynthetic DrivAerML-like dataset:")
    print(f"  {args.n_train} train + {args.n_test} test samples")
    print(f"  {args.n_points:,} points/sample, subset={args.subset_size:,}")
    print(f"  Input dim: {space_dim}, Output dim: {out_dim}")
    print(f"  Model config: {args.n_layers} layers, C={args.n_hidden}, "
          f"H={args.n_head}, M={args.slice_num}")

    # Create datasets
    train_ds = SyntheticDrivAerML(
        n_samples=args.n_train, n_points=args.n_points, seed=args.seed
    )
    test_ds = SyntheticDrivAerML(
        n_samples=args.n_test, n_points=args.n_points, seed=args.seed + 1000
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # --- Transolver v1 ---
    torch.manual_seed(args.seed)
    model_v1 = TransolverV1(
        space_dim=space_dim,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=args.slice_num,
        mlp_ratio=1,
        dropout=0.0,
    ).to(device)

    # --- Transolver v3 ---
    from transolver3.model import Transolver3

    torch.manual_seed(args.seed)
    model_v3 = Transolver3(
        space_dim=space_dim,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=args.slice_num,
        mlp_ratio=1,
        dropout=0.0,
        num_tiles=args.num_tiles,
    ).to(device)

    # Train both
    torch.manual_seed(args.seed)
    history_v1 = train_and_evaluate(
        model_v1, train_loader, test_loader, args, device, "Transolver v1"
    )

    torch.manual_seed(args.seed)
    history_v3 = train_and_evaluate(
        model_v3, train_loader, test_loader, args, device, "Transolver v3"
    )

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")

    # Get final test errors
    v1_final = [h for h in history_v1 if h['test_l2'] is not None][-1]
    v3_final = [h for h in history_v3 if h['test_l2'] is not None][-1]

    v1_params = sum(p.numel() for p in model_v1.parameters())
    v3_params = sum(p.numel() for p in model_v3.parameters())

    v1_avg_time = np.mean([h['time_s'] for h in history_v1])
    v3_avg_time = np.mean([h['time_s'] for h in history_v3])

    v1_peak_mem = max(h['peak_mem_mb'] for h in history_v1)
    v3_peak_mem = max(h['peak_mem_mb'] for h in history_v3)

    print(f"\n{'Metric':<25} {'Transolver v1':>15} {'Transolver v3':>15} {'Ratio v3/v1':>12}")
    print(f"{'-'*67}")
    print(f"{'Parameters':.<25} {v1_params:>15,} {v3_params:>15,} {v3_params/v1_params:>11.2f}x")
    print(f"{'Final test L2 (%)':.<25} {v1_final['test_l2']*100:>14.2f}% {v3_final['test_l2']*100:>14.2f}% {v3_final['test_l2']/v1_final['test_l2']:>11.2f}x")
    print(f"{'Avg epoch time (s)':.<25} {v1_avg_time:>15.2f} {v3_avg_time:>15.2f} {v3_avg_time/v1_avg_time:>11.2f}x")
    if v1_peak_mem > 0:
        print(f"{'Peak GPU mem (MB)':.<25} {v1_peak_mem:>15.0f} {v3_peak_mem:>15.0f} {v3_peak_mem/v1_peak_mem:>11.2f}x")

    # Convergence comparison
    print(f"\nConvergence (test L2 %):")
    v1_evals = [(h['epoch'], h['test_l2']) for h in history_v1 if h['test_l2'] is not None]
    v3_evals = [(h['epoch'], h['test_l2']) for h in history_v3 if h['test_l2'] is not None]
    print(f"  {'Epoch':>6}  {'v1 L2 %':>10}  {'v3 L2 %':>10}")
    for (e1, l1), (e3, l3) in zip(v1_evals, v3_evals):
        print(f"  {e1:>6}  {l1*100:>9.2f}%  {l3*100:>9.2f}%")

    # Save results
    results = {
        'args': vars(args),
        'v1_params': v1_params,
        'v3_params': v3_params,
        'v1_history': history_v1,
        'v3_history': history_v3,
        'summary': {
            'v1_final_l2': v1_final['test_l2'],
            'v3_final_l2': v3_final['test_l2'],
            'v1_avg_epoch_s': v1_avg_time,
            'v3_avg_epoch_s': v3_avg_time,
            'v1_peak_mem_mb': v1_peak_mem,
            'v3_peak_mem_mb': v3_peak_mem,
        }
    }
    result_path = os.path.join(args.save_dir, 'v1_v3_comparison.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':
    main()
