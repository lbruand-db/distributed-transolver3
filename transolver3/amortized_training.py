"""
Geometry Amortized Training for Transolver-3.

During training on industrial-scale meshes, the full mesh D_N is too large
to fit in GPU memory. Instead, each iteration trains on a random subset
D_n (n ~ 10^5 - 10^6) of the full mesh. This exposes the model to varying
geometry subsets, learning the continuous physics operator without relying
on a fixed discretization.

Reference: Transolver-3 paper, Section 3.2.
"""

import torch
import torch.nn as nn


class AmortizedMeshSampler:
    """Generates random subset indices for geometry amortized training.

    Each call to sample() returns a different random subset of mesh indices,
    enabling the model to see different parts of the geometry each iteration.

    Args:
        subset_size: number of mesh points per training iteration
        seed: optional random seed for reproducibility
    """

    def __init__(self, subset_size, seed=None):
        self.subset_size = subset_size
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def sample(self, total_points):
        """Sample random subset indices.

        Args:
            total_points: total number of mesh points (N)

        Returns:
            indices: (subset_size,) tensor of randomly chosen indices
        """
        if total_points <= self.subset_size:
            return torch.arange(total_points)
        return torch.randperm(total_points, generator=self.generator)[:self.subset_size]


def relative_l2_loss(pred, target):
    """Relative L2 loss used in Transolver-3 (paper Eq. 7).

    L2_rel = ||pred - target||_2 / ||target||_2

    Args:
        pred: (B, N, d_out) predictions
        target: (B, N, d_out) ground truth

    Returns:
        scalar loss averaged over the batch
    """
    diff_norm = torch.norm(pred - target, p=2, dim=(1, 2))
    target_norm = torch.norm(target, p=2, dim=(1, 2))
    return (diff_norm / (target_norm + 1e-8)).mean()


def create_optimizer(model, lr=1e-3, weight_decay=0.05):
    """Create AdamW optimizer per Transolver-3 training config (Table 6).

    Args:
        model: nn.Module
        lr: initial learning rate (default 1e-3)
        weight_decay: weight decay (default 0.05)

    Returns:
        optimizer: torch.optim.AdamW
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, total_steps, warmup_fraction=0.05, min_lr=1e-6):
    """Cosine LR schedule with linear warmup per Transolver-3 config.

    Args:
        optimizer: torch optimizer
        total_steps: total number of training steps
        warmup_fraction: fraction of steps for linear warmup (default 5%)
        min_lr: minimum learning rate

    Returns:
        scheduler: LambdaLR scheduler
    """
    warmup_steps = int(total_steps * warmup_fraction)

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))
            # Scale to ensure min_lr
            base_lr = optimizer.defaults['lr']
            target = min_lr / base_lr
            return target + (1.0 - target) * cosine_decay.item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(model, x, fx, target, optimizer, scheduler, sampler=None,
               num_tiles=0, tile_size=0, grad_clip=1.0, normalizer=None):
    """Single training step with optional geometry amortized training.

    Args:
        model: Transolver3 model
        x: (B, N, space_dim) mesh coordinates
        fx: (B, N, fun_dim) or None, input features
        target: (B, N, out_dim) ground truth (in original scale)
        optimizer: optimizer
        scheduler: LR scheduler
        sampler: AmortizedMeshSampler or None (None = use full mesh)
        num_tiles: number of tiles for attention
        tile_size: target points per tile (overrides num_tiles if >0).
                   Paper recommends 100_000 (Table 5).
        grad_clip: gradient clipping norm
        normalizer: TargetNormalizer or None. If provided, targets are
                    encoded before loss computation so the model learns
                    in normalized space (paper Appendix A.3).

    Returns:
        loss_value: scalar loss
    """
    model.train()
    optimizer.zero_grad()

    if sampler is not None:
        N = x.shape[1]
        indices = sampler.sample(N).to(x.device)
        pred = model(x, fx=fx, num_tiles=num_tiles, tile_size=tile_size,
                     subset_indices=indices)
        t = target[:, indices]
    else:
        pred = model(x, fx=fx, num_tiles=num_tiles, tile_size=tile_size)
        t = target

    if normalizer is not None:
        t = normalizer.encode(t)

    loss = relative_l2_loss(pred, t)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()

    return loss.item()
