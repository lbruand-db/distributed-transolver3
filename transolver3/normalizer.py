"""
Target normalization for Transolver-3.

Paper (Appendix A.3): "Target outputs are standardized to have zero mean
and unit variance across the dataset."

Provides TargetNormalizer with two fitting modes:
  - fit(): compute stats from a full tensor (small datasets)
  - fit_incremental(): streaming Welford's algorithm for large datasets
    that don't fit in memory at once
"""

import torch
import torch.nn as nn


class TargetNormalizer(nn.Module):
    """Standardizes targets to zero mean and unit variance.

    Computes per-channel statistics across all samples and spatial points:
        mean/std over dims (0, 1) for input shape (num_samples, N, out_dim).

    Registered as nn.Module so mean/std travel with model.to(device) and
    state_dict save/load.

    Usage:
        normalizer = TargetNormalizer()
        normalizer.fit(train_targets)           # (num_samples, N, out_dim)
        encoded = normalizer.encode(targets)    # for training
        decoded = normalizer.decode(predictions) # back to original scale
    """

    def __init__(self, out_dim=1, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer('mean', torch.zeros(1, 1, out_dim))
        self.register_buffer('std', torch.ones(1, 1, out_dim))
        self.register_buffer('fitted', torch.tensor(False))

    def fit(self, targets):
        """Compute normalization statistics from a full target tensor.

        Args:
            targets: (num_samples, N, out_dim) or (N, out_dim) training targets
        """
        if targets.ndim == 2:
            targets = targets.unsqueeze(0)

        # Mean and std across all samples and spatial points, per output channel
        # Result shape: (1, 1, out_dim)
        self.mean = targets.mean(dim=(0, 1), keepdim=True)
        self.std = targets.std(dim=(0, 1), keepdim=True) + self.eps
        self.fitted = torch.tensor(True)
        return self

    def fit_incremental(self, data_iter, count=None):
        """Compute stats via streaming Welford's algorithm.

        For datasets too large to load fully into memory. Iterate over
        batches of targets and accumulate running mean/variance.

        Args:
            data_iter: iterable yielding target tensors of shape
                       (B, N, out_dim) or (N, out_dim)
            count: optional total number of (sample, point) pairs.
                   If None, computed from the data.

        Returns:
            self
        """
        running_sum = None
        running_sq_sum = None
        total_count = 0

        for batch in data_iter:
            if batch.ndim == 2:
                batch = batch.unsqueeze(0)
            # Flatten samples and spatial dims: (B*N, out_dim)
            flat = batch.reshape(-1, batch.shape[-1]).double()
            n = flat.shape[0]

            if running_sum is None:
                running_sum = flat.sum(dim=0)
                running_sq_sum = (flat ** 2).sum(dim=0)
            else:
                running_sum = running_sum + flat.sum(dim=0)
                running_sq_sum = running_sq_sum + (flat ** 2).sum(dim=0)
            total_count += n

        mean = running_sum / total_count
        var = running_sq_sum / total_count - mean ** 2
        # Clamp variance to avoid sqrt of negative due to numerical issues
        std = torch.sqrt(var.clamp(min=0)) + self.eps

        self.mean = mean.float().reshape(1, 1, -1)
        self.std = std.float().reshape(1, 1, -1)
        self.fitted = torch.tensor(True)
        return self

    def encode(self, x):
        """Normalize targets: (x - mean) / std.

        Args:
            x: (..., out_dim) targets

        Returns:
            normalized x
        """
        return (x - self.mean) / self.std

    def decode(self, x):
        """Denormalize predictions: x * std + mean.

        Args:
            x: (..., out_dim) normalized predictions

        Returns:
            denormalized x
        """
        return x * self.std + self.mean

    def extra_repr(self):
        if self.fitted:
            return f"mean={self.mean.squeeze().tolist()}, std={self.std.squeeze().tolist()}"
        return "not fitted"
