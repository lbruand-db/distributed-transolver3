"""
Input and target normalization for Transolver-3.

Paper (Appendix A.3):
  - "Geometric features are typically first normalized using min-max scaling
     and then optionally multiplied by a constant scaling factor (e.g., 1000)."
  - "Target outputs are standardized to have zero mean and unit variance
     across the dataset."

Provides:
  - InputNormalizer: min-max scaling with optional constant scaling factor
  - TargetNormalizer: zero-mean / unit-variance standardization
"""

import torch
import torch.nn as nn


class InputNormalizer(nn.Module):
    """Min-max normalization for geometric input features.

    Paper (Appendix A.3): coordinates are normalized to [0, 1] via min-max
    scaling, then optionally multiplied by a constant factor (e.g., 1000).

    Supports two modes:
      - per_sample=True (default): min/max computed per sample independently.
        Suitable when each sample has different geometry (e.g., different meshes).
      - per_sample=False: min/max fitted from training data (dataset-level).
        Suitable when all samples share the same coordinate system.

    Usage:
        # Per-sample (no fitting needed):
        normalizer = InputNormalizer(scale=1000.0)
        x_norm = normalizer.encode(coords)  # (B, N, D) -> (B, N, D) in [0, scale]

        # Dataset-level:
        normalizer = InputNormalizer(scale=1000.0, per_sample=False)
        normalizer.fit(train_coords)
        x_norm = normalizer.encode(coords)
    """

    def __init__(self, scale=1.0, per_sample=True, eps=1e-8):
        super().__init__()
        self.scale = scale
        self.per_sample = per_sample
        self.eps = eps
        self.register_buffer('data_min', torch.tensor(0.0))
        self.register_buffer('data_max', torch.tensor(1.0))
        self.register_buffer('fitted', torch.tensor(False))

    def fit(self, coords):
        """Compute dataset-level min/max from training coordinates.

        Args:
            coords: (num_samples, N, D) or (N, D) training coordinates

        Returns:
            self
        """
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        # Min/max across all samples and points, per feature channel
        # Shape: (1, 1, D)
        self.data_min = coords.reshape(-1, coords.shape[-1]).min(dim=0).values.reshape(1, 1, -1)
        self.data_max = coords.reshape(-1, coords.shape[-1]).max(dim=0).values.reshape(1, 1, -1)
        self.fitted = torch.tensor(True)
        return self

    def fit_incremental(self, data_iter):
        """Compute dataset-level min/max via streaming.

        Args:
            data_iter: iterable yielding coordinate tensors of shape
                       (B, N, D) or (N, D)

        Returns:
            self
        """
        running_min = None
        running_max = None

        for batch in data_iter:
            if batch.ndim == 2:
                batch = batch.unsqueeze(0)
            flat = batch.reshape(-1, batch.shape[-1])
            batch_min = flat.min(dim=0).values
            batch_max = flat.max(dim=0).values

            if running_min is None:
                running_min = batch_min
                running_max = batch_max
            else:
                running_min = torch.min(running_min, batch_min)
                running_max = torch.max(running_max, batch_max)

        self.data_min = running_min.reshape(1, 1, -1)
        self.data_max = running_max.reshape(1, 1, -1)
        self.fitted = torch.tensor(True)
        return self

    def encode(self, x):
        """Normalize coordinates to [0, scale] via min-max scaling.

        Args:
            x: (..., D) coordinates

        Returns:
            normalized x in [0, self.scale]
        """
        if self.per_sample:
            # Compute min/max per sample (dims except last)
            orig_shape = x.shape
            if x.ndim == 2:
                # (N, D) — single sample
                x_min = x.min(dim=0, keepdim=True).values
                x_max = x.max(dim=0, keepdim=True).values
            else:
                # (B, N, D) — batch
                x_min = x.min(dim=-2, keepdim=True).values  # (B, 1, D)
                x_max = x.max(dim=-2, keepdim=True).values  # (B, 1, D)
        else:
            x_min = self.data_min
            x_max = self.data_max

        x_norm = (x - x_min) / (x_max - x_min + self.eps)
        return x_norm * self.scale

    def decode(self, x_norm):
        """Inverse of encode (only for dataset-level mode).

        Args:
            x_norm: (..., D) normalized coordinates

        Returns:
            original-scale coordinates
        """
        if self.per_sample:
            raise ValueError(
                "decode() is not supported in per_sample mode because "
                "per-sample min/max are not stored. Use per_sample=False "
                "and fit() for invertible normalization."
            )
        x = x_norm / self.scale
        return x * (self.data_max - self.data_min + self.eps) + self.data_min

    def extra_repr(self):
        parts = [f"scale={self.scale}", f"per_sample={self.per_sample}"]
        if self.fitted:
            parts.append(f"data_min={self.data_min.squeeze().tolist()}")
            parts.append(f"data_max={self.data_max.squeeze().tolist()}")
        return ", ".join(parts)


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
