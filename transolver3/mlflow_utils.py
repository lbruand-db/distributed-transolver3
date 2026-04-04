# Copyright 2024 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MLflow integration utilities for Transolver-3 experiment tracking.

Provides helpers to:
  - Log training runs with full hyperparameters and system metrics
  - Log models with signatures to Unity Catalog Model Registry
  - Save/restore normalizer artifacts for reproducible inference
  - Log prediction visualizations as artifacts

All functions require MLflow and are opt-in. Install with:
  pip install transolver3[databricks]
"""

import os
import tempfile

import torch


def _require_mlflow():
    try:
        import mlflow  # noqa: F401
    except ImportError:
        raise ImportError("mlflow is required for mlflow_utils. Install with: pip install transolver3[databricks]")


def log_training_run(model, config, normalizers=None):
    """Log model info, hyperparameters, and normalizer artifacts to the active MLflow run.

    Must be called inside an active mlflow.start_run() context.

    Args:
        model: Transolver3 model (or DDP-wrapped)
        config: dict of hyperparameters to log
        normalizers: optional dict of name -> normalizer (InputNormalizer/TargetNormalizer)
    """
    _require_mlflow()
    import mlflow

    if mlflow.active_run() is None:
        raise ValueError("No active MLflow run. Call this inside mlflow.start_run().")

    # Log hyperparameters
    mlflow.log_params(config)

    # Log model info
    from transolver3.distributed import unwrap_ddp_model

    raw_model = unwrap_ddp_model(model)
    n_params = sum(p.numel() for p in raw_model.parameters())
    mlflow.log_param("model_parameters", n_params)

    # Enable GPU/system metrics capture
    try:
        mlflow.enable_system_metrics_logging()
    except Exception:
        pass  # Not available in all MLflow versions

    # Log normalizer artifacts
    if normalizers:
        for name, norm in normalizers.items():
            log_normalization_artifacts(**{("input_norm" if "input" in name.lower() else "target_norm"): norm})


def log_model_with_signature(model, sample_input, registered_model_name=None):
    """Log a Transolver3 model with an inferred signature.

    Args:
        model: Transolver3 model (unwrapped)
        sample_input: tensor (B, N, space_dim) for signature inference
        registered_model_name: if set, registers in UC Model Registry
            (e.g., "catalog.schema.transolver3")

    Returns:
        MLflow ModelInfo
    """
    _require_mlflow()
    import mlflow
    import mlflow.pytorch
    import numpy as np

    from transolver3.distributed import unwrap_ddp_model

    raw_model = unwrap_ddp_model(model)
    raw_model.eval()
    with torch.no_grad():
        sample_output = raw_model(sample_input.to(next(raw_model.parameters()).device))

    signature = mlflow.models.infer_signature(
        sample_input.cpu().numpy().astype(np.float32),
        sample_output.cpu().numpy().astype(np.float32),
    )

    return mlflow.pytorch.log_model(
        raw_model,
        artifact_path="transolver3",
        signature=signature,
        registered_model_name=registered_model_name,
    )


def log_normalization_artifacts(input_norm=None, target_norm=None):
    """Save normalizer state_dicts as MLflow artifacts.

    Must be called inside an active MLflow run.

    Args:
        input_norm: optional InputNormalizer instance
        target_norm: optional TargetNormalizer instance
    """
    _require_mlflow()
    import mlflow

    with tempfile.TemporaryDirectory() as tmpdir:
        if input_norm is not None:
            path = os.path.join(tmpdir, "input_normalizer.pt")
            torch.save(input_norm.state_dict(), path)
            mlflow.log_artifact(path, artifact_path="normalizers")

        if target_norm is not None:
            path = os.path.join(tmpdir, "target_normalizer.pt")
            torch.save(target_norm.state_dict(), path)
            mlflow.log_artifact(path, artifact_path="normalizers")


def load_normalization_artifacts(run_id):
    """Restore normalizers from MLflow artifacts.

    Args:
        run_id: MLflow run ID

    Returns:
        tuple (InputNormalizer or None, TargetNormalizer or None)
    """
    _require_mlflow()
    import mlflow
    from transolver3.normalizer import InputNormalizer, TargetNormalizer

    artifact_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="normalizers")

    input_norm = None
    target_norm = None

    input_path = os.path.join(artifact_dir, "input_normalizer.pt")
    if os.path.exists(input_path):
        input_norm = InputNormalizer()
        input_norm.load_state_dict(torch.load(input_path, weights_only=True))

    target_path = os.path.join(artifact_dir, "target_normalizer.pt")
    if os.path.exists(target_path):
        target_norm = TargetNormalizer()
        target_norm.load_state_dict(torch.load(target_path, weights_only=True))

    return input_norm, target_norm


def log_prediction_visualization(pred, target, coords, step, title=None):
    """Create and log a pressure heatmap comparison as an MLflow artifact.

    Args:
        pred: (N, out_dim) or (N,) predicted values (first channel used)
        target: (N, out_dim) or (N,) ground truth values
        coords: (N, 2) or (N, 3) coordinates (first 2 dims used for scatter)
        step: training step number (used in filename)
        title: optional plot title
    """
    _require_mlflow()
    import mlflow

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib not available, skip visualization

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy()

    if pred_np.ndim == 2:
        pred_np = pred_np[:, 0]
    if target_np.ndim == 2:
        target_np = target_np[:, 0]

    x, y = coords_np[:, 0], coords_np[:, 1]
    error = abs(pred_np - target_np)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = min(pred_np.min(), target_np.min())
    vmax = max(pred_np.max(), target_np.max())

    axes[0].scatter(x, y, c=target_np, s=0.5, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[0].set_title("Ground Truth")
    axes[0].set_aspect("equal")

    axes[1].scatter(x, y, c=pred_np, s=0.5, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[1].set_title("Prediction")
    axes[1].set_aspect("equal")

    sc = axes[2].scatter(x, y, c=error, s=0.5, cmap="hot")
    axes[2].set_title(f"Absolute Error (mean={error.mean():.4f})")
    axes[2].set_aspect("equal")
    fig.colorbar(sc, ax=axes[2])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"prediction_step_{step}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path="predictions")
