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
Monitoring integration for Transolver-3 on Databricks.

Provides:
  - Inference table setup for automatic request/response logging
  - Lakehouse Monitoring for model quality and drift detection
  - Physical bounds checking for prediction validation
  - Distribution drift metrics (PSI) logged to Delta

All Databricks-specific imports are guarded. Install with:
  pip install transolver3[databricks]
"""

from datetime import datetime, timezone

import numpy as np


def setup_inference_table(endpoint_name, catalog, schema):
    """Enable inference table logging on a Databricks Model Serving endpoint.

    Auto-logs every request/response to a Delta table for monitoring.

    Args:
        endpoint_name: serving endpoint name
        catalog: Unity Catalog name
        schema: UC schema name

    Returns:
        Updated endpoint config
    """
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError(
            "databricks-sdk is required for setup_inference_table. Install with: pip install transolver3[databricks]"
        )

    client = WorkspaceClient()

    # Enable auto-capture of inference requests/responses
    result = client.serving_endpoints.update_config(
        name=endpoint_name,
        auto_capture_config={
            "catalog_name": catalog,
            "schema_name": schema,
            "table_name_prefix": endpoint_name.replace("-", "_"),
            "enabled": True,
        },
    )
    print(f"Inference table enabled for endpoint '{endpoint_name}' in {catalog}.{schema}")
    return result


def create_quality_monitor(catalog, schema, table_name):
    """Set up Lakehouse Monitoring on an inference table.

    Creates a monitor that tracks prediction quality, data drift,
    and model performance over time.

    Args:
        catalog: Unity Catalog name
        schema: UC schema name
        table_name: inference table name to monitor

    Returns:
        Monitor info
    """
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError(
            "databricks-sdk is required for create_quality_monitor. Install with: pip install transolver3[databricks]"
        )

    client = WorkspaceClient()
    full_table = f"{catalog}.{schema}.{table_name}"

    result = client.quality_monitors.create(
        table_name=full_table,
        assets_dir=f"/Workspace/Users/monitoring/{table_name}",
        output_schema_name=f"{catalog}.{schema}",
        schedule={"quartz_cron_expression": "0 0 * * * ?", "timezone_id": "UTC"},
    )
    print(f"Quality monitor created for {full_table}")
    return result


def check_prediction_bounds(predictions, physical_bounds):
    """Validate that predictions fall within physically plausible bounds.

    Pure PyTorch function — no Databricks dependency. Useful both for
    monitoring and as a sanity check in the serving layer.

    Args:
        predictions: tensor (B, N, out_dim) or (N, out_dim)
        physical_bounds: dict mapping channel index or name to (min, max) tuples.
            Example: {0: (-50000, 100000), 1: (0, 400)}
            For named channels: {"pressure": (-50000, 100000)}

    Returns:
        dict with:
            - "all_valid": bool, True if all predictions within bounds
            - "channels": dict per channel with "out_of_bounds_count",
              "out_of_bounds_fraction", "min_value", "max_value"
            - "total_out_of_bounds": int total across all channels
    """
    if predictions.ndim == 2:
        predictions = predictions.unsqueeze(0)

    result = {
        "all_valid": True,
        "channels": {},
        "total_out_of_bounds": 0,
    }

    for channel_key, (lo, hi) in physical_bounds.items():
        if isinstance(channel_key, int):
            channel_idx = channel_key
        else:
            channel_idx = channel_key  # caller should use int indices

        vals = predictions[..., channel_idx]
        below = (vals < lo).sum().item()
        above = (vals > hi).sum().item()
        oob = below + above
        total = vals.numel()

        result["channels"][channel_key] = {
            "out_of_bounds_count": oob,
            "out_of_bounds_fraction": oob / total if total > 0 else 0.0,
            "below_min": below,
            "above_max": above,
            "min_value": vals.min().item(),
            "max_value": vals.max().item(),
        }
        result["total_out_of_bounds"] += oob
        if oob > 0:
            result["all_valid"] = False

    return result


def log_drift_metrics(spark, catalog, schema, predictions, baseline_stats):
    """Compute and log distribution drift metrics to Delta.

    Compares current prediction distribution against baseline training
    statistics using Population Stability Index (PSI).

    Args:
        spark: SparkSession
        catalog: Unity Catalog name
        schema: UC schema name
        predictions: tensor (B, N, out_dim) — current predictions
        baseline_stats: dict with per-channel statistics from training:
            {"mean": [...], "std": [...], "percentiles": {...}}

    Returns:
        dict with per-channel PSI values and drift flags
    """
    try:
        from pyspark.sql import Row
    except ImportError:
        raise ImportError(
            "pyspark is required for log_drift_metrics. Install with: pip install transolver3[databricks]"
        )

    if predictions.ndim == 2:
        predictions = predictions.unsqueeze(0)

    pred_np = predictions.detach().cpu().numpy().reshape(-1, predictions.shape[-1])
    out_dim = pred_np.shape[-1]

    drift_results = {}
    rows = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for ch in range(out_dim):
        current_vals = pred_np[:, ch]
        current_mean = float(np.mean(current_vals))
        current_std = float(np.std(current_vals))

        baseline_mean = baseline_stats["mean"][ch]
        baseline_std = baseline_stats["std"][ch]

        # PSI approximation using normal distribution assumption
        psi = _compute_psi_normal(baseline_mean, baseline_std, current_mean, current_std)

        drift_flag = "high" if psi > 0.2 else ("medium" if psi > 0.1 else "low")

        drift_results[ch] = {
            "psi": psi,
            "drift": drift_flag,
            "current_mean": current_mean,
            "current_std": current_std,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
        }

        rows.append(
            Row(
                channel=ch,
                psi=psi,
                drift_flag=drift_flag,
                current_mean=current_mean,
                current_std=current_std,
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                logged_at=timestamp,
            )
        )

    # Write to Delta
    full_table = f"{catalog}.{schema}.drift_metrics"
    df = spark.createDataFrame(rows)
    df.write.format("delta").mode("append").saveAsTable(full_table)

    return drift_results


def _compute_psi_normal(mean1, std1, mean2, std2, n_bins=10):
    """Approximate PSI between two normal distributions.

    Uses binned comparison of CDFs.

    Args:
        mean1, std1: baseline distribution parameters
        mean2, std2: current distribution parameters
        n_bins: number of bins for PSI calculation

    Returns:
        PSI value (float). >0.2 indicates significant drift.
    """
    # Create bins spanning both distributions
    lo = min(mean1 - 3 * std1, mean2 - 3 * std2)
    hi = max(mean1 + 3 * std1, mean2 + 3 * std2)
    edges = np.linspace(lo, hi, n_bins + 1)

    eps = 1e-8

    # Compute bin probabilities from CDFs
    from scipy.stats import norm as _norm_dist

    p1 = np.diff(_norm_dist.cdf(edges, mean1, std1 + eps))
    p2 = np.diff(_norm_dist.cdf(edges, mean2, std2 + eps))

    # Clamp to avoid log(0)
    p1 = np.clip(p1, eps, 1.0)
    p2 = np.clip(p2, eps, 1.0)

    psi = float(np.sum((p2 - p1) * np.log(p2 / p1)))
    return psi
