"""Register a trained Transolver-3 model in UC Model Registry.

Promotes the model already logged to MLflow during training, rather than
re-logging it. Reads the MLflow run_id from a file written by the train task.

Fallback: if no run_id file is provided, loads from a checkpoint file and
re-registers (legacy behavior).

Usage (DAB workflow task):
  python register_model.py --catalog ml --schema transolver3 --model_name transolver3 \
                           --mlflow_run_id_file /Volumes/.../checkpoints/mlflow_run_id.txt
"""

import argparse
import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(_this_dir, ".."))

try:
    import mlflow  # noqa: E402

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def format_model_card(params, metrics, run_id="unknown"):
    """Format a model card from params and metrics dicts.

    Args:
        params: dict of hyperparameters (string keys and values)
        metrics: dict of metric names to float values
        run_id: MLflow run ID for reference

    Returns:
        Markdown-formatted model card string
    """
    field = params.get("field", "unknown")
    lines = [
        f"# Transolver-3 ({field})",
        "",
        "Physics-informed transformer solver for industrial-scale CFD.",
        "Trained on DrivAerML dataset with geometry amortized training.",
        "",
        "## Performance",
        "",
        f"- **Best test L2**: {float(metrics.get('best_test_l2', 0)):.4f} "
        f"({float(metrics.get('best_test_l2', 0)) * 100:.2f}%)",
    ]

    # Per-quantity metrics
    for key in ["test_l2_p_s", "test_l2_tau", "test_l2_u", "test_l2_p_v"]:
        if key in metrics:
            name = key.replace("test_l2_", "")
            lines.append(f"- **{name}**: {float(metrics[key]):.4f} ({float(metrics[key]) * 100:.2f}%)")

    lines += [
        "",
        "## Architecture",
        "",
        f"- Layers: {params.get('n_layers', '?')}",
        f"- Hidden dim: {params.get('n_hidden', '?')}",
        f"- Heads: {params.get('n_head', '?')}",
        f"- Slices: {params.get('slice_num', '?')}",
        f"- Parameters: {params.get('model_parameters', '?')}",
        f"- Input dim (space_dim): {params.get('space_dim', '?')}",
        f"- Output dim: {params.get('out_dim', '?')}",
        "",
        "## Training Configuration",
        "",
        f"- Field: {field}",
        f"- Epochs: {params.get('epochs', '?')}",
        f"- LR: {params.get('lr', '?')}",
        f"- Weight decay: {params.get('weight_decay', '?')}",
        f"- Subset size: {params.get('subset_size', '?')}",
        f"- Batch size: {params.get('batch_size', '?')} "
        f"(effective: {params.get('effective_batch_size', '?')})",
        f"- AMP: {params.get('amp', '?')}",
        f"- World size: {params.get('world_size', '?')}",
        f"- Seed: {params.get('seed', '?')}",
        "",
        "## Limitations",
        "",
        "- Trained on DrivAerML surface/volume meshes only",
        "- Expects min-max normalized coordinates scaled by 1000",
        "- Target normalizer (z-score) must be applied at inference time",
        f"- MLflow run: `{run_id}`",
    ]
    return "\n".join(lines)


def build_model_description(run_id):
    """Build a model card description from the MLflow run's params and metrics."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return format_model_card(run.data.params, run.data.metrics, run_id)


def compute_metric_deltas(current_metrics, previous_metrics):
    """Compute deltas between current and previous run metrics.

    Args:
        current_metrics: dict of current run metric names to float values
        previous_metrics: dict of previous run metric names to float values

    Returns:
        dict of metric deltas (negative = improvement). Keys: "aggregate"
        plus any per-quantity keys present in both runs.
    """
    current_l2 = current_metrics.get("best_test_l2")
    prev_l2 = previous_metrics.get("best_test_l2")

    if current_l2 is None or prev_l2 is None:
        return {}

    deltas = {"aggregate": current_l2 - prev_l2}

    for key in ["test_l2_p_s", "test_l2_tau", "test_l2_u", "test_l2_p_v"]:
        curr = current_metrics.get(key)
        prev = previous_metrics.get(key)
        if curr is not None and prev is not None:
            deltas[key] = curr - prev

    return deltas


def compare_with_previous_best(run_id):
    """Compare this run's metrics with the previous best run in the experiment.

    Returns a dict of metric deltas (negative = improvement).
    """
    client = mlflow.tracking.MlflowClient()
    current_run = client.get_run(run_id)
    experiment_id = current_run.info.experiment_id
    current_field = current_run.data.params.get("field", "")
    current_l2 = current_run.data.metrics.get("best_test_l2")

    if current_l2 is None:
        print("  No best_test_l2 metric in current run, skipping comparison.")
        return {}

    # Search for previous completed runs with the same field
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"params.field = '{current_field}' "
        f"AND metrics.best_test_l2 > 0 "
        f"AND attributes.run_id != '{run_id}'",
        order_by=["metrics.best_test_l2 ASC"],
        max_results=1,
    )

    if not runs:
        print("  No previous runs found for comparison.")
        return {}

    prev_run = runs[0]
    prev_l2 = prev_run.data.metrics.get("best_test_l2")
    prev_id = prev_run.info.run_id

    deltas = compute_metric_deltas(current_run.data.metrics, prev_run.data.metrics)
    delta = deltas.get("aggregate", 0)
    pct = (delta / prev_l2) * 100 if prev_l2 else 0
    improved = delta < 0

    print(f"\n{'=' * 50}")
    print(f"  Comparison with previous best (run {prev_id[:8]}...):")
    print(f"    Previous best L2: {prev_l2:.4f} ({prev_l2 * 100:.2f}%)")
    print(f"    Current L2:       {current_l2:.4f} ({current_l2 * 100:.2f}%)")
    print(f"    Delta:            {delta:+.4f} ({pct:+.1f}%)")
    print(f"    {'IMPROVED' if improved else 'REGRESSED'}")
    print(f"{'=' * 50}\n")

    # Log comparison metrics back to the current run
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_metric("delta_vs_previous", delta)
        mlflow.log_param("previous_best_run_id", prev_id)
        mlflow.log_param("previous_best_l2", prev_l2)

    for key, d in deltas.items():
        if key != "aggregate":
            prev = prev_run.data.metrics.get(key)
            curr = current_run.data.metrics.get(key)
            print(f"  {key}: {prev:.4f} -> {curr:.4f} (delta {d:+.4f})")

    return deltas


def register_from_mlflow_run(run_id, catalog, schema, model_name):
    """Promote an already-logged model from an MLflow run to UC Model Registry."""
    registered_name = f"{catalog}.{schema}.{model_name}"
    model_uri = f"runs:/{run_id}/transolver3"

    # Compare with previous best before registering
    try:
        compare_with_previous_best(run_id)
    except Exception as e:
        print(f"WARNING: Experiment comparison failed: {e}")

    print(f"Promoting model from run {run_id} to {registered_name}...")
    result = mlflow.register_model(model_uri, registered_name)
    print(f"Registered: {registered_name} version {result.version}")

    # Set model version description (model card)
    try:
        description = build_model_description(run_id)
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name=registered_name,
            version=result.version,
            description=description,
        )
        print(f"Set model card for version {result.version}")
    except Exception as e:
        print(f"WARNING: Could not set model description: {e}")

    return result


def register_from_checkpoint(checkpoint, catalog, schema, model_name):
    """Legacy: load checkpoint, infer config, and register as new MLflow model."""
    import torch  # noqa: E402
    from transolver3 import Transolver3  # noqa: E402
    from transolver3.serving import register_serving_model  # noqa: E402

    print(f"Loading model from {checkpoint}...")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

    # Infer architecture from checkpoint weights
    space_dim = state_dict["preprocess.linear_pre.0.weight"].shape[1]
    n_hidden = state_dict["preprocess.linear_pre.0.weight"].shape[0] // 2

    block_ids = set()
    for k in state_dict:
        if k.startswith("blocks.") and ".Attn." in k:
            block_ids.add(int(k.split(".")[1]))
    n_layers = len(block_ids)

    last_block = max(block_ids)
    out_dim = state_dict[f"blocks.{last_block}.mlp2.weight"].shape[0]

    head_dim = state_dict["blocks.0.Attn.to_q.weight"].shape[0]
    n_head = n_hidden // head_dim

    slice_num = state_dict["blocks.0.Attn.in_project_slice.weight"].shape[0] // n_head

    config = {
        "space_dim": space_dim,
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "n_head": n_head,
        "fun_dim": 0,
        "out_dim": out_dim,
        "slice_num": slice_num,
    }
    print(f"Inferred config: {config}")

    model = Transolver3(**config)
    model.load_state_dict(state_dict)

    print(f"Registering as {catalog}.{schema}.{model_name}...")
    mlflow.set_experiment("/Shared/transolver3-experiments")
    with mlflow.start_run(run_name="register-model"):
        info = register_serving_model(
            model,
            config,
            catalog=catalog,
            schema=schema,
            model_name=model_name,
        )
    print(f"Registered model: {info}")
    return info


def main():
    parser = argparse.ArgumentParser(description="Register model in UC Model Registry")
    parser.add_argument("--catalog", default="ml")
    parser.add_argument("--schema", default="transolver3")
    parser.add_argument("--model_name", default="transolver3")
    parser.add_argument(
        "--mlflow_run_id_file",
        type=str,
        default=None,
        help="Path to file containing MLflow run_id from training. "
        "If set, promotes the already-logged model (no re-logging).",
    )
    parser.add_argument("--checkpoint", default=None, help="Fallback: path to checkpoint file if no MLflow run_id.")
    args = parser.parse_args()

    mlflow.set_experiment("/Shared/transolver3-experiments")

    if args.mlflow_run_id_file and os.path.exists(args.mlflow_run_id_file):
        with open(args.mlflow_run_id_file) as f:
            run_id = f.read().strip()
        register_from_mlflow_run(run_id, args.catalog, args.schema, args.model_name)
    elif args.checkpoint:
        register_from_checkpoint(args.checkpoint, args.catalog, args.schema, args.model_name)
    else:
        print("ERROR: must provide either --mlflow_run_id_file or --checkpoint")
        sys.exit(1)


if __name__ == "__main__":
    main()
