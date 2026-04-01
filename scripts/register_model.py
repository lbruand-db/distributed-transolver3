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

import mlflow  # noqa: E402


def register_from_mlflow_run(run_id, catalog, schema, model_name):
    """Promote an already-logged model from an MLflow run to UC Model Registry."""
    registered_name = f"{catalog}.{schema}.{model_name}"
    model_uri = f"runs:/{run_id}/transolver3"

    print(f"Promoting model from run {run_id} to {registered_name}...")
    result = mlflow.register_model(model_uri, registered_name)
    print(f"Registered: {registered_name} version {result.version}")
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
