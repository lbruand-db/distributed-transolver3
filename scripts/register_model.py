"""Register a trained Transolver-3 model in UC Model Registry.

Usage (DAB workflow task):
  python register_model.py --catalog ml --schema transolver3 --model_name transolver3
"""

import argparse
import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(_this_dir, ".."))

import torch  # noqa: E402
import mlflow  # noqa: E402

from transolver3 import Transolver3  # noqa: E402
from transolver3.serving import register_serving_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Register model in UC Model Registry")
    parser.add_argument("--catalog", default="ml")
    parser.add_argument("--schema", default="transolver3")
    parser.add_argument("--model_name", default="transolver3")
    parser.add_argument("--checkpoint", default="./checkpoints/drivaer_ml_distributed/best_model.pt")
    parser.add_argument("--space_dim", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--out_dim", type=int, default=4)
    parser.add_argument("--slice_num", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Infer architecture from checkpoint weights
    space_dim = state_dict["preprocess.linear_pre.0.weight"].shape[1]
    n_hidden = state_dict["preprocess.linear_pre.0.weight"].shape[0] // 2

    # Count layers by counting unique block indices
    block_ids = set()
    for k in state_dict:
        if k.startswith("blocks.") and ".Attn." in k:
            block_ids.add(int(k.split(".")[1]))
    n_layers = len(block_ids)

    # out_dim from last block's output MLP (blocks.{n_layers-1}.mlp2.weight)
    last_block = max(block_ids)
    out_dim = state_dict[f"blocks.{last_block}.mlp2.weight"].shape[0]

    # n_head from attention head_dim: to_q.weight is [head_dim, head_dim]
    head_dim = state_dict["blocks.0.Attn.to_q.weight"].shape[0]
    n_head = n_hidden // head_dim

    # slice_num from in_project_slice.weight: [n_head * slice_num, n_hidden]
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

    print(f"Registering as {args.catalog}.{args.schema}.{args.model_name}...")
    mlflow.set_experiment("/Shared/transolver3-experiments")
    with mlflow.start_run(run_name="register-model"):
        info = register_serving_model(
            model,
            config,
            catalog=args.catalog,
            schema=args.schema,
            model_name=args.model_name,
        )
    print(f"Registered model: {info}")


if __name__ == "__main__":
    main()
