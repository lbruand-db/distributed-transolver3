"""Register a trained Transolver-3 model in UC Model Registry.

Usage (DAB workflow task):
  python register_model.py --catalog ml --schema transolver3 --model_name transolver3
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

    config = {
        "space_dim": args.space_dim,
        "n_layers": args.n_layers,
        "n_hidden": args.n_hidden,
        "n_head": args.n_head,
        "fun_dim": 0,
        "out_dim": args.out_dim,
        "slice_num": args.slice_num,
    }

    print(f"Loading model from {args.checkpoint}...")
    model = Transolver3(**config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    print(f"Registering as {args.catalog}.{args.schema}.{args.model_name}...")
    with mlflow.start_run(run_name="register-model"):
        info = register_serving_model(
            model, config,
            catalog=args.catalog,
            schema=args.schema,
            model_name=args.model_name,
        )
    print(f"Registered model: {info}")


if __name__ == "__main__":
    main()
