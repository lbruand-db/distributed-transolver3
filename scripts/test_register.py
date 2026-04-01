"""Quick test for register_model.py — inspects checkpoint and tests loading.

Run via DAB:
    databricks bundle run test_register -t a10g --profile DEFAULT
"""

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(_this_dir, ".."))

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    print(f"\nCheckpoint has {len(state_dict)} keys:")
    for k, v in sorted(state_dict.items()):
        print(f"  {k}: {list(v.shape)}")

    # Try to infer config
    print("\n--- Config inference ---")

    # Find space_dim from first linear layer
    pre_key = "preprocess.linear_pre.0.weight"
    if pre_key in state_dict:
        space_dim = state_dict[pre_key].shape[1]
        n_hidden = state_dict[pre_key].shape[0] // 2
        print(f"space_dim={space_dim}, n_hidden={n_hidden}")
    else:
        print(f"WARNING: {pre_key} not found")

    # Find out_dim from output layer
    out_keys = [k for k in state_dict if "output" in k.lower() or "out" in k.lower()]
    print(f"Output-related keys: {out_keys}")
    for k in out_keys:
        print(f"  {k}: {list(state_dict[k].shape)}")

    # Count layers
    attn_keys = [k for k in state_dict if ".Attn." in k and k.endswith(".weight")]
    unique_blocks = set(k.split(".")[1] for k in attn_keys if k.startswith("blocks."))
    print(f"n_layers={len(unique_blocks)} (blocks: {sorted(unique_blocks)})")

    # Try loading into Transolver3
    print("\n--- Attempting model load ---")
    try:
        from transolver3 import Transolver3

        space_dim = state_dict[pre_key].shape[1]
        n_hidden = state_dict[pre_key].shape[0] // 2

        n_layers = len(unique_blocks)
        last_block = max(int(b) for b in unique_blocks)

        # out_dim from last block's output MLP
        out_dim = state_dict[f"blocks.{last_block}.mlp2.weight"].shape[0]
        print(f"out_dim={out_dim} (from blocks.{last_block}.mlp2.weight)")

        # n_head from to_q head_dim
        head_dim = state_dict["blocks.0.Attn.to_q.weight"].shape[0]
        n_head = n_hidden // head_dim
        print(f"head_dim={head_dim}, n_head={n_head}")

        # slice_num from in_project_slice: [n_head * slice_num, n_hidden]
        slice_num = state_dict["blocks.0.Attn.in_project_slice.weight"].shape[0] // n_head
        print(f"slice_num={slice_num}")

        config = dict(
            space_dim=space_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            fun_dim=0,
            out_dim=out_dim,
            slice_num=slice_num,
            mlp_ratio=1,
        )
        print(f"Config: {config}")

        model = Transolver3(**config)
        model.load_state_dict(state_dict)
        print("SUCCESS: Model loaded correctly!")

    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
