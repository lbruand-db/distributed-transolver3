#!/usr/bin/env python3
"""Generate deterministic train/test/val split files for DrivAerML.

Paper (Appendix A.2): "we randomly allocate 400 samples for training and
50 for testing, while reserving 50 for validation."

Usage:
    python scripts/generate_split.py --data_dir /path/to/data
    python scripts/generate_split.py --data_dir /path/to/data --seed 42

Produces train.txt, test.txt, val.txt in data_dir.
"""

import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Generate DrivAerML train/test/val splits")
    parser.add_argument("--data_dir", required=True, help="Directory containing .npz sample files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--n_val", type=int, default=50)
    args = parser.parse_args()

    # Discover all sample files
    samples = sorted(f for f in os.listdir(args.data_dir) if f.endswith((".npz", ".h5")))
    total = len(samples)
    expected = args.n_train + args.n_test + args.n_val

    if total < expected:
        print(f"WARNING: found {total} samples, expected at least {expected}. "
              f"Using proportional split instead.")
        # Proportional split
        n_test = max(1, int(total * args.n_test / expected))
        n_val = max(1, int(total * args.n_val / expected))
        n_train = total - n_test - n_val
    else:
        n_train, n_test, n_val = args.n_train, args.n_test, args.n_val

    # Deterministic shuffle
    random.seed(args.seed)
    random.shuffle(samples)

    train = sorted(samples[:n_train])
    test = sorted(samples[n_train:n_train + n_test])
    val = sorted(samples[n_train + n_test:n_train + n_test + n_val])

    for name, split in [("train", train), ("test", test), ("val", val)]:
        path = os.path.join(args.data_dir, f"{name}.txt")
        with open(path, "w") as f:
            for s in split:
                f.write(s + "\n")
        print(f"Wrote {path} ({len(split)} samples)")

    # Any remaining samples beyond the split are not assigned
    remaining = total - n_train - n_test - n_val
    if remaining > 0:
        print(f"Note: {remaining} samples not assigned to any split")


if __name__ == "__main__":
    main()
