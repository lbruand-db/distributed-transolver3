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


def generate_split(samples, n_train=400, n_test=50, n_val=50, seed=42):
    """Split sample filenames into train/test/val sets.

    Args:
        samples: list of sample filenames
        n_train: number of training samples
        n_test: number of test samples
        n_val: number of validation samples
        seed: random seed for reproducible shuffle

    Returns:
        tuple of (train, test, val) — each a sorted list of filenames
    """
    total = len(samples)
    expected = n_train + n_test + n_val

    if total < expected:
        # Proportional split for smaller datasets
        n_test = max(1, int(total * n_test / expected))
        n_val = max(1, int(total * n_val / expected))
        n_train = total - n_test - n_val

    # Deterministic shuffle on a copy
    shuffled = list(samples)
    random.seed(seed)
    random.shuffle(shuffled)

    train = sorted(shuffled[:n_train])
    test = sorted(shuffled[n_train:n_train + n_test])
    val = sorted(shuffled[n_train + n_test:n_train + n_test + n_val])

    return train, test, val


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
    print(f"Found {total} samples in {args.data_dir}", flush=True)

    train, test, val = generate_split(
        samples, n_train=args.n_train, n_test=args.n_test, n_val=args.n_val, seed=args.seed
    )

    for name, split in [("train", train), ("test", test), ("val", val)]:
        path = os.path.join(args.data_dir, f"{name}.txt")
        with open(path, "w") as f:
            for s in split:
                f.write(s + "\n")
        print(f"Wrote {path} ({len(split)} samples)")

    remaining = total - len(train) - len(test) - len(val)
    if remaining > 0:
        print(f"Note: {remaining} samples not assigned to any split")


if __name__ == "__main__":
    main()
