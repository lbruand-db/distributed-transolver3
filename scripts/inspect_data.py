"""Inspect .npz mesh files in a UC Volume. Prints shapes and basic stats."""

import os
import sys
import numpy as np

data_dir = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/lucasbruand_catalog/cfd/data"

for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith(".npz"):
        continue
    path = os.path.join(data_dir, fname)
    print(f"\n=== {fname} ===")
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    for key in sorted(data.keys()):
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
