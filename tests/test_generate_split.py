# Copyright 2026 Databricks, Inc.
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

"""Tests for generate_split.py: deterministic data splitting."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from generate_split import generate_split


def _make_samples(n):
    """Create a list of fake sample filenames."""
    return [f"drivaer_{i:03d}.npz" for i in range(n)]


# ===== Split Sizes =====


def test_split_sizes_500():
    """With 500 samples, split should be exactly 400/50/50."""
    samples = _make_samples(500)
    train, test, val = generate_split(samples, n_train=400, n_test=50, n_val=50)

    assert len(train) == 400
    assert len(test) == 50
    assert len(val) == 50


def test_split_sizes_exact():
    """With exactly n_train + n_test + n_val samples, all are assigned."""
    samples = _make_samples(100)
    train, test, val = generate_split(samples, n_train=80, n_test=10, n_val=10)

    assert len(train) == 80
    assert len(test) == 10
    assert len(val) == 10
    assert len(train) + len(test) + len(val) == 100


def test_split_sizes_excess():
    """With more samples than needed, extras are unassigned."""
    samples = _make_samples(600)
    train, test, val = generate_split(samples, n_train=400, n_test=50, n_val=50)

    assert len(train) == 400
    assert len(test) == 50
    assert len(val) == 50
    # 100 unassigned


def test_split_proportional_fewer_samples():
    """With fewer samples than expected, split scales proportionally."""
    samples = _make_samples(20)
    train, test, val = generate_split(samples, n_train=400, n_test=50, n_val=50)

    total = len(train) + len(test) + len(val)
    assert total <= 20
    assert len(train) > 0
    assert len(test) > 0
    assert len(val) > 0


# ===== No Overlap =====


def test_no_overlap():
    """Train, test, and val should be disjoint."""
    samples = _make_samples(500)
    train, test, val = generate_split(samples)

    train_set = set(train)
    test_set = set(test)
    val_set = set(val)

    assert len(train_set & test_set) == 0, "Train and test overlap"
    assert len(train_set & val_set) == 0, "Train and val overlap"
    assert len(test_set & val_set) == 0, "Test and val overlap"


# ===== Determinism =====


def test_determinism_same_seed():
    """Same seed + same samples should produce identical splits."""
    samples = _make_samples(500)
    train1, test1, val1 = generate_split(samples, seed=42)
    train2, test2, val2 = generate_split(samples, seed=42)

    assert train1 == train2
    assert test1 == test2
    assert val1 == val2


def test_different_seeds_differ():
    """Different seeds should produce different splits."""
    samples = _make_samples(500)
    train1, _, _ = generate_split(samples, seed=42)
    train2, _, _ = generate_split(samples, seed=123)

    assert train1 != train2


# ===== Sorted Output =====


def test_splits_are_sorted():
    """All splits should be sorted for reproducible file ordering."""
    samples = _make_samples(500)
    train, test, val = generate_split(samples)

    assert train == sorted(train)
    assert test == sorted(test)
    assert val == sorted(val)


# ===== Input Not Mutated =====


def test_input_not_mutated():
    """generate_split should not modify the input list."""
    samples = _make_samples(500)
    original = list(samples)
    generate_split(samples)

    assert samples == original


# ===== Integration: main() writes files =====


def test_main_writes_split_files():
    """main() should create train.txt, test.txt, val.txt in data_dir."""
    from generate_split import main
    import sys as _sys

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake .npz files
        for i in range(50):
            open(os.path.join(tmpdir, f"drivaer_{i:03d}.npz"), "w").close()

        # Patch sys.argv for argparse
        old_argv = _sys.argv
        _sys.argv = ["generate_split.py", "--data_dir", tmpdir, "--seed", "42",
                      "--n_train", "30", "--n_test", "10", "--n_val", "10"]
        try:
            main()
        finally:
            _sys.argv = old_argv

        # Check files were created
        for name in ["train.txt", "test.txt", "val.txt"]:
            path = os.path.join(tmpdir, name)
            assert os.path.exists(path), f"{name} not created"
            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) > 0, f"{name} is empty"

        # Check sizes
        with open(os.path.join(tmpdir, "train.txt")) as f:
            assert len([line for line in f if line.strip()]) == 30
        with open(os.path.join(tmpdir, "test.txt")) as f:
            assert len([line for line in f if line.strip()]) == 10
        with open(os.path.join(tmpdir, "val.txt")) as f:
            assert len([line for line in f if line.strip()]) == 10
