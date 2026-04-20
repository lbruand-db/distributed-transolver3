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

"""
Tests for evaluate(), per-quantity metrics, and train_epoch integration.
"""

import sys
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Industrial-Scale-Benchmarks"))

from transolver3.model import Transolver3
from transolver3.amortized_training import relative_l2_loss, create_optimizer, create_scheduler
from transolver3.normalizer import TargetNormalizer
from exp_drivaer_ml_distributed import evaluate, _QUANTITY_SPLITS, train_epoch, get_field_key


def _make_model(space_dim=9, out_dim=4):
    """Create a small Transolver3 for testing."""
    return Transolver3(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=4,
        mlp_ratio=1,
        num_tiles=1,
    )


def _make_surface_dataloader(n_samples=2, n_points=50):
    """Create a dataloader that mimics surface field data."""
    # surface_x: (B, N, 9) = coords(3) + normals(3) + params(3)
    # surface_target: (B, N, 4) = pressure(1) + wall_shear(3)
    batches = []
    for _ in range(n_samples):
        x = torch.randn(1, n_points, 9)
        target = torch.randn(1, n_points, 4)
        batches.append({"surface_x": x, "surface_target": target})
    return batches


def _make_volume_dataloader(n_samples=2, n_points=50):
    """Create a dataloader that mimics volume field data."""
    # volume_x: (B, N, 6) = coords(3) + params(3)
    # volume_target: (B, N, 4) = velocity(3) + pressure(1)
    batches = []
    for _ in range(n_samples):
        x = torch.randn(1, n_points, 6)
        target = torch.randn(1, n_points, 4)
        batches.append({"volume_x": x, "volume_target": target})
    return batches


def _make_args(field="surface", num_tiles=1, cache_chunk_size=100, decode_chunk_size=50):
    return SimpleNamespace(
        field=field,
        num_tiles=num_tiles,
        cache_chunk_size=cache_chunk_size,
        decode_chunk_size=decode_chunk_size,
    )


# ===== Per-Quantity Splits =====


def test_quantity_splits_surface():
    """Surface splits should cover pressure(1) + wall_shear(3) = 4 channels."""
    splits = _QUANTITY_SPLITS["surface"]
    assert len(splits) == 2
    names = [s[0] for s in splits]
    assert "p_s" in names
    assert "tau" in names

    # Check channel ranges cover [0, 4) without gaps
    p_s = [s for s in splits if s[0] == "p_s"][0]
    tau = [s for s in splits if s[0] == "tau"][0]
    assert p_s[1] == 0 and p_s[2] == 1  # pressure: channel 0
    assert tau[1] == 1 and tau[2] == 4   # wall shear: channels 1-3


def test_quantity_splits_volume():
    """Volume splits should cover velocity(3) + pressure(1) = 4 channels."""
    splits = _QUANTITY_SPLITS["volume"]
    assert len(splits) == 2
    names = [s[0] for s in splits]
    assert "u" in names
    assert "p_v" in names

    u = [s for s in splits if s[0] == "u"][0]
    p_v = [s for s in splits if s[0] == "p_v"][0]
    assert u[1] == 0 and u[2] == 3    # velocity: channels 0-2
    assert p_v[1] == 3 and p_v[2] == 4  # pressure: channel 3


def test_quantity_splits_unknown_field():
    """Unknown fields should return empty splits (graceful degradation)."""
    assert _QUANTITY_SPLITS.get("both", []) == []
    assert _QUANTITY_SPLITS.get("nonexistent", []) == []


# ===== evaluate() Return Format =====


def test_evaluate_returns_dict_surface():
    """evaluate() must return a dict with 'aggregate' + per-quantity keys for surface."""
    torch.manual_seed(42)
    model = _make_model(space_dim=9, out_dim=4)
    dataloader = _make_surface_dataloader(n_samples=2, n_points=50)
    args = _make_args(field="surface")

    results = evaluate(model, dataloader, args, device=torch.device("cpu"))

    assert isinstance(results, dict)
    assert "aggregate" in results
    assert "p_s" in results
    assert "tau" in results
    assert isinstance(results["aggregate"], float)
    assert results["aggregate"] > 0
    assert results["p_s"] > 0
    assert results["tau"] > 0


def test_evaluate_returns_dict_volume():
    """evaluate() must return a dict with 'aggregate' + per-quantity keys for volume."""
    torch.manual_seed(42)
    model = _make_model(space_dim=6, out_dim=4)
    dataloader = _make_volume_dataloader(n_samples=2, n_points=50)
    args = _make_args(field="volume")

    results = evaluate(model, dataloader, args, device=torch.device("cpu"))

    assert isinstance(results, dict)
    assert "aggregate" in results
    assert "u" in results
    assert "p_v" in results


def test_evaluate_empty_dataloader():
    """evaluate() with no data should return inf."""
    model = _make_model()
    args = _make_args(field="surface")

    results = evaluate(model, [], args, device=torch.device("cpu"))

    assert results["aggregate"] == float("inf")


def test_evaluate_with_target_normalizer():
    """evaluate() should decode predictions when normalizer is provided."""
    torch.manual_seed(42)
    model = _make_model(space_dim=9, out_dim=4)
    dataloader = _make_surface_dataloader(n_samples=2, n_points=50)
    args = _make_args(field="surface")

    # Fit a normalizer with known stats
    normalizer = TargetNormalizer(out_dim=4)
    targets = torch.randn(10, 50, 4) * 3.0 + 5.0
    normalizer.fit(targets)

    # Should not crash, and results should differ from un-normalized eval
    results_with = evaluate(model, dataloader, args, device=torch.device("cpu"),
                            target_normalizer=normalizer)
    results_without = evaluate(model, dataloader, args, device=torch.device("cpu"))

    assert isinstance(results_with, dict)
    assert "aggregate" in results_with
    # With normalizer decoding, predictions are scaled so errors differ
    assert results_with["aggregate"] != results_without["aggregate"]


# ===== get_field_key =====


def test_get_field_key():
    """get_field_key should return correct x/target key pairs."""
    assert get_field_key("surface") == ("surface_x", "surface_target")
    assert get_field_key("volume") == ("volume_x", "volume_target")


# ===== Scheduler with accumulation_steps =====


def test_scheduler_total_steps_with_accumulation():
    """Verify correct total_steps calculation for gradient accumulation.

    total_steps should count optimizer steps, not backward passes.
    With accumulation_steps=4 and 100 batches, there are 25 optimizer steps.
    """
    import torch.nn as nn

    model = nn.Linear(10, 10)
    optimizer = create_optimizer(model, lr=1e-3)

    n_batches = 100
    accumulation_steps = 4
    epochs = 2

    # This matches the formula in exp_drivaer_ml_distributed.py
    steps_per_epoch = (n_batches + accumulation_steps - 1) // accumulation_steps
    total_steps = epochs * steps_per_epoch

    assert steps_per_epoch == 25, f"Expected 25, got {steps_per_epoch}"
    assert total_steps == 50, f"Expected 50, got {total_steps}"

    scheduler = create_scheduler(optimizer, total_steps, warmup_fraction=0.1)

    # Step through the full schedule — LR should reach near min_lr at the end
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()

    final_lr = optimizer.param_groups[0]["lr"]
    # min_lr default is 1e-6
    assert final_lr < 1e-4, f"Final LR {final_lr} too high — schedule may be miscounted"


def test_scheduler_accumulation_vs_no_accumulation():
    """With accumulation_steps=1, total_steps should equal epochs * batches."""
    n_batches = 100
    accumulation_steps = 1
    epochs = 5

    steps_per_epoch = (n_batches + accumulation_steps - 1) // accumulation_steps
    total_steps = epochs * steps_per_epoch

    assert steps_per_epoch == n_batches
    assert total_steps == epochs * n_batches
