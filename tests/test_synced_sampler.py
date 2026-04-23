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
Tests for SyncedSampler.

SyncedSampler replaces DistributedSampler in mesh-sharded DDP training.
All ranks must produce the *same* shuffled sample order per epoch so that
each optimizer step all-reduces spatial-shard gradients from the same geometry.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transolver3.samplers import SyncedSampler


class _FakeDataset:
    """Minimal dataset stub — SyncedSampler only needs len()."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def test_len_matches_dataset():
    """__len__ must equal len(dataset)."""
    ds = _FakeDataset(400)
    sampler = SyncedSampler(ds, seed=42)
    assert len(sampler) == 400


def test_full_permutation():
    """Every index appears exactly once — no drops, no duplicates."""
    n = 400
    sampler = SyncedSampler(_FakeDataset(n), seed=42)
    indices = list(sampler)
    assert sorted(indices) == list(range(n)), "Not a full permutation"


def test_same_order_on_all_simulated_ranks():
    """Core mesh-sharding property: identical indices on every rank.

    Simulated by constructing two independent sampler instances with the same
    seed and epoch — equivalent to two ranks initialising independently.
    """
    ds = _FakeDataset(400)
    rank0 = SyncedSampler(ds, seed=7)
    rank1 = SyncedSampler(ds, seed=7)

    for epoch in range(3):
        rank0.set_epoch(epoch)
        rank1.set_epoch(epoch)
        assert list(rank0) == list(rank1), f"Ranks diverged at epoch {epoch}"


def test_different_epochs_give_different_orders():
    """set_epoch must actually change the shuffle."""
    sampler = SyncedSampler(_FakeDataset(400), seed=42)

    sampler.set_epoch(0)
    order_0 = list(sampler)
    sampler.set_epoch(1)
    order_1 = list(sampler)
    sampler.set_epoch(2)
    order_2 = list(sampler)

    assert order_0 != order_1, "Epoch 0 and 1 produced identical order"
    assert order_1 != order_2, "Epoch 1 and 2 produced identical order"
    assert order_0 != order_2, "Epoch 0 and 2 produced identical order"


def test_deterministic_within_epoch():
    """Iterating twice at the same epoch must yield the same sequence."""
    sampler = SyncedSampler(_FakeDataset(200), seed=42)
    sampler.set_epoch(5)
    first = list(sampler)
    second = list(sampler)
    assert first == second, "Sampler is not deterministic within an epoch"


def test_default_epoch_is_zero():
    """Freshly constructed sampler should behave as epoch=0."""
    ds = _FakeDataset(100)
    sampler_default = SyncedSampler(ds, seed=42)
    sampler_explicit = SyncedSampler(ds, seed=42)
    sampler_explicit.set_epoch(0)
    assert list(sampler_default) == list(sampler_explicit)


def test_different_seeds_give_different_orders():
    """Seed parameter must actually influence the shuffle."""
    ds = _FakeDataset(400)
    s1 = SyncedSampler(ds, seed=1)
    s2 = SyncedSampler(ds, seed=2)
    s1.set_epoch(0)
    s2.set_epoch(0)
    assert list(s1) != list(s2), "Different seeds produced identical order"


def test_step_count_per_epoch():
    """With batch_size=1, steps/epoch == len(dataset) — not len/world_size.

    This is the regression guard: DistributedSampler(num_replicas=4) would
    give 100 steps; SyncedSampler must give 400.
    """
    n_train = 400
    sampler = SyncedSampler(_FakeDataset(n_train), seed=42)
    sampler.set_epoch(0)
    steps = sum(1 for _ in sampler)
    assert steps == n_train, (
        f"Expected {n_train} steps/epoch (paper-matched), got {steps}. "
        "DistributedSampler regression?"
    )


def test_small_dataset():
    """Works correctly for very small datasets (n=1, n=2)."""
    for n in [1, 2, 3]:
        sampler = SyncedSampler(_FakeDataset(n), seed=42)
        indices = list(sampler)
        assert sorted(indices) == list(range(n)), f"Failed for n={n}"
