# Copyright 2024 Databricks, Inc.
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
Tests for training utilities: sampler, scheduler, normalizers, and train_step.
"""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transolver3.model import Transolver3
from transolver3.amortized_training import (
    AmortizedMeshSampler,
    create_optimizer,
    create_scheduler,
    train_step,
)
from transolver3.normalizer import InputNormalizer, TargetNormalizer


# ===== Sampler & Scheduler =====


def test_sampler():
    """Test AmortizedMeshSampler."""
    sampler = AmortizedMeshSampler(subset_size=100, seed=42)

    indices = sampler.sample(1000)
    assert indices.shape == (100,), f"Expected (100,), got {indices.shape}"
    assert indices.max() < 1000
    assert indices.min() >= 0

    indices_small = sampler.sample(50)
    assert indices_small.shape == (50,)


def test_scheduler():
    """Test cosine scheduler with warmup."""
    model = nn.Linear(10, 10)
    optimizer = create_optimizer(model, lr=1e-3)
    scheduler = create_scheduler(optimizer, total_steps=1000, warmup_fraction=0.1)

    lrs = []
    for step in range(1000):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[50] > lrs[0], "LR should increase during warmup"
    assert lrs[500] < lrs[100], "LR should decrease after warmup"
    assert lrs[-1] < lrs[100], "Final LR should be lower than mid-training"


# ===== Target Normalizer =====


def test_target_normalizer_fit():
    """Test TargetNormalizer fit computes correct mean/std and encode/decode are inverse."""
    torch.manual_seed(42)
    num_samples, N, out_dim = 100, 50, 3
    targets = torch.randn(num_samples, N, out_dim) * 3.0 + 10.0

    normalizer = TargetNormalizer()
    normalizer.fit(targets)

    assert (normalizer.mean - 10.0).abs().max().item() < 0.5
    assert (normalizer.std - 3.0).abs().max().item() < 0.5

    encoded = normalizer.encode(targets)
    enc_mean = encoded.mean(dim=(0, 1))
    enc_std = encoded.std(dim=(0, 1))
    assert enc_mean.abs().max().item() < 0.1
    assert (enc_std - 1.0).abs().max().item() < 0.1

    decoded = normalizer.decode(encoded)
    diff = (decoded - targets).abs().max().item()
    assert diff < 1e-5, f"decode(encode(x)) != x, max diff: {diff}"


def test_target_normalizer_incremental():
    """Test incremental fitting matches full fit."""
    torch.manual_seed(42)
    num_samples, N, out_dim = 200, 50, 3
    targets = torch.randn(num_samples, N, out_dim) * 5.0 - 2.0

    norm_full = TargetNormalizer()
    norm_full.fit(targets)

    norm_inc = TargetNormalizer()
    batches = [targets[i : i + 20] for i in range(0, num_samples, 20)]
    norm_inc.fit_incremental(iter(batches))

    mean_diff = (norm_full.mean - norm_inc.mean).abs().max().item()
    std_diff = (norm_full.std - norm_inc.std).abs().max().item()

    assert mean_diff < 1e-3, f"Incremental mean diff: {mean_diff}"
    assert std_diff < 1e-3, f"Incremental std diff: {std_diff}"


def test_target_normalizer_2d_input():
    """Test TargetNormalizer works with 2D input (N, out_dim)."""
    targets = torch.randn(100, 3) * 2.0 + 5.0
    normalizer = TargetNormalizer()
    normalizer.fit(targets)

    encoded = normalizer.encode(targets)
    decoded = normalizer.decode(encoded)
    diff = (decoded - targets).abs().max().item()
    assert diff < 1e-5, f"2D roundtrip diff: {diff}"


def test_target_normalizer_state_dict():
    """Test TargetNormalizer saves/loads via state_dict (nn.Module)."""
    torch.manual_seed(42)
    targets = torch.randn(50, 30, 2) * 4.0 + 3.0

    norm1 = TargetNormalizer()
    norm1.fit(targets)

    state = norm1.state_dict()
    norm2 = TargetNormalizer(out_dim=2)
    norm2.load_state_dict(state)

    assert (norm1.mean - norm2.mean).abs().max().item() < 1e-7
    assert (norm1.std - norm2.std).abs().max().item() < 1e-7
    assert norm2.fitted.item() is True

    encoded1 = norm1.encode(targets)
    encoded2 = norm2.encode(targets)
    assert (encoded1 - encoded2).abs().max().item() < 1e-7


def test_target_normalizer_device():
    """Test TargetNormalizer follows model.to(device)."""
    targets = torch.randn(20, 10, 2)
    normalizer = TargetNormalizer()
    normalizer.fit(targets)

    encoded = normalizer.encode(targets)
    assert encoded.device.type == "cpu"

    normalizer.cpu()
    assert normalizer.mean.device.type == "cpu"


# ===== Input Normalizer =====


def test_input_normalizer_per_sample():
    """Test InputNormalizer per-sample mode (default)."""
    B, N, D = 2, 50, 3
    coords = torch.randn(B, N, D) * 100.0 + 500.0

    normalizer = InputNormalizer(scale=1000.0, per_sample=True)
    encoded = normalizer.encode(coords)

    for b in range(B):
        assert encoded[b].min().item() >= -1e-5, f"Sample {b} min={encoded[b].min().item()}, expected >= 0"
        assert encoded[b].max().item() <= 1000.0 + 1e-5, f"Sample {b} max={encoded[b].max().item()}, expected <= 1000"

    for b in range(B):
        per_ch_min = encoded[b].min(dim=0).values
        per_ch_max = encoded[b].max(dim=0).values
        assert per_ch_min.min().item() < 1.0, "Some channel min should be near 0"
        assert per_ch_max.max().item() > 999.0, "Some channel max should be near 1000"


def test_input_normalizer_dataset_level():
    """Test InputNormalizer dataset-level mode with fit/encode/decode roundtrip."""
    torch.manual_seed(42)
    num_samples, N, D = 20, 50, 3
    coords = torch.randn(num_samples, N, D) * 50.0 + 200.0

    normalizer = InputNormalizer(scale=1000.0, per_sample=False)
    normalizer.fit(coords)

    assert normalizer.fitted.item() is True
    assert normalizer.data_min.shape == (1, 1, D)
    assert normalizer.data_max.shape == (1, 1, D)

    encoded = normalizer.encode(coords)
    assert encoded.min().item() >= -1e-3, f"Min={encoded.min().item()}"
    assert encoded.max().item() <= 1000.0 + 1e-3, f"Max={encoded.max().item()}"

    decoded = normalizer.decode(encoded)
    diff = (decoded - coords).abs().max().item()
    assert diff < 1e-3, f"Roundtrip diff: {diff}"


def test_input_normalizer_incremental():
    """Test InputNormalizer incremental fitting matches full fit."""
    torch.manual_seed(42)
    num_samples, N, D = 100, 30, 3
    coords = torch.randn(num_samples, N, D) * 10.0

    norm_full = InputNormalizer(scale=1.0, per_sample=False)
    norm_full.fit(coords)

    norm_inc = InputNormalizer(scale=1.0, per_sample=False)
    batches = [coords[i : i + 10] for i in range(0, num_samples, 10)]
    norm_inc.fit_incremental(iter(batches))

    min_diff = (norm_full.data_min - norm_inc.data_min).abs().max().item()
    max_diff = (norm_full.data_max - norm_inc.data_max).abs().max().item()

    assert min_diff < 1e-6, f"Incremental min diff: {min_diff}"
    assert max_diff < 1e-6, f"Incremental max diff: {max_diff}"


def test_input_normalizer_scale_factor():
    """Test different scaling factors."""
    coords = torch.tensor([[[0.0, 10.0], [5.0, 20.0]]])  # B=1, N=2, D=2

    for scale in [1.0, 100.0, 1000.0]:
        norm = InputNormalizer(scale=scale)
        encoded = norm.encode(coords)
        assert encoded.min().item() < 1e-5
        assert abs(encoded.max().item() - scale) < 1e-3, f"scale={scale}: max={encoded.max().item()}"


def test_input_normalizer_2d():
    """Test InputNormalizer with 2D input (N, D)."""
    coords = torch.randn(50, 3) * 100.0

    normalizer = InputNormalizer(scale=1000.0)
    encoded = normalizer.encode(coords)
    assert encoded.shape == (50, 3)
    assert encoded.min().item() >= -1e-5
    assert encoded.max().item() <= 1000.0 + 1e-5


def test_input_normalizer_decode_per_sample_raises():
    """Test that decode raises in per_sample mode."""
    normalizer = InputNormalizer(per_sample=True)
    try:
        normalizer.decode(torch.randn(2, 10, 3))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "per_sample" in str(e)


# ===== Train Step =====


def test_train_step_with_normalizer():
    """Test train_step integrates with TargetNormalizer."""
    B, N = 1, 100
    space_dim = 3
    out_dim = 2

    model = Transolver3(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=8,
        mlp_ratio=1,
    )

    x = torch.randn(B, N, space_dim)
    target = torch.randn(B, N, out_dim) * 5.0 + 10.0

    normalizer = TargetNormalizer()
    normalizer.fit(target)

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, total_steps=10)

    loss_with = train_step(model, x, None, target, optimizer, scheduler, normalizer=normalizer)
    loss_without = train_step(model, x, None, target, optimizer, scheduler, normalizer=None)

    assert isinstance(loss_with, float) and loss_with > 0
    assert isinstance(loss_without, float) and loss_without > 0


def test_train_step_mixed_precision():
    """Test train_step with GradScaler for mixed precision training."""
    B, N = 1, 100
    space_dim = 3
    out_dim = 2

    model = Transolver3(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=8,
        mlp_ratio=1,
    )

    x = torch.randn(B, N, space_dim)
    target = torch.randn(B, N, out_dim)

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, total_steps=10)

    device_type = next(model.parameters()).device.type
    scaler = torch.amp.GradScaler(device=device_type, enabled=(device_type == "cuda"))

    losses = []
    for _ in range(3):
        loss = train_step(model, x, None, target, optimizer, scheduler, scaler=scaler)
        losses.append(loss)

    assert all(isinstance(v, float) and v > 0 for v in losses)

    loss_no_scaler = train_step(model, x, None, target, optimizer, scheduler)
    assert isinstance(loss_no_scaler, float) and loss_no_scaler > 0
