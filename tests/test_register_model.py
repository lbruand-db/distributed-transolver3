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

"""Tests for register_model.py: model card and experiment comparison."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from register_model import format_model_card, compute_metric_deltas


# ===== format_model_card =====


def test_model_card_basic():
    """Model card should contain key sections."""
    params = {
        "field": "surface",
        "n_layers": "24",
        "n_hidden": "256",
        "n_head": "8",
        "slice_num": "64",
        "epochs": "500",
        "lr": "0.001",
        "weight_decay": "0.05",
        "seed": "42",
    }
    metrics = {"best_test_l2": 0.0371}

    card = format_model_card(params, metrics, run_id="abc123")

    assert "# Transolver-3 (surface)" in card
    assert "## Performance" in card
    assert "## Architecture" in card
    assert "## Training Configuration" in card
    assert "## Limitations" in card
    assert "abc123" in card
    assert "0.0371" in card
    assert "3.71%" in card


def test_model_card_per_quantity_metrics():
    """Per-quantity metrics should appear when present."""
    params = {"field": "surface"}
    metrics = {
        "best_test_l2": 0.05,
        "test_l2_p_s": 0.0371,
        "test_l2_tau": 0.0585,
    }

    card = format_model_card(params, metrics)

    assert "p_s" in card
    assert "tau" in card
    assert "3.71%" in card
    assert "5.85%" in card


def test_model_card_no_per_quantity():
    """Model card should work without per-quantity metrics."""
    params = {"field": "volume"}
    metrics = {"best_test_l2": 0.04}

    card = format_model_card(params, metrics)

    assert "# Transolver-3 (volume)" in card
    assert "test_l2_p_s" not in card  # should not appear


def test_model_card_missing_params():
    """Missing params should show '?' not crash."""
    card = format_model_card({}, {})

    assert "unknown" in card  # field defaults to "unknown"
    assert "?" in card  # missing params show ?


def test_model_card_volume_metrics():
    """Volume per-quantity metrics should appear."""
    params = {"field": "volume"}
    metrics = {
        "best_test_l2": 0.05,
        "test_l2_u": 0.0414,
        "test_l2_p_v": 0.0572,
    }

    card = format_model_card(params, metrics)

    assert "**u**" in card
    assert "**p_v**" in card


# ===== compute_metric_deltas =====


def test_deltas_improvement():
    """Negative delta means improvement."""
    current = {"best_test_l2": 0.035}
    previous = {"best_test_l2": 0.050}

    deltas = compute_metric_deltas(current, previous)

    assert "aggregate" in deltas
    assert deltas["aggregate"] < 0  # improved
    assert abs(deltas["aggregate"] - (-0.015)) < 1e-9


def test_deltas_regression():
    """Positive delta means regression."""
    current = {"best_test_l2": 0.060}
    previous = {"best_test_l2": 0.050}

    deltas = compute_metric_deltas(current, previous)

    assert deltas["aggregate"] > 0  # regressed
    assert abs(deltas["aggregate"] - 0.010) < 1e-9


def test_deltas_per_quantity():
    """Per-quantity deltas should be computed when both runs have them."""
    current = {
        "best_test_l2": 0.04,
        "test_l2_p_s": 0.035,
        "test_l2_tau": 0.055,
    }
    previous = {
        "best_test_l2": 0.05,
        "test_l2_p_s": 0.040,
        "test_l2_tau": 0.060,
    }

    deltas = compute_metric_deltas(current, previous)

    assert "test_l2_p_s" in deltas
    assert "test_l2_tau" in deltas
    assert deltas["test_l2_p_s"] < 0  # improved
    assert deltas["test_l2_tau"] < 0  # improved


def test_deltas_missing_per_quantity():
    """Per-quantity deltas should be omitted if not in both runs."""
    current = {"best_test_l2": 0.04, "test_l2_p_s": 0.035}
    previous = {"best_test_l2": 0.05}  # no per-quantity

    deltas = compute_metric_deltas(current, previous)

    assert "aggregate" in deltas
    assert "test_l2_p_s" not in deltas  # previous doesn't have it


def test_deltas_no_best_test_l2():
    """Should return empty dict if best_test_l2 is missing."""
    deltas = compute_metric_deltas({}, {"best_test_l2": 0.05})
    assert deltas == {}

    deltas = compute_metric_deltas({"best_test_l2": 0.04}, {})
    assert deltas == {}


def test_deltas_identical_runs():
    """Identical metrics should produce zero deltas."""
    metrics = {"best_test_l2": 0.05, "test_l2_p_s": 0.04}

    deltas = compute_metric_deltas(metrics, metrics)

    assert deltas["aggregate"] == 0
    assert deltas["test_l2_p_s"] == 0
