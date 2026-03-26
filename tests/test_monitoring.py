"""Tests for transolver3.monitoring — prediction bounds and drift detection."""

import torch

from transolver3.monitoring import check_prediction_bounds


def test_check_prediction_bounds_all_valid():
    """All predictions within bounds returns all_valid=True."""
    predictions = torch.tensor([[[0.5, 100.0], [0.3, 200.0]]])  # (1, 2, 2)
    bounds = {0: (0.0, 1.0), 1: (0.0, 400.0)}

    result = check_prediction_bounds(predictions, bounds)
    assert result["all_valid"] is True
    assert result["total_out_of_bounds"] == 0
    assert result["channels"][0]["out_of_bounds_count"] == 0
    assert result["channels"][1]["out_of_bounds_count"] == 0


def test_check_prediction_bounds_some_invalid():
    """Out-of-bounds predictions are detected per channel."""
    predictions = torch.tensor([[[1.5, 100.0], [-0.5, 500.0]]])  # (1, 2, 2)
    bounds = {0: (0.0, 1.0), 1: (0.0, 400.0)}

    result = check_prediction_bounds(predictions, bounds)
    assert result["all_valid"] is False
    assert result["total_out_of_bounds"] == 3  # 1.5 above, -0.5 below, 500 above

    ch0 = result["channels"][0]
    assert ch0["out_of_bounds_count"] == 2  # 1.5 above, -0.5 below
    assert ch0["above_max"] == 1
    assert ch0["below_min"] == 1

    ch1 = result["channels"][1]
    assert ch1["out_of_bounds_count"] == 1  # 500 above


def test_check_prediction_bounds_2d_input():
    """2D input (N, out_dim) is handled by adding batch dim."""
    predictions = torch.tensor([[0.5, 100.0], [0.3, 200.0]])  # (2, 2)
    bounds = {0: (0.0, 1.0), 1: (0.0, 400.0)}

    result = check_prediction_bounds(predictions, bounds)
    assert result["all_valid"] is True


def test_check_prediction_bounds_min_max_values():
    """Min/max values are reported per channel."""
    predictions = torch.tensor([[[10.0], [-5.0], [3.0]]])  # (1, 3, 1)
    bounds = {0: (-100.0, 100.0)}

    result = check_prediction_bounds(predictions, bounds)
    assert result["channels"][0]["min_value"] == -5.0
    assert result["channels"][0]["max_value"] == 10.0


def test_check_prediction_bounds_empty_bounds():
    """Empty bounds dict returns valid with no channel info."""
    predictions = torch.randn(2, 50, 4)
    result = check_prediction_bounds(predictions, {})
    assert result["all_valid"] is True
    assert result["total_out_of_bounds"] == 0
