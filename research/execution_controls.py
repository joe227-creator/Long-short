"""Stateful target-weight controls shared by research and live execution."""

import json
from pathlib import Path

import torch


def load_live_execution_controls(spec_path=None):
    """Read selected research controls without coupling them to checkpoints."""
    path = Path(spec_path) if spec_path is not None else Path(__file__).with_name("optuna_spec.json")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    controls = {}
    if spec.get("fixed_uncertainty_strength") is not None:
        controls["uncertainty_strength"] = float(spec["fixed_uncertainty_strength"])
    if spec.get("fixed_partial_adjustment") is not None:
        controls["partial_adjustment"] = float(spec["fixed_partial_adjustment"])
    if spec.get("parameter") == "WEIGHT_BAND" and spec.get("selected_value") is not None:
        controls["weight_band"] = float(spec["selected_value"])
    elif spec.get("fixed_weight_band") is not None:
        controls["weight_band"] = float(spec["fixed_weight_band"])
    return controls


def apply_partial_adjustment(weights, rate):
    """Move each target toward its prior held weight by ``rate``."""
    rate = float(rate)
    if rate <= 0 or len(weights) <= 1:
        return weights
    if not 0 <= rate <= 1:
        raise ValueError(f"Partial-adjustment rate must be in [0, 1], got {rate}")

    adjusted = weights.clone()
    for row in range(1, len(adjusted)):
        adjusted[row] = adjusted[row - 1] + rate * (
            weights[row] - adjusted[row - 1]
        )
    return adjusted


def apply_state_band(values, band):
    """Hold each value until its change from the held state reaches ``band``."""
    band = float(band)
    if band <= 0 or len(values) <= 1:
        return values
    if values.ndim != 2:
        raise ValueError("State history must be rank-2")

    held = [values[0]]
    previous = values[0]
    for current in values[1:]:
        previous = torch.where(
            torch.abs(current - previous) >= band,
            current,
            previous,
        )
        held.append(previous)
    return torch.stack(held, dim=0)


def apply_weight_band(weights, band):
    """Hold each ETF target until its change reaches ``band``."""
    return apply_state_band(weights, band)


def apply_asymmetric_weight_band(weights, enter_band, exit_band):
    """Hold targets with asymmetric bands by position direction.

    Widens the retention band when a component is moving away from zero
    (entering/increasing) and narrows it when moving toward zero (exiting),
    per arXiv 2607.25258. ``enter_band >= exit_band`` is expected.
    """
    enter_band = float(enter_band)
    exit_band = float(exit_band)
    if (enter_band <= 0 and exit_band <= 0) or len(weights) <= 1:
        return weights
    if weights.ndim != 2:
        raise ValueError("State history must be rank-2")

    held = [weights[0]]
    previous = weights[0]
    for current in weights[1:]:
        reducing = current.abs() < previous.abs()
        band = torch.where(reducing, exit_band, enter_band)
        previous = torch.where(
            (current - previous).abs() >= band,
            current,
            previous,
        )
        held.append(previous)
    return torch.stack(held, dim=0)


def apply_live_weight_band(previous_weights, target_weights, band):
    """Retain prior live targets for ETF changes smaller than ``band``."""
    band = float(band)
    if previous_weights is None or band <= 0:
        return target_weights
    if previous_weights.shape != target_weights.shape:
        raise ValueError("Previous and target weights must have identical shapes")
    return torch.where(
        torch.abs(target_weights - previous_weights) >= band,
        target_weights,
        previous_weights,
    )
