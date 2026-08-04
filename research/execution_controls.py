"""Stateful target-weight controls shared by research and live execution."""

import torch


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
