"""Regression tests for stateful execution controls."""

import unittest

import torch

from research.execution_controls import (
    apply_live_weight_band,
    apply_partial_adjustment,
    apply_weight_band,
)


class ExecutionControlTests(unittest.TestCase):
    def test_partial_adjustment_moves_from_previous_adjusted_weight(self):
        weights = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.0, 1.0]])

        adjusted = apply_partial_adjustment(weights, 0.5)

        expected = torch.tensor([[0.0, 0.0], [0.5, -0.5], [0.25, 0.25]])
        self.assertTrue(torch.equal(adjusted, expected))

    def test_weight_band_holds_small_component_changes(self):
        weights = torch.tensor([
            [0.0, 0.0],
            [0.04, 0.20],
            [0.10, 0.21],
        ])

        held = apply_weight_band(weights, 0.05)

        expected = torch.tensor([
            [0.0, 0.0],
            [0.0, 0.20],
            [0.10, 0.20],
        ])
        self.assertTrue(torch.equal(held, expected))

    def test_live_weight_band_uses_prior_target(self):
        previous = torch.tensor([0.10, -0.20])
        target = torch.tensor([0.13, -0.30])

        held = apply_live_weight_band(previous, target, 0.05)

        self.assertTrue(torch.equal(held, torch.tensor([0.10, -0.30])))

    def test_live_weight_band_without_history_returns_target(self):
        target = torch.tensor([0.13, -0.30])

        self.assertTrue(torch.equal(apply_live_weight_band(None, target, 0.05), target))


if __name__ == "__main__":
    unittest.main()
