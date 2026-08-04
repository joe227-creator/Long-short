"""Regression tests for stateful execution controls."""

import unittest

import numpy as np
import torch

from research.execution_controls import (
    apply_drawdown_breaker,
    apply_live_weight_band,
    apply_partial_adjustment,
    apply_weight_band,
    load_live_execution_controls,
)
import trade


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

    def test_trade_applies_partial_adjustment_then_weight_band(self):
        etfs = [etf for pair in trade.ETF_PAIRS for etf in pair]
        previous = {etf: 0.0 for etf in etfs}
        target = np.zeros(len(etfs), dtype=float)
        target[:2] = [0.10, -0.20]

        adjusted = trade.apply_live_execution_controls(
            target,
            [{"weights": previous}],
            {"partial_adjustment": 0.5, "weight_band": 0.06},
        )

        self.assertEqual(adjusted[0], 0.0)
        self.assertEqual(adjusted[1], -0.1)

    def test_selected_research_controls_are_loadable_for_live_path(self):
        controls = load_live_execution_controls()

        self.assertAlmostEqual(controls["uncertainty_strength"], 1.1225169591437967)
        self.assertAlmostEqual(controls["partial_adjustment"], 0.44338242523523974)
        self.assertAlmostEqual(controls["weight_band"], 0.01683775438715695)

    def test_drawdown_breaker_no_drawdown_leaves_weights_unchanged(self):
        weights = torch.tensor([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
        targets = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        scaled = apply_drawdown_breaker(weights, targets, 0.10)

        self.assertTrue(torch.allclose(scaled, weights))

    def test_drawdown_breaker_scales_down_after_loss(self):
        weights = torch.tensor([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
        targets = torch.tensor([[-0.20, 0.0], [0.0, 0.0], [0.0, 0.0]])

        scaled = apply_drawdown_breaker(weights, targets, 0.10)

        self.assertAlmostEqual(float(scaled[0, 0]), 1.0)
        self.assertAlmostEqual(float(scaled[1, 0]), 0.0)
        self.assertAlmostEqual(float(scaled[2, 0]), 0.0)

    def test_drawdown_breaker_recovers_at_new_high(self):
        weights = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        targets = torch.tensor([[-0.10, 0.0], [0.15, 0.0], [0.10, 0.0], [0.0, 0.0]])

        scaled = apply_drawdown_breaker(weights, targets, 0.20)

        self.assertAlmostEqual(float(scaled[0, 0]), 1.0)
        self.assertAlmostEqual(float(scaled[1, 0]), 0.5)
        self.assertGreater(float(scaled[2, 0]), 0.5)
        self.assertAlmostEqual(float(scaled[3, 0]), 1.0)

    def test_drawdown_breaker_zero_threshold_is_noop(self):
        weights = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        targets = torch.tensor([[-0.10, 0.0], [0.0, 0.0]])

        scaled = apply_drawdown_breaker(weights, targets, 0.0)

        self.assertTrue(torch.equal(scaled, weights))


if __name__ == "__main__":
    unittest.main()
