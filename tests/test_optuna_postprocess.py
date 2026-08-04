"""Regression tests for validation-only Optuna postprocessing."""

import unittest

import torch

from research.optuna_postprocess import (
    _format_optuna_value,
    _select_dispersion,
    _select_signals,
)


class OptunaPostprocessTests(unittest.TestCase):
    def test_uncertainty_statistic_selects_named_dispersion(self):
        dispersion = {
            "std": torch.tensor([[1.0]]),
            "mad": torch.tensor([[0.5]]),
            "range": torch.tensor([[2.0]]),
        }
        spec = {"parameter": "UNCERTAINTY_STATISTIC"}

        selected = _select_dispersion(dispersion, spec, "mad")

        self.assertTrue(torch.equal(selected, torch.tensor([[0.5]])))

    def test_numeric_and_categorical_optuna_values_format(self):
        self.assertEqual(_format_optuna_value(0.016837754), "0.016837754")
        self.assertEqual(_format_optuna_value("mad"), "mad")

    def test_ensemble_aggregation_selects_named_signal_history(self):
        signals = {
            "trimmed_mean": torch.tensor([[1.0]]),
            "median": torch.tensor([[2.0]]),
            "mean": torch.tensor([[3.0]]),
        }
        spec = {"parameter": "ENSEMBLE_AGG"}

        selected = _select_signals(signals, spec, "median")

        self.assertTrue(torch.equal(selected, torch.tensor([[2.0]])))


if __name__ == "__main__":
    unittest.main()
