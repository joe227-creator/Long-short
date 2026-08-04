"""Regression tests for validation-only Optuna postprocessing."""

import unittest

import torch

from research.optuna_postprocess import _format_optuna_value, _select_dispersion


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


if __name__ == "__main__":
    unittest.main()
