"""Regression tests for forecast alignment and strict input validation."""

import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

import prepare
import train
from simulate import build_live_execution_targets


class ForecastAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.previous_forecast = train._VOL_FORECAST_TENSOR
        train._VOL_FORECAST_TENSOR = None

    def tearDown(self):
        train._VOL_FORECAST_TENSOR = self.previous_forecast

    def test_compute_portfolio_rejects_full_history_forecast(self):
        signals = torch.ones(2, prepare.NUM_PAIRS)
        targets = torch.ones(2, len(prepare.ETF_TICKERS))
        train._VOL_FORECAST_TENSOR = torch.tensor([0.01, 0.03, 0.04])

        with self.assertRaisesRegex(ValueError, "not aligned"):
            train.compute_portfolio(signals, targets)

    def test_compute_portfolio_uses_forecast_aligned_to_evaluation_rows(self):
        signals = torch.ones(2, prepare.NUM_PAIRS)
        targets = torch.ones(2, len(prepare.ETF_TICKERS))

        with mock.patch.object(train, "VOL_GATE", True), \
                mock.patch.object(train, "VOL_GATE_FORMULA", "timesfm"), \
                mock.patch.object(train, "VOL_GATE_THRESHOLD", 0.025), \
                mock.patch.object(train, "VOL_GATE_STRENGTH", 500.0):
            weights, _ = train.compute_portfolio(
                signals,
                targets,
                vol_forecast=torch.tensor([0.01, 0.04]),
            )

        self.assertGreater(weights[0].abs().sum().item(), weights[1].abs().sum().item())

    def test_timesfm_loader_preserves_requested_date_order(self):
        dates = pd.date_range("2022-01-03", periods=4, freq="B")
        cache = pd.DataFrame(
            {"forecast_vol": [0.01, 0.02, 0.03, 0.04]},
            index=dates,
        )

        with mock.patch.object(train.os.path, "exists", return_value=True), \
                mock.patch.object(pd, "read_parquet", return_value=cache):
            forecast = train._load_timesfm_vol_forecast(dates[2:])

        self.assertIsNotNone(forecast)
        assert forecast is not None
        np.testing.assert_allclose(forecast.detach().numpy(), [0.03, 0.04])

    def test_backward_volatility_excludes_current_period_return(self):
        quiet = torch.zeros(25)
        quiet[24] = 5.0
        vol = train._backward_volatility(quiet, 20)
        self.assertAlmostEqual(float(vol[24]), 0.0, places=6)

        prior_spike = torch.zeros(25)
        prior_spike[23] = 5.0
        vol_prior = train._backward_volatility(prior_spike, 20)
        self.assertGreater(float(vol_prior[24]), 0.0)

        early_rows = train._backward_volatility(quiet, 20)
        self.assertEqual(float(early_rows[10]), 0.0)


class StrictValidationTests(unittest.TestCase):
    def test_missing_model_feature_fails_closed(self):
        frame = pd.DataFrame({"present": [1.0]})

        with self.assertRaisesRegex(ValueError, "missing required model features"):
            prepare.validate_feature_columns(frame, ["present", "missing"])

    def test_missing_etf_source_fails_closed(self):
        dates = pd.date_range("2026-01-01", periods=2, freq="B")
        frame = pd.DataFrame({"SSO_Close": [1.0, 2.0]}, index=dates)

        with self.assertRaisesRegex(ValueError, "missing required yfinance columns"):
            prepare.validate_etf_data(frame)

    def test_missing_fred_source_fails_closed(self):
        dates = pd.date_range("2026-01-01", periods=2, freq="B")
        frame = pd.DataFrame({"DGS10": [1.0, 2.0]}, index=dates)

        with self.assertRaisesRegex(ValueError, "missing required FRED series"):
            prepare.validate_macro_data(frame)

    def test_missing_fred_api_key_fails_closed(self):
        with mock.patch.dict(os.environ, {"FRED_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "FRED_API_KEY not set"):
                prepare.download_macro_data(refresh=True)

    def test_missing_live_execution_price_fails_closed(self):
        dates = pd.date_range("2026-01-05", periods=2, freq="B")
        frame = pd.DataFrame(
            {"SSO_Open": [1.0, 2.0], "SSO_Close": [1.0, 2.0]},
            index=dates,
        )

        with self.assertRaisesRegex(ValueError, "missing required yfinance columns"):
            build_live_execution_targets(frame)


if __name__ == "__main__":
    unittest.main()
