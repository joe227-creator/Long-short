"""Regression tests for production checkpoint discovery and live loading."""

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import torch

import inference
import production_model
import trade


BASE_CONFIG = {
    "model_type": "lstm",
    "data_split_version": 2,
    "trade_frequency": "weekly",
    "seq_len": 90,
    "feature_columns": ["feature_a"],
    "scaler_params": {"mean": {"feature_a": 0.0}, "std": {"feature_a": 1.0}},
    "hidden_dim": 384,
    "num_layers": 2,
    "dropout": 0.18,
    "use_vsn": True,
    "vsn_hidden": 64,
    "vsn_residual": True,
    "vsn_learnable_alpha": False,
    "vsn_alpha": 0.5,
}


class DummyModel:
    def __init__(self, path):
        self.path = path
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, _x):
        return torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32)


class VaryingDummyModel(DummyModel):
    def __init__(self, signal):
        super().__init__(signal)
        self.signal = signal

    def __call__(self, _x):
        return torch.tensor(
            [[self.signal, -self.signal, self.signal, -self.signal]],
            dtype=torch.float32,
        )


def _touch(path):
    with open(path, "wb"):
        pass


class ProductionEnsembleLoadingTests(unittest.TestCase):
    def test_trade_load_ensemble_discovers_saved_seed_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _touch(os.path.join(tmpdir, "best_model_seed7.pt"))
            _touch(os.path.join(tmpdir, "best_model_seed42.pt"))

            loaded = []

            def fake_load_checkpoint(path, device):
                loaded.append(os.path.basename(path))
                return DummyModel(path), dict(BASE_CONFIG)

            with mock.patch.object(trade, "MODELS_DIR", tmpdir), \
                 mock.patch.object(trade, "load_checkpoint", side_effect=fake_load_checkpoint):
                models, config = trade.load_ensemble(device="cpu")

        self.assertEqual(loaded, ["best_model_seed7.pt", "best_model_seed42.pt"])
        self.assertEqual(len(models), 2)
        self.assertEqual(config["trade_frequency"], "weekly")

    def test_inference_load_ensemble_discovers_saved_seed_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _touch(os.path.join(tmpdir, "best_model_seed7.pt"))
            _touch(os.path.join(tmpdir, "best_model_seed42.pt"))

            loaded = []

            def fake_load_checkpoint(path, device):
                loaded.append(os.path.basename(path))
                return DummyModel(path), dict(BASE_CONFIG)

            with mock.patch.object(inference, "MODELS_DIR", tmpdir), \
                 mock.patch.object(inference, "load_checkpoint", side_effect=fake_load_checkpoint):
                models, config = inference.load_ensemble(device="cpu")

        self.assertEqual(loaded, ["best_model_seed7.pt", "best_model_seed42.pt"])
        self.assertEqual(len(models), 2)
        self.assertEqual(config["trade_frequency"], "weekly")

    def test_trade_load_ensemble_rejects_mixed_checkpoint_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _touch(os.path.join(tmpdir, "best_model_seed7.pt"))
            _touch(os.path.join(tmpdir, "best_model_seed42.pt"))

            def fake_load_checkpoint(path, device):
                config = dict(BASE_CONFIG)
                if path.endswith("best_model_seed42.pt"):
                    config["trade_frequency"] = "daily"
                return DummyModel(path), config

            with mock.patch.object(trade, "MODELS_DIR", tmpdir), \
                 mock.patch.object(trade, "load_checkpoint", side_effect=fake_load_checkpoint):
                with self.assertRaises(ValueError):
                    trade.load_ensemble(device="cpu")

    def test_trade_load_ensemble_rejects_checkpoint_without_fixed_split_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _touch(os.path.join(tmpdir, "best_model_seed7.pt"))

            def fake_load_checkpoint(path, device):
                config = dict(BASE_CONFIG)
                config.pop("data_split_version")
                return DummyModel(path), config

            with mock.patch.object(trade, "MODELS_DIR", tmpdir), \
                 mock.patch.object(trade, "load_checkpoint", side_effect=fake_load_checkpoint):
                with self.assertRaises(ValueError):
                    trade.load_ensemble(device="cpu")

    def test_run_live_ensemble_reports_weekly_signal_date_not_daily_tail(self):
        dates = pd.bdate_range("2026-01-01", periods=12)
        feat_df = pd.DataFrame({"feature_a": range(len(dates))}, index=dates)
        config = dict(BASE_CONFIG)
        config["seq_len"] = 3
        models = [DummyModel("a"), DummyModel("b"), DummyModel("c")]

        def fake_compute_portfolio(_signals, _targets):
            return torch.zeros(1, 8), torch.zeros(1)

        with mock.patch.object(production_model, "_load_timesfm_features", return_value=None), \
             mock.patch.object(production_model, "_load_timesfm_vol_forecast", return_value=None), \
             mock.patch.object(production_model, "compute_portfolio", side_effect=fake_compute_portfolio), \
             mock.patch.object(production_model, "SIGNAL_EMA_DECAY", 0.0):
            decision = production_model.run_live_ensemble(models, config, feat_df, device="cpu")

        # Weekly mode uses feat_df.iloc[::5] (same phase the model was trained
        # on), so the signal is based on Jan 15, not the newest daily row Jan 16.
        # Shifting the grid to the newest date degrades the signal (backtest
        # CAGR 50.75% -> 32.55%), so the trained grid phase must be preserved.
        self.assertEqual(decision["latest_date"], "2026-01-15")

    def test_run_live_ensemble_scales_disagreement_before_weight_conversion(self):
        dates = pd.bdate_range("2026-01-01", periods=12)
        feat_df = pd.DataFrame({"feature_a": range(len(dates))}, index=dates)
        config = dict(BASE_CONFIG)
        config["seq_len"] = 3
        models = [
            VaryingDummyModel(1.0),
            VaryingDummyModel(2.0),
            VaryingDummyModel(3.0),
            VaryingDummyModel(4.0),
        ]
        captured = {}

        def fake_compute_portfolio(signals, _targets):
            captured["signals"] = signals.clone()
            return torch.zeros(1, 8), torch.zeros(1)

        with mock.patch.object(production_model, "_load_timesfm_features", return_value=None), \
             mock.patch.object(production_model, "_load_timesfm_vol_forecast", return_value=None), \
             mock.patch.object(production_model, "compute_portfolio", side_effect=fake_compute_portfolio), \
             mock.patch.object(production_model, "load_live_execution_controls", return_value={"uncertainty_strength": 1.0}), \
             mock.patch.object(production_model, "SIGNAL_EMA_DECAY", 0.0), \
             mock.patch.object(production_model, "SIGNAL_CLIP", 0.0), \
             mock.patch.object(production_model, "SIGNAL_THRESHOLD", 0.0):
            decision = production_model.run_live_ensemble(models, config, feat_df, device="cpu")

        self.assertGreater(float(decision["dispersion"][0]), 0.0)
        self.assertLess(abs(float(captured["signals"][0, 0])), 2.0)


if __name__ == "__main__":
    unittest.main()
