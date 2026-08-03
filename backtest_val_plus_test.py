"""Backtest over a custom date window using the EXACT same pipeline as backtest.py.

Mirrors backtest.run_backtest() (same imports, same compute_portfolio, same signal
processing) but lets the split window be set via env vars instead of being fixed to
the val (2022-2024) or test (2024+) split. Used here to evaluate 2020-01-01 -> now,
i.e. early-stop val (2020-2021) + benchmark val (2022-2024) + test (2024+).

CAVEAT: the 2020-2021 early-stop window was used for checkpoint selection during
training, so it is NOT strictly out-of-sample. Treat the 2020->now aggregate as a
diagnostic, not a clean held-out estimate. The benchmark val (2022-2024) and test
(2024+) portions remain out-of-sample.

Usage:
    ARC_SPLIT_START=2020-01-01 ARC_SPLIT_END=2026-07-02 python backtest_val_plus_test.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

from prepare import (
    NUM_PAIRS, ETF_PAIRS, PAIR_NAMES, VAL_END, TRAIN_END,
    build_dataset, normalize_features, signals_to_weights, validate_feature_columns,
)
from train import (
    load_checkpoint, ENSEMBLE_SEEDS, SEQ_LEN,
    USE_TIMESFM_VOL, TIMESFM_DIM, TIMESFM_BLEND_ALPHA, TIMESFM_SIGNAL_SCALE,
    SIGNAL_POWER, SIGNAL_CLIP, ENSEMBLE_AGG, SIGNAL_EMA_DECAY, SIGNAL_THRESHOLD,
    LONG_ONLY, CASH_ENABLED, compute_portfolio,
    _load_timesfm_vol_forecast,
)
import train as _train_module
from production_model import _aggregate_ensemble, apply_signal_pipeline, load_timesfm_features_by_date
from backtest import load_ensemble, compute_metrics

MODELS_DIR = "models"


def run_window(split_start, split_end, split_label):
    """Run the backtest over [split_start, split_end) using backtest.py's pipeline."""
    device = torch.device("cpu")

    print("=" * 60)
    print(f"  Window Backtest — {split_label}")
    print(f"  Period: {split_start} -> {split_end}")
    print("=" * 60)

    models, config = load_ensemble(device)
    feature_columns = config["feature_columns"]
    trade_frequency = config.get("trade_frequency", "weekly")
    seq_len = config.get("seq_len", SEQ_LEN)

    print(f"\nLoading data (frequency={trade_frequency})...")
    features_df, targets_df = build_dataset(trade_frequency=trade_frequency)

    features_df = validate_feature_columns(
        features_df, feature_columns, context=f"{split_label} features"
    )

    scaler_params = config["scaler_params"]
    mean = pd.Series(scaler_params["mean"])
    std = pd.Series(scaler_params["std"])
    feat_norm = normalize_features(features_df, mean, std)

    # Weekly subsampling (matches make_dataloaders + backtest.py)
    if trade_frequency == "weekly":
        weekly_idx = feat_norm.index[::5]
        feat_norm = feat_norm.loc[weekly_idx]
        targets_df = targets_df.loc[weekly_idx]

    # Custom split window
    mask = (feat_norm.index >= split_start) & (feat_norm.index < split_end)
    split_indices = np.where(mask)[0]

    if len(split_indices) <= seq_len:
        print(f"ERROR: Not enough data ({len(split_indices)} rows, need >{seq_len})")
        return None

    split_start_idx = split_indices[0]
    if split_start_idx < seq_len - 1:
        split_start_idx = seq_len - 1
    split_end_idx = split_indices[-1]

    feat_np = feat_norm.values.astype(np.float32)
    tgt_np = targets_df.values.astype(np.float32)
    dates = feat_norm.index

    tsfm_features_by_date = load_timesfm_features_by_date(dates)
    if tsfm_features_by_date:
        _n_feat = len(next(iter(tsfm_features_by_date.values())))
        print(f"  TimesFM blending: alpha={TIMESFM_BLEND_ALPHA}, scale={TIMESFM_SIGNAL_SCALE}, "
              f"power={SIGNAL_POWER}, clip={SIGNAL_CLIP}, features={_n_feat}")

    all_signals, all_targets, all_dates = [], [], []
    ema_sig = None
    for i in range(split_start_idx, split_end_idx + 1):
        window = feat_np[i - seq_len + 1 : i + 1]
        if len(window) < seq_len:
            continue
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            stacked = torch.stack([m(x).cpu() for m in models])
            avg_sig = _aggregate_ensemble(stacked)
        _cash_sig = None
        if CASH_ENABLED and avg_sig.size(1) > NUM_PAIRS:
            _cash_sig = avg_sig[:, NUM_PAIRS:]
            avg_sig = avg_sig[:, :NUM_PAIRS]
        if SIGNAL_EMA_DECAY > 0:
            if ema_sig is None:
                ema_sig = avg_sig.clone()
            else:
                ema_sig = SIGNAL_EMA_DECAY * ema_sig + (1 - SIGNAL_EMA_DECAY) * avg_sig
            avg_sig = ema_sig
        tsfm_feat = tsfm_features_by_date.get(dates[i]) if tsfm_features_by_date else None
        avg_sig = apply_signal_pipeline(avg_sig, tsfm_feat)
        if _cash_sig is not None:
            avg_sig = torch.cat([avg_sig, _cash_sig], dim=1)
        all_signals.append(avg_sig)
        all_targets.append(torch.tensor(tgt_np[i:i + 1], dtype=torch.float32))
        all_dates.append(dates[i])

    all_signals_t = torch.cat(all_signals, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    evaluation_dates = pd.DatetimeIndex(all_dates)
    vol_forecast = _load_timesfm_vol_forecast(evaluation_dates)
    _train_module._VOL_FORECAST_TENSOR = vol_forecast

    weights, portfolio_returns = compute_portfolio(
        all_signals_t, all_targets_t, vol_forecast=vol_forecast
    )
    ret_np = portfolio_returns.numpy()
    w_np = weights.numpy()

    periods_per_year = 52 if trade_frequency == "weekly" else 252
    m = compute_metrics(ret_np, w_np, periods_per_year, trade_frequency)

    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS — {split_label}")
    print(f"{'=' * 60}")
    print(f"  Period:            {all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading frequency:  {trade_frequency}")
    print(f"  Number of periods: {m['n_periods']}")
    print(f"  Ensemble size:     {len(models)} models")
    print(f"{'=' * 60}")
    print(f"\n  CAGR:              {m['cagr']:+.4%}")
    print(f"  Total Return:      {m['total_return']:+.4%}")
    print(f"  Sharpe Ratio:      {m['sharpe']:.4f}")
    print(f"  Max Drawdown:      {m['max_drawdown']:.4%}")
    print(f"  CVaR (95%):        {m['cvar_95']:+.4%}")
    print(f"  Skewness:          {m['skewness']:.4f}")
    print(f"  Avg Turnover:      {m['turnover']:.4f}")
    print(f"  Win Rate:          {m['win_rate']:.1%}")
    print(f"  Mean Return/period:{m['mean_ret']:+.4%}")
    print(f"  Std Return/period: {m['std_ret']:.4%}")

    print("\n  Pair           Avg Weight   Contribution")
    print("  " + "-" * 40)
    w_per_pair = np.zeros(NUM_PAIRS)
    for i in range(NUM_PAIRS):
        bull_w = w_np[:, 2 * i]
        bear_w = w_np[:, 2 * i + 1]
        avg_bull = float(np.abs(bull_w).mean())
        avg_bear = float(np.abs(bear_w).mean())
        contrib = float(((weights[:, 2 * i] * all_targets_t[:, 2 * i] +
                          weights[:, 2 * i + 1] * all_targets_t[:, 2 * i + 1])
                         ).mean())
        print(f"  {PAIR_NAMES[i]:<12} {avg_bull:+.4f}/{avg_bear:+.4f}  {contrib:+.6f}/period")

    print(f"\n{'=' * 60}")
    print(f"METRIC cagr={m['cagr']:.6f}")
    print(f"METRIC cvar_95={m['cvar_95']:.6f}")
    print(f"METRIC skewness={m['skewness']:.6f}")
    print(f"METRIC sharpe={m['sharpe']:.6f}")
    print(f"METRIC max_drawdown={m['max_drawdown']:.6f}")
    print(f"METRIC turnover={m['turnover']:.6f}")
    print(f"{'=' * 60}")
    return m


if __name__ == "__main__":
    # Default window: 2020-01-01 -> now (early-stop val + benchmark val + test)
    start = os.environ.get("ARC_SPLIT_START", "2020-01-01")
    end = os.environ.get("ARC_SPLIT_END", "2026-07-02")
    run_window(start, end, f"{start} -> {end} (val+test)")
