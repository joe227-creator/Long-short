"""
Simulate trading P&L on the out-of-sample test period using the trained ensemble.

Loads all seed models, averages signals, reconstructs weekly portfolio returns,
and reports: annualized return, max drawdown, Sharpe ratio, cumulative P&L.

Usage: uv run simulate.py
"""

import os
import glob

import numpy as np
import pandas as pd
import torch

from prepare import (
    ETF_TICKERS, ETF_PAIRS, PAIR_NAMES, NUM_PAIRS, VAL_END,
    build_dataset, make_dataloaders, signals_to_weights, download_etf_data,
    normalize_features, validate_feature_columns,
    validate_target_columns,
)
from train import (
    load_checkpoint, TRADE_FREQUENCY, SEQ_LEN, BATCH_SIZE, ENSEMBLE_SEEDS,
    USE_TIMESFM_VOL, TIMESFM_DIM, TIMESFM_BLEND_ALPHA, TIMESFM_SIGNAL_SCALE,
    SIGNAL_POWER, SIGNAL_CLIP, ENSEMBLE_AGG, SIGNAL_EMA_DECAY, SIGNAL_THRESHOLD,
    compute_portfolio, _load_timesfm_vol_forecast,
)
import train as _train_module
from production_model import _aggregate_ensemble, apply_signal_pipeline, load_timesfm_features_by_date

MODELS_DIR = "models"


def build_live_execution_targets(etf_df, trade_frequency="weekly"):
    """Forward returns assuming orders execute at the next trading day's open.

    The canonical backtest target is close(t) -> close(t+horizon). Live trading
    runs after the signal date close, so the tradable return is instead
    open(t+1) -> close(t+horizon). This models the Sunday/Monday execution gap.
    """
    if etf_df is None or not isinstance(etf_df, pd.DataFrame) or etf_df.empty:
        raise ValueError("Live execution ETF data is empty")
    required_price_columns = [
        column
        for ticker in ETF_TICKERS
        for column in (f"{ticker}_Open", f"{ticker}_Close")
    ]
    missing_price_columns = [
        column for column in required_price_columns if column not in etf_df.columns
    ]
    if missing_price_columns:
        raise ValueError(
            "Live execution ETF data missing required yfinance columns: "
            + ", ".join(missing_price_columns)
        )
    if not np.isfinite(etf_df[required_price_columns].to_numpy(dtype=float)).all():
        raise ValueError("Live execution ETF data contains missing/non-finite prices")
    targets = {}
    horizon = 1 if trade_frequency == "daily" else 5
    for ticker in ETF_TICKERS:
        open_col = f"{ticker}_Open"
        close_col = f"{ticker}_Close"
        entry = np.log(etf_df[open_col].shift(-1))
        exit_ = np.log(etf_df[close_col].shift(-horizon))
        targets[f"{ticker}_fwd_ret"] = exit_ - entry
    target_df = pd.DataFrame(targets, index=etf_df.index)
    target_df.index.name = "Date"
    validate_target_columns(
        target_df,
        [f"{ticker}_fwd_ret" for ticker in ETF_TICKERS],
        context="Live execution targets",
        allow_nan=True,
    )
    return target_df


def load_ensemble(device):
    """Load the ensemble models for the configured seeds."""
    seed_paths = []
    for seed in ENSEMBLE_SEEDS:
        p = os.path.join(MODELS_DIR, f"best_model_seed{seed}.pt")
        if os.path.exists(p):
            seed_paths.append(p)
    if not seed_paths:
        # Fallback to single model
        single = os.path.join(MODELS_DIR, "best_model.pt")
        if os.path.exists(single):
            seed_paths = [single]
        else:
            raise FileNotFoundError("No trained models found in models/")

    models = []
    for p in seed_paths:
        model, config = load_checkpoint(p, device)
        model.eval()
        models.append(model)
        print(f"  Loaded {os.path.basename(p)}")
    return models, config


def run_simulation():
    device = torch.device("cpu")

    print("=" * 60)
    print("  Trading P&L Simulation — Test Period")
    print("=" * 60)

    # Load ensemble
    print("\nLoading ensemble models...")
    models, config = load_ensemble(device)
    print(f"  Ensemble size: {len(models)} models")

    trade_frequency = config.get("trade_frequency", "daily")
    seq_len = config.get("seq_len", SEQ_LEN)
    feature_columns = config["feature_columns"]

    # Load data
    print(f"\nLoading data (frequency={trade_frequency})...")
    features_df, targets_df = build_dataset(trade_frequency=trade_frequency)
    etf_df = download_etf_data(refresh=False)
    targets_df = build_live_execution_targets(etf_df, trade_frequency=trade_frequency).reindex(features_df.index)
    valid_targets = targets_df.notna().all(axis=1)
    if not valid_targets.any():
        raise ValueError("Live execution targets contain no complete evaluation rows")
    features_df = features_df.loc[valid_targets]
    targets_df = targets_df.loc[valid_targets]
    validate_target_columns(
        targets_df,
        [f"{ticker}_fwd_ret" for ticker in ETF_TICKERS],
        context="Aligned live execution targets",
    )

    # Filter to only the feature columns used by the model
    features_df = validate_feature_columns(
        features_df, feature_columns, context="simulation features"
    )

    # Normalize using training scaler from checkpoint
    scaler_params = config["scaler_params"]
    mean = pd.Series(scaler_params["mean"])
    std = pd.Series(scaler_params["std"])
    feat_norm = normalize_features(features_df, mean, std)

    # Weekly subsampling (matches make_dataloaders logic)
    if trade_frequency == "weekly":
        weekly_idx = feat_norm.index[::5]
        feat_norm = feat_norm.loc[weekly_idx]
        targets_df = targets_df.loc[weekly_idx]

    # Identify test period
    test_mask = feat_norm.index >= VAL_END
    test_indices = np.where(test_mask)[0]

    if len(test_indices) <= seq_len:
        print(f"ERROR: Not enough test data ({len(test_indices)} rows, need >{seq_len})")
        return

    # Build test sequences manually so we can align dates
    feat_np = feat_norm.values.astype(np.float32)
    tgt_np = targets_df.values.astype(np.float32)
    dates = feat_norm.index

    # The test dataloader uses indices from the start of the test region
    # Each sample i uses features[i:i+lookback] to predict targets[i+lookback-1]
    # To get the test set with proper alignment, we need lookback rows BEFORE the first test index
    test_start = test_indices[0]
    usable_start = test_start  # first index in test_mask
    # We need seq_len rows prior for the first sequence
    seq_start = usable_start - seq_len + 1 if usable_start >= seq_len else 0

    # Load TimesFM pair-exhaustion features, aligned to the weekly grid via
    # closest-prior (single-source helper shared with live inference).
    tsfm_features_by_date = load_timesfm_features_by_date(dates)
    if tsfm_features_by_date:
        _n_feat = len(next(iter(tsfm_features_by_date.values())))
        print(f"  TimesFM blending: alpha={TIMESFM_BLEND_ALPHA}, scale={TIMESFM_SIGNAL_SCALE}, "
              f"power={SIGNAL_POWER}, clip={SIGNAL_CLIP}, features={_n_feat}")
    elif USE_TIMESFM_VOL and TIMESFM_BLEND_ALPHA < 1.0:
        print("  WARNING: TimesFM cache not found or empty, skipping blend")

    test_signals = []
    test_targets = []
    test_dates = []
    ema_sig = None  # for signal EMA smoothing

    for i in range(usable_start, len(feat_np)):
        if i - seq_len + 1 < 0:
            continue
        window = feat_np[i - seq_len + 1 : i + 1]  # (seq_len, F)
        if len(window) < seq_len:
            continue
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, T, F)

        # Ensemble aggregation (mean/median/trimmed_mean)
        with torch.no_grad():
            stacked = torch.stack([m(x).cpu() for m in models])
            avg_sig = _aggregate_ensemble(stacked)

        # Signal EMA smoothing: reduce signal noise across time
        if SIGNAL_EMA_DECAY > 0:
            if ema_sig is None:
                ema_sig = avg_sig.clone()
            else:
                ema_sig = SIGNAL_EMA_DECAY * ema_sig + (1 - SIGNAL_EMA_DECAY) * avg_sig
            avg_sig = ema_sig

        # Load TimesFM features for this date (closest-prior, point-in-time, no look-ahead)
        tsfm_feat = tsfm_features_by_date.get(dates[i]) if tsfm_features_by_date else None

        # Apply signal processing pipeline (power, clip, threshold, TimesFM blend)
        avg_sig = apply_signal_pipeline(avg_sig, tsfm_feat)

        test_signals.append(avg_sig)
        test_targets.append(torch.tensor(tgt_np[i:i+1], dtype=torch.float32))
        test_dates.append(dates[i])

    all_signals = torch.cat(test_signals, dim=0)  # (N, 4)
    all_targets = torch.cat(test_targets, dim=0)  # (N, 8)
    evaluation_dates = pd.DatetimeIndex(test_dates)
    vol_forecast = _load_timesfm_vol_forecast(evaluation_dates)
    _train_module._VOL_FORECAST_TENSOR = vol_forecast

    # Convert to weights + apply vol gate (circuit breaker) — matches backtest.py
    weights, portfolio_returns = compute_portfolio(
        all_signals, all_targets, vol_forecast=vol_forecast
    )  # (N, 8), (N,)
    ret_np = portfolio_returns.numpy()

    # --- Metrics ---
    n_periods = len(ret_np)
    periods_per_year = 52 if trade_frequency == "weekly" else 252

    # Cumulative returns
    cum_returns = np.cumprod(1 + ret_np)
    total_return = cum_returns[-1] / cum_returns[0] * (1 + ret_np[0]) - 1  # total from start
    total_return = cum_returns[-1] - 1  # simpler: starts at $1

    # Annualized return
    years = n_periods / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Sharpe ratio
    mean_ret = ret_np.mean()
    std_ret = ret_np.std()
    sharpe = (mean_ret / max(std_ret, 1e-8)) * np.sqrt(periods_per_year)

    # Max drawdown
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / np.maximum(running_max, 1e-8)
    max_drawdown = abs(drawdowns.min())

    # Turnover
    w_np = weights.numpy()
    weight_changes = np.abs(w_np[1:] - w_np[:-1]).sum(axis=1)
    avg_turnover = weight_changes.mean()

    # Win rate
    win_rate = (ret_np > 0).sum() / n_periods

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  SIMULATION RESULTS — Test Period")
    print(f"{'=' * 60}")
    print(f"  Period:              {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading frequency:   {trade_frequency}")
    print(f"  Number of periods:   {n_periods} weeks")
    print(f"  Ensemble size:       {len(models)} models")
    print(f"{'=' * 60}")
    print(f"\n  Annualized Return:   {annualized_return:+.2%}")
    print(f"  Total Return:        {total_return:+.2%}")
    print(f"  Sharpe Ratio:        {sharpe:.3f}")
    print(f"  Max Drawdown:        {max_drawdown:.2%}")
    print(f"  Avg Weekly Turnover: {avg_turnover:.4f}")
    print(f"  Win Rate:            {win_rate:.1%}")
    print(f"  Mean Weekly Return:  {mean_ret:+.4%}")
    print(f"  Std Weekly Return:   {std_ret:.4%}")
    print()

    # Per-pair breakdown
    print(f"  {'Pair':<12} {'Avg Weight':>12} {'Contribution':>14}")
    print(f"  {'-'*40}")
    for i, (pair_name, (bull, bear)) in enumerate(zip(PAIR_NAMES, ETF_PAIRS)):
        bull_w = w_np[:, 2*i].mean()
        bear_w = w_np[:, 2*i+1].mean()
        pair_ret = (weights[:, 2*i] * all_targets[:, 2*i] +
                    weights[:, 2*i+1] * all_targets[:, 2*i+1]).sum().item() / n_periods
        print(f"  {pair_name:<12} {bull_w:+.4f}/{bear_w:+.4f}  {pair_ret:+.6f}/wk")

    # Weekly P&L series
    print(f"\n  {'Week':<14} {'Return':>10} {'Cumulative':>12}")
    print(f"  {'-'*38}")
    # Show first 10 and last 10 weeks
    show_n = min(10, n_periods)
    for i in range(show_n):
        print(f"  {test_dates[i].strftime('%Y-%m-%d'):<14} {ret_np[i]:+.4%} {cum_returns[i]:>11.4f}")
    if n_periods > 20:
        print(f"  {'... (' + str(n_periods - 20) + ' weeks omitted) ...'}")
    for i in range(max(show_n, n_periods - 10), n_periods):
        print(f"  {test_dates[i].strftime('%Y-%m-%d'):<14} {ret_np[i]:+.4%} {cum_returns[i]:>11.4f}")

    print(f"\n  Final portfolio value (starting $1.00): ${cum_returns[-1]:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_simulation()
