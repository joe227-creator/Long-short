"""
Canonical backtest + metric emitter for the autoresearch harness.

Loads the 5-seed ensemble from models/, reconstructs weekly test-period
portfolio returns, and prints METRIC lines for the autoresearch framework:

    METRIC cagr=<value>          (primary  — higher is better)
    METRIC cvar_95=<value>       (secondary — higher is better, less negative)
    METRIC skewness=<value>      (secondary — higher is better)
    METRIC sharpe=<value>
    METRIC max_drawdown=<value>
    METRIC turnover=<value>

Usage:  python backtest.py
        python backtest.py --val   # use validation split instead of test
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

from prepare import (
    NUM_PAIRS, ETF_PAIRS, PAIR_NAMES, VAL_END, TRAIN_END,
    build_dataset, signals_to_weights, normalize_features, validate_feature_columns,
)
from train import (
    load_checkpoint, ENSEMBLE_SEEDS, SEQ_LEN,
    USE_TIMESFM_VOL, TIMESFM_DIM, TIMESFM_BLEND_ALPHA, TIMESFM_SIGNAL_SCALE,
    SIGNAL_POWER, SIGNAL_CLIP, ENSEMBLE_AGG, SIGNAL_EMA_DECAY, SIGNAL_THRESHOLD,
    LONG_ONLY, CASH_ENABLED, compute_portfolio,
    _load_timesfm_vol_forecast,
)
import train as _train_module
from production_model import apply_signal_pipeline, load_timesfm_features_by_date
from research.evidence import write_evidence
from research.optuna_postprocess import optimize_or_load

MODELS_DIR = "models"
ENSEMBLE_METHODS = ("trimmed_mean", "median", "mean")


def _aggregate_ensemble_method(stacked_signals, method):
    if method == "median":
        return stacked_signals.median(dim=0).values
    if method == "trimmed_mean":
        sorted_stack, _ = stacked_signals.sort(dim=0)
        return sorted_stack[1:-1].mean(dim=0)
    if method == "mean":
        return stacked_signals.mean(dim=0)
    raise ValueError(f"Unsupported ensemble aggregation: {method}")


def load_ensemble(device):
    """Load the ensemble models for the configured seeds."""
    seed_paths = []
    for seed in ENSEMBLE_SEEDS:
        p = os.path.join(MODELS_DIR, f"best_model_seed{seed}.pt")
        if os.path.exists(p):
            seed_paths.append(p)
    if not seed_paths:
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


def compute_metrics(ret_np, w_np, periods_per_year, trade_frequency):
    """Compute all portfolio metrics from a return series."""
    n_periods = len(ret_np)

    # --- CAGR (primary) ---
    cum_returns = np.cumprod(1.0 + ret_np)
    total_return = cum_returns[-1] - 1.0
    years = n_periods / periods_per_year
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0

    # --- Sharpe ---
    mean_ret = ret_np.mean()
    std_ret = ret_np.std()
    sharpe = (mean_ret / max(std_ret, 1e-8)) * np.sqrt(periods_per_year)

    # --- Max drawdown ---
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / np.maximum(running_max, 1e-8)
    max_drawdown = abs(drawdowns.min())

    # --- Turnover ---
    if len(w_np) > 1:
        weight_changes = np.abs(w_np[1:] - w_np[:-1]).sum(axis=1)
        turnover = weight_changes.mean()
    else:
        turnover = 0.0

    # --- CVaR 95% (Conditional Value at Risk) ---
    # VaR_95 = 5th percentile of returns; CVaR_95 = mean of returns <= VaR_95
    # Higher (less negative) is better.
    var_95 = np.percentile(ret_np, 5)
    tail = ret_np[ret_np <= var_95]
    cvar_95 = tail.mean() if len(tail) > 0 else var_95

    # --- Skewness (Fisher-Pearson) ---
    if std_ret > 1e-8:
        skewness = float(((ret_np - mean_ret) ** 3).mean() / (std_ret ** 3))
    else:
        skewness = 0.0

    # --- Win rate ---
    win_rate = (ret_np > 0).sum() / n_periods

    return {
        "cagr": float(cagr),
        "cvar_95": float(cvar_95),
        "skewness": float(skewness),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "turnover": float(turnover),
        "total_return": float(total_return),
        "win_rate": float(win_rate),
        "mean_ret": float(mean_ret),
        "std_ret": float(std_ret),
        "n_periods": int(n_periods),
    }


def run_backtest(split="test"):
    """Run the backtest and print metrics. split = 'test' or 'val'."""
    device = torch.device("cpu")

    print("=" * 60)
    print(f"  Autoresearch Backtest — {split.upper()} period")
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

    # Filter to model's feature columns
    features_df = validate_feature_columns(
        features_df, feature_columns, context=f"{split} backtest features"
    )

    # Normalize using checkpoint scaler
    scaler_params = config["scaler_params"]
    mean = pd.Series(scaler_params["mean"])
    std = pd.Series(scaler_params["std"])
    feat_norm = normalize_features(features_df, mean, std)

    # Weekly subsampling (matches make_dataloaders)
    if trade_frequency == "weekly":
        weekly_idx = feat_norm.index[::5]
        feat_norm = feat_norm.loc[weekly_idx]
        targets_df = targets_df.loc[weekly_idx]

    # Identify split period
    if split == "val":
        split_start = TRAIN_END
        split_end = VAL_END
        mask = (feat_norm.index >= split_start) & (feat_norm.index < split_end)
    else:
        mask = feat_norm.index >= VAL_END
    split_indices = np.where(mask)[0]

    if len(split_indices) <= seq_len:
        print(f"ERROR: Not enough {split} data ({len(split_indices)} rows, need >{seq_len})")
        return None

    split_start_idx = split_indices[0]
    if split_start_idx < seq_len - 1:
        split_start_idx = seq_len - 1
    split_end_idx = split_indices[-1]  # stop at the split boundary (val won't leak into test)

    feat_np = feat_norm.values.astype(np.float32)
    tgt_np = targets_df.values.astype(np.float32)
    dates = feat_norm.index

    # Load TimesFM pair-exhaustion features (closest-prior, shared with live inference)
    tsfm_features_by_date = load_timesfm_features_by_date(dates)
    if tsfm_features_by_date:
        _n_feat = len(next(iter(tsfm_features_by_date.values())))
        print(f"  TimesFM blending: alpha={TIMESFM_BLEND_ALPHA}, scale={TIMESFM_SIGNAL_SCALE}, "
              f"power={SIGNAL_POWER}, clip={SIGNAL_CLIP}, features={_n_feat}")
    elif USE_TIMESFM_VOL and TIMESFM_BLEND_ALPHA < 1.0:
        print("  WARNING: TimesFM cache not found or empty, skipping blend")

    all_signals = {method: [] for method in ENSEMBLE_METHODS}
    all_dispersion = {"std": [], "mad": [], "range": []}
    all_targets = []
    all_dates = []
    ema_sig = {method: None for method in ENSEMBLE_METHODS}
    for i in range(split_start_idx, split_end_idx + 1):
        window = feat_np[i - seq_len + 1 : i + 1]
        if len(window) < seq_len:
            continue
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            stacked = torch.stack([m(x).cpu() for m in models])
            raw_signals = {
                method: _aggregate_ensemble_method(stacked, method)
                for method in ENSEMBLE_METHODS
            }
            ensemble_center = stacked.mean(dim=0)
            all_dispersion["std"].append(stacked.std(dim=0))
            all_dispersion["mad"].append(
                torch.abs(stacked - ensemble_center).mean(dim=0)
            )
            all_dispersion["range"].append(
                stacked.max(dim=0).values - stacked.min(dim=0).values
            )
        tsfm_feat = tsfm_features_by_date.get(dates[i]) if tsfm_features_by_date else None
        for method, avg_sig in raw_signals.items():
            # When cash enabled, split cash signal before pipeline so threshold/clip
            # only applies to pair signals (cash signal should NOT be zeroed).
            _cash_sig = None
            if CASH_ENABLED and avg_sig.size(1) > 4:  # NUM_PAIRS=4
                _cash_sig = avg_sig[:, 4:]
                avg_sig = avg_sig[:, :4]
            # Signal EMA smoothing (temporal — applies across consecutive predictions)
            if SIGNAL_EMA_DECAY > 0:
                if ema_sig[method] is None:
                    ema_sig[method] = avg_sig.clone()
                else:
                    ema_sig[method] = (
                        SIGNAL_EMA_DECAY * ema_sig[method]
                        + (1 - SIGNAL_EMA_DECAY) * avg_sig
                    )
                avg_sig = ema_sig[method]
            # Apply shared signal pipeline to pair signals only.
            avg_sig = apply_signal_pipeline(avg_sig, tsfm_feat)
            # Reattach cash signal (unprocessed — model learned it directly).
            if _cash_sig is not None:
                avg_sig = torch.cat([avg_sig, _cash_sig], dim=1)
            all_signals[method].append(avg_sig)
        all_targets.append(torch.tensor(tgt_np[i:i + 1], dtype=torch.float32))
        all_dates.append(dates[i])

    all_signals_t = {
        method: torch.cat(values, dim=0)
        for method, values in all_signals.items()
    }
    dispersion_t = {
        name: torch.cat(values, dim=0)
        for name, values in all_dispersion.items()
    }
    all_targets_t = torch.cat(all_targets, dim=0)
    evaluation_dates = pd.DatetimeIndex(all_dates)
    vol_forecast = _load_timesfm_vol_forecast(evaluation_dates)
    _train_module._VOL_FORECAST_TENSOR = vol_forecast

    weights, portfolio_returns = optimize_or_load(
        split,
        all_signals_t,
        all_targets_t,
        vol_forecast,
        evaluation_dates,
        trade_frequency,
        dispersion_t,
        compute_metrics,
    )
    ret_np = portfolio_returns.numpy()
    w_np = weights.numpy()

    periods_per_year = 52 if trade_frequency == "weekly" else 252
    m = compute_metrics(ret_np, w_np, periods_per_year, trade_frequency)
    evidence = write_evidence(
        split, m, ret_np, w_np, evaluation_dates, config
    )
    score = evidence["score"]

    # Print human-readable report
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS — {split.upper()}")
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
    print(f"  Research Score:    {score['research_score']:+.8f}")
    print()

    # Per-pair breakdown
    print(f"  {'Pair':<12} {'Avg Weight':>12} {'Contribution':>14}")
    print(f"  {'-' * 40}")
    for i, (pair_name, (bull, bear)) in enumerate(zip(PAIR_NAMES, ETF_PAIRS)):
        bull_w = w_np[:, 2 * i].mean()
        bear_w = w_np[:, 2 * i + 1].mean()
        pair_ret = (weights[:, 2 * i] * all_targets_t[:, 2 * i] +
                    weights[:, 2 * i + 1] * all_targets_t[:, 2 * i + 1]).sum().item() / m['n_periods']
        print(f"  {pair_name:<12} {bull_w:+.4f}/{bear_w:+.4f}  {pair_ret:+.6f}/period")

    print(f"\n{'=' * 60}")

    # --- Emit validation METRIC lines; test lines are informational only. ---
    metric_prefix = "METRIC" if split == "val" else "TEST_METRIC"
    # Primary
    print(f"{metric_prefix} cagr={m['cagr']:.6f}")
    # Secondary (per user priority: CVaR then Skewness)
    print(f"{metric_prefix} cvar_95={m['cvar_95']:.6f}")
    print(f"{metric_prefix} skewness={m['skewness']:.6f}")
    # Additional info metrics
    print(f"{metric_prefix} sharpe={m['sharpe']:.6f}")
    print(f"{metric_prefix} max_drawdown={m['max_drawdown']:.6f}")
    print(f"{metric_prefix} turnover={m['turnover']:.6f}")
    print(f"{metric_prefix} mean_rolling_6m_return={score['mean_rolling_6m_return']:.6f}")
    print(f"{metric_prefix} return_on_risk={score['return_on_risk']:.6f}")
    print(f"{metric_prefix} win_rate_126_session={score['win_rate_126_session']:.6f}")
    print(f"{metric_prefix} maximum_drawdown={score['maximum_drawdown']:.6f}")
    print(f"{metric_prefix} research_score={score['research_score']:.6f}")

    return m


if __name__ == "__main__":
    split = "test"
    if "--val" in sys.argv:
        split = "val"
    result = run_backtest(split=split)
    if result is None:
        sys.exit(1)
