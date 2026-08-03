"""Single source of truth for live checkpoint loading and inference.

trade.py and inference.py both call this module so production checkpoint discovery,
config validation, feature alignment, and weekly subsampling remain identical.

Signal processing pipeline (must match backtest.py):
  ensemble aggregation → clip → power → threshold → TimesFM blend → tanh → weights
"""

import os
import re

import numpy as np
import pandas as pd
import torch

from prepare import (
    DATA_SPLIT_VERSION, normalize_features, signals_to_weights, ETF_TICKERS,
    validate_feature_columns,
)
from train import (
    load_checkpoint,
    SIGNAL_CLIP, SIGNAL_POWER, SIGNAL_THRESHOLD, SIGNAL_EMA_DECAY,
    USE_TIMESFM_VOL, TIMESFM_DIM, TIMESFM_BLEND_ALPHA, TIMESFM_SIGNAL_SCALE,
    ENSEMBLE_AGG, SEQ_LEN,
    compute_portfolio, _load_timesfm_vol_forecast,
)
import train as _train_module


MODELS_DIR = "models"

_SEED_CHECKPOINT_RE = re.compile(r"^best_model_seed(\d+)\.pt$")
_CONSISTENCY_KEYS = (
    "model_type",
    "data_split_version",
    "trade_frequency",
    "seq_len",
    "feature_columns",
    "scaler_params",
    "hidden_dim",
    "num_layers",
    "dropout",
    "use_feature_gate",
    "use_vsn",
    "vsn_hidden",
    "vsn_residual",
    "vsn_learnable_alpha",
    "vsn_alpha",
)


def _discover_checkpoint_paths(models_dir=MODELS_DIR):
    if not os.path.isdir(models_dir):
        return []

    seed_paths = []
    for name in os.listdir(models_dir):
        match = _SEED_CHECKPOINT_RE.match(name)
        if match:
            seed_paths.append((int(match.group(1)), os.path.join(models_dir, name)))

    if seed_paths:
        return sorted(seed_paths, key=lambda item: item[0])

    single_path = os.path.join(models_dir, "best_model.pt")
    if os.path.exists(single_path):
        return [(None, single_path)]

    return []


def _find_config_mismatch(base_config, other_config):
    for key in _CONSISTENCY_KEYS:
        if base_config.get(key) != other_config.get(key):
            return key
    return None


def load_production_ensemble(device, models_dir=MODELS_DIR, checkpoint_loader=load_checkpoint):
    """Load the saved production ensemble from disk.

    Seeded checkpoints are discovered from models/ directly so live trading does not
    depend on mutable experiment globals in train.py.
    """
    checkpoint_paths = _discover_checkpoint_paths(models_dir)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No trained models found in {models_dir}/")

    models = []
    config = None
    loaded_seeds = []
    baseline_file = None

    for seed, path in checkpoint_paths:
        model, checkpoint_config = checkpoint_loader(path, device)
        if config is None:
            config = dict(checkpoint_config)
            baseline_file = os.path.basename(path)
            if config.get("data_split_version") != DATA_SPLIT_VERSION:
                raise ValueError(
                    f"{baseline_file} was not trained with data_split_version={DATA_SPLIT_VERSION}; "
                    "retrain before running live trading"
                )
        else:
            mismatch_key = _find_config_mismatch(config, checkpoint_config)
            if mismatch_key is not None:
                raise ValueError(
                    "Inconsistent checkpoint configs between "
                    f"{baseline_file} and {os.path.basename(path)}: field '{mismatch_key}' differs"
                )

        model.eval()
        models.append(model)
        if seed is not None:
            loaded_seeds.append(seed)

    config["loaded_seeds"] = loaded_seeds
    config["loaded_checkpoint_files"] = [os.path.basename(path) for _, path in checkpoint_paths]
    return models, config


def describe_model(config):
    model_name = config.get("model_type", "model").upper()
    if config.get("use_vsn"):
        suffix = "VSN_RESIDUAL" if config.get("vsn_residual") else "VSN"
        return f"{model_name} + {suffix}"
    if config.get("use_feature_gate"):
        return f"{model_name} + FEATURE_GATE"
    return model_name


def _aggregate_ensemble(stacked_signals):
    """Aggregate ensemble signals using the configured method.

    Args:
        stacked_signals: tensor of shape (num_models, 1, num_pairs)
    Returns:
        aggregated signal tensor of shape (1, num_pairs)
    """
    if ENSEMBLE_AGG == "median":
        return stacked_signals.median(dim=0).values
    elif ENSEMBLE_AGG == "trimmed_mean":
        sorted_stack, _ = stacked_signals.sort(dim=0)
        return sorted_stack[1:-1].mean(dim=0)
    else:  # "mean"
        return stacked_signals.mean(dim=0)


def _load_timesfm_dataframe():
    """Load the TimesFM pair-exhaustion cache, filtered to TIMESFM_DIM columns.

    Single source of truth for the cache path, the `tsfm_pair_exhaustion_`
    column filter, and the TIMESFM_DIM cap. Returns a date-sorted DataFrame or
    None if TimesFM is disabled or the cache is missing.
    """
    if not (USE_TIMESFM_VOL and TIMESFM_BLEND_ALPHA < 1.0):
        return None

    tsfm_cache = os.path.join(os.path.expanduser("~"), ".cache", "etf_autoresearch",
                               "timesfm_vol_features.parquet")
    if not os.path.exists(tsfm_cache):
        return None

    tsfm_df = pd.read_parquet(tsfm_cache)
    keep = [c for c in tsfm_df.columns if c.startswith("tsfm_pair_exhaustion_")]
    keep = keep[:TIMESFM_DIM]
    return tsfm_df[keep].sort_index()


def load_timesfm_features_by_date(weekly_dates):
    """TimesFM pair-exhaustion features aligned to weekly_dates via closest-prior.

    Reindexes the cache onto the weekly date grid with forward-fill, so each
    weekly date gets the most recent on-or-before TimesFM row (point-in-time,
    no look-ahead). This matches the live `_load_timesfm_features` lookup, so
    offline backtest/simulate/year_breakdown and live inference share the SAME
    closest-prior semantics — previously the offline scripts used an exact-date
    match that would silently skip the blend on any date not present in the
    cache.

    Returns a {date: np.ndarray} dict; dates before the first TimesFM row are
    omitted.
    """
    tsfm_df = _load_timesfm_dataframe()
    if tsfm_df is None or len(tsfm_df) == 0:
        return {}
    aligned = tsfm_df.reindex(weekly_dates, method="ffill")
    return {d: row.values for d, row in aligned.dropna(how="all").iterrows()}


def _load_timesfm_features(latest_date):
    """Load TimesFM pair-exhaustion features for a single date (closest-prior, no look-ahead).

    Live path. Returns None if TimesFM is disabled, the cache is missing, or no
    TimesFM row exists on or before `latest_date`.
    """
    tsfm_df = _load_timesfm_dataframe()
    if tsfm_df is None or len(tsfm_df) == 0:
        return None

    if isinstance(latest_date, str):
        latest_date = pd.Timestamp(latest_date)

    available_dates = tsfm_df.index[tsfm_df.index <= latest_date]
    if len(available_dates) == 0:
        return None
    closest_date = available_dates[-1]
    return tsfm_df.loc[closest_date].values


def apply_signal_pipeline(avg_sig, tsfm_feat=None):
    """Apply the full signal processing pipeline to a single ensemble signal.

    This MUST match backtest.py's signal processing order:
      1. Signal power transform
      2. Signal clipping
      3. Signal threshold
      4. TimesFM stacked generalization blend

    Args:
        avg_sig: tensor of shape (1, num_pairs) — aggregated ensemble signal
        tsfm_feat: optional numpy array of TimesFM features for this date
    Returns:
        processed signal tensor of shape (1, num_pairs)
    """
    # Signal power transform: compress large signals
    if SIGNAL_POWER != 1.0:
        avg_sig = torch.sign(avg_sig) * torch.abs(avg_sig) ** SIGNAL_POWER

    # Signal clipping: cap extreme signals to reduce tanh saturation
    if SIGNAL_CLIP > 0:
        avg_sig = torch.clamp(avg_sig, -SIGNAL_CLIP, SIGNAL_CLIP)

    # Signal threshold: zero out small signals
    if SIGNAL_THRESHOLD > 0:
        avg_sig = torch.where(torch.abs(avg_sig) < SIGNAL_THRESHOLD,
                              torch.zeros_like(avg_sig), avg_sig)

    # Stacked generalization: blend LSTM signals with TimesFM volume signals
    if tsfm_feat is not None:
        tsfm_sig = torch.tensor((tsfm_feat - 1.0) * TIMESFM_SIGNAL_SCALE,
                                dtype=avg_sig.dtype).unsqueeze(0)
        num_pairs = avg_sig.size(1)
        if tsfm_sig.size(1) >= num_pairs:
            tsfm_sig = tsfm_sig[:, :num_pairs]
        else:
            pad = torch.zeros(1, num_pairs - tsfm_sig.size(1), dtype=avg_sig.dtype)
            tsfm_sig = torch.cat([tsfm_sig, pad], dim=1)
        avg_sig = TIMESFM_BLEND_ALPHA * avg_sig + (1 - TIMESFM_BLEND_ALPHA) * tsfm_sig

    return avg_sig


def run_live_ensemble(models, config, feat_df, device):
    """Run the saved ensemble on the latest feature window and return live decisions.

    Applies the full signal processing pipeline (clip, TimesFM blend, etc.)
    to match backtest.py exactly.
    """
    feature_columns = config["feature_columns"]
    scaler_params = config["scaler_params"]

    feat_aligned = validate_feature_columns(
        feat_df, feature_columns, context="Live model features"
    )
    missing_columns = []

    mean = pd.Series(scaler_params["mean"])
    std = pd.Series(scaler_params["std"])
    feat_norm = normalize_features(feat_aligned, mean, std)

    trade_frequency = config.get("trade_frequency", "daily")
    if trade_frequency == "weekly":
        feat_norm = feat_norm.iloc[::5]

    seq_len = config.get("seq_len", SEQ_LEN)
    if len(feat_norm) < seq_len:
        raise ValueError(f"Not enough data ({len(feat_norm)} rows, need {seq_len})")

    latest_date = feat_norm.index[-1]

    def _ensemble_signal_at(end_pos):
        """Aggregated ensemble signal for the window ending at feat_norm row `end_pos` (inclusive)."""
        window = feat_norm.iloc[end_pos - seq_len + 1 : end_pos + 1].values.astype(np.float32)
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            stacked = torch.stack([model(x).cpu() for model in models])
        return _aggregate_ensemble(stacked)

    last_pos = len(feat_norm) - 1
    if SIGNAL_EMA_DECAY > 0:
        # Temporal EMA smoothing (must match backtest.py/simulate.py, which chain
        # ema = d*ema + (1-d)*sig across consecutive weekly predictions). The EMA
        # anchor decays as d^k, so replaying the trailing EMA_WARMUP_WEEKS windows
        # reproduces the backtest EMA to numerical precision (0.4^30 ~ 1e-12).
        EMA_WARMUP_WEEKS = 30
        first_pos = max(seq_len - 1, last_pos - EMA_WARMUP_WEEKS + 1)
        ema_sig = _ensemble_signal_at(first_pos).clone()
        for pos in range(first_pos + 1, last_pos + 1):
            sig = _ensemble_signal_at(pos)
            ema_sig = SIGNAL_EMA_DECAY * ema_sig + (1 - SIGNAL_EMA_DECAY) * sig
        avg_sig = ema_sig
    else:
        avg_sig = _ensemble_signal_at(last_pos)

    # Load TimesFM features for the latest date (point-in-time, no look-ahead)
    tsfm_feat = _load_timesfm_features(latest_date)

    # Apply signal processing pipeline (clip, power, threshold, TimesFM blend)
    processed_sig = apply_signal_pipeline(avg_sig, tsfm_feat)

    # Apply vol gate (circuit breaker) — MUST match backtest.py's compute_portfolio.
    # Scales down positions in high-volatility (bear) periods using the TimesFM
    # forward-looking vol forecast (zero-shot, past data only — no look-ahead).
    vol_forecast = _load_timesfm_vol_forecast(pd.DatetimeIndex([latest_date]))
    _train_module._VOL_FORECAST_TENSOR = vol_forecast
    # Dummy targets: only the row count (1) is used by the portfolio pipeline;
    # with VOL_GATE_FORMULA="timesfm" actual target values are not consumed.
    dummy_targets = torch.zeros(1, len(ETF_TICKERS), dtype=torch.float32)
    weights, _ = compute_portfolio(processed_sig, dummy_targets)

    # Convert to weights via tanh + normalize (fixed contract in prepare.py)
    signals = torch.tanh(processed_sig)

    return {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "missing_columns": missing_columns,
        "raw_signals": avg_sig.squeeze(0).cpu().numpy(),
        "signals": signals.squeeze(0).cpu().numpy(),
        "weights": weights.squeeze(0).cpu().numpy(),
    }
