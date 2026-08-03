"""Regenerate/extend the TimesFM caches used by the vol gate and the signal blend.

Two caches in ~/.cache/etf_autoresearch/:
  timesfm_raw_vol_forecasts.parquet  — market vol forecast (vol gate / circuit breaker)
  timesfm_vol_features.parquet       — per-ETF vol ratios + pair exhaustion (signal blend)

APPEND-ONLY: existing rows are never modified, so historical backtests remain
byte-reproducible. Only dates after each cache's last row are computed and appended.

Methodology (reverse-engineered 2026-07-03; the original generation script was lost):
  - Underlying series: cross-ETF mean of the 20-day rolling std (ddof=0) of daily
    log returns. This matches the cached `backward_vol` column EXACTLY (err ~1e-17).
  - Forecast: TimesFM 2.5 zero-shot (local checkpoint timesfm-2.5-200m-transformers/),
    context = past 256 daily values, horizon = 25 trading days,
    forecast_vol = mean of the point forecast over the horizon.
    Against the original cache this reproduces forecast_vol with MAE ~9e-4
    (corr 0.963); byte-exact reproduction is impossible because the original run
    used a different TimesFM runtime. Appending (instead of rewriting) keeps the
    historical rows identical to what the backtest was validated on.
  - Pair features: per-ETF vol_ratio = forecast_vol / backward_vol (same config,
    per-ETF series); tsfm_pair_exhaustion_<pair> = min(bull, bear vol_ratio);
    tsfm_pair_asymmetry_<pair> = |bull - bear|. Only the exhaustion columns are
    consumed by the model (TIMESFM_DIM=4 filter in production_model.py); at
    TIMESFM_SIGNAL_SCALE=0.001 their signal contribution is near-noise, so the
    fidelity requirement is low. NOTE: the historical file names the equity bull
    columns "SPUU" (pre-migration ticker); appended rows keep that schema but are
    computed from SSO (the current equity bull ETF).

Usage:
    uv run generate_timesfm_vol.py             # extend caches from cached prices
    uv run generate_timesfm_vol.py --refresh   # re-download prices first

Called automatically by trade.py on every weekly run (failure is non-fatal there).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from prepare import CACHE_DIR, ETF_TICKERS, ETF_PAIRS, PAIR_NAMES, download_etf_data

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "timesfm-2.5-200m-transformers")
RAW_VOL_CACHE = os.path.join(CACHE_DIR, "timesfm_raw_vol_forecasts.parquet")
FEATURES_CACHE = os.path.join(CACHE_DIR, "timesfm_vol_features.parquet")

CONTEXT = 256          # past trading days fed to TimesFM (zero-shot, no fine-tuning)
HORIZON = 25           # forecast horizon (trading days) averaged into forecast_vol
VOL_WINDOW = 20        # rolling window for the realized (backward) vol series
BATCH = 32             # contexts per model call

# Historical column naming in timesfm_vol_features.parquet (equity bull was SPUU
# before the SPUU->SSO migration). Values are computed from the CURRENT tickers.
_LEGACY_NAME = {"SSO": "SPUU"}

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        from transformers import TimesFm2_5ModelForPrediction
        _MODEL = TimesFm2_5ModelForPrediction.from_pretrained(
            MODEL_DIR, local_files_only=True).to(torch.float32).eval()
    return _MODEL


def _vol_series(etf_df):
    """(aggregate market vol, per-ETF vol) — 20d rolling std of daily log returns."""
    close = etf_df[[f"{t}_Close" for t in ETF_TICKERS]]
    lr = np.log(close).diff()
    per_etf = lr.rolling(VOL_WINDOW).std(ddof=0)
    per_etf.columns = ETF_TICKERS
    return per_etf.mean(axis=1), per_etf


def _forecast_batch(contexts):
    """Run TimesFM on a list of 1-D numpy contexts. Returns (point (B,128), quantiles (B,128,10))."""
    model = _load_model()
    points, quants = [], []
    for i in range(0, len(contexts), BATCH):
        chunk = [torch.tensor(c.astype(np.float32)) for c in contexts[i:i + BATCH]]
        with torch.no_grad():
            out = model(past_values=chunk, return_dict=True)
        points.append(out.mean_predictions.numpy())
        quants.append(out.full_predictions.numpy())
    return np.concatenate(points), np.concatenate(quants)


def _new_dates(series_index, cache_path):
    """Dates in series_index that are strictly after the cache's last row."""
    if os.path.exists(cache_path):
        existing = pd.read_parquet(cache_path)
        return existing, series_index[series_index > existing.index.max()]
    return None, series_index


def _extend_raw_vol(vol_agg, verbose=True):
    """Extend timesfm_raw_vol_forecasts.parquet (vol gate input)."""
    vol = vol_agg.dropna()
    usable = vol.index[CONTEXT - 1:]  # need a full context window
    existing, new_dates = _new_dates(usable, RAW_VOL_CACHE)
    if len(new_dates) == 0:
        if verbose:
            print(f"  vol-gate cache fresh (last: {existing.index.max().date()})")
        return existing

    contexts = []
    for d in new_dates:
        pos = vol.index.get_loc(d)
        contexts.append(vol.iloc[pos - CONTEXT + 1: pos + 1].values)
    point, quant = _forecast_batch(contexts)

    fv = point[:, :HORIZON].mean(axis=1)
    lt = quant[:, :HORIZON, 3].mean(axis=1)          # q30 (closest to original left tail)
    spread = (quant[:, :HORIZON, 9] - quant[:, :HORIZON, 1]).mean(axis=1)  # q90 - q10
    bwd = vol.loc[new_dates].values
    rows = pd.DataFrame({
        "forecast_vol": fv,
        "forecast_left_tail": lt,
        "forecast_left_tail_abs": np.abs(lt),
        "forecast_quantile_spread": spread,
        "backward_vol": bwd,
        "vol_ratio": fv / np.where(bwd > 0, bwd, np.nan),
    }, index=pd.DatetimeIndex(new_dates, name="date"))

    combined = rows if existing is None else pd.concat([existing, rows])
    combined.to_parquet(RAW_VOL_CACHE)
    if verbose:
        print(f"  vol-gate cache: appended {len(rows)} rows "
              f"(now through {combined.index.max().date()})")
    return combined


def _extend_features(vol_per_etf, verbose=True):
    """Extend timesfm_vol_features.parquet (pair-exhaustion blend features)."""
    vol = vol_per_etf.dropna()
    usable = vol.index[CONTEXT - 1:]
    existing, new_dates = _new_dates(usable, FEATURES_CACHE)
    if len(new_dates) == 0:
        if verbose:
            print(f"  blend-feature cache fresh (last: {existing.index.max().date()})")
        return existing

    # Batch all (date x ETF) contexts in one pass
    contexts = []
    for d in new_dates:
        pos = vol.index.get_loc(d)
        for t in ETF_TICKERS:
            contexts.append(vol[t].iloc[pos - CONTEXT + 1: pos + 1].values)
    point, _ = _forecast_batch(contexts)
    point = point.reshape(len(new_dates), len(ETF_TICKERS), -1)

    fv = point[:, :, :HORIZON].mean(axis=2)                      # (D, 8)
    slope = (point[:, :, HORIZON - 1] - point[:, :, 0]) / np.clip(point[:, :, 0], 1e-12, None)
    bwd = vol.loc[new_dates].values                               # (D, 8)
    ratio = fv / np.where(bwd > 0, bwd, np.nan)

    data = {}
    for j, t in enumerate(ETF_TICKERS):
        name = _LEGACY_NAME.get(t, t)
        data[f"tsfm_vol_ratio_{name}"] = ratio[:, j]
        data[f"tsfm_vol_slope_{name}"] = slope[:, j]
    for (bull, bear), pair in zip(ETF_PAIRS, PAIR_NAMES):
        bi, ei = ETF_TICKERS.index(bull), ETF_TICKERS.index(bear)
        data[f"tsfm_pair_exhaustion_{pair}"] = np.minimum(ratio[:, bi], ratio[:, ei])
        data[f"tsfm_pair_asymmetry_{pair}"] = np.abs(ratio[:, bi] - ratio[:, ei])

    rows = pd.DataFrame(data, index=pd.DatetimeIndex(new_dates, name="Date"))
    if existing is not None:
        rows = rows.reindex(columns=existing.columns)  # keep historical column order
        combined = pd.concat([existing, rows])
    else:
        combined = rows
    combined.to_parquet(FEATURES_CACHE)
    if verbose:
        print(f"  blend-feature cache: appended {len(rows)} rows "
              f"(now through {combined.index.max().date()})")
    return combined


def refresh_timesfm_caches(etf_df=None, refresh_prices=False, verbose=True):
    """Extend both TimesFM caches to the latest available price date (append-only).

    Args:
        etf_df: optional pre-downloaded ETF price DataFrame (from download_etf_data).
        refresh_prices: force re-download of prices when etf_df is None.
    Returns (raw_vol_df, features_df).
    """
    if etf_df is None:
        etf_df = download_etf_data(refresh=refresh_prices)
    if verbose:
        print("TimesFM cache refresh (zero-shot, append-only):")
    vol_agg, vol_per = _vol_series(etf_df)
    raw_df = _extend_raw_vol(vol_agg, verbose=verbose)
    feat_df = _extend_features(vol_per, verbose=verbose)

    # Loud staleness check — the vol gate (circuit breaker) depends on this.
    latest_price = etf_df.index.max()
    for label, df in [("vol-gate", raw_df), ("blend-feature", feat_df)]:
        lag = (latest_price - df.index.max()).days
        if lag > 7:
            print(f"  *** WARNING: {label} TimesFM cache is {lag} days behind the "
                  f"latest price date ({latest_price.date()}) — vol gate/blend may be stale ***")
    return raw_df, feat_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extend TimesFM vol caches (append-only)")
    parser.add_argument("--refresh", action="store_true", help="Re-download prices first")
    args = parser.parse_args()
    refresh_timesfm_caches(refresh_prices=args.refresh)
    print("Done.")
