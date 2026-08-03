"""Year-by-year breakdown using the canonical portfolio and vol-gate pipeline."""
import os, sys, numpy as np, pandas as pd, torch
from prepare import (
    build_dataset, VAL_END, ETF_PAIRS, PAIR_NAMES, normalize_features,
    validate_feature_columns,
)
from train import (
    load_checkpoint, ENSEMBLE_SEEDS, SEQ_LEN, USE_TIMESFM_VOL, TIMESFM_DIM,
    TIMESFM_BLEND_ALPHA, TIMESFM_SIGNAL_SCALE, SIGNAL_POWER, SIGNAL_CLIP,
    ENSEMBLE_AGG, SIGNAL_EMA_DECAY, SIGNAL_THRESHOLD, compute_portfolio,
    _load_timesfm_vol_forecast,
)
from production_model import _aggregate_ensemble, apply_signal_pipeline, load_timesfm_features_by_date

device = torch.device("cpu")
print("Loading ensemble...")
models, config = [], None
for seed in ENSEMBLE_SEEDS:
    p = os.path.join("models", f"best_model_seed{seed}.pt")
    if os.path.exists(p):
        m, c = load_checkpoint(p, device); m.eval(); models.append(m); config = c
        print(f"  Loaded best_model_seed{seed}.pt")
print(f"  Ensemble size: {len(models)}")

tf = config.get("trade_frequency", "daily"); sl = config.get("seq_len", SEQ_LEN)
fc = config["feature_columns"]
print(f"\nLoading data (freq={tf})...")
features_df, targets_df = build_dataset(trade_frequency=tf)
features_df = validate_feature_columns(
    features_df, fc, context="year breakdown features"
)

sp = config["scaler_params"]
feat_norm = normalize_features(
    features_df, pd.Series(sp["mean"]), pd.Series(sp["std"])
)

if tf == "weekly":
    weekly_idx = feat_norm.index[::5]; feat_norm = feat_norm.loc[weekly_idx]; targets_df = targets_df.loc[weekly_idx]

mask = feat_norm.index >= VAL_END; indices = np.where(mask)[0]
start = indices[0]; end = indices[-1]
feat_np = feat_norm.values.astype(np.float32); tgt_np = targets_df.values.astype(np.float32); dates = feat_norm.index

# Load TimesFM pair-exhaustion features aligned to the weekly grid via closest-prior
# (single-source helper shared with live inference).
tsfm_by_date = load_timesfm_features_by_date(dates)

all_sigs, all_tgts, all_dates = [], [], []
for i in range(start, end + 1):
    w = feat_np[i - sl + 1 : i + 1]
    if len(w) < sl: continue
    x = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        stacked = torch.stack([m(x).cpu() for m in models])
        avg = _aggregate_ensemble(stacked)
    tsfm = tsfm_by_date.get(dates[i]) if tsfm_by_date else None
    avg = apply_signal_pipeline(avg, tsfm)
    all_sigs.append(avg); all_tgts.append(torch.tensor(tgt_np[i:i+1], dtype=torch.float32)); all_dates.append(dates[i])

all_sigs = torch.cat(all_sigs, dim=0); all_tgts = torch.cat(all_tgts, dim=0)
dates_arr = pd.DatetimeIndex(all_dates)
vol_forecast = _load_timesfm_vol_forecast(dates_arr)
weights, portfolio_returns = compute_portfolio(
    all_sigs, all_tgts, vol_forecast=vol_forecast
)
rets = portfolio_returns.numpy()

ppy = 52 if tf == "weekly" else 252
print(f"\n{'='*70}")
print(f"  YEAR-BY-YEAR BREAKDOWN")
print(f"{'='*70}")
print(f"  {'Year':<8} {'Periods':>8} {'CAGR':>12} {'Return':>12} {'Sharpe':>10} {'WinRate':>9} {'MaxDD':>9} {'Turnover':>10}")
print(f"  {'-'*78}")

years = sorted(set(dates_arr.year))
for y in years:
    ym = dates_arr.year == y
    r = rets[ym]; n = len(r)
    if n < 2: continue
    cum = np.cumprod(1 + r); tr = cum[-1] - 1
    yrs = n / ppy; cagr = (1 + tr) ** (1 / yrs) - 1 if yrs > 0 else 0
    mr, sr = r.mean(), r.std(); sh = (mr / max(sr, 1e-8)) * np.sqrt(ppy)
    wr = (r > 0).sum() / n
    rm = np.maximum.accumulate(cum); dd = (cum - rm) / np.maximum(rm, 1e-8); mdd = abs(dd.min())
    w = weights.numpy()[ym]; wc = np.abs(w[1:] - w[:-1]).sum(axis=1).mean() if len(w) > 1 else 0
    print(f"  {y:<8} {n:>8} {cagr:>+11.2%} {tr:>+11.2%} {sh:>9.3f} {wr:>8.1%} {mdd:>8.2%} {wc:>10.4f}")

# Full period
cum_all = np.cumprod(1 + rets); tr_all = cum_all[-1] - 1
yrs_all = len(rets) / ppy; cagr_all = (1 + tr_all) ** (1 / yrs_all) - 1
mr_all, sr_all = rets.mean(), rets.std(); sh_all = (mr_all / max(sr_all, 1e-8)) * np.sqrt(ppy)
wr_all = (rets > 0).sum() / len(rets)
rm_all = np.maximum.accumulate(cum_all); dd_all = (cum_all - rm_all) / np.maximum(rm_all, 1e-8); mdd_all = abs(dd_all.min())
w_all = weights.numpy(); wc_all = np.abs(w_all[1:] - w_all[:-1]).sum(axis=1).mean()
print(f"  {'-'*78}")
print(f"  {'ALL':<8} {len(rets):>8} {cagr_all:>+11.2%} {tr_all:>+11.2%} {sh_all:>9.3f} {wr_all:>8.1%} {mdd_all:>8.2%} {wc_all:>10.4f}")
print(f"{'='*70}")

# Per-year per-pair contribution
print(f"\n{'='*70}")
print(f"  PER-PAIR CONTRIBUTION BY YEAR (avg weekly return)")
print(f"{'='*70}")
header = f"  {'Year':<8}"
for pn in PAIR_NAMES:
    header += f" {pn:>10}"
print(header)
print(f"  {'-' * (8 + 10 * len(PAIR_NAMES))}")
w_np = weights.numpy()
for y in years:
    ym = np.where(dates_arr.year == y)[0]
    if len(ym) < 2: continue
    line = f"  {y:<8}"
    for i, (pn, (bull, bear)) in enumerate(zip(PAIR_NAMES, ETF_PAIRS)):
        pr = (w_np[ym, 2*i] * all_tgts.numpy()[ym, 2*i] + w_np[ym, 2*i+1] * all_tgts.numpy()[ym, 2*i+1]).mean()
        line += f" {pr:>+10.4f}"
    print(line)

# Per-year stats table
print(f"\n{'='*70}")
print(f"  MONTHLY RETURN DISTRIBUTION BY YEAR")
print(f"{'='*70}")
for y in years:
    ym = dates_arr.year == y
    r = rets[ym]
    if len(r) < 2: continue
    print(f"  {y}:  min={r.min():+.2%}  p25={np.percentile(r,25):+.2%}  median={np.median(r):+.2%}  p75={np.percentile(r,75):+.2%}  max={r.max():+.2%}  std={r.std():.2%}")

print(f"\n{'='*70}")
print(f"  HALF-YEAR BREAKDOWN")
print(f"{'='*70}")
print(f"  {'Period':<14} {'Weeks':>6} {'CAGR':>12} {'Return':>12} {'Sharpe':>10} {'WinRate':>9} {'MaxDD':>9}")
print(f"  {'-'*72}")
half_labels = [f"H1 {y}" for y in years[:1]] + [f"H2 {y}" for y in years] + [f"H1 {y}" for y in years if y > years[0]]
for y in years:
    for h in [1, 2]:
        if h == 1:
            ym = (dates_arr.year == y) & (dates_arr.month <= 6)
        else:
            ym = (dates_arr.year == y) & (dates_arr.month >= 7)
        r = rets[ym]; n = len(r)
        if n < 2: continue
        cum = np.cumprod(1 + r); tr = cum[-1] - 1
        yrs = n / ppy; cagr = (1 + tr) ** (1 / yrs) - 1 if yrs > 0 else 0
        mr, sr = r.mean(), r.std(); sh = (mr / max(sr, 1e-8)) * np.sqrt(ppy)
        wr = (r > 0).sum() / n
        rm = np.maximum.accumulate(cum); dd = (cum - rm) / np.maximum(rm, 1e-8); mdd = abs(dd.min())
        label = f"H{h} {y}"
        print(f"  {label:<14} {n:>6} {cagr:>+11.2%} {tr:>+11.2%} {sh:>9.3f} {wr:>8.1%} {mdd:>8.2%}")
print(f"{'='*70}\n")
