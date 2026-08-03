"""Sub-period diagnostic for overfitting checklist Item 3.
Reports ensemble Calmar/CAGR/MaxDD split by half-year within the val window.
Read-only — does NOT modify any off-limits file. Does NOT emit METRIC calmar."""
import os, numpy as np, pandas as pd, torch
from prepare import (
    NUM_PAIRS, ETF_PAIRS, PAIR_NAMES, TRAIN_END, VAL_END, build_dataset,
    normalize_features, validate_feature_columns,
)
from train import (
    load_checkpoint, ENSEMBLE_SEEDS, SEQ_LEN, USE_TIMESFM_VOL, TIMESFM_BLEND_ALPHA,
    TIMESFM_SIGNAL_SCALE, SIGNAL_POWER, SIGNAL_CLIP, ENSEMBLE_AGG, SIGNAL_EMA_DECAY,
    compute_portfolio, _load_timesfm_vol_forecast,
)
from production_model import _aggregate_ensemble, apply_signal_pipeline, load_timesfm_features_by_date
from backtest import load_ensemble, compute_metrics

def main():
    device = torch.device("cpu")
    models, config = load_ensemble(device)
    seq_len = config.get("seq_len", SEQ_LEN); fc = config["feature_columns"]; tf = config.get("trade_frequency","weekly")
    features_df, targets_df = build_dataset(trade_frequency=tf)
    features_df = validate_feature_columns(
        features_df, fc, context="sub-period diagnostic features"
    )
    sp = config["scaler_params"]; mean = pd.Series(sp["mean"]); std = pd.Series(sp["std"])
    fn = normalize_features(features_df, mean, std)
    if tf == "weekly": wi = fn.index[::5]; fn = fn.loc[wi]; targets_df = targets_df.loc[wi]
    feat_np = fn.values.astype(np.float32); tgt_np = targets_df.values.astype(np.float32); dates = fn.index
    tsf = load_timesfm_features_by_date(dates); ppy = 52 if tf == "weekly" else 252
    te, ve = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
    sigs, tgts, dts = [], [], []; ema = None
    for i in range(seq_len-1, len(feat_np)):
        if dates[i] < te or dates[i] >= ve: continue
        x = torch.tensor(feat_np[i-seq_len+1:i+1], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): st = torch.stack([m(x).cpu() for m in models])
        a = _aggregate_ensemble(st)
        if SIGNAL_EMA_DECAY > 0:
            ema = a.clone() if ema is None else SIGNAL_EMA_DECAY*ema + (1-SIGNAL_EMA_DECAY)*a; a = ema
        a = apply_signal_pipeline(a, tsf.get(dates[i]) if tsf else None)
        sigs.append(a); tgts.append(torch.tensor(tgt_np[i:i+1], dtype=torch.float32)); dts.append(dates[i])
    es = torch.cat(sigs,dim=0); et = torch.cat(tgts,dim=0); ed = pd.DatetimeIndex(dts)
    vol_forecast = _load_timesfm_vol_forecast(ed)
    w, portfolio_returns = compute_portfolio(es, et, vol_forecast=vol_forecast)
    pr = portfolio_returns.numpy(); wn = w.numpy()
    m = compute_metrics(pr, wn, ppy, tf)
    print(f"\n{'='*70}\n  ENSEMBLE VAL ({ed[0].date()}->{ed[-1].date()}, {m['n_periods']}p)\n{'='*70}")
    print(f"  Calmar={m['cagr']/max(m['max_drawdown'],0.01):.3f}  CAGR={m['cagr']:+.2%}  MaxDD={m['max_drawdown']:.2%}  Sharpe={m['sharpe']:.2f}")
    print(f"\n{'='*70}\n  ITEM 3: SUB-PERIOD ANALYSIS\n{'='*70}")
    subs = [("2022-H1 (rate-hike start)","2022-01-01","2022-07-01"),("2022-H2 (bear bottom)","2022-07-01","2023-01-01"),
            ("2023-H1 (recovery)","2023-01-01","2023-07-01"),("2023-H2 (bull resume)","2023-07-01","2024-01-01")]
    print(f"  {'Sub-period':<32} {'N':>4} {'Calmar':>8} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'WinRate':>8}")
    print(f"  {'-'*88}")
    all_pos = True
    for label, s, e in subs:
        mask = (ed >= pd.Timestamp(s)) & (ed < pd.Timestamp(e))
        if mask.sum() < 4: print(f"  {label:<32} {int(mask.sum()):>4}  (too few)"); continue
        ms = compute_metrics(pr[mask], wn[mask], ppy, tf)
        c = ms['cagr']/max(ms['max_drawdown'],0.01)
        if ms['cagr'] <= 0: all_pos = False
        print(f"  {label:<32} {ms['n_periods']:>4} {c:>8.3f} {ms['cagr']:>+8.2%} {ms['max_drawdown']:>8.2%} {ms['sharpe']:>8.2f} {ms['win_rate']:>8.1%}")
    print(f"  {'-'*88}\n  [All positive CAGR = robust. Any negative = regime-specific red flag.]")
    print(f"  ALL_SUBPERIODS_POSITIVE={'YES' if all_pos else 'NO (regime-specific)'}")
    print(f"{'='*70}")

if __name__ == "__main__": main()
