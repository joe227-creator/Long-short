"""Validation-only Optuna selection for cost-aware portfolio exposure."""

import csv
import json
from pathlib import Path

import numpy as np
import optuna
import torch

import train as _train
from research.evidence import _baseline_reference, _score, _window_records
from research.execution_controls import (
    apply_partial_adjustment,
    apply_state_band,
    apply_weight_band,
)


def _read_spec():
    return json.loads(Path("research/optuna_spec.json").read_text(encoding="utf-8"))


def _apply_hysteresis(signals, band):
    return apply_state_band(signals, band)


def _evaluate(split, value, spec, signals, targets, vol_forecast, dates, frequency, dispersion, metric_fn):
    fixed_strength = spec.get("fixed_uncertainty_strength")
    if fixed_strength is not None:
        signals = signals / (1.0 + float(fixed_strength) * dispersion)
    if spec["parameter"] == "HYSTERESIS":
        signals = _apply_hysteresis(signals, float(value))
    elif spec["parameter"] == "UNCERTAINTY_STRENGTH":
        signals = signals / (1.0 + value * dispersion)
    elif spec["parameter"] == "PARTIAL_ADJUSTMENT":
        strength = float(spec.get("uncertainty_strength", 0.0))
        signals = signals / (1.0 + strength * dispersion)
    elif spec["parameter"] in {"VOL_GATE_THRESHOLD", "VOL_GATE_STRENGTH", "CASH_BIAS"}:
        setattr(_train, spec["parameter"], float(value))
    elif spec["parameter"] == "WEIGHT_BAND":
        pass
    else:
        raise ValueError(f"Unsupported Optuna parameter: {spec['parameter']}")
    weights, returns = _train.compute_portfolio(
        signals, targets, vol_forecast=vol_forecast
    )
    partial_rate = spec.get("fixed_partial_adjustment")
    if spec["parameter"] == "PARTIAL_ADJUSTMENT":
        partial_rate = value
    if partial_rate is not None and len(weights) > 1:
        weights = apply_partial_adjustment(weights, partial_rate)
        returns = (weights * targets).sum(dim=1)
    if spec["parameter"] == "WEIGHT_BAND":
        weights = apply_weight_band(weights, value)
        returns = (weights * targets).sum(dim=1)
    returns_np = returns.detach().cpu().numpy()
    weights_np = weights.detach().cpu().numpy()
    turnover = np.zeros(len(returns_np), dtype=float)
    if len(weights_np) > 1:
        turnover[1:] = np.abs(weights_np[1:] - weights_np[:-1]).sum(axis=1)
    cost_rate = float(spec["cost_bps"]) / 10000.0
    net_returns_np = returns_np - cost_rate * turnover
    net_returns = torch.as_tensor(net_returns_np, dtype=returns.dtype)
    gross_metrics = metric_fn(
        returns_np,
        weights_np,
        52 if frequency == "weekly" else 252,
        frequency,
    )
    net_metrics = metric_fn(
        net_returns_np,
        weights_np,
        52 if frequency == "weekly" else 252,
        frequency,
    )
    window_periods = 26 if frequency == "weekly" else 126
    gross_score = _score(
        gross_metrics,
        _window_records(returns_np, dates, window_periods),
        _baseline_reference(gross_metrics)[0],
    )
    net_score = _score(
        net_metrics,
        _window_records(net_returns_np, dates, window_periods),
        _baseline_reference(net_metrics)[0],
    )
    stress = []
    for bps in spec.get("stress_bps", [spec["cost_bps"]]):
        stress_returns = returns_np - (float(bps) / 10000.0) * turnover
        stress_metrics = metric_fn(
            stress_returns,
            weights_np,
            52 if frequency == "weekly" else 252,
            frequency,
        )
        stress_score = _score(
            stress_metrics,
            _window_records(stress_returns, dates, window_periods),
            _baseline_reference(stress_metrics)[0],
        )
        stress.append({"bps": bps, "score": stress_score})
    Path(".openresearch/artifacts").mkdir(parents=True, exist_ok=True)
    Path(f".openresearch/artifacts/cost_overlay_{split}.json").write_text(
        json.dumps({
            "cost_bps": spec["cost_bps"],
            "selected_parameter": spec["parameter"],
            "selected_value": value,
            "mean_period_cost": float((cost_rate * turnover).mean()),
            "gross": {"metrics": gross_metrics, "score": gross_score},
            "net": {"metrics": net_metrics, "score": net_score},
            "stress": stress,
        }, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return net_score, weights, net_returns, gross_score


def _write_trials(path, study):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "state", "value", "parameter"])
        for trial in study.trials:
            writer.writerow([
                trial.number,
                trial.state.name,
                "" if trial.value is None else f"{trial.value:.12g}",
                trial.params.get(study.user_attrs.get("parameter", ""), ""),
            ])


def optimize_or_load(split, signals, targets, vol_forecast, dates, frequency, dispersion, metric_fn):
    """Select disagreement strength on net validation, then reuse on test."""
    spec = _read_spec()
    artifact_dir = Path(".openresearch/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_path = artifact_dir / f"optuna_{spec['study_name']}_best.json"

    if spec.get("selected_value") is not None:
        best_value = float(spec["selected_value"])
        print(f"OPTUNA fixed {spec['parameter']}={best_value:.8g} (selected_value preset)")
        _, weights, returns, _ = _evaluate(
            split, best_value, spec, signals, targets, vol_forecast, dates,
            frequency, dispersion, metric_fn,
        )
        print(f"COST_BPS={spec['cost_bps']} net_score_artifact=cost_overlay_{split}.json")
        return weights, returns

    if split == "val":
        db_path = artifact_dir / f"optuna_{spec['study_name']}.db"
        study = optuna.create_study(
            study_name=spec["study_name"],
            storage=f"sqlite:///{db_path.resolve().as_posix()}",
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(spec["seed"])),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        )
        study.set_user_attr("parameter", spec["parameter"])

        def objective(trial):
            value = trial.suggest_float(
                spec["parameter"],
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
            score, _, _, _ = _evaluate(
                split, value, spec, signals, targets, vol_forecast, dates,
                frequency, dispersion, metric_fn,
            )
            trial.set_user_attr("sharpe", score["sharpe"])
            trial.set_user_attr("turnover", score["turnover"])
            trial.report(score["research_score"], step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score["research_score"]

        study.optimize(objective, n_trials=int(spec["n_trials"]), catch=(ValueError,))
        if study.best_trial is None:
            raise RuntimeError("Optuna produced no completed trials")
        best_value = study.best_trial.params[spec["parameter"]]
        best_score = study.best_value
        best_path.write_text(json.dumps({
            "parameter": spec["parameter"],
            "value": best_value,
            "research_score": best_score,
            "trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "seed": spec["seed"],
        }, indent=2) + "\n", encoding="utf-8")
        _write_trials(artifact_dir / f"optuna_{spec['study_name']}_trials.csv", study)
        print(f"OPTUNA best {spec['parameter']}={best_value:.8g} score={best_score:.8f}")
    else:
        if not best_path.exists():
            raise RuntimeError("Validation Optuna result missing before test evaluation")
        best_value = json.loads(best_path.read_text(encoding="utf-8"))["value"]
        print(f"OPTUNA reused {spec['parameter']}={float(best_value):.8g}")

    _, weights, returns, _ = _evaluate(
        split, best_value, spec, signals, targets, vol_forecast, dates,
        frequency, dispersion, metric_fn,
    )
    print(f"COST_BPS={spec['cost_bps']} net_score_artifact=cost_overlay_{split}.json")
    return weights, returns
