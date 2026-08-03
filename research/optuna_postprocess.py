"""Validation-only Optuna selection for disagreement-aware exposure."""

import csv
import json
from pathlib import Path

import optuna

import train as _train
from research.evidence import _baseline_reference, _score, _window_records


def _read_spec():
    return json.loads(Path("research/optuna_spec.json").read_text(encoding="utf-8"))


def _evaluate(value, spec, signals, targets, vol_forecast, dates, frequency, dispersion, metric_fn):
    if spec["parameter"] == "UNCERTAINTY_STRENGTH":
        signals = signals / (1.0 + value * dispersion)
    elif spec["parameter"] in {"VOL_GATE_THRESHOLD", "VOL_GATE_STRENGTH", "CASH_BIAS"}:
        setattr(_train, spec["parameter"], float(value))
    else:
        raise ValueError(f"Unsupported Optuna parameter: {spec['parameter']}")
    weights, returns = _train.compute_portfolio(
        signals, targets, vol_forecast=vol_forecast
    )
    returns_np = returns.detach().cpu().numpy()
    weights_np = weights.detach().cpu().numpy()
    metrics = metric_fn(
        returns_np,
        weights_np,
        52 if frequency == "weekly" else 252,
        frequency,
    )
    windows = _window_records(
        returns_np,
        dates,
        26 if frequency == "weekly" else 126,
    )
    baseline_turnover, _ = _baseline_reference(metrics)
    return _score(metrics, windows, baseline_turnover), weights, returns


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
    """Select disagreement strength on validation, then reuse it on test."""
    spec = _read_spec()
    artifact_dir = Path(".openresearch/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_path = artifact_dir / f"optuna_{spec['study_name']}_best.json"

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
            score, _, _ = _evaluate(
                value, spec, signals, targets, vol_forecast, dates,
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

    _, weights, returns = _evaluate(
        best_value, spec, signals, targets, vol_forecast, dates,
        frequency, dispersion, metric_fn,
    )
    return weights, returns
