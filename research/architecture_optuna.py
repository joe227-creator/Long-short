"""Clean categorical Optuna screen for temporal model architectures."""

import csv
import glob
import json
import os
import subprocess
from pathlib import Path

import optuna


SEED = 20260803
SEEDS = "42,6,123"
TIME_BUDGET = 120
ARCHITECTURES = ["lstm", "gru", "lstm_attn", "tcn", "tcn_vsn"]
ARTIFACT_DIR = Path(".openresearch/artifacts")


def _clear_models():
    for path in glob.glob("models/best_model_seed*.pt"):
        os.remove(path)


def _score(output):
    for line in output.splitlines():
        if line.startswith("METRIC research_score="):
            return float(line.split("=", 1)[1])
    return None


def _run(model_type):
    _clear_models()
    env = os.environ.copy()
    env.update({
        "ARC_MODEL_TYPE": model_type,
        "ARC_SEEDS": SEEDS,
        "ARC_TIME_BUDGET": str(TIME_BUDGET),
        "PYTHONHASHSEED": "0",
    })
    result = subprocess.run(
        "/home/user/venv/bin/python train.py && /home/user/venv/bin/python backtest.py --val",
        shell=True,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout + result.stderr
    return _score(output), output


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="temporal_architecture",
        storage=f"sqlite:///{(ARTIFACT_DIR / 'temporal_architecture.db').resolve().as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def objective(trial):
        model_type = trial.suggest_categorical("ARC_MODEL_TYPE", ARCHITECTURES)
        score, output = _run(model_type)
        if score is None:
            print(output[-1000:])
            raise optuna.TrialPruned()
        print(f"Trial {trial.number}: ARC_MODEL_TYPE={model_type} score={score:.8f}")
        return score

    study.optimize(objective, n_trials=8, catch=(Exception,))
    best = study.best_trial
    result = {
        "parameter": "ARC_MODEL_TYPE",
        "value": best.params["ARC_MODEL_TYPE"],
        "research_score": best.value,
        "trial": best.number,
        "n_trials": len(study.trials),
        "seeds": SEEDS,
        "time_budget": TIME_BUDGET,
    }
    (ARTIFACT_DIR / "temporal_architecture_best.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (ARTIFACT_DIR / "temporal_architecture_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "state", "value", "ARC_MODEL_TYPE"])
        for trial in study.trials:
            writer.writerow([
                trial.number, trial.state.name,
                "" if trial.value is None else f"{trial.value:.12g}",
                trial.params.get("ARC_MODEL_TYPE", ""),
            ])
    print(f"BEST ARC_MODEL_TYPE={result['value']} score={result['research_score']:.8f}")


if __name__ == "__main__":
    main()
