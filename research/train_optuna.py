"""Training-side Optuna harness (clean screen).

Retunes a training hyperparameter via env-var override. Each trial retrains
a reduced ensemble (3 seeds, 120s budget) from scratch (models cleared first)
and scores the validation backtest with the winner stack's fixed execution
controls. ARC_SEEDS is propagated to backtest so ensemble loading matches
exactly the trained seeds.
"""

import csv
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import optuna


SPEC = {
    "turn_pen": {"env": "ARC_TURN_PEN", "low": 0.0, "high": 60.0, "n_trials": 4},
    "ret_scale": {"env": "ARC_RET_SCALE", "low": 10.0, "high": 400.0, "n_trials": 4},
    "patience": {"env": "ARC_PATIENCE", "low": 3, "high": 30, "n_trials": 4, "int": True},
}

SEED = 20260803
SCREEN_SEEDS = "42,6,123"
SCREEN_TIME_BUDGET = 120

ARTIFACT_DIR = Path(".openresearch/artifacts")
MODELS_DIR = Path("models")


def _clear_models():
    for f in glob.glob(str(MODELS_DIR / "best_model_seed*.pt")):
        os.remove(f)


def _run_command(cmd, env_override):
    env = os.environ.copy()
    env.update(env_override)
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, env=env, cwd="."
    )
    return result.stdout + result.stderr


def _extract_score(output):
    for line in output.splitlines():
        if line.startswith("METRIC research_score="):
            return float(line.split("=")[1])
    return None


def run_trial(param_name, value):
    spec = SPEC[param_name]
    env_override = {
        spec["env"]: str(value),
        "ARC_SEEDS": SCREEN_SEEDS,
        "ARC_TIME_BUDGET": str(SCREEN_TIME_BUDGET),
        "PYTHONHASHSEED": "0",
    }
    _clear_models()
    output = _run_command(
        "/home/user/venv/bin/python train.py && /home/user/venv/bin/python backtest.py --val",
        env_override,
    )
    return _extract_score(output), output


def optimize(param_name):
    spec = SPEC[param_name]
    study_name = f"train_{param_name}"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = ARTIFACT_DIR / f"{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path.resolve().as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def objective(trial):
        if spec.get("int"):
            value = trial.suggest_int(spec["env"], int(spec["low"]), int(spec["high"]))
        else:
            value = trial.suggest_float(spec["env"], spec["low"], spec["high"])
        score, output = run_trial(param_name, value)
        if score is None:
            print(f"Trial {trial.number} failed: no score in output")
            print(output[-500:])
            raise optuna.TrialPruned()
        print(f"Trial {trial.number}: {spec['env']}={value} score={score:.6f}")
        return score

    study.optimize(objective, n_trials=spec["n_trials"], catch=(Exception,))

    best = study.best_trial
    print(f"BEST {param_name}: {spec['env']}={best.params[spec['env']]} score={best.value:.6f}")

    (ARTIFACT_DIR / f"{study_name}_best.json").write_text(json.dumps({
        "parameter": param_name,
        "env": spec["env"],
        "value": best.params[spec["env"]],
        "research_score": best.value,
        "trial": best.number,
        "n_trials": len(study.trials),
        "screen_seeds": SCREEN_SEEDS,
        "screen_time_budget": SCREEN_TIME_BUDGET,
    }, indent=2) + "\n", encoding="utf-8")

    with (ARTIFACT_DIR / f"{study_name}_trials.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["number", "state", "value", spec["env"]])
        for t in study.trials:
            writer.writerow([
                t.number, t.state.name,
                "" if t.value is None else f"{t.value:.12g}",
                t.params.get(spec["env"], ""),
            ])
    return best.params[spec["env"]], best.value


if __name__ == "__main__":
    import sys
    optimize(sys.argv[1])