"""
Optuna hyperparameter sweep for train_model.py.

Usage:
  python sweep_params.py [--n_trials 100] [--db sqlite:///sweep.db] [--save_dir sweep_runs]

Parallel workers: run multiple processes pointing at the same --db.
Monitor with: optuna-dashboard <db>
"""

import argparse
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import optuna
from train_model import Config, Pruned, _log, _preprocess_data, train


class SweepConfig:
    n_trials: int = 100
    db: str = "sqlite:///sweep.db"
    save_dir: str = "sweep_runs"
    study_name: str = "ppo_stock_sweep"
    sampler_startup_trials: int = 10
    pruner_startup_trials: int = 5
    pruner_warmup_steps: int = 20


SEARCH_SPACE = {
    "dropout":       (0.0, 0.2, False),
    "train_peers":   (1, 10, False),
    "value_coeff":   (0.01, 1.0, True),
    "entropy_coeff": (0.001, 0.1, True),
    "lr":            (1e-5, 1e-3, True),
}


def _param_hash(params):
    keys = sorted(SEARCH_SPACE.keys())
    serialized = json.dumps({k: params[k] for k in keys}).encode()
    return hashlib.md5(serialized).hexdigest()[:12]


def objective(trial, preprocess_path, base_save_dir):
    cfg = Config()

    for name, (low, high, log) in SEARCH_SPACE.items():
        setattr(cfg, name, trial.suggest_float(name, low, high, log=log))

    cfg.save_dir = os.path.join(base_save_dir, _param_hash(trial.params))
    cfg.ablation = False
    cfg.grad_diag_freq = 0.0

    _log()
    _log(f"[Trial {trial.number}]")

    def epoch_callback(epoch, score):
        trial.report(score, epoch)
        if trial.should_prune():
            _log(f"    Pruned at epoch {epoch}")
            raise Pruned(f"Pruned at epoch {epoch}")

    try:
        best_score = train(
            cfg,
            preprocess_path=preprocess_path,
            epoch_callback=epoch_callback,
        )
    except Pruned:
        raise optuna.TrialPruned()

    return best_score


def main():
    scfg = SweepConfig()
    parser = argparse.ArgumentParser(description="Optuna sweep for train_model")

    for name, ann_type in SweepConfig.__annotations__.items():
        default = getattr(scfg, name)
        flag = f"--{name}"
        parser.add_argument(flag, type=ann_type, default=default,
                            help=f"(default: {default})")

    args = parser.parse_args()
    for name in SweepConfig.__annotations__:
        setattr(scfg, name, getattr(args, name))

    os.makedirs(scfg.save_dir, exist_ok=True)

    cfg = Config()
    cfg.save_dir = scfg.save_dir
    preprocess_path = _preprocess_data(cfg)

    study = optuna.create_study(
        study_name=scfg.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=scfg.sampler_startup_trials),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=scfg.pruner_startup_trials,
            n_warmup_steps=scfg.pruner_warmup_steps,
        ),
        storage=scfg.db,
        load_if_exists=True,
    )

    # Re-enqueue interrupted trials that have a checkpoint on disk
    retryable = (optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING)
    for t in study.trials:
        if t.state in retryable and t.params:
            h = _param_hash(t.params)
            if os.path.exists(os.path.join(scfg.save_dir, h, "model_latest.pt")):
                study.enqueue_trial(t.params)
                _log()
                _log(f"Re-enqueued trial {t.number} ({h})")

    def _after_trial(study, trial):
        plot_sweep(study, scfg.save_dir)

    study.optimize(
        lambda trial: objective(trial, preprocess_path, scfg.save_dir),
        n_trials=scfg.n_trials,
        callbacks=[_after_trial],
    )

    if os.path.exists(preprocess_path):
        os.remove(preprocess_path)

    _log()
    _log("[Sweep Summary]")
    _log(f"    {'Best trial':<20s}: {study.best_trial.number}")
    _log(f"    {'Best score':<20s}: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        _log(f"    {key:<20s}: {value}")


def plot_sweep(study, out_dir):
    """Plot sweep diagnostics: optimization history, parameter importances,
    parallel coordinate."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        return

    param_names = sorted(completed[0].params.keys())
    scores = np.array([t.value for t in completed])
    param_vals = {k: np.array([t.params[k] for t in completed])
                  for k in param_names}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Optimization history
    trial_numbers = [t.number for t in completed]
    axes[0].scatter(trial_numbers, scores, c=scores, cmap="RdYlGn",
                    s=30, alpha=0.7)
    best_so_far = np.maximum.accumulate(scores)
    axes[0].plot(trial_numbers, best_so_far, linewidth=2, label="Best so far")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Optimization History")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # 2. Parameter importances (correlation-based approximation)
    importances = {}
    for k, vals in param_vals.items():
        if vals.std() > 0:
            importances[k] = abs(float(np.corrcoef(vals, scores)[0, 1]))
        else:
            importances[k] = 0.0
    sorted_params = sorted(importances, key=importances.get, reverse=True)
    imp_vals = [importances[k] for k in sorted_params]
    axes[1].barh(range(len(sorted_params)), imp_vals, color="steelblue")
    axes[1].set_yticks(range(len(sorted_params)))
    axes[1].set_yticklabels(sorted_params)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Importance (|correlation|)")
    axes[1].set_title("Parameter Importances")
    axes[1].grid(True, alpha=0.3, axis="x")

    # 3. Parallel coordinate
    all_axes = param_names + ["score"]
    all_vals = {**param_vals, "score": scores}
    normed = {}
    for k, v in all_vals.items():
        vmin, vmax = v.min(), v.max()
        normed[k] = (v - vmin) / (vmax - vmin + 1e-10)

    ax = axes[2]
    x = np.arange(len(all_axes))
    norm = plt.Normalize(scores.min(), scores.max())
    cmap = plt.cm.RdYlGn
    for i in range(len(completed)):
        y = [normed[k][i] for k in all_axes]
        ax.plot(x, y, c=cmap(norm(scores[i])), alpha=0.5, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(all_axes, rotation=30, ha="right")
    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel Coordinate (green=high score)")
    ax.grid(True, alpha=0.3)

    for a in axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(out_dir, "sweep_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"    {'Saved file':<20s}: {path}")


if __name__ == "__main__":
    main()
