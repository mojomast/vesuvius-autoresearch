from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AutoResearch Optuna studies.")
    parser.add_argument("--storage", default="logs/optuna.db", help="Optuna SQLite DB path")
    parser.add_argument("--study", default=None, help="Study name; prints all studies when omitted")
    args = parser.parse_args()

    import optuna

    storage = f"sqlite:///{Path(args.storage)}"
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if not summaries:
        print("No Optuna studies found.")
        return 0
    names = [args.study] if args.study else [summary.study_name for summary in summaries]
    for name in names:
        study = optuna.load_study(study_name=name, storage=storage)
        print(f"study={name} trials={len(study.trials)} directions={[str(d) for d in study.directions]}")
        try:
            best = study.best_trial
        except RuntimeError:
            best = None
        if best is not None:
            print(f"best trial={best.number} value={best.value} params={best.params} run_id={best.user_attrs.get('run_id')}")
        if len(study.directions) > 1:
            for trial in study.best_trials:
                print(f"pareto trial={trial.number} values={trial.values} params={trial.params} run_id={trial.user_attrs.get('run_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
