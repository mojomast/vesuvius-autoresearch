# Research ML F: Bayesian Readiness

## Current Wiring

The Optuna stub is candidate-ordering only, not a full main-loop Bayesian optimizer.

- `src/autoresearch/search_strategy.py` defines `OptunaBayesianSearchStrategy.order_candidates()`.
- `autoresearch.py::_propose_configs()` calls `strategy_from_env(...).order_candidates(candidates, runs)` only when `AUTORESEARCH_SEARCH_STRATEGY` is non-heuristic.
- The strategy builds an in-memory `Study`, converts historical runs to completed trials with only `candidate_index=0`, asks for one integer `candidate_index`, and rotates the existing heuristic candidate list from that index.
- The main loop in `autoresearch.py::main()` still generates YAML configs, runs `run_experiment.py`, and relies on experiment DB rows afterward. It does not create durable Optuna trials, reserve asked params, pass trial identity into runs, or call `study.tell()` after an experiment finishes.
- `IMPROVEMENTS.md` correctly describes this limitation: Bayesian mode conservatively reorders existing bounded candidates and does not yet persist an Optuna trial ledger or perform full ask/tell experiment management.

The schema/proposal/strategy package modules are thin legacy exports. Real proposal generation, bounds, signatures, cost-tier filtering, and experiment execution still live in `autoresearch.py`.

## Readiness Assessment

The project is close to being able to support real Bayesian guidance because it already has:

- A typed config surface in `src/autoresearch/schemas.py`.
- `PARAM_BOUNDS` for safe numeric ranges.
- Search signatures for dedupe and reservation.
- Cost-tier guards before execution.
- Historical experiment rows with decoded config, metrics, artifact directory, and run id.
- A quality score `_run_quality_score()` that combines F1, AP, F0.5, calibration penalties, promotion evidence, and warnings.

The missing parts are:

- A stable Optuna storage or explicit trial table.
- Trial identity linked to generated config metadata.
- Conversion from historical DB rows into meaningful parameter vectors instead of a dummy `candidate_index`.
- A main-loop ask/run/tell path that asks Optuna for params, materializes a config, runs it, loads the resulting DB row, computes the objective, and tells Optuna the result.
- Handling for failed or skipped runs as failed/pruned trials.
- A fallback path when asked params produce duplicate signatures or violate cost/scope constraints.

## Recommended Search Space

Base the search space on `PARAM_BOUNDS`, but do not expose every key for every model family at once. Conditional spaces should start from the selected base config and preserve dataset scope.

Recommended continuous or log-scaled params:

- `training.learning_rate`: float, log scale, `0.001..0.25` globally; for torch runs use a narrower practical subrange such as `0.0002..0.006` if bounds are split by family.
- `training.weight_decay`: float, log-ish with explicit zero option, `0.0..0.05`.
- `training.pos_weight`: float, log scale, `0.25..25.0`; exclude when a model path treats `auto` specially unless converted to a numeric effective value.
- `training.dice_loss_weight`: float, `0.0..0.8`, torch only.
- `training.positive_rate_loss_weight`: float, `0.0..0.1`, torch only.
- `training.positive_rate_loss_tolerance`: float, log scale, `0.001..0.02`, torch only.
- `training.tversky_loss_weight`: float, `0.0..0.5`, torch only.
- `training.tversky_beta`: float, `0.1..0.9`, torch only; set `tversky_alpha = 1.0 - beta` when used.
- `evaluation.threshold`: float, `0.05..0.95`.
- `evaluation.max_pred_positive_rate_ratio`: float, `1.5..3.5`, when positive-rate cap is enabled.

Recommended integer/categorical params:

- `training.epochs`: int, `2..20`, but cap at the current cost-tier limit unless promotion/diversify context allows expensive runs.
- `training.batch_size`: categorical or int, `2..32`, torch only.
- `training.max_train_samples`: categorical, values such as `512`, `1024`, `2048`, `4096`, torch only and guarded by `AUTORESEARCH_TORCH_MAX_TRAIN_SAMPLES`.
- `training.max_train_pixels`: int/categorical, `100000..1200000`, NumPy only.
- `training.sample_positive_fraction`: float, `0.05..0.95`, NumPy MLP only.
- `training.hard_negative_fraction`: float, `0.1..0.9`, only when `training.sampling_strategy == hard_mining`.
- `model.base_channels`: categorical/int, `4..16`, torch only.
- `model.depth`: int, `1..3`; depths above 3 are already canonicalized away.
- `model.hidden_units`: int/categorical, `8..96`, `tiny_numpy_mlp` only.
- `model.name`: categorical only for explicit family switches such as `tiny_numpy_ink_logreg`, `tiny_numpy_mlp`, and selected torch bases; do not mix model families in the same dense numeric study unless encoded conditionally.

Recommended exclusions:

- Do not optimize dataset paths, `dataset.research_scope`, validation mode, `resolved_data`, or `validation_setup` inside the Bayesian parameter vector.
- Do not optimize `training.seed`, `training.seeds`, or `training.deterministic` as model-quality parameters. Treat seed repeats as replication/promotion evidence, not hyperparameter improvement.
- Do not optimize `evaluation.tta_flips` and `training.augment_flips` in the first real Bayesian version unless represented as explicit categorical booleans with cost-tier accounting.
- Do not optimize `BALANCED_CALIBRATION_PATH` as a synthetic path. Represent its component fields directly.
- Do not ask Optuna for configs that fail `_cost_tier_allowed()`, duplicate `_search_signature()`, violate required mutation families, or leave the selected scope.

## Objective Choices

Single objective is the best minimum implementation.

- Use `_run_quality_score(run)` as the first objective because it already encodes the current promotion-aware tradeoff.
- Keep `direction="maximize"`.
- Store component metrics in `trial.user_attrs` for later analysis: `val_f1`, `average_precision`, `val_f05`, `precision`, `recall`, positive-rate ratio, fixed-threshold status, promotion eligibility, cost tier, scope, and run id.
- If historical backtesting shows `_run_quality_score()` is too opinionated, use `val_f1 + 0.25 * average_precision + 0.10 * val_f05` plus the existing calibration penalties as a transparent variant.

Multi-objective can come later.

- Candidate objectives: maximize `val_f1`, maximize `average_precision`, maximize `val_f05` or precision, minimize positive-rate calibration error, and minimize cost.
- Use NSGA-II only after the single-objective loop has correct trial persistence and enough completed trials.
- Pareto selection still needs search-signature dedupe, scope policy, mutation-family diversity, and cost guards, so it is not a simpler replacement for the existing proposal pipeline.

## Minimum Implementation For Real Bayesian Guidance

The smallest useful implementation is not to replace the entire proposal system. It is to add one ask/run/tell path that generates one full config from Optuna params and falls back to heuristic proposals when no safe novel trial can be asked.

Minimum pieces:

- Add persistent Optuna storage, e.g. `sqlite:///experiments/optuna.db`, or equivalent tables in `experiments/experiments.db`.
- Define a stable study name that includes model family and scope, such as `autoresearch:{scope}:{model_family}`.
- Implement `run_to_trial(run)` that extracts params from `run["config"]`, skips rows missing required params for the active conditional space, computes `_run_quality_score(run)`, and stores `run_id` as a user attr.
- Backfill historical completed trials idempotently by checking `run_id` user attrs or a side table.
- Implement `ask_config(base, study, runs, context)` that calls `study.ask()`, samples conditional params, materializes a config, applies the same post-processing currently done in `_propose_configs()`, and rejects duplicates/cost violations.
- Add `autoresearch.optuna_trial_number`, `autoresearch.optuna_study_name`, and `autoresearch.search_strategy = bayesian_ask_tell` to generated config metadata.
- After `_run_experiment_checked(cfg_path)`, reload recent runs, find the new run by search signature or latest artifact/config metadata, compute objective, and call `study.tell(trial, value)`.
- On subprocess failure, call `study.tell(trial, state=TrialState.FAIL)` before re-raising or continuing according to existing behavior.
- Keep heuristic candidate-ordering mode separate or remove it once ask/tell is stable; do not conflate candidate-index trials with real hyperparameter trials.

## Ask/Tell Pseudocode

```python
def load_bayesian_study(scope, model_family):
    sampler = optuna.samplers.TPESampler(seed=20260529, multivariate=True, group=True)
    return optuna.create_study(
        study_name=f"autoresearch:{scope}:{model_family}",
        storage="sqlite:///experiments/optuna.db",
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )


def backfill_completed_trials(study, runs, search_space):
    known_run_ids = {
        trial.user_attrs.get("run_id")
        for trial in study.trials
        if trial.user_attrs.get("run_id")
    }
    for run in runs:
        run_id = str(run.get("run_id") or "")
        if not run_id or run_id in known_run_ids:
            continue
        params = extract_params_from_config(run["config"], search_space)
        if params is None:
            continue
        value = _run_quality_score(run)
        trial = optuna.trial.create_trial(
            params=params,
            distributions=search_space.distributions_for(params),
            value=value,
            user_attrs={
                "run_id": run_id,
                "search_signature": list(_search_signature(run["config"])),
                "metrics": compact_metrics(run),
            },
        )
        try:
            study.add_trial(trial)
        except ValueError:
            pass


def ask_bayesian_proposal(study, base, runs, context):
    tested = _reserved_signatures(runs)
    for attempt in range(32):
        trial = study.ask()
        cfg = copy.deepcopy(base)
        params = sample_conditional_params(trial, cfg)
        apply_params_to_config(cfg, params)
        normalize_dependent_fields(cfg)
        cfg.pop("resolved_data", None)
        cfg.pop("validation_setup", None)
        if context.lock_to_baseline_scope:
            cfg["dataset"] = copy.deepcopy(load_config(BASELINE).get("dataset", {}))
        cost_tier = _config_cost_tier(cfg)
        signature = _search_signature(cfg)
        if signature in tested or not _cost_tier_allowed(cost_tier, allow_expensive=context.allow_expensive):
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        cfg.setdefault("autoresearch", {}).update({
            "search_strategy": "bayesian_ask_tell",
            "optuna_study_name": study.study_name,
            "optuna_trial_number": trial.number,
            "search_signature": list(signature),
            "cost_tier": cost_tier,
            "intent": "cron_exploration",
            "run_profile": "exploration",
            "promotable": False,
            "proposal_status": "generated",
        })
        return trial, cfg, "Optuna TPE ask over bounded autoresearch search space"
    return None


def run_one_bayesian_trial(base, runs, context):
    scope = str(base.get("dataset", {}).get("research_scope") or context.scope_policy)
    model_family = str(_get_nested(base, ("model", "name"), "unknown"))
    study = load_bayesian_study(scope, model_family)
    search_space = build_search_space(base, PARAM_BOUNDS)
    backfill_completed_trials(study, runs, search_space)
    asked = ask_bayesian_proposal(study, base, runs, context)
    if asked is None:
        return heuristic_fallback(base, runs, context)
    trial, cfg, reason = asked
    cfg_path = write_generated_config(cfg, reason)
    try:
        _run_experiment_checked(cfg_path)
    except subprocess.CalledProcessError:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        safe_unlink(cfg_path)
        raise
    latest_runs = _recent_runs(limit=0)
    run = find_completed_run_for_trial(latest_runs, trial.number, _search_signature(cfg))
    if run is None:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        raise RuntimeError("Bayesian trial completed subprocess but no matching DB row was found")
    value = _run_quality_score(run)
    trial.set_user_attr("run_id", run["run_id"])
    trial.set_user_attr("metrics", compact_metrics(run))
    trial.set_user_attr("search_signature", list(_search_signature(run["config"])))
    study.tell(trial, value)
    return run
```

## Recommendation

Keep the current candidate-ordering Optuna mode as experimental, but do not treat it as Bayesian optimization. The next ML-search milestone should be a narrow single-objective ask/tell loop over `PARAM_BOUNDS`, conditionalized by model family, with historical DB backfill and persistent trial identity. That is the minimum change that would let Optuna guide actual parameter selection rather than only rotate the order of hand-written candidates.
