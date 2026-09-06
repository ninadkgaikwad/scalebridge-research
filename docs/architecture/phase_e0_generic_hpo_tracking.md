# Phase E0-8 Generic HPO + Optuna + MLflow + Frozen Configuration

## Status and authority

This document describes the implementation of the ratified E0-8 contract:

`docs/mathematics/thermal_modeling/phase_e0/contracts/ScaleBridge_PhaseE0_E0-8_Generic_HPO_Tracking_FrozenConfiguration_Contract_v1.tex`

E0-8 is a **method-neutral orchestration layer**. It does not define the
hyperparameters, fitted-model mathematics, scientific objective, representative
subset policy, or final multi-objective selection rule of E.1/E.2/E.3/E.4.

The responsibility chain is:

```text
E.x provider defines HPO science
    -> E0-8 creates/runs/tracks the study
    -> E0-8 freezes h*
    -> E.x performs the final production fit
    -> E0-7 packages the fitted model and HPO provenance
```

## Production package

The implementation lives entirely under:

```text
src/scalebridge/tuning/e0_hpo/
```

No method-specific model package is imported by this layer.

### `contracts.py`

Defines study/data/objective/trial/frozen-configuration identities. Canonical
JSON and SHA-256 fingerprints reject NaN/Inf and unsupported unstable metadata.
The study fingerprint contains study identity, method/provider identity,
search-space snapshot, objective contract, Phase-D HPO-data fingerprint, study
seed, and the E0-8 schema version exactly as ratified.

### `provider.py`

Defines `BaseHPOProvider`, the interface future E.1/E.2/E.3/E.4 methods
implement. A provider owns:

- method/provider identity and version;
- immutable search-space snapshot;
- objective names/directions;
- exact Phase-D TRAIN-derived data selection;
- conditional hyperparameter suggestions;
- method-specific fit/score execution for one trial;
- whether pruning is scientifically meaningful;
- optional deterministic selection of one trial from a Pareto set.

The common layer contains no hyperparameter names belonging to a real method.

### `data_policy.py`

Reuses the E0-2 `TemporalOwnershipContract` as the scientific leakage guard.
E0-8 v1 requires the outer HPO source to be literal Phase-D `train` only.
Validation/test cannot influence HPO selection. Any representative subset,
inner fit/score split, or cross-validation policy remains provider-owned but is
fingerprinted in `selection_payload`.

### `optuna_backend.py`

Optuna is the optimization authority. E0-8 supplies generic sampler/pruner
registries and persistent-storage support. Resume requires persistent Optuna
storage, an already-existing study, and exact E0-8 fingerprint equality.
Existing studies with trials but without an E0-8 fingerprint are rejected
rather than adopted ambiguously.

When a persisted study is resumed, E0-8 recreates the Optuna sampler with a
deterministic **segment seed** derived from the immutable study seed, study ID,
and current trial count. This prevents a reconstructed seeded sampler from
replaying the first pseudo-random suggestion segment while preserving
reproducibility of the resumed segment. Trial-level method seeds remain derived
independently from study seed + study ID + absolute trial number.

### `mlflow_tracking.py`

MLflow is the experiment/provenance authority. When enabled, E0-8 creates one
study parent run and nested trial runs. It records study identity/fingerprint,
provider identity, objective/data/search fingerprints, sampled hyperparameters,
trial seeds/states, objective/diagnostic metrics, failures, and provider trial
artifacts. MLflow is never used to decide which trial is best.

### `artifacts.py`

Every study materializes the ratified generic artifacts:

```text
study_manifest.json
search_space_snapshot.json
objective_contract.json
data_selection_manifest.json
trials.parquet
study_summary.json
selection_manifest.json
frozen_hyperparameters.json        # when one final trial is selected
pareto_trials.parquet              # multi-objective studies
```

`FAILED`, `PRUNED`, and `COMPLETE` remain distinct in `trials.parquet`.
Provider artifact references are retained, and when MLflow is enabled the
artifacts are attached to the corresponding nested trial run.

### `runner.py`

`run_hpo_study()` is the generic orchestration entry point. It:

1. asks the provider for objective/search/data contracts;
2. enforces provider HPO-configuration compatibility;
3. enforces Phase-D TRAIN-only source ownership;
4. creates the immutable E0-8 study identity/fingerprint;
5. creates or scientifically resumes Optuna storage;
6. starts optional MLflow study/trial tracking;
7. derives a deterministic seed for every trial;
8. delegates suggestion + method fitting/scoring to the provider;
9. preserves `COMPLETE`, `PRUNED`, and recoverable `FAILED` states;
10. lets infrastructure exceptions stop the study;
11. asks Optuna for the single-objective best trial or multi-objective Pareto set;
12. asks the provider, never E0-8, for an optional Pareto final-selection rule;
13. writes standardized artifacts and freezes the selected hyperparameters.

A trial may be skipped after a method deliberately raises
`RecoverableTrialError`. Storage/tracking/programming errors are not converted
into ordinary bad trials.

### `handoff.py`

Produces a small HPO provenance record for future E.x exporters/E0-7 bundles.
It records study/fingerprint/data/search/objective identities, selected trial or
Pareto evidence, MLflow lineage, and the frozen-configuration SHA-256. E0-7 does
not rerun HPO.

## Reproducibility

A trial seed is deterministically derived from:

```text
(study seed, trial number, E0-8 study ID)
```

The method provider receives that seed and is responsible for applying it to
its own stochastic libraries where scientifically possible.

A resume request is accepted only when both the standardized artifact directory
and the persistent Optuna study represent the same immutable study fingerprint.
Changing provider version, search-space snapshot, objectives, HPO data
selection, study identity, seed, or E0-8 schema therefore creates an
incompatible study.

## Single vs multi-objective

For one objective, Optuna's best completed trial is frozen directly.

For multiple objectives, E0-8 always retains the Pareto set. A final frozen
configuration exists only when the E.x provider supplies a deterministic
selection rule and selects exactly one member of that Pareto set. Otherwise the
study intentionally ends with Pareto evidence and no false single `best` trial.

## What E0-8 does not own

E0-8 does not:

- define real ML/SciML/optimization/Bayesian hyperparameter names;
- define E.1/E.2/E.3/E.4 fitting algorithms;
- treat fitted physical/neural/posterior parameters as hyperparameters;
- decide what representative TRAIN subset is scientifically valid;
- create a scientific scalarization for a method;
- access Phase-D validation/test for HPO selection;
- fit the final production model;
- replace E0-7 portable packaging;
- replace E0-10 comprehensive real testing.

## Qualification boundary

The E0-8 standalone validator uses the controlled RestaurantFastFood/Buffalo
Phase-D product only to prove that the framework can consume real Phase-D
partition authority. Its synthetic HPO provider and `demo_*` hyperparameters
are framework fixtures only and are not production E.1-E.4 science.

E0-9 later qualifies infrastructure across the four ScaleBridge environments.
E0-10 remains the authoritative comprehensive real method/training/bundle/
evaluation testing campaign after E.1-E.4 exist.
