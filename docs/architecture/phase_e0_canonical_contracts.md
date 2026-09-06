# Phase E.0 E0-2 — Canonical Data + Scientific Contracts

## Status

E0-2 defines the method-neutral boundary between the authoritative Phase D final
thermal-model data products and all later Phase E model families.

It does **not** implement a thermal model, RC topology, PINN, Neural ODE,
optimization estimator, Bayesian estimator, or simulator.

The E0-2 contract is intentionally shared by future:

- ML methods,
- SciML methods,
- optimization methods,
- Bayesian methods.

Before any specific method family or method is implemented, its detailed
mathematics must be developed and locked in a versioned `.tex` contract.  E0-2
provides only the cross-family semantics those later math contracts consume.

## Authority boundary

Phase D remains authoritative for:

- final `data.parquet` / `manifest.json` realizations,
- state/control/disturbance column semantics,
- zone-specific feature availability,
- spatial product identity (`independent`, `dependent1`, `dependent2`),
- heat-input representation,
- lag and target horizon,
- temporal-policy identity,
- train/validation/test/excluded outer partition labels,
- aggregation and Phase C lineage carried in the manifest.

Phase E consumes those definitions.  Phase E must not reconstruct them from
folder names or redefine the outer split.

## Canonical physical signals

| Signal | E0 role | Units | Physical domain | Locked interpretation |
|---|---|---:|---|---|
| `zone_temperature` | observed state | degC | temperature | measured zone-air state supplied by Phase D |
| `outdoor_temperature` | disturbance | degC | temperature | common boundary temperature |
| `qac` | control input | W | thermal power | signed delivered HVAC heat; positive heating, negative cooling |
| `qsol1` | disturbance | W | thermal power | positive thermal addition |
| `qsol2` | disturbance | W | thermal power | positive thermal addition |
| `zic` | disturbance | W | thermal power | grouped convective/internal thermal addition |
| `zir` | disturbance | W | thermal power | grouped radiative/internal thermal addition |
| `qzic_*`, `qzir_*`, `qzivr_l` | disturbance | W | thermal power | component representation when Phase D exposes it |
| `phvac` | auxiliary output | W | electrical power | HVAC electrical power; never a thermal balance input |

### QAC versus PHVAC

This distinction is non-negotiable.

`qac` is the thermal HVAC quantity used by thermal dynamics.

`phvac` is an electrical-power quantity produced by the corresponding Phase C
runtime/model.  It may be composed downstream for evaluation, grid studies, or
control objectives, but E0-2 explicitly rejects PHVAC as a thermal model input.

## Exact manifest-derived bindings

E0-2 never resolves model features by substring matching.

A Phase E input or target is bound by the exact Phase D manifest tuple:

- base signal,
- aggregate zone ID,
- temporal role,
- lag/target offset,
- materialized column name,
- units,
- Phase D physical role.

Example:

```text
base_signal       = qac
aggregate_zone_id = Dining
lag               = 0
column_name       = Dining__qac__lag_0
```

If that exact binding does not exist, Phase E reports it as unavailable for that
realization.  It does not select a vaguely similar column and does not silently
insert zero.

## Zone-specific availability

Phase D already supports mixed applicability across zones.  E0-2 preserves that.

Examples that are legal:

- one zone has QAC and another is structurally uncontrolled;
- one zone has solar terms and another does not;
- grouped heat signals may exist for one zone but not another;
- Dep2 uses current-zone state/QAC with disturbances from a compatible
  all-to-one source zone.

Therefore no Phase E method may assume that all zones share an identical feature
vector.  A later method contract must explicitly declare its required and
optional signals and validate them against the E0-2 bindings.

## Spatial products

### Independent

One final Phase D product models one aggregate zone.

E0-2 keeps only that product zone as `modeled_zone_ids`, even though a Phase D
manifest may retain the full current-zone inventory for lineage.

### Dependent 1

All current aggregate zones are modeled jointly and their available state,
control, and disturbance signals are exposed as manifest-defined bindings.

E0-2 does **not** invent RC coupling edges.  Physical adjacency and inter-zone
resistances belong to E0-3/E0-4.

### Dependent 2

Current aggregate zones provide states and available QAC controls.

The Phase-D-selected compatible all-to-one zone provides disturbances.

E0-2 preserves:

- current modeled zone IDs,
- dependent-2 source zone ID,
- disturbance source zone IDs,
- aggregation/Phase C lineage.

Any later disturbance-allocation mathematics must be derived from the
authoritative aggregation definition/weights.  The paper-specific two-zone
`lambda_D + lambda_K = 2` rule is not encoded in E0-2.

## Observed versus latent states

Phase D supplies measured zone-air temperature as the observed thermal state.

E0-2 deliberately fabricates **no latent RC states**.

Mass, envelope, wall, or other hidden states belong to the later topology/method
math contract and require an explicit initialization/estimation strategy.

This prevents Phase E from creating artificial hidden-state targets merely
because a chosen RC topology has more states than Phase D measures.

## Temporal ownership and leakage

The final Phase D product owns:

- input lag,
- target horizon,
- policy family,
- policy realization,
- outer partition labels,
- leakage-safe excluded rows.

E0-2 preserves those values.

The locked HPO rule is stricter:

> The representative hyperparameter-tuning source rows are selected from Phase D
> TRAIN only.  Inner fit/validation splits may be created inside that
> training-only subset.

Phase D validation, test, and excluded rows are rejected as source rows for the
HPO subset by the E0-2 contract.

In all cases Phase D TEST is forbidden for hyperparameter tuning.

## Provenance

E0-2 carries provenance directly from the final manifest, including when present:

- campaign ID,
- case ID,
- aggregation matrix run ID,
- aggregation run ID,
- aggregation ID,
- weight mode,
- Phase C campaign run ID,
- Dep2 match status,
- Dep2 source aggregation run ID.

Unknown additional provenance keys are retained.

E0-2 does not parse the filesystem path to fill missing provenance.

A deterministic SHA-256 of canonicalized manifest content is used to identify the
exact source manifest and derive an E0 contract ID.

## Relationship to the controlled PINODE/EPSR paper code

`Paper_PINODE_EPSR` is a scientific/reference oracle, not the generic
implementation.

Reusable science includes:

- QAC/PHVAC separation,
- observed/latent state distinction,
- normalized-time discipline,
- IND/DEP1/DEP2 information architecture,
- exact solver-stage physics/projection concepts,
- train-only scaling/tuning,
- explicit mathematical-invariant tests.

The following paper-specific details are intentionally not copied:

- RestaurantFastFood/Buffalo IDs,
- Dining/Kitchen fixed indexing,
- fixed 1-zone/2-zone dimensions,
- hard-coded forcing vectors,
- substring feature resolution,
- fixed 300-second timestep,
- pair-specific Dep2 allocation,
- paper output directories,
- duplicated manual RC equations.

## API

Primary modules:

```text
scalebridge.data.thermal_modeling.phase_e_contracts
scalebridge.data.thermal_modeling.phase_e_adapter
```

Primary functions:

```python
build_phase_e_data_contract(manifest)
load_phase_e_data_contract(manifest_path)
validate_materialized_columns(contract, columns)
validate_partition_values(contract, partitions)
```

Primary contract object:

```python
PhaseEDataContract
```

Exact input resolution:

```python
contract.require_input(
    "qac",
    aggregate_zone_id="Dining",
    lag=0,
)
```

Exact target resolution:

```python
contract.require_target(
    "zone_temperature",
    aggregate_zone_id="Dining",
    horizon=1,
)
```

No path guessing or substring matching is involved.

## What E0-2 intentionally leaves for later

E0-3:
- generic RC topology graph,
- capacity/state nodes,
- resistance edges,
- heat-routing maps,
- physical parameters,
- generated RHS/residual/energy constraints.

E0-4:
- arbitrary-zone coupling,
- shared inter-zone resistance parameters,
- latent-state initialization contracts,
- aggregation-aware disturbance allocation.

E0-5:
- Euler, Heun/RK2, RK4, precise sixth-order RK,
- physical/normalized coordinate handling,
- ZOH/interpolation policy.

E0-6:
- NumPy/PyTorch/Neuromancer/CasADi/Pyomo/Bayesian parity.

E0-7:
- runtime API,
- QACRuntime/PHVACRuntime composition,
- portable bundles.

E0-8:
- complete Optuna/MLflow trainer and tuning lifecycle.

E0-9/E0-10:
- full test matrix and end-to-end qualification.
