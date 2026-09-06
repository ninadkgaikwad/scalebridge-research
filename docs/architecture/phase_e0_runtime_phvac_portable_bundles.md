# Phase E0-7 Runtime + PHVAC + Portable Model Bundles

## Authority

Mathematical authority:

`docs/mathematics/thermal_modeling/phase_e0/contracts/ScaleBridge_PhaseE0_E0-7_Runtime_PHVAC_Portable_Model_Bundle_Contract_v1_1.tex`

E0-7 is a **post-training/post-estimation** layer. It does not own optimization,
backpropagation, sampling, HPO, Sim1/Sim2/Sim3 policy, Gymnasium, MPC, or RL.

## Package

`src/scalebridge/models/portable/`

The package contains:

- `contracts.py` — immutable bundle envelope, logical B/C/D locators,
  normalization, runtime I/O schema, generic future method payload descriptor,
  and PHVAC bundle contract.
- `bundle.py` — directory-based portable bundle materialization, manifest,
  embedded-file SHA256 inventory, and optional integrity qualification.
- `lineage.py` — machine-local data-root registry and logical artifact resolver.
- `historical.py` — Phase-D historical replay table loader/contract validator;
  downstream evaluators own Sim1/Sim2/Sim3 policy.
- `normalization.py` — fitted normalization and inverse normalization execution.
- `phvac.py` — Phase-C PHVAC artifact ingestion, `abs(QAC)` execution from the
  stored/current Phase-C contract, and building reconstruction.
- `rc_payload.py` — portable final physical-RC payload for ODE/RC,
  Inverse-PINN deployment, optimization output, and deterministic Bayesian
  point summaries.
- `runtime.py` — forward-only stateful RC runtime over E0-3/E0-4/E0-5.

## Portable authority

A model bundle stores static fitted information. Runtime state is separate.
Time-varying controls and disturbances are not embedded in the model artifact.

Upstream data is represented with logical `DataLocator` objects:

`root alias + portable relative path + IDs + optional SHA256`.

Absolute Windows paths are not bundle authority. Historical evaluation resolves
these locators through a `DataRootRegistry`; future SmartBuildingsSim will
provide controls/disturbances through its simulator layer.

## PHVAC

Current Phase-C PHVAC contract:

- predictor transform: `absolute_value` of QAC;
- target allocation: `equal_across_aggregate_zones`.

Let `N` be total aggregate zones and `M` the number of zones with no available
PHVAC model. Then `N-M` PHVAC models are available.

- `M = 0`: direct sum of all `N` PHVAC predictions.
- `0 < M < N`: multiply the available-model sum by `N/(N-M)`.
- `M = N`: model-based PHVAC reconstruction is unavailable.

The difference between the corrected total and the available-model sum is named
**allocation completion**. It is not attributed as physical HVAC consumption of
a non-HVAC zone.

PHVAC models are embedded into the portable bundle because they are static
models required for forward deployment. Their original Phase-C locations may
also be retained as lineage.

## Validation versus runtime cost

Bundle SHA256/inventory checks, disturbance echo, and detailed forward
provenance are available for qualification but are not repeated on every normal
runtime step. `RCForwardRuntime.step(..., include_diagnostics=False)` is the
lightweight default.

## Future E.1/E.2/E.3/E.4 payloads

E0-7 freezes only the outer envelope now. Exact payload schemas are extended as
the four estimation-method families are implemented. Deployment semantics
already frozen by E0-7 are:

- ODE/RC and Inverse PINN-RC: final physical Theta, no training NN weights.
- NODE: learned vector-field weights.
- Base PINODE / EBP-PINODE: required physical + neural executable payloads.
- CasADi/IPOPT: final physical Theta, not the NLP object.
- Bayesian: physical point estimate or later-defined posterior representation.

E0-10 remains the authoritative comprehensive testing campaign.
