# Phase E0 E0-4 Runtime State/Input Realization

## Authority

Mathematical authority:

`docs/mathematics/thermal_modeling/phase_e0/contracts/ScaleBridge_PhaseE0_E0-4_Runtime_State_Input_Contract_v1.tex`

E0-4 is a backend-neutral runtime/state layer above the qualified E0-3B RC
compiler.  It does not implement a numerical integrator.

## Responsibility boundary

- E0-2: canonical Phase-D-to-Phase-E scientific/data contracts.
- E0-3: RC topology, parameters, heat routing, and continuous RHS.
- E0-4: runtime initialization, timestamped canonical frame realization,
  IND/DEP1/DEP2 input binding, and recursive-state ownership.
- E0-5: interpolation, discretization, and numerical time advance.

## Runtime initialization

`initialization.py` implements the locked policies:

- `auto` (default): user > observed air temperature > resolved setpoint > 22 C;
- `user_fixed`;
- `observed`;
- `setpoint`;
- `default`.

The resolved zone vector `T0*` is lifted into compiler state order through `S0`:

`X0 = S0 T0*`

and is checked against:

`H S0 = I`, `H X0 = T0*`.

The older E0-3B `default_initial_state()` helper remains unchanged for backward
compatibility.  New runtime code should use `initialize_runtime_state()`.

## Canonical runtime frame

`runtime_binding.py` defines one timestamped `CanonicalRuntimeFrame`.  Rich
frames are allowed, but only inputs required by the compiled model are copied
into the low-level E0-3 `RCInputSnapshot`.

The frame explicitly distinguishes:

- local thermal source values;
- aggregate/all-to-one thermal source values;
- boundary temperatures;
- observed air temperatures;
- auxiliary electrical powers such as PHVAC;
- local/aggregate source-availability metadata.

Model-forcing applicability is derived independently from the E0-3 compiled
thermal ports.

## Spatial forcing modes

- IND: required forcing is local.
- DEP1: required forcing is local; E0-3 supplies coupled physics.
- DEP2: QAC stays local; every applicable non-HVAC signal is supplied from its
  authoritative all-to-one source and explicit E0-3 `B_g` allocation result.

DEP2 source availability is therefore not tied to local Phase-C source
availability for non-HVAC disturbances.

## PHVAC

PHVAC may be present in `auxiliary_electrical_powers`, but any attempt to place
PHVAC in a thermal-power mapping fails immediately.  PHVAC is never copied into
`Q`.

## Recursive state ownership

`runtime_state.py` exposes three explicit state origins:

- initialization;
- model evolution;
- explicit reset.

`accept_model_evolved_state()` accepts only the state produced by the future
E0-5 numerical layer.  It has no measured-temperature argument, preventing a
hidden teacher-forced reset.  A reset requires the separate
`explicit_state_reset()` API and a non-empty reason.

## Runtime invariants

`runtime_invariants.py` checks the directly realizable E0-4 invariants,
including initialization projection, `H S0 = I`, structural port isolation,
DEP2 aggregate-coordinate consistency, PHVAC exclusion, and immutable E0-3
physics signatures.  DEP1/DEP2 physical equivalence remains a hard contract.

## Validation

Focused implementation tests:

`tests/thermal_modeling/test_phase_e0_runtime_state_input.py`

Standalone scientific validator:

`scripts/thermal_modeling/validate_phase_e0_runtime_state_input.py`

Validation artifact:

`validated_artifacts/phase_e0/e04_runtime_state_input_validation.json`
