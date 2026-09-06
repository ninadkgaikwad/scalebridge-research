# Phase E0-5 Common Discretization Architecture

## Authority boundary

E0-3 owns the continuous RC graph and matrices. E0-4 owns canonical runtime
state/input realization. E0-5 owns only the fixed-step numerical map from
`X_k` and the already-bound `T_B,k`, `Q_k` to `X_{k+1}`.

The implementation authority is the frozen contract:

`docs/mathematics/thermal_modeling/phase_e0/contracts/ScaleBridge_PhaseE0_E0-5_Common_Discretization_Contract_v1.tex`

## Normal execution path

1. Materialize E0-3 matrices through `CompiledRCModel.matrices`.
2. Convert them without topology-specific equations to
   `A=-C^-1 L_CC`, `B_T=-C^-1 L_CB`, `B_Q=C^-1 Gamma` using row-wise
   division by the positive capacitance vector.
3. Consume the ordered E0-4 `RuntimeBinding.boundary_vector` and
   `RuntimeBinding.effective_thermal_vector`.
4. Hold those forcing tensors constant for the full canonical interval and all
   internal stages/substeps (left-endpoint ZOH).
5. Advance with the selected fixed-step solver.
6. Return the raw evolved state; E0-5 performs no clipping, smoothing,
   projection, observation reset, or hidden forcing reinterpretation.

The default solver is Neuromancer `RK4`, but the user can select any compatible
native solver registered for the audited Neuromancer 1.5.6 environment.

## Neuromancer registry

The compatible first-order, one-step, fixed-h methods are:

- `euler` -> `Euler`
- `euler_trap` -> `Euler_Trap`
- `rk2` -> `RK2`
- `rk4` -> `RK4` (default)
- `rk4_trap` -> `RK4_Trap`
- `luther` -> `Luther`
- `runge_kutta_fehlberg` -> `Runge_Kutta_Fehlberg`

`DiffEqIntegrator`, SDE integrators, multistep-history methods, and second-order
state methods are deliberately excluded from this registry because they do not
share the frozen first-order fixed-step drop-in contract.

Neuromancer imports are lazy. The independent exact linear oracle can therefore
be used even in a runtime where Neuromancer is not importable.

## Graph-general exact linear ZOH oracle

The oracle never branches on RC flavour, zone count, adjacency, or spatial
mode. It consumes only the authoritative E0-3 compiled matrices and forms

`A=-C^-1 L_CC`

`B=[-C^-1 L_CB, C^-1 Gamma]`.

For held input `U_k=[T_B,k, Q_k]`, it computes the augmented matrix exponential

`exp([[A,B],[0,0]] dt) = [[A_d,B_d],[0,I]]`

and returns

`X_{k+1}=A_d X_k + B_d U_k`.

The implementation uses `torch.linalg.matrix_exp`; it does not require `A^-1`
and therefore remains valid for singular `A`. Exact transition matrices are
cached by timestep for a fixed materialized graph/parameter realization.

## Substeps

For `N_s` explicit substeps, `h=dt/N_s`. Native Neuromancer methods execute
exactly `N_s` calls with the same held forcing. `exact_zoh_linear` also executes
the requested number of exact substeps so provenance is truthful; by the exact
semigroup property this agrees with one exact full-interval step up to floating
point error.

## Diagnostics

`diagnostics_per_step=False` is the default. The normal path does not perform
an eigendecomposition, exact-oracle comparison, convergence study, or other
expensive scientific analysis.

When enabled, E0-5 can report:

- input/output finiteness;
- exact-ZOH error for the linear RC graph;
- the largest RC modal decay rate;
- Euler/RK4 stability metrics where theory is frozen;
- a recommended minimum substep count;
- Neuromancer RKF local-error magnitude.

Diagnostics observe the already-computed numerical state. They never replace or
repair it. The implementation tests exact equality of the normal and diagnostic
solver result for the same configuration.

## Recursive runtime handoff

`CommonDiscretizationEngine.step_runtime` requires the E0-4 runtime-state and
runtime-binding timestamps to match. The evolved state is returned through
E0-4 `accept_model_evolved_state`, so the resulting origin is
`MODEL_EVOLUTION`; no observed-temperature argument exists in this path.
