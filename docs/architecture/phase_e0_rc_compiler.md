# Phase E0 E0-3B — Generic Continuous-Time RC Physics Compiler

## Status

E0-3B is the first implementation slice of the ratified Phase E0 RC mathematics.

Mathematical authorities:

- `ScaleBridge_PhaseE0_E0-3_Generic_Elemental_RC_Contract_v2.tex`
- `ScaleBridge_PhaseE0_E0-3_Current_RC_Flavours_v2.tex`
- `ScaleBridge_PhaseE0_E0-3_Generic_Spatial_RC_Compiler_Contract_v1.tex`

The implementation is backend-neutral in model structure and uses NumPy only as
the reference numerical evaluator of the continuous-time right-hand side.

It deliberately does **not** implement numerical integration, training,
Neuromancer, TorchDiffEq, CasADi, Optuna, MLflow, or Phase-D split logic.

## Package boundary

Reusable production code lives in:

```text
src/scalebridge/models/grey_box/rc_networks/
```

The controlled `Paper_PINODE_EPSR` implementation is used only as a scientific
test oracle. Production RC code does not import it.

The E0-2 Phase-D adapter remains authoritative for canonical data bindings.
E0-3B does not infer Phase-D identity from paths and does not recreate Phase-D
partition logic.

## Math-to-code map

| Frozen math | E0-3B implementation |
|---|---|
| elemental state/boundary graph | `flavours.py`, `compiler.py` |
| 1R1C/2R2C/3R2C/4R3C | `flavours.py` |
| modeled-zone replication | `compile_rc_model()` |
| adjacency + chain fallback | `compiler.py::_resolve_adjacency()` |
| connection-rule expansion | `compiler.py::_resolve_connection_rules()` + compiler edge expansion |
| independent-by-default parameters | `parameters.py` |
| explicit compatible sharing | `ParameterSharingRule`, `build_parameter_registry()` |
| `A_g B_g = 1` DEP2 allocation | `allocation.py` |
| `p=softmax(alpha)`, `lambda=p/w` | `estimated_allocation_result()` |
| `N_z-1` allocation DOF | `allocation_degrees_of_freedom()` |
| partial-fixed residual allocation | `allocation.py` |
| `D`, `G`, `L=DGD^T` | `CompiledRCModel.matrices()` |
| conservative heat routing `Gamma` | `CompiledRCModel._build_gamma()` |
| `Y=HX` | compiled `observation` matrix / `observe()` |
| continuous RHS | `runtime.py::rhs()` |
| latent-state initialization equals observed air | `default_initial_state()` |
| compiler invariants | `invariants.py` |

## Built-in RC flavours

The exact ScaleBridge topologies are:

- 1R1C: state `a`, boundary edge `a-o`.
- 2R2C: states `a,m`, edges `a-o`, `a-m`.
- 3R2C: states `a,m`, edges `a-o`, `a-m`, `m-o`.
- 4R3C: states `a,e,m`, edges `a-o`, `a-e`, `e-o`, `a-m`.

Core atomic thermal ports use the frozen grouping:

- convective/direct: `qac`, `zic`, `qsol1`;
- radiative: `zir`, `qsol2`.

Additional canonical thermal signals can be admitted only when their routing
group is explicitly supplied in `RCCompilerSpec.port_groups`.

`phvac` is rejected as a thermal port.

## Spatial modes

`IND`
: no inter-zone physical resistance edges; local forcing.

`DEP1`
: compiled inter-zone RC graph; local forcing.

`DEP2`
: exactly the DEP1 physical graph; local state/QAC plus explicitly allocated
  all-to-one non-HVAC thermal forcing.

DEP2 does not silently create an allocation. Every applicable non-HVAC thermal
signal must be covered by exactly one explicit allocation family.

## Zone-specific applicability

`zone_port_availability` is structural, not a runtime missing-value policy.

If a port is declared structurally applicable, its instantaneous value is
required at runtime. Missing required values raise `RCCompileError`.

If a port is structurally unavailable, the compiler omits it from the thermal
port vector and `Gamma`; it is not silently inserted as zero.

For a shared DEP2 allocation family, every signal in that family must have the
same participating-zone set.

## Parameters

Each physical capacitance, resistance, routing coefficient, and inter-zone
resistance is a distinct parameter instance by default.

Explicit compatible sharing maps multiple physical instances to one master
parameter without merging state nodes or changing topology.

Current E0-3B evaluates supplied numerical master/instance values. Training-time
positive transforms, priors, and optimizer ownership remain later Phase-E work.

## Continuous-time authority

E0-3B evaluates only:

```text
C Xdot = -L_CC X - L_CB T_B + Gamma Q
```

No time-stepping method is part of this package slice.

The later E0-5 discretization engine must consume this continuous-time model
rather than duplicating RC equations.

## Validation

Focused tests include:

- parity with the controlled paper 1R1C/2R2C single and coupled equations;
- hand-equation tests for 3R2C and 4R3C;
- adjacency and cross-state rule expansion;
- parameter sharing;
- equal/unequal DEP2 allocation;
- `N_z-1` DOF and neutral initialization;
- partial-fixed feasibility;
- DEP1/DEP2 graph equivalence;
- latent-state initialization;
- PHVAC exclusion;
- no silent zero-fill of structurally required signals;
- Laplacian/routing compiler invariants.

The focused validator is:

```powershell
python .\scripts\thermal_modeling\validate_phase_e0_rc_compiler.py
```
