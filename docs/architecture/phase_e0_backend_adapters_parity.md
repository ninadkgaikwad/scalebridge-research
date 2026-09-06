# Phase E0-6 Backend Adapters + Numerical Parity — v2 Architecture

E0-6 uses **physical RC parameters `Theta` as the cross-backend scientific authority**.  Backend choice and estimation-coordinate choice are separate concerns.

## Method-family boundary

- Pure black-box ML and pure Neural ODE models do not require the RC backend.
- Inverse PINN-RC, differentiable RC ODE fitting, Base PINODE, and EBP-PINODE use PyTorch/Neuromancer and unconstrained raw coordinates `rho`, mapped to physical `Theta` by differentiable transforms.
- Deterministic grey-box parameter estimation uses `CasadiPhysicalRCBackend`: IPOPT decision variables are physical `Theta` directly, with box bounds and explicit conservation/simplex constraints.
- `CasadiTransformedRCBackend` is retained only as a transformed-coordinate parity/reference realization. `CasadiRCBackend` remains a backward-compatible alias for it.
- Bayesian RC likelihood/reference evaluation can use `NumpyPhysicalRCBackend` directly in physical `Theta` coordinates.

## Frozen ownership

E0-3 remains the only topology/flavour authority. E0-6 consumes the compiled 1R1C/2R2C/3R2C/4R3C graph, parameter masters, sharing, routing metadata, and DEP2 allocation specification. E0-6 does not redefine flavour equations.

## Direct physical decision plan

`build_physical_parameterization_plan(model)` emits:

- one physical decision coordinate for each estimated scalar master;
- one physical decision coordinate for each estimated 4R3C radiative routing component;
- one physical contribution `p_i = w_i lambda_i` coordinate for each estimated DEP2 allocation participant;
- `x0`, lower bounds, and upper bounds;
- explicit linear equality rows for routing-simplex mass and DEP2 contribution mass.

Consequently a constrained physical decision vector may have more entries than the unconstrained transformed `rho` vector. This is intentional.

## CasADi/IPOPT production path

`CasadiPhysicalRCBackend` exposes physical matrix/RHS realization, symbolic Euler/RK2/RK4 propagation, physical constraint expressions, and an IPOPT-ready NLP schema. Generic nonlinear parameter-estimation formulations should use symbolic RK4/direct transcription unless a method explicitly chooses another E0-5-compatible representation. Exact ZOH remains an independent linear-RC value oracle/optional propagation path and is not required for IPOPT symbolic operation.

## P4 parity

PyTorch differentiates in raw coordinates while the physical CasADi backend differentiates in physical coordinates. The required parity identity is

`grad_rho L = J_T(rho)^T grad_Theta L`,

where `J_T = d Theta / d rho`. Raw-gradient equality across the two coordinate systems is not required.

## Compatibility

The v1 transformed NumPy/PyTorch/Neuromancer behavior is preserved. Existing E0-3, E0-4, E0-5, and controlled PINODE/EPSR code is not modified by the v2 implementation revision.
