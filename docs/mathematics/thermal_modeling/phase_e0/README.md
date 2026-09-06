# Phase E.0 mathematical contracts

This directory contains mathematical foundations that are shared across
thermal-model method families.

## Draft versus locked contracts

- `drafts/` contains active mathematical work that is still under scientific
  review.
- `contracts/` is reserved for reviewed and explicitly locked versions.

The current E0-3A generic RC-network document is intentionally a draft.
No E0-3 RC implementation should be treated as scientifically locked until a
corresponding reviewed contract is promoted to `contracts/`.

## Current E0-3 design decisions

The generic RC-network contract is being developed around these agreed
principles:

- 1R1C, 2R2C, 3R2C, and 4R3C are current ScaleBridge RC model flavours,
  implemented as specifications over one generic RC-network contract.
- Custom RC structures may use arbitrary state/capacitance nodes, boundary
  nodes, resistance edges, canonical ports, and heat-routing rules.
- Outdoor temperature is a disturbance/boundary-temperature input.
- Canonical heat ports remain atomic (for example QAC, QZIC, QZIR, QSol1,
  QSol2); the user/model specification defines routing.
- QAC defaults to the zone-air state for built-in RC flavours.
- Generic radiative routing uses a constrained routing vector; the familiar
  two-state `eta` is a special parameterization of that vector.
- User-defined zonal adjacency is authoritative. If absent, the simple
  fallback is a chain following the supplied zone order.
- Inter-zone connection rules are general and may contain one or multiple
  state-node pairings (for example air-air and mass-mass).
- Different inter-zone connection types use separate physical resistance
  parameters unless an explicit parameter-sharing rule is supplied.
- The user RC specification is independent of IND/DEP1/DEP2; a spatial
  compiler produces those information-architecture realizations.
- Aggregation is represented mathematically by signal-aware operators
  `A_g`; DEP2 allocation uses `B_g` with hard aggregate conservation
  `A_g B_g = I` when the stated operator dimensions/rank permit it.
- Equal, floor-area, volume, identity, all-to-one, and custom aggregation
  styles are represented through authoritative aggregation operators rather
  than hard-coded labels.
- A simple neutral DEP2 allocation fallback is allowed for normalized
  single-aggregate weights; custom or estimated allocations remain available.
- Phase D is a training/testing provider of canonical ports. A future runtime
  provider will recreate those same ports using Phase-B-like
  measurements/actions plus Phase-C model outputs.

The four current RC flavours must be explicitly drawn and mathematically
locked side-by-side before implementation.
