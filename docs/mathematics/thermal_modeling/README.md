# Thermal-modeling mathematics

This tree contains generic ScaleBridge thermal-model mathematical contracts.

## Current structure

- `phase_e0/` — cross-method mathematical foundations developed during
  Phase E.0.
- Future method-specific contracts should be organized under stable family
  paths such as:
  - `methods/ml/`
  - `methods/sciml/`
  - `methods/optimization/`
  - `methods/bayesian/`

## Source-independent model rule

Generic thermal models consume canonical physical ports and model
specifications. Phase D is a training/testing data provider, not part of the
model physics itself. Future runtime/simulator adapters may recreate the same
canonical ports from Phase-B-like measurements/actions and Phase-C models.

## Paper references

The EPSR PINODE mathematical contracts are stored once, under:

`Paper_PINODE_EPSR/docs/mathematics/contracts/`

Generic Phase-E contracts may cite those files, but should not duplicate them.
