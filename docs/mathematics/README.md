# ScaleBridge mathematical contracts

This directory is the repository home for versioned mathematical specifications
that define ScaleBridge scientific behavior before implementation.

## Authority rule

Mathematical contracts are separated by scope:

- `thermal_modeling/` contains generic, reusable ScaleBridge thermal-model
  mathematics.
- Paper-specific mathematical contracts remain with the paper that owns them.
  For the current EPSR PINODE work, those files live under
  `Paper_PINODE_EPSR/docs/mathematics/contracts/`.

Paper-specific files may be used as scientific references when generalizing
ScaleBridge infrastructure, but they are not automatically generic contracts.

## Math-first development rule

For each new thermal-model method or shared scientific mechanism:

1. develop the mathematics;
2. write a versioned `.tex` contract;
3. review and lock the mathematics;
4. design software architecture;
5. implement;
6. write math-traceable tests;
7. validate.

Draft and locked contracts must be distinguishable by path and filename.
