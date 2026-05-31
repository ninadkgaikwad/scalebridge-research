# ScaleBridge Legacy Source Policy

Date: 2026-05-31  
Project: ScaleBridge Research  
Repository: scalebridge-research  
Python package: scalebridge  

## Purpose

This document defines how legacy code should be treated during the ScaleBridge refactor.

The legacy codebase is reference logic only. Production code must be rewritten cleanly inside the `src/scalebridge/` package.

## Ignore Completely

The following legacy sources should not be migrated:

- `Code/BuildingThermalModeling/`
- `BuildingModeling_BlackGreyBox_Condensed/`

Reason:

These folders are considered older or lower-quality code paths and should not guide the new production architecture.

## Reference Only

The following legacy source is useful only as behavioral reference:

- `BasicANN_Journal/`

Use it to understand:

- feature definitions,
- target variables,
- train/validation/test splitting,
- baseline ANN/RNN/GRU behavior,
- old result naming,
- paper-related output expectations.

Do not copy TensorFlow/Keras implementation into production code.

## Active Conceptual Source

The following source is the current conceptual source for data pipeline and grey-box work:

- `Code/`

Use it to understand:

- current data pipeline assumptions,
- grey-box estimation workflows,
- advanced grey-box model logic,
- experiment conventions.

However, this source is not modularized and should not be imported directly into production package code.

## TensorFlow to PyTorch Rule

ScaleBridge is PyTorch-first.

TensorFlow/Keras code may be read for reference but should not become production code unless there is a documented exception.

Production model code should live under:

- `src/scalebridge/models/`
- `src/scalebridge/training/`
- `src/scalebridge/evaluation/`

## No Direct Legacy Imports

Production package code must not import from:

- `legacy_reference/`
- old manuscript folders,
- old unstructured experiment folders,
- TensorFlow legacy scripts.

Correct migration process:

```text
legacy script
→ identify purpose
→ identify inputs
→ identify outputs
→ identify paper relevance
→ extract conceptual logic
→ rewrite cleanly in ScaleBridge
→ test against legacy behavior when useful
→ track new experiments with MLflow
```

## Week 0 Status

This policy is part of the Week 0 foundation freeze and must be considered active for Week 1 development.
