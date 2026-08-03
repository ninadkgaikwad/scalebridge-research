PHASE C PATCH GROUP 2 — C2-C4 APPLICABILITY PROPAGATION

Files replaced:
  scripts/heat_input_regression/build_heat_input_regression_features.py
  scripts/heat_input_regression/build_heat_input_regression_splits.py
  scripts/heat_input_regression/build_heat_input_regression_datasets.py
  scripts/heat_input_regression/validate_heat_input_regression_features.py
  scripts/heat_input_regression/validate_heat_input_regression_datasets.py

No changes required:
  feature_engineering.py (already conditionally builds only needed feature families)
  datasets.py (already maps only requested/applicable model IDs)
  build/validate C3 core logic (already model-count agnostic; builder only receives metadata propagation)
  canonical-aware wrapper (standard C2 validator is already canonicalized)

Behavior:
  - C2 uses C1 applicable_models.csv and snapshots full applicability.
  - Zero-applicable-model zones complete with timestamp-only C2 data.
  - C3 completes and carries applicability metadata.
  - C4 creates only applicable datasets.
  - Zero-applicable-model zones complete C4 with zero datasets.
  - C2 validator enforces exact C1/C2 applicable-set equality.
  - C4 validator enforces exact C1/C4 model-set equality and validates zero-model zones.
