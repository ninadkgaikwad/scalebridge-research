PHASE C PATCH GROUP 3 — C5-C8 AVAILABILITY-AWARE EXECUTION
===========================================================

Replace these files:

scripts/heat_input_regression/train_heat_input_regression_models.py
scripts/heat_input_regression/evaluate_heat_input_regression_models.py
scripts/heat_input_regression/run_heat_input_regression_full_year_inference.py
scripts/heat_input_regression/validate_heat_input_regression_training.py
scripts/heat_input_regression/run_phase_c_campaign.py

The bundle also includes the reviewed current C5/C7/C8 validators for convenience,
but they are not modified by this patch.

Key behavior:
- C5 continues to validate only discovered C4 model_dataset_manifest.json files.
- C6 accepts a validated zero-model C4 selection as a successful zero-task run.
- C7 accepts a zero-artifact C6 run and writes schema-stable empty indexes.
- C8 receives feature_root and dataset_root from the orchestrator.
- C8 preserves every C2 zone, including zones with zero evaluated components.
- Zero-component zones receive timestamp-only annual outputs and a valid manifest.
- Every C8 zone receives component_applicability.csv copied from the C2 snapshot.
- Structurally inapplicable components are not converted into failed tasks.

Validation target for the current ApartmentMidRise five-zone chain:
- C5: all 38 C4 datasets compatible with pytorch_linear CUDA.
- C6: 38/38 training tasks complete.
- C7: 38/38 evaluations complete.
- C8: 5/5 zones complete; component counts 2,9,9,9,9.
