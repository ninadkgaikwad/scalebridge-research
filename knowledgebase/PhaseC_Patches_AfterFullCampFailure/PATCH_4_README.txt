PHASE C PATCH GROUP 4 — C9 MLFLOW + CAMPAIGN SUMMARY
=====================================================

Modified files
--------------
scripts/heat_input_regression/validate_phase_c_mlflow_tracking.py
scripts/heat_input_regression/run_phase_c_campaign.py
src/scalebridge/tracking/mlflow/heat_input_regression.py

Purpose
-------
1. Log C1-C8 availability-aware metrics to MLflow.
2. Add a campaign-level availability summary to the C9 registration manifest.
3. Add the same summary to phase_c_campaign_run_manifest.json.
4. Verify C6 task runs are nested under the C6 stage run.
5. Verify C7 task runs are nested under the C7 stage run.
6. Verify C8 zone runs are nested under the C8 stage run.
7. Log zero-component and component-applicability metadata for C8 zones.
8. Preserve dynamic task counts; structurally unavailable models are not fake failures.

Install
-------
Copy the three files into their matching repository paths.
Do not delete any other tracking modules.

Validation
----------
Compile all three files, then run C9 registration and the C9 validator using
an existing complete C1-C8 run or the final controlled campaign.
