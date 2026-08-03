Phase C Patch Group 4A — Runtime C9 Expected Counts

Problem fixed
-------------
The unified campaign runner previously calculated expected C6/C7/C8 MLflow task
counts while constructing the command list. At that point the training,
evaluation, and inference artifacts did not yet exist, so a fresh full run
hard-coded expected counts of 0, 0, and 0 into the final C9 validator command.

Resolution
----------
1. run_phase_c_campaign.py now passes only --expected-stage-runs 8.
2. validate_phase_c_mlflow_tracking.py resolves omitted task expectations at
   runtime from the completed registration manifest availability_summary:
   - trained_model_count
   - evaluated_model_count
   - inference_zone_count
3. Explicit CLI expected counts still override manifest-derived values.

Modified files
--------------
scripts/heat_input_regression/run_phase_c_campaign.py
scripts/heat_input_regression/validate_phase_c_mlflow_tracking.py
