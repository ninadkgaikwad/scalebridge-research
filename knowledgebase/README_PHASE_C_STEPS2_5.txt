ScaleBridge Phase C Steps 2-5 Integrated Patch
================================================

Purpose
-------
This patch completes the C5-C9 changes required by the C1-C4 QAC/PHVAC patch.

Implemented behavior
--------------------
1. C5 reads fit_intercept from every C4 model_dataset_manifest.json.
2. C6 defaults to manifest-driven intercept policy:
   - PHVAC: fit_intercept=True
   - all thermal/heat-input models including QAC: fit_intercept=False
   The CLI --fit-intercept/--no-fit-intercept remains only as an explicit override.
3. C6 manifests record model_role, input_transform, dependency_model_id,
   target_allocation, and intercept_policy_source.
4. C7 PHVAC evaluation writes:
   - oracle prediction from abs(measured QAC)
   - chained prediction from abs(predicted QAC)
5. C7 creates building-level PHVAC reconstruction by summing aggregate-zone
   PHVAC targets/predictions for each split.
6. C8 executes QAC before PHVAC and uses abs(predicted_QAC) for deployed/chained
   PHVAC inference. It also preserves an oracle PHVAC output based on the C2
   measured-QAC predictor.
7. C8 writes building-level PHVAC predictions by summing aggregate-zone PHVAC.
8. C8 validation understands chained PHVAC predictor provenance.
9. MLflow registration records the new physical/model policy metadata.
10. The campaign runner expected-task counts now use the actual manifest names:
    training_manifest.json and annual_component_predictions_manifest.json.

Installation
------------
Extract this ZIP into the scalebridge-research repository root and allow the
listed files to replace their current versions.

Important
---------
Apply this only after the earlier C1-C4 QAC/PHVAC patch is present. The new
C5-C9 code expects C4 manifests to contain fit_intercept, model_role,
input_transform, dependency_model_id, and target_allocation.
