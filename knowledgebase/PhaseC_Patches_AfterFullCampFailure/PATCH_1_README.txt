PHASE C PATCH 1 — C1 MODEL APPLICABILITY
=========================================

Replace:
    scripts/heat_input_regression/audit_aggregation_for_heat_input_regression.py

Purpose:
    - Prevent structurally unavailable model inputs from aborting a zone audit.
    - Classify each candidate model independently.
    - Preserve existing applicable_models.csv and unavailable_models.csv contracts.
    - Add model_applicability.csv and inapplicable_models.csv.
    - Add reason_code, applicability_class, missing_required_signals,
      dependency_status, and fatal_for_zone fields.
    - Make PHVAC honor its declared QAC dependency.
    - Preserve unknown exceptions as real fatal errors.

Expected Apartment_Corridors behavior:
    QAC:
        structurally_inapplicable because system-node mass flow and temperature
        are absent.

    PHVAC:
        structurally_inapplicable because dependency model QAC is unavailable.

    Other models:
        evaluated independently and retained when applicable.

Validation order:
    1. Compile the patched script.
    2. Run C1 for one Apartment_Corridors zone.
    3. Inspect candidate_models.csv, applicable_models.csv,
       unavailable_models.csv, model_applicability.csv,
       inapplicable_models.csv, and zone_audit_manifest.json.
    4. Do not run the full campaign until the focused result is reviewed.
