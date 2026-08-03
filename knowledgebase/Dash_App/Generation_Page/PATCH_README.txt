BGIRS GENERATION PATCH 02 — FULL BUILDER, EXECUTION, RESULTS
==========================================================

Adds:
- Three lazy Generation tabs: Campaign Builder, Execution, Results.
- ASHRAE 2013/2016/2019 source discovery using ../../Data.
- Any selected IDF × any selected EPW case matrices.
- Secure ZIP import using idf/*.idf and epw/*.epw.
- Fixed P1 Generation signal profile (35 variables).
- Shared downstream campaign lifecycle matching run_p1_compact_campaign.py.
- Definition-driven scripts/energyplus/run_generation_campaign.py.
- Managed subprocess Start/Stop and live console output.
- Selected-campaign metadata and multi-signal parquet plotting with Full/Custom range.

Apply from repository root:
  powershell -ExecutionPolicy Bypass -File .\APPLY_BGIRS_GENERATION_PATCH_02.ps1

Then run:
  pytest tests\integration\energyplus\test_general_case_builder.py tests\dashapp\unit tests\dashapp\smoke -v
  python scripts\dashapp\validation\validate_shell.py

The installer backs up overwritten files under .patch_backups.
