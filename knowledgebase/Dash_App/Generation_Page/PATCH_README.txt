BGIRS Generation Patch 08 — Custom Interactive Legend + Variable Column/Key Filter
=================================================================================

This patch supersedes Patch 07. If Patch 07 has NOT been applied, apply Patch 08 directly.

REPLACE:
  src/scalebridge/dashapp/pages/data_pipeline/phase_a_generation/results/page.py
  src/scalebridge/dashapp/pages/data_pipeline/phase_a_generation/callbacks.py
  src/scalebridge/dashapp/services/generation/results_data.py
  tests/dashapp/unit/test_generation_results_export.py

ADD:
  tests/dashapp/unit/test_generation_custom_interactive_legend.py
  tests/dashapp/unit/test_generation_variable_key_filter.py

Behavior:
- Results layout is a true 75/25 plot/custom-legend split.
- Custom legend is independently vertically scrollable.
- Clicking legend entries hides/shows individual traces.
- Adds a multi-select "Variable column / key" filter after Variable name.
- The filter is populated from canonical parquet key_value values.
- Available keys are auto-selected when the variable context changes; users can deselect/narrow them.
- Plot traces are split by variable + key_value. Different key columns are never mixed into one trace.
- Only selected key_values are plotted.
- Only selected key_values are exported to CSV/Parquet ZIPs.
- Exported per-signal filenames, combined data, and selection_manifest.json preserve key_value nomenclature.
- Existing Building/Weather/Case/Run/Variable filters and Full/Custom datetime range are preserved.
