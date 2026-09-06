"""Contextual help for the Phase B Aggregation workspace."""

AGGREGATION_HELP_ENTRIES = {
    "aggregation.input.parent_generation_campaign": {
        "title": "Parent Generation Campaign",
        "summary": "Choose the Phase A campaign that supplies Aggregation source cases.",
        "details": (
            "B5 discovers existing Generation campaigns from the configured generated-data "
            "root. Only each case's latest successful Generation run is eligible. "
            "Both completed and completed_with_warnings are accepted."
        ),
    },
    "aggregation.action.refresh_generation_campaigns": {
        "title": "Refresh Generation Campaigns",
        "summary": "Rescan the configured campaign root for Generation campaigns.",
        "details": "This is read-only discovery. It does not start Generation or Aggregation.",
    },
    "aggregation.input.building_type_filter": {
        "title": "Building Type Filter",
        "summary": "Limit selectable Generation cases by building type.",
        "details": "Filtering changes only the case selection presented by the builder.",
    },
    "aggregation.input.weather_filter": {
        "title": "Weather Location Filter",
        "summary": "Limit selectable Generation cases by weather location.",
        "details": "Weather metadata comes from the selected Generation run manifest.",
    },
    "aggregation.input.climate_filter": {
        "title": "Climate Zone Filter",
        "summary": "Limit selectable Generation cases by climate zone.",
        "details": "Climate-zone metadata is read from Generation lineage; B6 does not infer it.",
    },
    "aggregation.input.case_selection": {
        "title": "Generation Case Selection",
        "summary": "Choose the Generation cases that this Aggregation campaign will process.",
        "details": (
            "Selected case IDs are persisted directly in the B1 campaign definition. "
            "No Aggregation plan is constructed while editing the builder."
        ),
    },
    "aggregation.input.strategies": {
        "title": "Aggregation Strategies",
        "summary": "Select one or more scientifically implemented grouping strategies.",
        "details": (
            "Available strategies are all_thermal_zones_to_one, custom_groups, and identity. "
            "These values are the authoritative AggregationStrategy enum values."
        ),
    },
    "aggregation.input.weight_modes": {
        "title": "Aggregation Weight Modes",
        "summary": "Select equal, floor-area, and/or volume weighting.",
        "details": (
            "B6 creates one B1 plan request for each selected strategy × weight-mode "
            "combination. Scientific weighting remains in Phase B Aggregation."
        ),
    },
    "aggregation.input.rule_set": {
        "title": "Aggregation Rule Set",
        "summary": "Choose the scientific signal-aggregation rules.",
        "details": "legacy_v1 is currently the only implemented authoritative rule set.",
    },
    "aggregation.input.custom_aggregation_id": {
        "title": "Custom Aggregation ID",
        "summary": "Name the custom grouping partition represented by the editable table.",
        "details": (
            "The ID is written into every custom-group CSV row and referenced by the "
            "B1 custom_groups plan request."
        ),
    },
    "aggregation.input.custom_grouping": {
        "title": "Custom Zone Grouping",
        "summary": "Assign every selected source thermal zone to one aggregate zone.",
        "details": (
            "B6 uses the thermal-zone inventory from the exact selected Generation run. "
            "Before save, the existing scientific custom-partition validator checks for "
            "missing, extra, or duplicate source zones."
        ),
    },
    "aggregation.input.campaign_id": {
        "title": "Aggregation Campaign ID",
        "summary": "Stable identifier for the saved Phase B campaign definition.",
        "details": "The ID follows the authoritative B1 campaign-ID validation contract.",
    },
    "aggregation.input.machine_id": {
        "title": "Machine ID",
        "summary": "Record the intended execution-machine identity in the campaign definition.",
        "details": "B6 stores this value only; it does not start execution.",
    },
    "aggregation.input.case_limit": {
        "title": "Case Limit",
        "summary": "Optional cap applied after the selected Generation case IDs.",
        "details": "Leave blank to process every selected eligible case.",
    },
    "aggregation.input.variable_limit": {
        "title": "Variable Limit",
        "summary": "Optional testing cap on Aggregation variable groups.",
        "details": "Leave blank for all available variable groups. This is a B1 execution option.",
    },
    "aggregation.input.preview_rows": {
        "title": "Preview Rows",
        "summary": "Number of preview rows written by scientific Aggregation outputs.",
        "details": "This does not change full canonical Aggregation computations.",
    },
    "aggregation.input.aggregate_zone_stem": {
        "title": "Aggregate Zone Name Stem",
        "summary": "Base name used by built-in grouping strategies.",
        "details": "Default is Aggregated_Zone, matching the authoritative plan builder.",
    },
    "aggregation.input.system_node_pattern": {
        "title": "System Node Pattern",
        "summary": "Pattern used by scientific system-node mapping rules.",
        "details": "Default is DIRECT AIR INLET NODE, matching the authoritative plan builder.",
    },
    "aggregation.input.mlflow_uri": {
        "title": "MLflow Tracking URI",
        "summary": "Tracking server URI to persist with the Aggregation definition.",
        "details": "B6 does not connect to MLflow or create runs.",
    },
    "aggregation.input.mlflow_experiment": {
        "title": "MLflow Experiment",
        "summary": "Optional experiment name for later B7 execution.",
        "details": "Leave blank to allow the scientific runner's normal default behavior.",
    },
    "aggregation.input.mlflow_run_name": {
        "title": "MLflow Run Name",
        "summary": "Optional MLflow run name persisted for later execution.",
        "details": "This is definition metadata only during B6.",
    },
    "aggregation.page.execution": {
        "title": "Aggregation Execution",
        "summary": "Launch and monitor a saved Phase B Aggregation campaign.",
        "details": (
            "B7 manages only the subprocess lifecycle. Scientific plan construction and "
            "Aggregation execution remain in the authoritative B2 generic campaign runner."
        ),
    },
    "aggregation.execution.saved_definition": {
        "title": "Saved Aggregation Campaign",
        "summary": "Choose a validated B1 definition previously saved by Campaign Builder.",
        "details": (
            "The definition summary shows the parent Generation campaign, selected case "
            "count, plan requests, strategies, weights, machine identity, and MLflow state."
        ),
    },
    "aggregation.execution.command": {
        "title": "Resolved B2 Runner Command",
        "summary": "Preview the exact command B7 will launch.",
        "details": (
            "B7 uses the current Python executable with unbuffered output and invokes "
            "scripts/aggregation/run_aggregation_campaign.py --campaign-definition <path>."
        ),
    },
    "aggregation.execution.console": {
        "title": "Live Aggregation Console",
        "summary": "Display combined stdout/stderr from the managed B2 subprocess.",
        "details": (
            "The console is captured by a background reader thread. The Stop button "
            "terminates the complete process tree so child processes are not orphaned."
        ),
    },
    "aggregation.page.results": {
        "title": "Aggregation Results",
        "summary": "Filter, plot, and export completed Phase B Aggregation signals.",
        "details": (
            "B8 is read-only. It reads B2 matrix JSON/CSV summary artifacts and does not "
            "load aggregated time-series Parquet files. Time-series plotting begins in B9."
        ),
    },
    "aggregation.results.campaign": {
        "title": "Aggregation Campaign",
        "summary": "Choose an Aggregation campaign that has one or more matrix results.",
        "details": (
            "Current B2 manifests are keyed by aggregation_campaign_id and retain the "
            "parent Generation campaign separately. Older matrices are surfaced as legacy."
        ),
    },
    "aggregation.results.matrix_run": {
        "title": "Matrix Run",
        "summary": "Choose one execution of the selected Aggregation campaign.",
        "details": (
            "A matrix run contains the campaign manifest, selected-plan table, case-run "
            "table, output index, and optional missing-Generation rows."
        ),
    },
    "aggregation.results.case_runs": {
        "title": "Case / Plan Runs",
        "summary": "Inspect one row per executed Aggregation case/plan combination.",
        "details": (
            "Values are read directly from aggregation_matrix_case_runs.csv, including "
            "strategy, weight, rule set, zone counts, runtime, status, and errors."
        ),
    },
    "aggregation.results.issues": {
        "title": "Errors / Missing Inputs",
        "summary": "Surface failed case runs and unavailable Generation inputs.",
        "details": (
            "B8 does not reinterpret failures. It displays recorded case-run errors and "
            "missing_generation_rows.csv when that optional artifact exists."
        ),
    },
    "aggregation.results.artifacts": {
        "title": "Artifact Paths",
        "summary": "Show the filesystem artifacts backing the selected matrix result.",
        "details": (
            "Paths are displayed for provenance and debugging. B8 does not modify or "
            "delete any Aggregation artifact."
        ),
    },
    "aggregation.results.strategy": {
        "title": "Strategy",
        "summary": "Filter completed results by Aggregation strategy.",
        "details": "Values come from the executed Aggregation plan, such as all_thermal_zones_to_one, custom_groups, or identity.",
    },
    "aggregation.results.weight": {
        "title": "Weight Mode",
        "summary": "Filter completed results by Aggregation weighting mode.",
        "details": "Values come directly from the executed plan, such as equal, floor_area, or volume.",
    },
    "aggregation.results.ruleset": {
        "title": "Rule Set",
        "summary": "Filter completed results by the Aggregation rule set.",
        "details": "The selector remains visible even when only one rule set is available.",
    },
    "aggregation.results.zone": {
        "title": "Aggregation Zone",
        "summary": "Choose one or more stored aggregate-zone outputs.",
        "details": "Zones are discovered from the selected completed Aggregation run directories.",
    },
    "aggregation.results.variable": {
        "title": "Variable",
        "summary": "Choose one or more source variables to plot.",
        "details": "This maps to source_variable_name in the Phase B long-form output.",
    },
    "aggregation.results.variable_column": {
        "title": "Variable Column",
        "summary": "Choose one or more aggregated output columns for the selected variables.",
        "details": "This maps to output_variable_name. Most variables have one output column; Schedule Value can have several equipment-specific outputs.",
    },
    "aggregation.results.run": {
        "title": "Run",
        "summary": "Choose one or more completed case/plan executions.",
        "details": "Run labels include building, weather, strategy, weight, and Aggregation run ID so multi-case campaigns remain distinguishable.",
    },
    "aggregation.results.time_range": {
        "title": "Time Range",
        "summary": "Plot the full stored range or restrict it to start/end datetimes.",
        "details": "The start/end controls remain visible; they are disabled while Full range is selected.",
    },
    "aggregation.results.building": {
        "title": "Building Type",
        "summary": "Multi-select building types represented in the selected Aggregation campaigns.",
        "details": "The selection intersects with every other Results filter. Only matching completed Aggregation runs remain eligible.",
    },
    "aggregation.results.weather": {
        "title": "Weather Location",
        "summary": "Multi-select weather locations represented by completed Aggregation runs.",
        "details": "Weather location remains separate from climate zone so either dimension can be compared independently.",
    },
    "aggregation.results.climate": {
        "title": "Climate Zone Metadata",
        "summary": "Climate zone is retained as lineage metadata rather than a separate Results filter.",
        "details": "Weather Location provides the Results filtering dimension; climate zone remains available in run/export lineage.",
    },
}
