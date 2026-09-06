"""Contextual help for Phase C Heat-Input Regression."""
from __future__ import annotations

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig


HEAT_INPUT_HELP_ENTRIES = {
    "subpage.data_pipeline.phase_c_heat_input": {
        "title": "Phase C: Heat-Input Regression",
        "summary": "Build, execute, and inspect Phase C heat-input regression campaigns.",
        "details": (
            "The Phase C workspace owns Campaign Builder, Execution, and Results. "
            "The Dash UI is a curated front end over PhaseCCampaignConfig; scientific "
            "C1-C9 behavior stays in the authoritative generalized runner."
        ),
    },
    "heat_input.page.campaign_builder": {
        "title": "Phase C Campaign Builder",
        "summary": "Create a reusable Phase C campaign from Aggregation outputs.",
        "details": (
            "The normal Builder exposes only campaign-level scientific choices. Runner "
            "implementation details and recovery controls remain automatic or CLI-only."
        ),
    },
    "heat_input.page.execution": {
        "title": "Phase C Execution",
        "summary": "Run one saved campaign through the complete Phase C workflow.",
        "details": (
            "Dash always launches the complete C1-through-C9 workflow without overwrite. "
            "A Dry Run checkbox changes the same Start action into plan-only execution."
        ),
    },
    "heat_input.page.results": {
        "title": "Phase C Results",
        "summary": "Inspect model datasets, fitted predictions, metrics, and annual inference.",
        "details": (
            "Results is manifest-first and read-only. Detailed C4/C7/C8 files are opened "
            "only after the model/context selection is narrowed."
        ),
    },
    "heat_input.section.identity": {
        "title": "Campaign Identity",
        "summary": "Name and describe the reusable Phase C campaign definition.",
        "details": (
            "phase_c_campaign_id identifies the saved UI definition. It is distinct from "
            "the parent Generation campaign and an individual execution run ID."
        ),
    },
    "heat_input.section.upstream": {
        "title": "Aggregation Source",
        "summary": "Choose any usable Aggregation campaign and matrix run.",
        "details": (
            "Discovery scans actual Phase B matrix artifacts as well as modern saved Phase B "
            "definitions. Valid legacy matrices therefore remain selectable even when they "
            "predate the current Dash campaign-definition format."
        ),
    },
    "heat_input.section.scope": {
        "title": "Campaign Scope",
        "summary": "Optionally narrow the selected Aggregation matrix before Phase C.",
        "details": (
            "Generation case, Aggregation strategy, custom grouping ID, weight mode, rule set, "
            "and model relationships are kept as separate concepts. The UI compiles these "
            "modeler-facing choices back to the existing Phase C runner scope fields."
        ),
    },
    "heat_input.input.case": {
        "title": "Generation Case",
        "summary": "Optionally select one upstream Generation case using readable building/weather lineage.",
        "details": "The stored value remains the authoritative case_id; the label shows building and weather so opaque case IDs are not the primary modeler-facing information.",
    },
    "heat_input.input.aggregation_strategy": {
        "title": "Aggregation Strategy",
        "summary": "Choose the Phase B strategy without mixing it with weight or rule-set identifiers.",
        "details": "Supported normalized values include all_thermal_zones_to_one, custom_groups, and identity. This selector is distinct from the internal aggregation_id consumed by the Phase C runner.",
    },
    "heat_input.input.custom_grouping_id": {
        "title": "Custom Grouping ID",
        "summary": "Choose the Phase B custom zone-group definition when strategy is custom_groups.",
        "details": "For example, custom_v1 identifies the zone grouping itself. Weight Mode remains separate, so custom_v1 can be used with both equal and floor_area when both completed Phase B plans exist.",
    },
    "heat_input.input.weight_mode": {
        "title": "Weight Mode",
        "summary": "Choose the Phase B aggregation weighting independently of strategy/grouping.",
        "details": "Examples include equal and floor_area. Leaving this empty includes all applicable weights when the selected scope can be represented by the authoritative Phase C runner.",
    },
    "heat_input.input.rule_set": {
        "title": "Rule Set",
        "summary": "Show the Phase B aggregation rule set as its own lineage dimension.",
        "details": "legacy_v1 is currently the supported rule set for the accepted campaign. It is no longer hidden inside an aggregation plan identifier.",
    },
    "heat_input.section.scientific": {
        "title": "Scientific Setup",
        "summary": "Minimal scientific policies needed to define a diverse Phase C campaign.",
        "details": (
            "Choose predictor/target policy, split strategy/fractions, estimator/device, "
            "validation, and MLflow tracking. Stage-specific implementation settings use "
            "the authoritative PhaseCCampaignConfig defaults."
        ),
    },
    "heat_input.section.preview_save": {
        "title": "Preview and Save",
        "summary": "Validate the compact campaign definition before persistence.",
        "details": (
            "Preview constructs the same typed definition as Save but writes nothing. "
            "Save refuses replacement unless Replace existing definition is checked."
        ),
    },
    "heat_input.input.phase_c_campaign_id": {
        "title": "Phase C Campaign ID",
        "summary": "Stable identifier for the reusable Phase C definition.",
        "details": (
            "This is not the parent Generation campaign_id and not an individual "
            "phase_c_run_id."
        ),
    },
    "heat_input.input.display_name": {
        "title": "Display Name",
        "summary": "Optional human-readable label for the saved Phase C definition.",
        "details": "This metadata does not alter scientific runner behavior.",
    },
    "heat_input.input.machine_id": {
        "title": "Machine ID",
        "summary": "Record the current ScaleBridge machine identity.",
        "details": "The machine identity is stored for provenance and is read-only here.",
    },
    "heat_input.input.notes": {
        "title": "Notes",
        "summary": "Optional notes attached to the reusable definition.",
        "details": "Notes are UI metadata only and do not alter Phase C science.",
    },
    "heat_input.input.parent_aggregation_campaign": {
        "title": "Aggregation Campaign",
        "summary": "Choose the Phase B campaign whose matrix output supplies Phase C.",
        "details": (
            "The list is artifact-first: modern definition-backed campaigns and legacy "
            "Aggregation matrix groups are both discoverable."
        ),
    },
    "heat_input.action.refresh_aggregation_campaigns": {
        "title": "Refresh Aggregation Campaigns",
        "summary": "Rescan actual Phase B matrix artifacts and saved definitions.",
        "details": "This is read-only discovery and does not execute Aggregation.",
    },
    "heat_input.input.matrix_run": {
        "title": "Aggregation Matrix Run",
        "summary": "Choose the executed Phase B matrix consumed by Phase C.",
        "details": (
            "The matrix determines the available cases, aggregation strategies/levels, "
            "weights, and downstream aggregate zones."
        ),
    },
    "heat_input.output.matrix_ownership": {
        "title": "Aggregation Matrix Source",
        "summary": "Show modern definition ownership or legacy artifact provenance.",
        "details": (
            "Legacy matrices remain valid upstream sources; they are labeled explicitly "
            "instead of being hidden merely because a newer definition JSON is absent."
        ),
    },
    "heat_input.execution.saved_definition": {
        "title": "Saved Phase C Campaign",
        "summary": "Select the reusable Builder definition to execute.",
        "details": "Execution reads the saved definition without editing it in place.",
    },
    "heat_input.execution.phase_c_run_id": {
        "title": "Phase C Run ID",
        "summary": "Identity of this individual complete Phase C execution.",
        "details": (
            "This is distinct from the reusable campaign definition. A timestamped run ID "
            "is suggested automatically."
        ),
    },
    "heat_input.execution.dry_run": {
        "title": "Dry Run",
        "summary": "Plan the complete Phase C workflow without executing C1-C9 science.",
        "details": (
            "When checked, Start Phase C invokes the generalized runner in dry-run mode. "
            "Uncheck it for the normal complete Phase C execution."
        ),
    },
    "heat_input.execution.command": {
        "title": "Technical Details",
        "summary": "Inspect the resolved runner command and effective configuration.",
        "details": (
            "These details are read-only and collapsed by default. The normal Dash workflow "
            "does not expose arbitrary shell editing or stage-recovery controls."
        ),
    },
    "heat_input.execution.progress": {
        "title": "Phase C Progress",
        "summary": "Passive progress through the complete Phase C workflow.",
        "details": (
            "C1-C9 labels are monitoring detail only. The user does not select individual "
            "stages from the normal Dash execution surface."
        ),
    },
    "heat_input.execution.console": {
        "title": "Live Console",
        "summary": "One combined live process stream from the generalized runner.",
        "details": (
            "stdout and stderr are combined into one bounded monitoring console. Scientific "
            "stage logs remain in their normal output folders."
        ),
    },
    "heat_input.results.run": {
        "title": "Phase C Result Run",
        "summary": "Choose one recorded Phase C run for read-only inspection.",
        "details": (
            "Discovery reads campaign manifests first and resolves stage roots from the "
            "recorded command plan."
        ),
    },
    "heat_input.results.filters": {
        "title": "Model / Context Filters",
        "summary": "Narrow the compact indexes before loading detailed model data.",
        "details": (
            "Building, weather, case, aggregation, weight, zone, model, and estimator are "
            "available. Weather is the location/climate selector, so no separate Climate "
            "filter is required in this workspace."
        ),
    },
    "heat_input.results.dataset_trajectory": {
        "title": "Dataset X / Y Trajectory",
        "summary": "Inspect C4 predictor X and target Y for one model dataset.",
        "details": (
            "Preview uses regression_pairs_preview.csv. Full mode reads only timestamp and "
            "the selected predictor/target columns from regression_pairs_full.parquet."
        ),
    },
    "heat_input.results.evaluation": {
        "title": "Observed Y vs Predicted Ŷ",
        "summary": "Inspect C7 fitted-model trajectories and evaluation diagnostics.",
        "details": (
            "The default plot is the Y/Ŷ time trajectory. Scatter and residual diagnostics "
            "remain available, with detailed artifacts loaded only after narrowing."
        ),
    },
    "heat_input.results.phvac_modes": {
        "title": "PHVAC Oracle and Chained Modes",
        "summary": "Keep PHVAC oracle and chained evaluations scientifically separate.",
        "details": (
            "Oracle PHVAC uses measured QAC-derived input. Chained PHVAC uses predicted QAC. "
            "They remain separate metric rows and traces."
        ),
    },
    "heat_input.results.annual": {
        "title": "Full-Year Predicted Ŷ",
        "summary": "Inspect C8 full-year prediction trajectories for selected components.",
        "details": (
            "C8 indexes zone packages first. After selecting one zone, component predictions "
            "are filtered by the selected model when a model filter is active."
        ),
    },
    "heat_input.results.inventory": {
        "title": "Metrics and Comparative Diagnostics",
        "summary": "Compare metrics and inspect compact lineage/model inventories.",
        "details": (
            "Estimator metrics, fitted coefficients, context error, split coverage, component "
            "availability, and PHVAC reconstruction remain available. All main plots use the "
            "75/25 graph-to-scrollable-legend layout."
        ),
    },
    "heat_input.results.validation": {
        "title": "Validation",
        "summary": "Inspect validator summaries and bounded diagnostic tables.",
        "details": (
            "Validation is read-only. Detailed diagnostic CSVs are loaded only after a "
            "validator stage is selected."
        ),
    },
}


HEAT_INPUT_HELP_ENTRIES.update(
    {
        "heat_input.results.plot_download": {
            "title": "Download Plotted Data",
            "summary": (
                "Download a self-describing ZIP for the traces currently visible in this plot."
            ),
            "details": (
                "The CSV/Parquet selector controls the data file inside the ZIP. The ZIP also "
                "contains selection_manifest.json and README.txt. The export is built from the "
                "displayed Plotly figure, not from a fresh query, so it follows the plotted "
                "snapshot and excludes traces hidden with the custom legend. The manifest records "
                "trace visibility, plot/filter context captured when the figure was built, and "
                "Phase C run lineage."
            ),
        },
        "heat_input.results.artifact_downloads": {
            "title": "Artifacts and Run Information",
            "summary": "Download non-plot Phase C run or model artifacts.",
            "details": (
                "These downloads are deliberately separated from plot data. Use the run summary "
                "for compact campaign/status/availability/validation metadata, and the selected "
                "model bundle for one exact trained model plus its training and dataset lineage."
            ),
        },
        "heat_input.results.summary_download": {
            "title": "Download Run Summary",
            "summary": "Download compact Phase C run metadata and validation summaries.",
            "details": (
                "The ZIP contains campaign_summary.json, stage_summary.csv, "
                "structural_availability.csv, validation_overview.csv, and a provenance manifest. "
                "It is not the data currently shown in a plot."
            ),
        },
        "heat_input.results.model_download": {
            "title": "Download Selected Model Bundle",
            "summary": "Download one exact persisted C6 trained-model artifact with lineage.",
            "details": (
                "This requires the current filters to resolve to exactly one completed trained "
                "model. The bundle includes the persisted model files, model/training manifests, "
                "source model-dataset manifest when available, selected model metadata, and a "
                "selection manifest."
            ),
        },
    }
)


_LINEAGE_MANAGED = {
    "campaign_root",
    "campaign_id",
    "generated_data_root",
    "matrix_run_id",
}

for field in PhaseCCampaignConfig.capability_manifest()["fields"]:
    name = str(field["name"])
    stages = ", ".join(field.get("phase_c_stages") or []) or "runner"
    visibility = str(field.get("ui_visibility") or "basic")
    managed = name in _LINEAGE_MANAGED
    if managed:
        detail_prefix = (
            "The Dash UI represents this backend field through upstream lineage or the "
            "dedicated matrix selector. "
        )
    else:
        detail_prefix = (
            "This remains part of the authoritative PhaseCCampaignConfig backend contract. "
            "It may be automatic/CLI-only even when it is not exposed in the simplified UI. "
        )
    HEAT_INPUT_HELP_ENTRIES[f"heat_input.field.{name}"] = {
        "title": name.replace("_", " ").title(),
        "summary": str(field.get("description") or ""),
        "details": f"{detail_prefix}Visibility: {visibility}. Phase C stage(s): {stages}.",
    }
