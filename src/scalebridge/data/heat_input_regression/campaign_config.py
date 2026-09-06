# -*- coding: utf-8 -*-
"""Scientific-neutral configuration contract for the Phase C campaign runner.

This module is intentionally outside Dash.  It is the single typed contract for
all user/config-file options that the authoritative C1-C9 Phase C runner can
honor.  Stage-local output paths and C1-C9 provenance IDs remain runner-owned
plumbing and are therefore not configurable here.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scalebridge.data.heat_input_regression.splitting import SUPPORTED_SPLIT_STRATEGIES


Stage = Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
Estimator = Literal["closed_form_linear", "pytorch_linear"]
Device = Literal["cpu", "cuda", "auto"]
ValidationProfile = Literal["full", "some", "none"]
MLflowValidationMode = Literal["full", "lightweight", "none"]


def _field(
    default: Any,
    *,
    stages: tuple[str, ...],
    group: str,
    visibility: str = "basic",
    description: str,
    **kwargs: Any,
):
    extra = {
        "phase_c_stages": list(stages),
        "ui_group": group,
        "ui_visibility": visibility,
    }
    return Field(default=default, description=description, json_schema_extra=extra, **kwargs)


class PhaseCCampaignConfig(BaseModel):
    """Complete public configuration for one Phase C C1-C9 campaign.

    Defaults preserve the behavior of the previously validated top-level runner
    where it intentionally differed from stage-script defaults (notably C6 uses
    ``pytorch_linear`` and device ``auto`` by default).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.2.0"

    # Parent/identity resolution.
    campaign_root: str | None = _field(
        None, stages=("C1",), group="identity",
        description="Existing parent campaign root containing Aggregation outputs.",
    )
    campaign_id: str | None = _field(
        None, stages=("C1", "C9"), group="identity",
        description="Campaign identifier. Derived from campaign_root when omitted.",
    )
    generated_data_root: str | None = _field(
        None, stages=("C1",), group="identity", visibility="advanced",
        description="Generated-data root used with campaign_id when campaign_root is omitted.",
    )
    matrix_run_id: str | None = _field(
        None, stages=("C1", "C2", "C3", "C4"), group="identity",
        description="Aggregation matrix run. Latest successful matrix is selected when omitted.",
    )
    c1_aggregation_run_root: str | None = _field(
        None, stages=("C1",), group="identity", visibility="expert",
        description=("Expert C1-only alternative to matrix discovery. Because C2-C4 require matrix "
                     "identity, this is valid only when start_stage=stop_stage=C1."),
    )
    phase_c_run_id: str | None = _field(
        None, stages=("C1", "C9"), group="identity", visibility="advanced",
        description="Optional Phase C run ID ending in YYYYMMDD_HHMMSS.",
    )

    # Scope filters supported by the stage CLIs.
    case_id: str | None = _field(
        None, stages=("C1", "C2", "C3", "C4"), group="scope",
        description="Optional single case filter.",
    )
    aggregation_id: str | None = _field(
        None, stages=("C1", "C2", "C3", "C4"), group="scope",
        description="Optional aggregation-ID filter.",
    )
    weight_mode: str | None = _field(
        None, stages=("C1", "C2", "C3", "C4"), group="scope",
        description="Optional Aggregation weight-mode filter.",
    )
    aggregate_zone_id: str | None = _field(
        None, stages=("C1", "C2", "C3", "C4", "C6", "C7", "C8"), group="scope",
        description="Optional aggregate-zone filter propagated through all compatible stages.",
    )
    model_ids: tuple[str, ...] = _field(
        (), stages=("C2", "C4", "C6", "C7", "C8"), group="scope",
        description="Optional repeatable model-ID filter. Empty means all applicable models.",
    )
    downstream_aggregate_zone_ids: tuple[str, ...] = _field(
        (), stages=("C6", "C7", "C8"), group="scope", visibility="expert",
        description=("Optional repeatable C6-C8 aggregate-zone recovery filter. Empty inherits "
                     "aggregate_zone_id when one is set, otherwise consumes every upstream zone."),
    )

    # C1/C2 scientific feature policy.
    minimum_sample_count: int = _field(
        1000, stages=("C1", "C2"), group="features_targets", ge=1,
        description="Minimum aligned valid samples required for a candidate relationship.",
    )
    internal_gain_predictor_method: Literal["aggregate_average", "contribution_sum"] = _field(
        "aggregate_average", stages=("C1", "C2"), group="features_targets",
        description="Internal-gain predictor construction method.",
    )
    hvac_target_method: Literal["signed_zone_sensible", "absolute_zone_sensible"] = _field(
        "signed_zone_sensible", stages=("C1", "C2"), group="features_targets",
        description="QAC/HVAC sensible target construction method.",
    )
    feature_preview_rows: int = _field(
        100, stages=("C2",), group="features_targets", visibility="advanced", ge=0,
        description="Rows written to each C2 preview CSV.",
    )

    # C3 split policy.
    split_strategy: Literal["monthly_distributed_holdout", "chronological_fraction"] = _field(
        "monthly_distributed_holdout", stages=("C3",), group="split",
        description="Deterministic train/validation/test split strategy.",
    )
    train_fraction: float = _field(
        0.70, stages=("C3",), group="split", ge=0.0, le=1.0,
        description="Requested training fraction.",
    )
    validation_fraction: float = _field(
        0.15, stages=("C3",), group="split", ge=0.0, le=1.0,
        description="Requested validation fraction.",
    )
    test_fraction: float = _field(
        0.15, stages=("C3",), group="split", ge=0.0, le=1.0,
        description="Requested test fraction.",
    )
    minimum_split_samples: int = _field(
        1000, stages=("C3",), group="split", ge=1,
        description="Minimum included samples required by C3 split validation.",
    )
    fraction_tolerance: float = _field(
        0.01, stages=("C3",), group="split", visibility="advanced", ge=0.0,
        description="Absolute fraction tolerance used by C3 split validation.",
    )
    split_random_seed: int = _field(
        42, stages=("C3",), group="split", visibility="advanced",
        description="Persisted split seed. Current supported splitters are deterministic.",
    )
    split_preview_rows: int = _field(
        100, stages=("C3",), group="split", visibility="advanced", ge=0,
        description="Rows written to each C3 preview CSV.",
    )

    # C4 dataset construction.
    dataset_minimum_split_samples: int = _field(
        1000, stages=("C4",), group="datasets", visibility="advanced", ge=1,
        description="Minimum rows required in each model-specific split during C4 construction.",
    )
    dataset_preview_rows: int = _field(
        100, stages=("C4",), group="datasets", visibility="advanced", ge=0,
        description="Rows written to each C4 dataset preview CSV.",
    )

    # C5 model API validation.
    c5_max_c4_models: int | None = _field(
        None, stages=("C5",), group="model_api", visibility="advanced", ge=1,
        description="Optional C4 model-dataset limit used only by C5 API compatibility validation.",
    )
    c5_skip_pytorch: bool = _field(
        False, stages=("C5",), group="model_api", visibility="advanced",
        description="Skip PyTorch checks in C5. Closed-form checks always run.",
    )
    c5_pytorch_devices: tuple[Device, ...] = _field(
        ("cpu",), stages=("C5",), group="model_api", visibility="advanced",
        description="Repeatable PyTorch devices validated by C5.",
    )

    # C6 estimator/training controls.
    estimator_types: tuple[Estimator, ...] = _field(
        ("pytorch_linear",), stages=("C6", "C7", "C8"), group="training",
        description="Estimator families trained and propagated to evaluation/inference.",
    )
    pytorch_devices: tuple[Device, ...] = _field(
        ("auto",), stages=("C6", "C7", "C8"), group="training",
        description="Repeatable requested PyTorch devices.",
    )
    fit_intercept_override: bool | None = _field(
        None, stages=("C6",), group="training", visibility="expert",
        description=("Expert global override for fit_intercept. None preserves the authoritative "
                     "per-model registry/dataset policy."),
    )
    ridge_alpha: float = _field(
        0.0, stages=("C6",), group="training", visibility="advanced", ge=0.0,
        description="Closed-form ridge penalty.",
    )
    learning_rate: float = _field(
        0.03, stages=("C6",), group="training", visibility="advanced", gt=0.0,
        description="PyTorch Adam learning rate.",
    )
    max_epochs: int = _field(
        3000, stages=("C6",), group="training", visibility="advanced", ge=1,
        description="Maximum PyTorch epochs.",
    )
    tolerance: float = _field(
        1e-10, stages=("C6",), group="training", visibility="advanced", ge=0.0,
        description="PyTorch early-stopping improvement tolerance.",
    )
    patience: int = _field(
        200, stages=("C6",), group="training", visibility="advanced", ge=1,
        description="PyTorch early-stopping patience.",
    )
    training_seed: int = _field(
        42, stages=("C6",), group="training", visibility="advanced",
        description="PyTorch deterministic training seed.",
    )
    reload_atol: float = _field(
        1e-12, stages=("C6",), group="training", visibility="advanced", ge=0.0,
        description="Absolute tolerance for C6 save/reload prediction identity.",
    )
    reload_rtol: float = _field(
        1e-12, stages=("C6",), group="training", visibility="advanced", ge=0.0,
        description="Relative tolerance for C6 save/reload prediction identity.",
    )
    training_prediction_preview_rows: int = _field(
        100, stages=("C6",), group="training", visibility="advanced", ge=0,
        description="Rows written to C6 prediction previews.",
    )

    # C7 evaluation and C8 annual inference.
    evaluation_prediction_preview_rows: int = _field(
        200, stages=("C7",), group="evaluation", visibility="advanced", ge=0,
        description="Rows written to C7 prediction preview files.",
    )
    write_full_predictions: bool = _field(
        True, stages=("C7",), group="evaluation",
        description="Write complete C7 prediction/residual tables in addition to previews.",
    )
    evaluation_requested_devices: tuple[Device, ...] = _field(
        (), stages=("C7",), group="evaluation", visibility="expert",
        description=("Optional C7 recovery filter on requested_device. Empty consumes every device "
                     "present in the selected C6 run."),
    )
    inference_preview_rows: int = _field(
        100, stages=("C8",), group="inference", visibility="advanced", ge=0,
        description="Rows written to each C8 annual inference preview.",
    )
    inference_requested_devices: tuple[Device, ...] = _field(
        (), stages=("C8",), group="inference", visibility="expert",
        description=("Optional C8 recovery filter on requested_device. Empty consumes every device "
                     "present in the selected C7 run."),
    )

    # Validation thresholds. These are intentionally explicit so the runner is
    # a complete contract rather than silently relying on validator defaults.
    feature_validation_absolute_tolerance: float = _field(
        1e-9, stages=("C2",), group="validation", visibility="advanced", ge=0.0,
        description="C2 feature-validation absolute tolerance.",
    )
    feature_validation_relative_tolerance: float = _field(
        1e-9, stages=("C2",), group="validation", visibility="advanced", ge=0.0,
        description="C2 feature-validation relative tolerance.",
    )
    expected_canonical_row_count: int = _field(
        105120, stages=("C2",), group="validation", visibility="advanced", ge=1,
        description="Expected canonical C2 row count used by canonical-aware/coalescence validation.",
    )
    canonical_timestamp_expected_row_count: int | None = _field(
        None, stages=("C2",), group="validation", visibility="advanced", ge=1,
        description="Optional exact row count for canonical-timestamp validation.",
    )
    expected_cadence_seconds: float = _field(
        300.0, stages=("C2",), group="validation", visibility="advanced", gt=0.0,
        description="Expected canonical timestamp cadence in seconds.",
    )
    fail_on_conflicting_source_values: bool = _field(
        False, stages=("C2",), group="validation", visibility="advanced",
        description="Fail C2 coalescence validation when conflicting duplicate source values exist.",
    )
    dataset_validation_absolute_tolerance: float = _field(
        1e-9, stages=("C4",), group="validation", visibility="advanced", ge=0.0,
        description="C4 dataset-validation absolute tolerance.",
    )
    dataset_validation_relative_tolerance: float = _field(
        1e-9, stages=("C4",), group="validation", visibility="advanced", ge=0.0,
        description="C4 dataset-validation relative tolerance.",
    )
    training_validation_coefficient_atol: float = _field(
        0.0, stages=("C6",), group="validation", visibility="advanced", ge=0.0,
        description="C6 coefficient absolute tolerance.",
    )
    training_validation_prediction_atol: float = _field(
        1e-12, stages=("C6",), group="validation", visibility="advanced", ge=0.0,
        description="C6 prediction absolute tolerance.",
    )
    training_validation_prediction_rtol: float = _field(
        1e-12, stages=("C6",), group="validation", visibility="advanced", ge=0.0,
        description="C6 prediction relative tolerance.",
    )
    evaluation_validation_metric_atol: float = _field(
        1e-12, stages=("C7",), group="validation", visibility="advanced", ge=0.0,
        description="C7 metric absolute tolerance.",
    )
    evaluation_validation_metric_rtol: float = _field(
        1e-12, stages=("C7",), group="validation", visibility="advanced", ge=0.0,
        description="C7 metric relative tolerance.",
    )
    inference_validation_prediction_atol: float = _field(
        1e-12, stages=("C8",), group="validation", visibility="advanced", ge=0.0,
        description="C8 prediction absolute tolerance.",
    )
    inference_validation_prediction_rtol: float = _field(
        1e-12, stages=("C8",), group="validation", visibility="advanced", ge=0.0,
        description="C8 prediction relative tolerance.",
    )

    # C9 MLflow controls.
    mlflow_enabled: bool = _field(
        True, stages=("C9",), group="mlflow",
        description="Register the completed Phase C run with MLflow.",
    )
    mlflow_validation_mode: MLflowValidationMode = _field(
        "full", stages=("C9",), group="mlflow",
        description="C9 registration validation mode.",
    )
    mlflow_experiment_name: str | None = _field(
        None, stages=("C9",), group="mlflow", visibility="advanced",
        description="Optional MLflow experiment-name override.",
    )
    mlflow_run_name: str | None = _field(
        None, stages=("C9",), group="mlflow", visibility="advanced",
        description="Optional MLflow parent run-name override.",
    )
    mlflow_strict: bool = _field(
        True, stages=("C9",), group="mlflow", visibility="advanced",
        description="Strict C9 registration behavior. False maps to --non-strict.",
    )
    mlflow_log_compact_artifacts: bool = _field(
        True, stages=("C9",), group="mlflow", visibility="advanced",
        description="Log compact Phase C artifacts to MLflow.",
    )
    mlflow_log_model_artifacts: bool = _field(
        False, stages=("C9",), group="mlflow", visibility="advanced",
        description="Also log model artifact directories to MLflow.",
    )
    mlflow_max_artifact_bytes: int = _field(
        20_000_000, stages=("C9",), group="mlflow", visibility="advanced", ge=1,
        description="Maximum individual artifact size accepted by C9 logging.",
    )

    # Optional C8 diagnostics already implemented as standalone scripts.
    run_inference_missing_value_audit: bool = _field(
        False, stages=("C8",), group="diagnostics", visibility="advanced",
        description="Run the existing C8 missing-value root-cause audit after inference.",
    )
    inspect_source_files: bool = _field(
        False, stages=("C8",), group="diagnostics", visibility="advanced",
        description="Inspect source files during the optional C8 missing-value audit.",
    )
    run_residual_gap_audit: bool = _field(
        False, stages=("C8",), group="diagnostics", visibility="advanced",
        description="Run the existing residual-gap deep audit after C8 inference.",
    )
    residual_gap_neighbor_radius: int = _field(
        2, stages=("C8",), group="diagnostics", visibility="advanced", ge=1,
        description="Neighbor radius used by the optional residual-gap audit.",
    )

    # Execution/recovery/truncation.
    validation_profile: ValidationProfile = _field(
        "full", stages=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"),
        group="execution", description="Overall validator profile.",
    )
    start_stage: Stage = _field(
        "C1", stages=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"),
        group="execution", visibility="advanced", description="First pipeline stage to execute.",
    )
    stop_stage: Stage = _field(
        "C9", stages=("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"),
        group="execution", visibility="advanced", description="Last pipeline stage to execute.",
    )
    continue_on_error: bool = _field(
        False, stages=("C1", "C2", "C3", "C4", "C6", "C7", "C8"),
        group="execution", visibility="advanced", description="Continue after stage/task failures where supported.",
    )
    overwrite_existing: bool = _field(
        False, stages=("C6", "C8"), group="execution", visibility="advanced",
        description="Allow replacement only in stages whose CLIs implement overwrite semantics.",
    )
    max_zones: int | None = _field(
        None, stages=("C1",), group="execution", visibility="advanced", ge=1,
        description="Development truncation: first N discovered aggregate zones at C1.",
    )
    max_model_datasets: int | None = _field(
        None, stages=("C6",), group="execution", visibility="advanced", ge=1,
        description="Development truncation: maximum C4 model datasets trained by C6.",
    )
    max_artifacts: int | None = _field(
        None, stages=("C7", "C8"), group="execution", visibility="advanced", ge=1,
        description="Development truncation: maximum artifacts evaluated/inferred.",
    )

    @field_validator(
        "model_ids", "downstream_aggregate_zone_ids", "estimator_types", "pytorch_devices",
        "c5_pytorch_devices", "evaluation_requested_devices", "inference_requested_devices",
    )
    @classmethod
    def _unique_tuple(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(str(v).strip() for v in values if str(v).strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Repeatable selections must not contain duplicates")
        return cleaned

    @model_validator(mode="after")
    def _validate_contract(self):
        if self.split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
            raise ValueError(f"Unsupported split strategy: {self.split_strategy}")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if not np.isclose(sum(fractions), 1.0, atol=1e-12):
            raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1")
        order = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        if order.index(self.start_stage) > order.index(self.stop_stage):
            raise ValueError("start_stage must not come after stop_stage")
        if not self.estimator_types:
            raise ValueError("estimator_types must contain at least one estimator")
        if "pytorch_linear" in self.estimator_types and not self.pytorch_devices:
            raise ValueError("pytorch_devices must not be empty when pytorch_linear is selected")
        if not self.c5_skip_pytorch and not self.c5_pytorch_devices:
            raise ValueError("c5_pytorch_devices must not be empty when C5 PyTorch validation is enabled")
        if self.inspect_source_files and not self.run_inference_missing_value_audit:
            raise ValueError("inspect_source_files requires run_inference_missing_value_audit=true")
        if self.aggregate_zone_id and self.downstream_aggregate_zone_ids:
            if set(self.downstream_aggregate_zone_ids) != {self.aggregate_zone_id}:
                raise ValueError(
                    "downstream_aggregate_zone_ids cannot broaden an upstream aggregate_zone_id filter"
                )
        if self.c1_aggregation_run_root:
            if not (self.start_stage == "C1" and self.stop_stage == "C1"):
                raise ValueError("c1_aggregation_run_root is supported only for a C1-only run")
            if self.matrix_run_id:
                raise ValueError("c1_aggregation_run_root and matrix_run_id are mutually exclusive")
        if self.generated_data_root and not self.campaign_id and not self.campaign_root:
            raise ValueError("generated_data_root requires campaign_id when campaign_root is omitted")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def capability_manifest(cls) -> dict[str, Any]:
        schema = cls.model_json_schema()
        fields: list[dict[str, Any]] = []
        for name, model_field in cls.model_fields.items():
            if name == "schema_version":
                continue
            info = schema.get("properties", {}).get(name, {})
            extra = model_field.json_schema_extra or {}
            fields.append(
                {
                    "name": name,
                    "default": model_field.default,
                    "required": model_field.is_required(),
                    "description": model_field.description or "",
                    "phase_c_stages": extra.get("phase_c_stages", []),
                    "ui_group": extra.get("ui_group", ""),
                    "ui_visibility": extra.get("ui_visibility", ""),
                    "json_schema": info,
                }
            )
        return {
            "schema_version": "0.2.0",
            "contract": "PhaseCCampaignConfig",
            "fields": fields,
            "internal_only": [
                "stage output roots",
                "C1-C8 stage run IDs derived from phase_c_run_id timestamp suffix",
                "C1-C8 manifest override paths used only by standalone C9 recovery",
                "registration_output_dir",
                "source-syntax validator path list",
                "C9 expected stage/task-run counts derived from the fixed C1-C8 hierarchy and completed manifests",
            ],
            "internal_only_cli_options": [
                "--audit-run-id", "--dataset-root", "--dataset-run-id",
                "--evaluation-root", "--evaluation-run-id", "--feature-root",
                "--feature-run-id", "--inference-root", "--inference-run-id",
                "--output-root", "--paths", "--registration-manifest",
                "--registration-output-dir", "--split-run-id", "--training-root",
                "--training-run-id", "--expected-stage-runs",
                "--expected-training-task-runs", "--expected-evaluation-task-runs",
                "--expected-inference-task-runs",
            ],
        }
