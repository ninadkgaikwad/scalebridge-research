from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    CanonicalRuntimeFrame,
    DiscretizationConfig,
    RCCompilerSpec,
)
from scalebridge.models.heat_input_regression.linear_closed_form import (
    ClosedFormLinearRegression,
)
from scalebridge.models.portable import (
    ArtifactStage,
    DataLocator,
    DataRootRegistry,
    MethodPayloadDescriptor,
    ModelFamily,
    NormalizationContract,
    PHVACBundleContract,
    PHVACRuntime,
    PHVACZoneModelSpec,
    PortableModelBundle,
    PortableModelError,
    PortableModelManifest,
    RCForwardRuntime,
    RCPhysicalPayload,
    RuntimeInputSchema,
    ScalarTransform,
    default_final_allocation_results,
    denormalize_named_outputs,
    locator_from_path,
    normalize_named_inputs,
    write_portable_model_bundle,
    write_rc_physical_payload,
)


def _fit_phvac(path: Path, *, coefficient: float, intercept: float = 100.0) -> Path:
    x = np.array([0.0, 1000.0, 2000.0, 3000.0])
    y = intercept + coefficient * x
    model = ClosedFormLinearRegression(
        fit_intercept=True,
        model_id="PHVAC",
        metadata={
            "input_transform": "absolute_value",
            "target_allocation": "equal_across_aggregate_zones",
        },
    ).fit(x, y)
    model.save(path)
    return path


def _simple_rc_payload() -> RCPhysicalPayload:
    spec = RCCompilerSpec(flavour="1r1c", zone_ids=("Z1",), mode="ind")
    theta = {
        "C|Z1|C_a": 1.0e6,
        "R|boundary|Z1::a--boundary::outdoor_temperature|R_ao": 0.01,
    }
    return RCPhysicalPayload(
        compiler_spec=spec,
        theta=theta,
        discretization=DiscretizationConfig(solver="exact_zoh_linear", substeps=1),
    )


def _manifest(*, phvac=None, normalization=None, payload_metadata=None) -> PortableModelManifest:
    return PortableModelManifest(
        model_id="demo_model",
        payload=MethodPayloadDescriptor(
            family=ModelFamily.OPTIMIZATION,
            method_id="casadi_ipopt_rc",
            deployment_kind="physical_rc_theta",
            metadata=dict(payload_metadata or {}),
        ),
        runtime_inputs=RuntimeInputSchema(
            controls=("qac",),
            disturbances=("outdoor_temperature", "zic", "zir", "qsol1", "qsol2"),
            observed_outputs=("zone_temperature",),
        ),
        normalization=normalization or NormalizationContract(),
        phvac=phvac,
    )


def test_data_locator_rejects_absolute_paths() -> None:
    with pytest.raises(PortableModelError):
        DataLocator(
            stage=ArtifactStage.PHASE_D,
            root_alias="generated_data",
            relative_path="C:/bad/path.json",
            artifact_kind="phase_d_manifest",
        )


def test_data_root_registry_roundtrip_and_optional_hash(tmp_path: Path) -> None:
    root = tmp_path / "data_root"
    root.mkdir()
    artifact = root / "phase_d" / "manifest.json"
    artifact.parent.mkdir()
    artifact.write_text('{"ok": true}', encoding="utf-8")
    locator = locator_from_path(
        artifact,
        root_alias="generated_data",
        root=root,
        stage="phase_d",
        artifact_kind="phase_d_manifest",
        include_sha256=True,
        required_for_historical_replay=True,
    )
    registry = DataRootRegistry({"generated_data": root})
    assert registry.resolve(locator, must_exist=True, verify_sha256=True) == artifact.resolve()
    assert locator.required_for_historical_replay is True


def test_normalization_is_immutable_model_data_and_roundtrips() -> None:
    contract = NormalizationContract(
        inputs={"temperature": ScalarTransform(offset=20.0, scale=5.0)},
        outputs={"temperature": ScalarTransform(offset=20.0, scale=5.0)},
    )
    normalized = normalize_named_inputs({"temperature": 25.0}, contract, strict=True)
    assert normalized["temperature"] == pytest.approx(1.0)
    physical = denormalize_named_outputs(normalized, contract, strict=True)
    assert physical["temperature"] == pytest.approx(25.0)


def test_generic_payload_envelope_supports_all_four_families() -> None:
    for family in (
        ModelFamily.CLASSICAL_ML,
        ModelFamily.SCIML,
        ModelFamily.OPTIMIZATION,
        ModelFamily.BAYESIAN,
    ):
        descriptor = MethodPayloadDescriptor(
            family=family,
            method_id=f"{family.value}_future_method",
            deployment_kind="future_payload",
        )
        assert descriptor.family is family


def test_bundle_roundtrip_and_integrity(tmp_path: Path) -> None:
    payload_file = tmp_path / "payload.json"
    payload_file.write_text('{"theta": 1}', encoding="utf-8")
    bundle = write_portable_model_bundle(
        tmp_path / "bundle",
        _manifest(payload_metadata={"rc_payload_relpath": "payload/model.json"}),
        embedded_artifacts={"payload/model.json": payload_file},
    )
    assert bundle.manifest.model_id == "demo_model"
    assert bundle.path("payload/model.json").is_file()
    PortableModelBundle.load(bundle.root, validate_integrity=True)


def test_bundle_integrity_is_validation_capability_not_per_step(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("original", encoding="utf-8")
    bundle = write_portable_model_bundle(
        tmp_path / "bundle",
        _manifest(),
        embedded_artifacts={"payload/a.txt": source},
    )
    bundle.path("payload/a.txt").write_text("tampered", encoding="utf-8")
    # Loading without qualification is intentionally cheap.
    PortableModelBundle.load(bundle.root, validate_integrity=False)
    with pytest.raises(PortableModelError):
        PortableModelBundle.load(bundle.root, validate_integrity=True)


def test_phvac_m0_direct_sum_all_n_models(tmp_path: Path) -> None:
    a = _fit_phvac(tmp_path / "A", coefficient=0.2)
    b = _fit_phvac(tmp_path / "B", coefficient=0.3)
    contract = PHVACBundleContract(
        total_aggregate_zones=2,
        zone_models=(
            PHVACZoneModelSpec("A", "phvac/A"),
            PHVACZoneModelSpec("B", "phvac/B"),
        ),
    )
    bundle = write_portable_model_bundle(
        tmp_path / "bundle",
        _manifest(phvac=contract),
        embedded_artifacts={"phvac/A": a, "phvac/B": b},
    )
    pred = PHVACRuntime.from_bundle(bundle).predict({"A": -1000.0, "B": 1000.0})
    assert pred.missing_model_count == 0
    assert pred.reconstruction_factor == pytest.approx(1.0)
    assert pred.per_zone_w["A"] == pytest.approx(300.0)
    assert pred.per_zone_w["B"] == pytest.approx(400.0)
    assert pred.partial_sum_w == pytest.approx(700.0)
    assert pred.allocation_completion_w == pytest.approx(0.0)
    assert pred.building_total_w == pytest.approx(700.0)


def test_phvac_missing_model_uses_n_over_n_minus_m_completion(tmp_path: Path) -> None:
    a = _fit_phvac(tmp_path / "A", coefficient=0.2)
    b = _fit_phvac(tmp_path / "B", coefficient=0.3)
    contract = PHVACBundleContract(
        total_aggregate_zones=3,
        zone_models=(
            PHVACZoneModelSpec("A", "phvac/A"),
            PHVACZoneModelSpec("B", "phvac/B"),
        ),
    )
    bundle = write_portable_model_bundle(
        tmp_path / "bundle",
        _manifest(phvac=contract),
        embedded_artifacts={"phvac/A": a, "phvac/B": b},
    )
    pred = PHVACRuntime.from_bundle(bundle).predict({"A": -1000.0, "B": 1000.0})
    assert pred.available_model_count == 2
    assert pred.missing_model_count == 1
    assert pred.partial_sum_w == pytest.approx(700.0)
    assert pred.reconstruction_factor == pytest.approx(3.0 / 2.0)
    assert pred.building_total_w == pytest.approx(1050.0)
    assert pred.allocation_completion_w == pytest.approx(350.0)


def test_phvac_all_models_missing_is_unavailable_not_fabricated(tmp_path: Path) -> None:
    contract = PHVACBundleContract(total_aggregate_zones=3, zone_models=())
    bundle = write_portable_model_bundle(tmp_path / "bundle", _manifest(phvac=contract))
    pred = PHVACRuntime.from_bundle(bundle).predict({})
    assert pred.available is False
    assert pred.missing_model_count == 3
    assert pred.reconstruction_factor is None
    assert pred.building_total_w is None
    assert pred.allocation_completion_w is None


def test_phvac_uses_absolute_qac_transform(tmp_path: Path) -> None:
    a = _fit_phvac(tmp_path / "A", coefficient=0.2)
    contract = PHVACBundleContract(
        total_aggregate_zones=1,
        zone_models=(PHVACZoneModelSpec("A", "phvac/A", input_transform="absolute_value"),),
    )
    bundle = write_portable_model_bundle(
        tmp_path / "bundle",
        _manifest(phvac=contract),
        embedded_artifacts={"phvac/A": a},
    )
    runtime = PHVACRuntime.from_bundle(bundle)
    assert runtime.predict({"A": -1000.0}).building_total_w == pytest.approx(
        runtime.predict({"A": 1000.0}).building_total_w
    )


def test_rc_physical_payload_roundtrip(tmp_path: Path) -> None:
    payload = _simple_rc_payload()
    path = write_rc_physical_payload(tmp_path / "rc.json", payload)
    from scalebridge.models.portable import load_rc_physical_payload

    loaded = load_rc_physical_payload(path)
    assert loaded.compiler_spec.flavour == "1r1c"
    assert loaded.compiler_spec.zone_ids == ("Z1",)
    assert dict(loaded.theta) == dict(payload.theta)
    assert loaded.discretization.normalized_solver == "exact_zoh_linear"


def test_rc_forward_runtime_save_load_equivalence_and_optional_diagnostics(tmp_path: Path) -> None:
    payload = _simple_rc_payload()
    payload_file = write_rc_physical_payload(tmp_path / "rc.json", payload)
    bundle = write_portable_model_bundle(
        tmp_path / "bundle",
        _manifest(payload_metadata={"rc_payload_relpath": "payload/rc.json"}),
        embedded_artifacts={"payload/rc.json": payload_file},
    )

    def run_once(include_diagnostics: bool):
        runtime = RCForwardRuntime.from_bundle(bundle)
        runtime.initialize(timestamp=0, observed_air_temperatures_c={"Z1": 22.0})
        frame = CanonicalRuntimeFrame(
            timestamp=0,
            boundary_temperatures={"outdoor_temperature": 10.0},
            local_thermal_powers={
                ("Z1", "qac"): 1000.0,
                ("Z1", "zic"): 0.0,
                ("Z1", "zir"): 0.0,
                ("Z1", "qsol1"): 0.0,
                ("Z1", "qsol2"): 0.0,
            },
        )
        return runtime.step(
            frame,
            next_timestamp=300,
            sample_dt_s=300.0,
            include_diagnostics=include_diagnostics,
        )

    a = run_once(False)
    b = run_once(True)
    np.testing.assert_allclose(a.state.state, b.state.state, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(a.observed_output, b.observed_output, atol=1e-12, rtol=1e-12)
    assert a.diagnostics is None
    assert b.diagnostics is not None
    assert "disturbances_used" in b.diagnostics


def test_rc_runtime_does_not_implicitly_reset_state(tmp_path: Path) -> None:
    payload = _simple_rc_payload()
    runtime = RCForwardRuntime(payload)
    runtime.initialize(timestamp=0, observed_air_temperatures_c={"Z1": 22.0})
    frame0 = CanonicalRuntimeFrame(
        timestamp=0,
        boundary_temperatures={"outdoor_temperature": 10.0},
        local_thermal_powers={
            ("Z1", "qac"): 0.0,
            ("Z1", "zic"): 0.0,
            ("Z1", "zir"): 0.0,
            ("Z1", "qsol1"): 0.0,
            ("Z1", "qsol2"): 0.0,
        },
        observed_air_temperatures={"Z1": 99.0},  # rich frame; must not teacher-force
    )
    out = runtime.step(frame0, next_timestamp=300, sample_dt_s=300.0)
    assert out.state.state[0] < 22.0
    assert out.state.state[0] != pytest.approx(99.0)


def test_dep2_neutral_allocation_can_be_portably_embedded() -> None:
    spec = RCCompilerSpec(
        flavour="1r1c",
        zone_ids=("A", "B"),
        mode="dep2",
        dep2_allocations=(
            AllocationFamilySpec(
                name="disturbance_family",
                signals=("zic", "zir", "qsol1", "qsol2"),
                weights={"A": 0.5, "B": 0.5},
                mode=AllocationMode.NEUTRAL_FIXED,
            ),
        ),
    )
    results = default_final_allocation_results(spec)
    assert results["disturbance_family"].lambda_by_zone == {"A": 1.0, "B": 1.0}


def _minimal_phase_d_manifest() -> dict[str, object]:
    def col(name, physical_role, temporal_role, base_signal, zone, offset, units):
        return {
            "name": name,
            "physical_role": physical_role,
            "temporal_role": temporal_role,
            "aggregate_zone_id": zone,
            "base_signal": base_signal,
            "offset_steps": offset,
            "units": units,
        }

    return {
        "schema_version": "phase_d_d6_silo_contract_v1",
        "d7_schema_version": "phase_d_d7_final_dataset_v1",
        "silo": "ml_sciml",
        "mode": "independent",
        "independent_zone_id": "Z1",
        "current_zone_ids": ["Z1"],
        "dependent_2_source_zone_id": None,
        "heat_representation": {
            "representation": "grouped_qzic_qzir",
            "include_visible_lighting_in_qzir": True,
            "folder_name": "grp_vrin",
        },
        "temporal_config": {
            "silo": "ml_sciml",
            "input_lag": 1,
            "target_horizon": 1,
            "policy_name": "monthly_distributed_holdout",
            "policy_realization_id": None,
            "policy_parameters": {
                "train_fraction": 0.70,
                "test_fraction": 0.15,
                "validation_fraction": 0.15,
            },
        },
        "final_columns": [
            col("timestamp", "metadata", "anchor_timestamp", "timestamp", None, None, None),
            col("included", "metadata", "selection", "included", None, None, None),
            col("partition", "metadata", "partition", "partition", None, None, None),
            col("window_id", "metadata", "selection_window", "window_id", None, None, None),
            col("season", "metadata", "season", "season", None, None, None),
            col("outdoor_temperature__lag_0", "disturbance", "model_input", "outdoor_temperature", None, 0, "degC"),
            col("Z1__zone_temperature__lag_0", "state", "model_input", "zone_temperature", "Z1", 0, "degC"),
            col("Z1__qac__lag_0", "control_input", "model_input", "qac", "Z1", 0, "W"),
            col("Z1__zic__lag_0", "disturbance", "model_input", "zic", "Z1", 0, "W"),
            col("Z1__zir__lag_0", "disturbance", "model_input", "zir", "Z1", 0, "W"),
            col("Z1__qsol1__lag_0", "disturbance", "model_input", "qsol1", "Z1", 0, "W"),
            col("Z1__qsol2__lag_0", "disturbance", "model_input", "qsol2", "Z1", 0, "W"),
            col("Z1__zone_temperature__target_1", "target", "prediction_target", "zone_temperature", "Z1", 1, "degC"),
        ],
        "provenance": {
            "campaign_id": "campaign",
            "case_id": "case",
            "aggregation_matrix_run_id": "matrix",
            "aggregation_run_id": "run",
            "aggregation_id": "identity",
            "weight_mode": "equal",
            "phase_c_campaign_run_id": "phase_c",
        },
        "row_count": 2,
        "included_row_count": 1,
        "partition_counts": {"train": 1, "excluded": 1},
    }


def test_historical_replay_resolves_phase_d_manifest_and_data(tmp_path: Path) -> None:
    import importlib.util
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        pytest.skip("Local packaging environment has no Parquet engine; laptop environment is authoritative")
    import pandas as pd
    from scalebridge.models.portable import HistoricalReplayDataset

    root = tmp_path / "generated"
    run = root / "phase_d" / "run"
    run.mkdir(parents=True)
    manifest = _minimal_phase_d_manifest()
    manifest_path = run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    columns = [item["name"] for item in manifest["final_columns"]]
    rows = []
    for included, partition in [(True, "train"), (False, "excluded")]:
        row = {name: 0.0 for name in columns}
        row.update({
            "timestamp": "2017-09-01 00:00:00",
            "included": included,
            "partition": partition,
            "window_id": "w",
            "season": "summer",
        })
        rows.append(row)
    data_path = run / "data.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)

    manifest_locator = locator_from_path(
        manifest_path,
        root_alias="generated_data",
        root=root,
        stage="phase_d",
        artifact_kind="phase_d_manifest",
        required_for_historical_replay=True,
    )
    data_locator = locator_from_path(
        data_path,
        root_alias="generated_data",
        root=root,
        stage="phase_d",
        artifact_kind="phase_d_data",
        required_for_historical_replay=True,
    )
    replay = HistoricalReplayDataset.load(
        registry=DataRootRegistry({"generated_data": root}),
        manifest_locator=manifest_locator,
        data_locator=data_locator,
    )
    assert replay.contract.modeled_zone_ids == ("Z1",)
    selected = replay.select(partitions=("train",), included_only=True)
    assert len(selected) == 1


def test_prepare_phvac_bundle_contract_ingests_phase_c_artifact_metadata(tmp_path: Path) -> None:
    from scalebridge.models.portable import prepare_phvac_bundle_contract

    a = _fit_phvac(tmp_path / "A", coefficient=0.2)
    contract, embeds = prepare_phvac_bundle_contract(
        total_aggregate_zones=2,
        zone_artifact_dirs={"A": a},
    )
    assert contract.total_aggregate_zones == 2
    assert contract.available_model_count == 1
    assert contract.missing_model_count == 1
    assert contract.zone_models[0].input_transform == "absolute_value"
    assert contract.zone_models[0].target_allocation == "equal_across_aggregate_zones"
    assert embeds["phvac/A"] == a
