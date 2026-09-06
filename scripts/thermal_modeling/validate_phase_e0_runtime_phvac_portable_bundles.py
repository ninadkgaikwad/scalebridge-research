from __future__ import annotations

"""Authoritative laptop validator for ScaleBridge Phase E0-7 implementation."""

import json
from pathlib import Path
import tempfile

import numpy as np

from scalebridge.models.grey_box.rc_networks import (
    CanonicalRuntimeFrame,
    DiscretizationConfig,
    RCCompilerSpec,
)
from scalebridge.models.heat_input_regression.linear_closed_form import (
    ClosedFormLinearRegression,
)
from scalebridge.models.heat_input_regression.registry import get_model_specification
from scalebridge.models.portable import (
    MethodPayloadDescriptor,
    ModelFamily,
    PHVACRuntime,
    PortableModelBundle,
    PortableModelManifest,
    RCForwardRuntime,
    RCPhysicalPayload,
    RuntimeInputSchema,
    prepare_phvac_bundle_contract,
    write_portable_model_bundle,
    write_rc_physical_payload,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _fit_phvac(path: Path, coefficient: float) -> Path:
    x = np.asarray([0.0, 1000.0, 2000.0, 3000.0])
    y = 100.0 + coefficient * x
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


def main() -> int:
    spec = get_model_specification("PHVAC")
    _require(spec.input_transform == "absolute_value", "Phase-C PHVAC transform drifted")
    _require(
        spec.target_allocation == "equal_across_aggregate_zones",
        "Phase-C PHVAC allocation drifted",
    )

    with tempfile.TemporaryDirectory(prefix="scalebridge_e07_validate_") as td:
        root = Path(td)
        ph_a = _fit_phvac(root / "phase_c_A", 0.2)
        ph_b = _fit_phvac(root / "phase_c_B", 0.3)

        # Qualification 1: current Phase-C equal-allocation completion with one
        # missing PHVAC model (N=3, M=1, available=2).
        completion_contract, completion_embeds = prepare_phvac_bundle_contract(
            total_aggregate_zones=3,
            zone_artifact_dirs={"A": ph_a, "B": ph_b},
        )
        phvac_manifest = PortableModelManifest(
            model_id="e07_validator_phvac",
            payload=MethodPayloadDescriptor(
                family=ModelFamily.GENERIC,
                method_id="phvac_only_validator",
                deployment_kind="forward_auxiliary",
            ),
            runtime_inputs=RuntimeInputSchema(controls=("qac",)),
            phvac=completion_contract,
        )
        phvac_bundle = write_portable_model_bundle(
            root / "phvac_bundle",
            phvac_manifest,
            embedded_artifacts=completion_embeds,
        )
        pred = PHVACRuntime.from_bundle(phvac_bundle).predict({"A": -1000.0, "B": 1000.0})
        _require(pred.missing_model_count == 1, "Expected one missing PHVAC model")
        _require(abs(pred.reconstruction_factor - 1.5) < 1e-12, "N/(N-M) factor incorrect")
        _require(abs(pred.building_total_w - 1.5 * pred.partial_sum_w) < 1e-12, "PHVAC completion mismatch")

        # Qualification 2: one-zone physical RC + matching one-zone PHVAC
        # carried together in one deployable bundle.
        runtime_phvac_contract, runtime_phvac_embeds = prepare_phvac_bundle_contract(
            total_aggregate_zones=1,
            zone_artifact_dirs={"A": ph_a},
        )
        rc_spec = RCCompilerSpec(flavour="1r1c", zone_ids=("A",), mode="ind")
        rc_payload = RCPhysicalPayload(
            compiler_spec=rc_spec,
            theta={
                "C|A|C_a": 1.0e6,
                "R|boundary|A::a--boundary::outdoor_temperature|R_ao": 0.01,
            },
            discretization=DiscretizationConfig(solver="exact_zoh_linear", substeps=1),
        )
        rc_file = write_rc_physical_payload(root / "rc_payload.json", rc_payload)
        manifest = PortableModelManifest(
            model_id="e07_validator_model",
            payload=MethodPayloadDescriptor(
                family=ModelFamily.OPTIMIZATION,
                method_id="validator_physical_rc",
                deployment_kind="physical_rc_theta",
                metadata={"rc_payload_relpath": "payload/rc_payload.json"},
            ),
            runtime_inputs=RuntimeInputSchema(
                controls=("qac",),
                disturbances=("outdoor_temperature", "zic", "zir", "qsol1", "qsol2"),
                observed_outputs=("zone_temperature",),
            ),
            phvac=runtime_phvac_contract,
        )
        embeds = {"payload/rc_payload.json": rc_file, **runtime_phvac_embeds}
        bundle = write_portable_model_bundle(root / "bundle", manifest, embedded_artifacts=embeds)
        PortableModelBundle.load(bundle.root, validate_integrity=True)

        runtime = RCForwardRuntime.from_bundle(bundle)
        runtime.initialize(timestamp=0, observed_air_temperatures_c={"A": 22.0})
        frame = CanonicalRuntimeFrame(
            timestamp=0,
            boundary_temperatures={"outdoor_temperature": 10.0},
            local_thermal_powers={
                ("A", "qac"): 1000.0,
                ("A", "zic"): 0.0,
                ("A", "zir"): 0.0,
                ("A", "qsol1"): 0.0,
                ("A", "qsol2"): 0.0,
            },
        )
        out = runtime.step(
            frame,
            next_timestamp=300,
            sample_dt_s=300.0,
            include_diagnostics=True,
        )
        _require(np.all(np.isfinite(out.observed_output)), "Forward runtime produced non-finite output")
        _require(out.diagnostics is not None, "Optional diagnostics were not emitted")
        _require(out.phvac is not None and out.phvac.building_total_w is not None, "PHVAC was not attached to runtime output")

        reloaded = PortableModelBundle.load(bundle.root, validate_integrity=True)
        runtime2 = RCForwardRuntime.from_bundle(reloaded)
        runtime2.initialize(timestamp=0, observed_air_temperatures_c={"A": 22.0})
        out2 = runtime2.step(frame, next_timestamp=300, sample_dt_s=300.0)
        np.testing.assert_allclose(out.observed_output, out2.observed_output, atol=1e-12, rtol=1e-12)
        _require(abs(out.phvac.building_total_w - out2.phvac.building_total_w) < 1e-12, "PHVAC save/load parity failed")

    print("[PASS] E0-7 portable bundle write/load/integrity")
    print("[PASS] E0-7 Phase-C PHVAC ingestion + N/(N-M) reconstruction")
    print("[PASS] E0-7 physical RC forward runtime + PHVAC attachment")
    print("[PASS] E0-7 deterministic reload parity + optional diagnostics")
    print("E0-7 RUNTIME + PHVAC + PORTABLE MODEL BUNDLES VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
