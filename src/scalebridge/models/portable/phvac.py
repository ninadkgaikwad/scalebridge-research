from __future__ import annotations

"""E0-7 Phase-C PHVAC attachment and equal-allocation reconstruction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from scalebridge.models.heat_input_regression.serialization import (
    load_heat_input_regression_model,
)

from .bundle import PortableModelBundle
from .contracts import PHVACBundleContract, PHVACZoneModelSpec, PortableModelError


@dataclass(frozen=True)
class PHVACPrediction:
    available: bool
    per_zone_w: Mapping[str, float]
    partial_sum_w: float | None
    allocation_completion_w: float | None
    building_total_w: float | None
    total_aggregate_zones: int
    available_model_count: int
    missing_model_count: int
    reconstruction_factor: float | None


class PHVACRuntime:
    """Forward-only runtime over PHVAC models embedded from authoritative Phase C."""

    def __init__(
        self,
        bundle_root: str | Path,
        contract: PHVACBundleContract,
    ) -> None:
        self.bundle_root = Path(bundle_root).resolve()
        self.contract = contract
        self._models = {}
        for item in contract.zone_models:
            artifact = (self.bundle_root / Path(*item.artifact_relpath.split("/"))).resolve()
            try:
                artifact.relative_to(self.bundle_root)
            except ValueError as exc:
                raise PortableModelError("PHVAC artifact escapes portable bundle root") from exc
            if not artifact.is_dir():
                raise PortableModelError(f"Embedded PHVAC artifact directory missing: {artifact}")
            model = load_heat_input_regression_model(artifact)
            if getattr(model, "model_id", "") not in {"", "PHVAC"}:
                raise PortableModelError(
                    f"Embedded PHVAC artifact for {item.zone_id!r} has model_id={model.model_id!r}"
                )
            self._models[item.zone_id] = model

    @classmethod
    def from_bundle(cls, bundle: PortableModelBundle) -> "PHVACRuntime | None":
        if bundle.manifest.phvac is None:
            return None
        return cls(bundle.root, bundle.manifest.phvac)

    @staticmethod
    def _transform(value: float, spec: PHVACZoneModelSpec) -> float:
        x = float(value)
        if not np.isfinite(x):
            raise PortableModelError(f"Non-finite QAC for PHVAC zone {spec.zone_id!r}")
        if spec.input_transform == "absolute_value":
            return abs(x)
        if spec.input_transform == "identity":
            return x
        raise PortableModelError(f"Unsupported PHVAC input transform: {spec.input_transform}")

    def predict(self, qac_by_zone_w: Mapping[str, float]) -> PHVACPrediction:
        n = self.contract.total_aggregate_zones
        available = self.contract.available_model_count
        missing = self.contract.missing_model_count

        if available == 0:
            return PHVACPrediction(
                available=False,
                per_zone_w={},
                partial_sum_w=None,
                allocation_completion_w=None,
                building_total_w=None,
                total_aggregate_zones=n,
                available_model_count=0,
                missing_model_count=n,
                reconstruction_factor=None,
            )

        per_zone: dict[str, float] = {}
        specs = {item.zone_id: item for item in self.contract.zone_models}
        for zone_id, model in self._models.items():
            if zone_id not in qac_by_zone_w:
                raise PortableModelError(
                    f"QAC is required for embedded PHVAC model zone {zone_id!r}"
                )
            predictor = self._transform(float(qac_by_zone_w[zone_id]), specs[zone_id])
            value = float(model.predict_one(predictor))
            if not np.isfinite(value):
                raise PortableModelError(f"PHVAC prediction is non-finite for zone {zone_id!r}")
            per_zone[zone_id] = value

        partial = float(sum(per_zone.values()))
        if missing == 0:
            factor = 1.0
            completion = 0.0
            total = partial
        else:
            # Current Phase-C production contract: every PHVAC zone model was
            # trained against P_HVAC,facility / N, so missing allocated shares
            # are completed by N/(N-M), where M is the missing-model count.
            factor = float(n / available)
            total = float(factor * partial)
            completion = float(total - partial)

        return PHVACPrediction(
            available=True,
            per_zone_w=per_zone,
            partial_sum_w=partial,
            allocation_completion_w=completion,
            building_total_w=total,
            total_aggregate_zones=n,
            available_model_count=available,
            missing_model_count=missing,
            reconstruction_factor=factor,
        )


def prepare_phvac_bundle_contract(
    *,
    total_aggregate_zones: int,
    zone_artifact_dirs: Mapping[str, str | Path],
    source_locators: Mapping[str, object] | None = None,
    embedded_root: str = "phvac",
) -> tuple[PHVACBundleContract, dict[str, Path]]:
    """Inspect current Phase-C model artifacts and prepare E0-7 embedding.

    Stored model metadata is preferred.  The current authoritative Phase-C
    registry supplies the fallback policy for legacy artifacts that predate
    propagation of ``input_transform``/``target_allocation`` into metadata.
    """
    import json
    from scalebridge.models.heat_input_regression.registry import get_model_specification
    from .contracts import DataLocator

    current = get_model_specification("PHVAC")
    specs: list[PHVACZoneModelSpec] = []
    embeds: dict[str, Path] = {}
    locators = dict(source_locators or {})
    for zone_id, source in sorted(zone_artifact_dirs.items()):
        source_path = Path(source)
        manifest_path = source_path / "model_manifest.json"
        if not manifest_path.is_file():
            raise PortableModelError(
                f"Phase-C PHVAC artifact for {zone_id!r} lacks model_manifest.json"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(payload.get("model_id", "")) != "PHVAC":
            raise PortableModelError(
                f"Expected PHVAC model artifact for {zone_id!r}; got {payload.get('model_id')!r}"
            )
        metadata = dict(payload.get("metadata", {}))
        transform = str(metadata.get("input_transform", current.input_transform))
        allocation = str(metadata.get("target_allocation", current.target_allocation))
        if allocation != "equal_across_aggregate_zones":
            raise PortableModelError(
                "E0-7 v1 current production implementation requires equal Phase-C PHVAC allocation"
            )
        relative = f"{embedded_root.strip('/')}/{zone_id}"
        locator = locators.get(zone_id)
        if locator is not None and not isinstance(locator, DataLocator):
            raise PortableModelError("source_locators values must be DataLocator instances")
        specs.append(
            PHVACZoneModelSpec(
                zone_id=str(zone_id),
                artifact_relpath=relative,
                input_transform=transform,
                target_allocation=allocation,
                source_locator=locator,
            )
        )
        embeds[relative] = source_path
    return (
        PHVACBundleContract(
            total_aggregate_zones=int(total_aggregate_zones),
            zone_models=tuple(specs),
        ),
        embeds,
    )
