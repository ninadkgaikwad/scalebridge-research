from __future__ import annotations

"""Portable physical-RC payload used by ODE/Inverse-PINN/Opt/Bayes deployment."""

from dataclasses import dataclass
from typing import Any, Mapping

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    AllocationResult,
    ConnectionRule,
    DiscretizationConfig,
    HeatPortGroup,
    ParameterSharingRule,
    RCCompilerSpec,
    SpatialMode,
    ZoneAdjacency,
    compile_rc_model,
    fixed_allocation_result,
    neutral_allocation_result,
)

from .contracts import PortableModelError


RC_PHYSICAL_PAYLOAD_SCHEMA_VERSION = "phase_e0_e07_rc_physical_payload_v1"


def _spec_to_dict(spec: RCCompilerSpec) -> dict[str, Any]:
    return {
        "flavour": spec.flavour,
        "zone_ids": list(spec.zone_ids),
        "mode": spec.mode.value,
        "adjacency": None
        if spec.adjacency is None
        else [{"zone_a": item.zone_a, "zone_b": item.zone_b} for item in spec.adjacency],
        "connection_rules": [
            {"state_a": item.state_a, "state_b": item.state_b}
            for item in spec.connection_rules
        ],
        "zone_port_availability": {
            str(k): list(v) for k, v in spec.zone_port_availability.items()
        },
        "port_groups": {
            str(k): (v.value if isinstance(v, HeatPortGroup) else str(v))
            for k, v in spec.port_groups.items()
        },
        "parameter_sharing": [
            {
                "name": item.name,
                "instance_ids": list(item.instance_ids),
                "family": item.family,
                "zone_ids": list(item.zone_ids),
            }
            for item in spec.parameter_sharing
        ],
        "dep2_allocations": [
            {
                "name": item.name,
                "signals": list(item.signals),
                "weights": dict(item.weights),
                "mode": item.mode.value,
                "participating_zone_ids": list(item.participating_zone_ids),
                "fixed_lambdas": dict(item.fixed_lambdas),
                "lower_bounds": dict(item.lower_bounds),
                "upper_bounds": dict(item.upper_bounds),
            }
            for item in spec.dep2_allocations
        ],
    }


def _spec_from_dict(payload: Mapping[str, Any]) -> RCCompilerSpec:
    adjacency_payload = payload.get("adjacency")
    adjacency = None
    if adjacency_payload is not None:
        adjacency = tuple(
            ZoneAdjacency(str(item["zone_a"]), str(item["zone_b"]))
            for item in adjacency_payload
        )
    return RCCompilerSpec(
        flavour=str(payload["flavour"]),
        zone_ids=tuple(str(v) for v in payload["zone_ids"]),
        mode=SpatialMode.normalize(str(payload["mode"])),
        adjacency=adjacency,
        connection_rules=tuple(
            ConnectionRule(str(item["state_a"]), str(item["state_b"]))
            for item in payload.get("connection_rules", ())
        ),
        zone_port_availability={
            str(k): tuple(str(x) for x in v)
            for k, v in dict(payload.get("zone_port_availability", {})).items()
        },
        port_groups={str(k): str(v) for k, v in dict(payload.get("port_groups", {})).items()},
        parameter_sharing=tuple(
            ParameterSharingRule(
                name=str(item["name"]),
                instance_ids=tuple(str(v) for v in item.get("instance_ids", ())),
                family=None if item.get("family") is None else str(item["family"]),
                zone_ids=tuple(str(v) for v in item.get("zone_ids", ())),
            )
            for item in payload.get("parameter_sharing", ())
        ),
        dep2_allocations=tuple(
            AllocationFamilySpec(
                name=str(item["name"]),
                signals=tuple(str(v) for v in item["signals"]),
                weights={str(k): float(v) for k, v in dict(item["weights"]).items()},
                mode=AllocationMode(str(item["mode"])),
                participating_zone_ids=tuple(
                    str(v) for v in item.get("participating_zone_ids", ())
                ),
                fixed_lambdas={
                    str(k): float(v) for k, v in dict(item.get("fixed_lambdas", {})).items()
                },
                lower_bounds={
                    str(k): float(v) for k, v in dict(item.get("lower_bounds", {})).items()
                },
                upper_bounds={
                    str(k): float(v) for k, v in dict(item.get("upper_bounds", {})).items()
                },
            )
            for item in payload.get("dep2_allocations", ())
        ),
    )


def _allocation_to_dict(value: AllocationResult) -> dict[str, Any]:
    return {
        "family_name": value.family_name,
        "lambda_by_zone": dict(value.lambda_by_zone),
        "p_by_zone": dict(value.p_by_zone),
        "residual": float(value.residual),
        "ab_error": float(value.ab_error),
        "logits": list(value.logits),
    }


def _allocation_from_dict(payload: Mapping[str, Any]) -> AllocationResult:
    return AllocationResult(
        family_name=str(payload["family_name"]),
        lambda_by_zone={str(k): float(v) for k, v in dict(payload["lambda_by_zone"]).items()},
        p_by_zone={str(k): float(v) for k, v in dict(payload["p_by_zone"]).items()},
        residual=float(payload.get("residual", 0.0)),
        ab_error=float(payload.get("ab_error", 0.0)),
        logits=tuple(float(v) for v in payload.get("logits", ())),
    )


@dataclass(frozen=True)
class RCPhysicalPayload:
    compiler_spec: RCCompilerSpec
    theta: Mapping[str, float]
    discretization: DiscretizationConfig = DiscretizationConfig()
    dep2_allocation_results: Mapping[str, AllocationResult] = None  # type: ignore[assignment]
    schema_version: str = RC_PHYSICAL_PAYLOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RC_PHYSICAL_PAYLOAD_SCHEMA_VERSION:
            raise PortableModelError(f"Unsupported RC payload schema {self.schema_version!r}")
        theta = {str(k): float(v) for k, v in self.theta.items()}
        if not theta:
            raise PortableModelError("RC physical payload requires final physical theta")
        object.__setattr__(self, "theta", theta)
        allocations = dict(self.dep2_allocation_results or {})
        object.__setattr__(self, "dep2_allocation_results", allocations)

        model = compile_rc_model(self.compiler_spec)
        # This resolves and validates all physical master values, sharing and bounds.
        model.parameter_registry.resolve_instance_values(theta)
        if model.spec.mode is SpatialMode.DEP2:
            expected = set(model.allocation_families)
            if set(allocations) != expected:
                raise PortableModelError(
                    "DEP2 physical payload requires one final allocation result per family"
                )
        elif allocations:
            raise PortableModelError("DEP2 allocation results are invalid for IND/DEP1 payloads")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "compiler_spec": _spec_to_dict(self.compiler_spec),
            "theta": dict(self.theta),
            "discretization": {
                "solver": self.discretization.solver,
                "substeps": self.discretization.substeps,
                "diagnostics_per_step": self.discretization.diagnostics_per_step,
                "stability_safety_factor": self.discretization.stability_safety_factor,
            },
            "dep2_allocation_results": {
                str(k): _allocation_to_dict(v)
                for k, v in self.dep2_allocation_results.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RCPhysicalPayload":
        d = dict(payload.get("discretization", {}))
        return cls(
            schema_version=str(payload.get("schema_version", RC_PHYSICAL_PAYLOAD_SCHEMA_VERSION)),
            compiler_spec=_spec_from_dict(payload["compiler_spec"]),
            theta={str(k): float(v) for k, v in dict(payload["theta"]).items()},
            discretization=DiscretizationConfig(
                solver=str(d.get("solver", "rk4")),
                substeps=int(d.get("substeps", 1)),
                diagnostics_per_step=bool(d.get("diagnostics_per_step", False)),
                stability_safety_factor=float(d.get("stability_safety_factor", 0.9)),
            ),
            dep2_allocation_results={
                str(k): _allocation_from_dict(v)
                for k, v in dict(payload.get("dep2_allocation_results", {})).items()
            },
        )

    def compiled_model(self):
        return compile_rc_model(self.compiler_spec)


def default_final_allocation_results(spec: RCCompilerSpec) -> dict[str, AllocationResult]:
    """Convenience for non-estimated DEP2 families; estimated results must be supplied."""
    model = compile_rc_model(spec)
    out: dict[str, AllocationResult] = {}
    for name, family in model.allocation_families.items():
        if family.mode is AllocationMode.NEUTRAL_FIXED:
            out[name] = neutral_allocation_result(family, spec.zone_ids)
        elif family.mode is AllocationMode.FIXED:
            out[name] = fixed_allocation_result(family, spec.zone_ids)
        else:
            raise PortableModelError(
                f"Estimated DEP2 family {name!r} requires its fitted final allocation result"
            )
    return out


def write_rc_physical_payload(path: str | "Path", payload: RCPhysicalPayload):
    """Write a portable physical-RC payload JSON file."""
    import json
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload.to_dict(), indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return target


def load_rc_physical_payload(path: str | "Path") -> RCPhysicalPayload:
    import json
    from pathlib import Path

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortableModelError(f"Unable to load physical RC payload: {target}") from exc
    return RCPhysicalPayload.from_dict(payload)
