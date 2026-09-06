# -*- coding: utf-8 -*-
"""Phase E.0 canonical data and scientific contracts.

E0-2 is an adaptation layer over the authoritative Phase D final-product
manifest.  It intentionally does not recreate Phase D split logic, infer
scientific identity from folder names, fabricate latent states, or silently
replace unavailable signals with zero.

Method-family mathematics (ML/SciML/Optimization/Bayesian) sits downstream of
these contracts and must be specified separately before family-specific code is
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from .constants import ModelingSilo, PhaseDMode


PHASE_E0_CONTRACT_SCHEMA_VERSION = "phase_e0_e02_contract_v1"


class PhaseEContractError(ValueError):
    """Raised when a Phase D product cannot be consumed safely by Phase E."""


class PhaseESignalRole(str, Enum):
    """Role of one exact Phase D column at the Phase D -> Phase E boundary."""

    OBSERVED_STATE = "observed_state"
    CONTROL_INPUT = "control_input"
    DISTURBANCE = "disturbance"
    TARGET = "target"
    AUXILIARY_OUTPUT = "auxiliary_output"
    METADATA = "metadata"


class PhysicalDomain(str, Enum):
    """Physical domain of a canonical thermal-model signal."""

    TIME = "time"
    TEMPERATURE = "temperature"
    THERMAL_POWER = "thermal_power"
    ELECTRICAL_POWER = "electrical_power"
    METADATA = "metadata"


@dataclass(frozen=True)
class ScientificSignalSpec:
    """Cross-family scientific meaning of one canonical signal.

    These definitions are intentionally method independent.  They establish the
    Phase E semantic boundary; later method-specific TeX contracts decide how a
    signal is used by a particular model.
    """

    canonical_name: str
    role: PhaseESignalRole
    units: str | None
    domain: PhysicalDomain
    zone_scoped: bool
    sign_convention: str
    thermal_balance_input: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "role": self.role.value,
            "units": self.units,
            "domain": self.domain.value,
            "zone_scoped": self.zone_scoped,
            "sign_convention": self.sign_convention,
            "thermal_balance_input": self.thermal_balance_input,
        }


def build_scientific_signal_registry() -> dict[str, ScientificSignalSpec]:
    """Return canonical E0 signal semantics shared by all method families."""

    registry = {
        "timestamp": ScientificSignalSpec(
            "timestamp",
            PhaseESignalRole.METADATA,
            None,
            PhysicalDomain.TIME,
            False,
            "not_applicable",
            False,
        ),
        "zone_temperature": ScientificSignalSpec(
            "zone_temperature",
            PhaseESignalRole.OBSERVED_STATE,
            "degC",
            PhysicalDomain.TEMPERATURE,
            True,
            "absolute_temperature_state",
            False,
        ),
        "outdoor_temperature": ScientificSignalSpec(
            "outdoor_temperature",
            PhaseESignalRole.DISTURBANCE,
            "degC",
            PhysicalDomain.TEMPERATURE,
            False,
            "absolute_boundary_temperature",
            False,
        ),
        "qac": ScientificSignalSpec(
            "qac",
            PhaseESignalRole.CONTROL_INPUT,
            "W",
            PhysicalDomain.THERMAL_POWER,
            True,
            "positive_heating_negative_cooling",
            True,
        ),
        "phvac": ScientificSignalSpec(
            "phvac",
            PhaseESignalRole.AUXILIARY_OUTPUT,
            "W",
            PhysicalDomain.ELECTRICAL_POWER,
            True,
            "electrical_hvac_power_not_thermal_heat",
            False,
        ),
        "qsol1": ScientificSignalSpec(
            "qsol1",
            PhaseESignalRole.DISTURBANCE,
            "W",
            PhysicalDomain.THERMAL_POWER,
            True,
            "positive_adds_thermal_energy",
            True,
        ),
        "qsol2": ScientificSignalSpec(
            "qsol2",
            PhaseESignalRole.DISTURBANCE,
            "W",
            PhysicalDomain.THERMAL_POWER,
            True,
            "positive_adds_thermal_energy",
            True,
        ),
        "zic": ScientificSignalSpec(
            "zic",
            PhaseESignalRole.DISTURBANCE,
            "W",
            PhysicalDomain.THERMAL_POWER,
            True,
            "positive_adds_thermal_energy",
            True,
        ),
        "zir": ScientificSignalSpec(
            "zir",
            PhaseESignalRole.DISTURBANCE,
            "W",
            PhysicalDomain.THERMAL_POWER,
            True,
            "positive_adds_thermal_energy",
            True,
        ),
    }

    # Component-level internal gains remain legitimate disturbances when the
    # Phase D product uses the component representation instead of zic/zir.
    for prefix in ("qzic", "qzir"):
        for suffix in ("p", "l", "ee", "ge", "oe", "hwe", "se"):
            name = f"{prefix}_{suffix}"
            registry[name] = ScientificSignalSpec(
                name,
                PhaseESignalRole.DISTURBANCE,
                "W",
                PhysicalDomain.THERMAL_POWER,
                True,
                "positive_adds_thermal_energy",
                True,
            )

    registry["qzivr_l"] = ScientificSignalSpec(
        "qzivr_l",
        PhaseESignalRole.DISTURBANCE,
        "W",
        PhysicalDomain.THERMAL_POWER,
        True,
        "positive_adds_thermal_energy",
        True,
    )
    return registry


SCIENTIFIC_SIGNAL_REGISTRY = build_scientific_signal_registry()


def get_scientific_signal_spec(name: str) -> ScientificSignalSpec:
    """Return a locked cross-family semantic definition."""

    try:
        return SCIENTIFIC_SIGNAL_REGISTRY[name]
    except KeyError as exc:
        raise PhaseEContractError(
            f"No canonical Phase E scientific signal definition for {name!r}"
        ) from exc


@dataclass(frozen=True)
class PhaseESignalBinding:
    """Exact manifest-derived binding to one materialized Phase D column."""

    column_name: str
    base_signal: str
    role: PhaseESignalRole
    aggregate_zone_id: str | None
    temporal_role: str
    offset_steps: int | None
    units: str | None

    def __post_init__(self) -> None:
        if not self.column_name:
            raise PhaseEContractError("column_name cannot be empty")
        if not self.base_signal:
            raise PhaseEContractError("base_signal cannot be empty")

        if self.base_signal == "phvac" and self.temporal_role == "model_input":
            raise PhaseEContractError(
                "PHVAC is electrical HVAC power and is forbidden as a thermal "
                "model input. Use QAC as the thermal HVAC input and compose "
                "PHVAC downstream through the Phase C runtime."
            )

        spec = SCIENTIFIC_SIGNAL_REGISTRY.get(self.base_signal)
        if spec is None:
            return

        if (
            self.role is not PhaseESignalRole.TARGET
            and spec.role is not self.role
        ):
            raise PhaseEContractError(
                f"Signal {self.base_signal!r} has canonical role "
                f"{spec.role.value!r}, not {self.role.value!r}"
            )

        if spec.zone_scoped and self.aggregate_zone_id is None:
            raise PhaseEContractError(
                f"Signal {self.base_signal!r} requires an aggregate_zone_id"
            )
        if not spec.zone_scoped and self.aggregate_zone_id is not None:
            raise PhaseEContractError(
                f"Signal {self.base_signal!r} is common/unqualified and cannot "
                "carry aggregate_zone_id"
            )

        if (
            spec.units is not None
            and self.units is not None
            and spec.units != self.units
        ):
            raise PhaseEContractError(
                f"Signal {self.base_signal!r} expected units {spec.units!r}; "
                f"manifest reports {self.units!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "base_signal": self.base_signal,
            "role": self.role.value,
            "aggregate_zone_id": self.aggregate_zone_id,
            "temporal_role": self.temporal_role,
            "offset_steps": self.offset_steps,
            "units": self.units,
        }


@dataclass(frozen=True)
class SpatialDependencyContract:
    """Phase-D-owned information/spatial organization consumed by Phase E.

    E0-2 deliberately does not create physical RC coupling edges.  Those belong
    to E0-3/E0-4 topology contracts.  Here we preserve only which zones provide
    states/controls and, for Dep2, which compatible aggregate zone provides
    disturbances.
    """

    mode: PhaseDMode
    modeled_zone_ids: tuple[str, ...]
    independent_zone_id: str | None = None
    dependent_2_source_zone_id: str | None = None
    disturbance_source_zone_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.modeled_zone_ids:
            raise PhaseEContractError("modeled_zone_ids cannot be empty")
        if len(self.modeled_zone_ids) != len(set(self.modeled_zone_ids)):
            raise PhaseEContractError("modeled_zone_ids must be unique")
        if len(self.disturbance_source_zone_ids) != len(
            set(self.disturbance_source_zone_ids)
        ):
            raise PhaseEContractError("disturbance_source_zone_ids must be unique")

        if self.mode is PhaseDMode.INDEPENDENT:
            if len(self.modeled_zone_ids) != 1:
                raise PhaseEContractError(
                    "Independent Phase E product must model exactly one Phase D "
                    "aggregate zone"
                )
            if self.independent_zone_id != self.modeled_zone_ids[0]:
                raise PhaseEContractError(
                    "independent_zone_id must equal the modeled independent zone"
                )
            if self.dependent_2_source_zone_id is not None:
                raise PhaseEContractError(
                    "independent products cannot have a Dep2 source zone"
                )
        elif self.mode is PhaseDMode.DEPENDENT2:
            if self.independent_zone_id is not None:
                raise PhaseEContractError(
                    "dependent products cannot define independent_zone_id"
                )
            if self.dependent_2_source_zone_id is None:
                raise PhaseEContractError(
                    "Dependent2 requires the Phase-D-selected compatible "
                    "all-to-one disturbance source zone"
                )
        else:
            if self.independent_zone_id is not None:
                raise PhaseEContractError(
                    "dependent products cannot define independent_zone_id"
                )
            if self.dependent_2_source_zone_id is not None:
                raise PhaseEContractError(
                    "Only dependent2 may define a dependent_2_source_zone_id"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "modeled_zone_ids": list(self.modeled_zone_ids),
            "independent_zone_id": self.independent_zone_id,
            "dependent_2_source_zone_id": self.dependent_2_source_zone_id,
            "disturbance_source_zone_ids": list(self.disturbance_source_zone_ids),
            "physical_coupling_defined_here": False,
        }


@dataclass(frozen=True)
class TemporalOwnershipContract:
    """Phase D temporal/partition contract preserved at the Phase E boundary."""

    silo: ModelingSilo
    input_lag: int
    target_horizon: int
    policy_name: str
    policy_parameters: Mapping[str, Any] = field(default_factory=dict)
    primary_partitions: tuple[str, ...] = ()
    allowed_partition_values: tuple[str, ...] = ()
    outer_partition_owner: str = "phase_d"
    hyperparameter_tuning_source_partitions: tuple[str, ...] = ("train",)

    def __post_init__(self) -> None:
        if self.input_lag < 1:
            raise PhaseEContractError("input_lag must be >= 1")
        if self.target_horizon < 1:
            raise PhaseEContractError("target_horizon must be >= 1")
        if self.outer_partition_owner != "phase_d":
            raise PhaseEContractError(
                "Phase E must consume, not redefine, Phase D outer partitions"
            )
        if "test" in self.hyperparameter_tuning_source_partitions:
            raise PhaseEContractError(
                "Phase D test rows may never be used for hyperparameter tuning"
            )

    def assert_hyperparameter_tuning_partitions(
        self,
        partitions: Iterable[str],
    ) -> None:
        """Reject leakage from Phase D validation/test/excluded partitions.

        The locked Phase E tuning workflow samples the representative HPO subset
        from Phase D TRAIN and may create inner fit/validation partitions inside
        that training-only subset.
        """

        seen = {str(value) for value in partitions}
        forbidden = seen - set(self.hyperparameter_tuning_source_partitions)
        if forbidden:
            raise PhaseEContractError(
                "Hyperparameter tuning source rows must come only from Phase D "
                f"TRAIN. Forbidden partitions found: {sorted(forbidden)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "silo": self.silo.value,
            "input_lag": self.input_lag,
            "target_horizon": self.target_horizon,
            "policy_name": self.policy_name,
            "policy_parameters": dict(self.policy_parameters),
            "primary_partitions": list(self.primary_partitions),
            "allowed_partition_values": list(self.allowed_partition_values),
            "outer_partition_owner": self.outer_partition_owner,
            "hyperparameter_tuning_source_partitions": list(
                self.hyperparameter_tuning_source_partitions
            ),
            "phase_e_may_redefine_outer_partitions": False,
        }


@dataclass(frozen=True)
class AggregationLineageBinding:
    """Aggregation/Phase-C lineage carried forward without path inference."""

    campaign_id: str | None = None
    case_id: str | None = None
    aggregation_matrix_run_id: str | None = None
    aggregation_run_id: str | None = None
    aggregation_id: str | None = None
    weight_mode: str | None = None
    phase_c_campaign_run_id: str | None = None
    dependent_2_match_status: str | None = None
    dependent_2_source_aggregation_run_id: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "case_id": self.case_id,
            "aggregation_matrix_run_id": self.aggregation_matrix_run_id,
            "aggregation_run_id": self.aggregation_run_id,
            "aggregation_id": self.aggregation_id,
            "weight_mode": self.weight_mode,
            "phase_c_campaign_run_id": self.phase_c_campaign_run_id,
            "dependent_2_match_status": self.dependent_2_match_status,
            "dependent_2_source_aggregation_run_id": (
                self.dependent_2_source_aggregation_run_id
            ),
            "extra": dict(self.extra),
            "identity_inferred_from_filesystem": False,
        }


@dataclass(frozen=True)
class PhaseEDataContract:
    """Canonical, method-neutral Phase E view of one final Phase D product."""

    contract_id: str
    source_manifest_sha256: str
    phase_d_schema_version: str
    phase_d_d7_schema_version: str | None
    spatial: SpatialDependencyContract
    temporal: TemporalOwnershipContract
    heat_representation: Mapping[str, Any]
    input_bindings: tuple[PhaseESignalBinding, ...]
    target_bindings: tuple[PhaseESignalBinding, ...]
    metadata_columns: tuple[str, ...]
    provenance: AggregationLineageBinding
    row_count: int | None = None
    included_row_count: int | None = None
    partition_counts: Mapping[str, int] = field(default_factory=dict)
    schema_version: str = PHASE_E0_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise PhaseEContractError("contract_id cannot be empty")
        names = [
            binding.column_name
            for binding in (*self.input_bindings, *self.target_bindings)
        ]
        if len(names) != len(set(names)):
            raise PhaseEContractError(
                "input/target bindings must have unique materialized column names"
            )
        if any(
            binding.base_signal == "phvac"
            for binding in self.input_bindings
        ):
            raise PhaseEContractError(
                "PHVAC cannot appear in Phase E thermal-model input bindings"
            )

        state_zones = {
            binding.aggregate_zone_id
            for binding in self.input_bindings
            if binding.role is PhaseESignalRole.OBSERVED_STATE
            and binding.offset_steps == 0
        }
        expected = set(self.spatial.modeled_zone_ids)
        if state_zones != expected:
            raise PhaseEContractError(
                "Each modeled zone must have exactly its Phase D lag-0 observed "
                f"zone-temperature state. expected={sorted(expected)} "
                f"found={sorted(value for value in state_zones if value is not None)}"
            )

    @property
    def modeled_zone_ids(self) -> tuple[str, ...]:
        return self.spatial.modeled_zone_ids

    @property
    def latent_state_bindings(self) -> tuple[PhaseESignalBinding, ...]:
        """Phase D supplies no fabricated targets for latent RC states.

        E0-3/E0-4 topology and method contracts define latent states and their
        initialization/estimation strategy.
        """

        return ()

    def find_input(
        self,
        base_signal: str,
        *,
        aggregate_zone_id: str | None,
        lag: int = 0,
    ) -> PhaseESignalBinding | None:
        offset = -int(lag)
        matches = [
            item
            for item in self.input_bindings
            if item.base_signal == base_signal
            and item.aggregate_zone_id == aggregate_zone_id
            and item.offset_steps == offset
        ]
        if len(matches) > 1:
            raise PhaseEContractError(
                "Ambiguous exact input binding for "
                f"signal={base_signal!r}, zone={aggregate_zone_id!r}, lag={lag}"
            )
        return matches[0] if matches else None

    def require_input(
        self,
        base_signal: str,
        *,
        aggregate_zone_id: str | None,
        lag: int = 0,
    ) -> PhaseESignalBinding:
        """Resolve an exact Phase D binding; never invent/zero a missing signal."""

        binding = self.find_input(
            base_signal,
            aggregate_zone_id=aggregate_zone_id,
            lag=lag,
        )
        if binding is None:
            raise PhaseEContractError(
                "Required Phase D signal is not bound for this realization: "
                f"signal={base_signal!r}, zone={aggregate_zone_id!r}, lag={lag}. "
                "Phase E must not infer a column by substring or silently replace "
                "an unavailable applicable signal with zero."
            )
        return binding

    def find_target(
        self,
        base_signal: str,
        *,
        aggregate_zone_id: str,
        horizon: int = 1,
    ) -> PhaseESignalBinding | None:
        matches = [
            item
            for item in self.target_bindings
            if item.base_signal == base_signal
            and item.aggregate_zone_id == aggregate_zone_id
            and item.offset_steps == int(horizon)
        ]
        if len(matches) > 1:
            raise PhaseEContractError(
                "Ambiguous exact target binding for "
                f"signal={base_signal!r}, zone={aggregate_zone_id!r}, "
                f"horizon={horizon}"
            )
        return matches[0] if matches else None

    def require_target(
        self,
        base_signal: str,
        *,
        aggregate_zone_id: str,
        horizon: int = 1,
    ) -> PhaseESignalBinding:
        binding = self.find_target(
            base_signal,
            aggregate_zone_id=aggregate_zone_id,
            horizon=horizon,
        )
        if binding is None:
            raise PhaseEContractError(
                "Required Phase D target is not bound for this realization: "
                f"signal={base_signal!r}, zone={aggregate_zone_id!r}, "
                f"horizon={horizon}"
            )
        return binding

    def available_lag0_signals(
        self,
        aggregate_zone_id: str | None,
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item.base_signal
                for item in self.input_bindings
                if item.aggregate_zone_id == aggregate_zone_id
                and item.offset_steps == 0
            )
        )

    def required_materialized_columns(self) -> tuple[str, ...]:
        return (
            *self.metadata_columns,
            *(item.column_name for item in self.input_bindings),
            *(item.column_name for item in self.target_bindings),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "phase_d_schema_version": self.phase_d_schema_version,
            "phase_d_d7_schema_version": self.phase_d_d7_schema_version,
            "spatial": self.spatial.to_dict(),
            "temporal": self.temporal.to_dict(),
            "heat_representation": dict(self.heat_representation),
            "input_bindings": [item.to_dict() for item in self.input_bindings],
            "target_bindings": [item.to_dict() for item in self.target_bindings],
            "metadata_columns": list(self.metadata_columns),
            "provenance": self.provenance.to_dict(),
            "row_count": self.row_count,
            "included_row_count": self.included_row_count,
            "partition_counts": dict(self.partition_counts),
            "latent_state_source": "method_or_topology_contract_not_phase_d",
        }
