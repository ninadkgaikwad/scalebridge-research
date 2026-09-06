# -*- coding: utf-8 -*-
"""Phase D D6 final silo, temporal, partition, and column-schema contracts.

D6 defines contracts only. It does not construct or persist annual datasets.
D7 consumes these contracts to build final Independent / Dependent 1 /
Dependent 2 Parquets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import ModelingSilo, PhaseDMode


D6_SCHEMA_VERSION = "phase_d_d6_silo_contract_v1"

COMMON_POLICY_COLUMNS = (
    "timestamp",
    "included",
    "partition",
    "window_id",
    "season",
)

STATE_SIGNAL = "zone_temperature"
CONTROL_SIGNAL = "qac"
COMMON_DISTURBANCE_SIGNAL = "outdoor_temperature"

QZIC_COMPONENTS = (
    "qzic_p",
    "qzic_l",
    "qzic_ee",
    "qzic_ge",
    "qzic_oe",
    "qzic_hwe",
    "qzic_se",
)

QZIR_COMPONENTS = (
    "qzir_p",
    "qzir_l",
    "qzir_ee",
    "qzir_ge",
    "qzir_oe",
    "qzir_hwe",
    "qzir_se",
)

VISIBLE_LIGHTING_COMPONENT = "qzivr_l"
SOLAR_DISTURBANCES = ("qsol1", "qsol2")


SILO_FOLDER_TOKENS = {
    ModelingSilo.ML_SCIML: "ml",
    ModelingSilo.OPT_BAYES: "ob",
}

MODE_FOLDER_TOKENS = {
    PhaseDMode.INDEPENDENT: "ind",
    PhaseDMode.DEPENDENT1: "dep1",
    PhaseDMode.DEPENDENT2: "dep2",
}

POLICY_FOLDER_TOKENS = {
    # ML / SciML
    "monthly_distributed_holdout": "mdh",
    "chronological_holdout": "ch",
    "seasonal_holdout": "sh",
    # Optimization / Bayesian
    "seasonal_distributed": "sd",
    "seasonal_block_holdout": "sbh",
    "contiguous_identification": "ci",
    "custom_datetime_ranges": "cdr",
}


class D6ContractError(ValueError):
    """Raised when a requested D6 silo contract is internally inconsistent."""


class PhysicalRole(str, Enum):
    """Thermal-model physical role used by final Phase D products."""

    STATE = "state"
    CONTROL_INPUT = "control_input"
    DISTURBANCE = "disturbance"
    METADATA = "metadata"
    TARGET = "target"


class HeatInputRepresentation(str, Enum):
    """Representation of Phase C internal heat inputs in final Phase D data."""

    GROUPED = "grouped_qzic_qzir"
    COMPONENTS = "component_heat_inputs"


class Partition(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EXCLUDED = "excluded"


class Season(str, Enum):
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"


@dataclass(frozen=True)
class PolicyContract:
    """Contract for one temporal partition / selection policy."""

    name: str
    silo: ModelingSilo
    partitions: tuple[Partition, ...]
    uses_full_year: bool
    annual_data_retained: bool = True
    allows_excluded: bool = True
    requires_window_id_for_included: bool = False
    requires_season_for_included: bool = False
    description: str = ""
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.name:
            raise D6ContractError("policy name cannot be empty")
        if len(self.partitions) != len(set(self.partitions)):
            raise D6ContractError("policy partitions must be unique")
        if Partition.EXCLUDED in self.partitions:
            raise D6ContractError(
                "EXCLUDED is an allowed non-partition state, not a primary partition"
            )

    @property
    def allowed_partition_values(self) -> tuple[Partition, ...]:
        if self.allows_excluded:
            return (*self.partitions, Partition.EXCLUDED)
        return self.partitions

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "silo": self.silo.value,
            "partitions": [item.value for item in self.partitions],
            "allowed_partition_values": [
                item.value for item in self.allowed_partition_values
            ],
            "uses_full_year": self.uses_full_year,
            "annual_data_retained": self.annual_data_retained,
            "allows_excluded": self.allows_excluded,
            "requires_window_id_for_included": self.requires_window_id_for_included,
            "requires_season_for_included": self.requires_season_for_included,
            "description": self.description,
        }


ML_SCIML_POLICY_CATALOG: dict[str, PolicyContract] = {
    "monthly_distributed_holdout": PolicyContract(
        name="monthly_distributed_holdout",
        silo=ModelingSilo.ML_SCIML,
        partitions=(Partition.TRAIN, Partition.VALIDATION, Partition.TEST),
        uses_full_year=True,
        allows_excluded=True,
        description=(
            "Partition essentially all temporally valid annual samples into "
            "train/validation/test with holdouts distributed across months. "
            "Boundary samples may be excluded to prevent temporal leakage."
        ),
    ),
    "chronological_holdout": PolicyContract(
        name="chronological_holdout",
        silo=ModelingSilo.ML_SCIML,
        partitions=(Partition.TRAIN, Partition.VALIDATION, Partition.TEST),
        uses_full_year=True,
        allows_excluded=True,
        description=(
            "Partition annual samples into chronological train/validation/test "
            "blocks with leakage-safe temporal boundaries."
        ),
    ),
    "seasonal_holdout": PolicyContract(
        name="seasonal_holdout",
        silo=ModelingSilo.ML_SCIML,
        partitions=(Partition.TRAIN, Partition.VALIDATION, Partition.TEST),
        uses_full_year=True,
        allows_excluded=True,
        requires_season_for_included=True,
        description=(
            "Partition annual samples using deterministic seasonal holdout rules "
            "while retaining full-year ML/SciML coverage."
        ),
    ),
}


OPT_BAYES_POLICY_CATALOG: dict[str, PolicyContract] = {
    "seasonal_distributed": PolicyContract(
        name="seasonal_distributed",
        silo=ModelingSilo.OPT_BAYES,
        partitions=(Partition.TRAIN, Partition.TEST),
        uses_full_year=False,
        allows_excluded=True,
        requires_window_id_for_included=True,
        requires_season_for_included=True,
        description=(
            "Select deterministic train/test identification windows distributed "
            "across winter/spring/summer/fall. Non-selected annual samples are "
            "retained but marked excluded."
        ),
    ),
    "seasonal_block_holdout": PolicyContract(
        name="seasonal_block_holdout",
        silo=ModelingSilo.OPT_BAYES,
        partitions=(Partition.TRAIN, Partition.TEST),
        uses_full_year=False,
        allows_excluded=True,
        requires_window_id_for_included=True,
        requires_season_for_included=True,
        description=(
            "Use explicit seasonal blocks for identification and held-out "
            "evaluation; retain the rest of the year as excluded."
        ),
    ),
    "contiguous_identification": PolicyContract(
        name="contiguous_identification",
        silo=ModelingSilo.OPT_BAYES,
        partitions=(Partition.TRAIN, Partition.TEST),
        uses_full_year=False,
        allows_excluded=True,
        requires_window_id_for_included=True,
        description=(
            "Use one deterministic contiguous train/test window for "
            "system identification and evaluation."
        ),
    ),
    "custom_datetime_ranges": PolicyContract(
        name="custom_datetime_ranges",
        silo=ModelingSilo.OPT_BAYES,
        partitions=(Partition.TRAIN, Partition.TEST),
        uses_full_year=False,
        allows_excluded=True,
        requires_window_id_for_included=True,
        description=(
            "Use explicitly configured datetime ranges for train/test selection; "
            "all other annual samples remain excluded."
        ),
    ),
}


def get_policy_contract(
    silo: ModelingSilo | str,
    policy_name: str,
) -> PolicyContract:
    silo_value = ModelingSilo(silo)
    catalog = (
        ML_SCIML_POLICY_CATALOG
        if silo_value is ModelingSilo.ML_SCIML
        else OPT_BAYES_POLICY_CATALOG
    )
    if policy_name not in catalog:
        raise D6ContractError(
            f"Unsupported {silo_value.value} policy: {policy_name!r}"
        )
    return catalog[policy_name]


@dataclass(frozen=True)
class TemporalConfig:
    """Temporal realization of a final silo dataset."""

    silo: ModelingSilo
    input_lag: int
    target_horizon: int
    policy_name: str
    policy_parameters: Mapping[str, Any] = field(default_factory=dict)
    policy_realization_id: str | None = None

    def __post_init__(self) -> None:
        if self.input_lag < 1:
            raise D6ContractError("input_lag must be >= 1")
        if self.target_horizon < 1:
            raise D6ContractError("target_horizon must be >= 1")
        if self.silo is ModelingSilo.OPT_BAYES:
            if self.input_lag != 1 or self.target_horizon != 1:
                raise D6ContractError(
                    "opt_bayes requires input_lag=1 and target_horizon=1"
                )
        get_policy_contract(self.silo, self.policy_name)
        if self.policy_realization_id is not None:
            _validate_path_token(
                self.policy_realization_id,
                field_name="policy_realization_id",
            )

    @property
    def policy(self) -> PolicyContract:
        return get_policy_contract(self.silo, self.policy_name)

    @property
    def lag_horizon_folder(self) -> str:
        return f"l{self.input_lag}_h{self.target_horizon}"

    @property
    def policy_token(self) -> str:
        try:
            return POLICY_FOLDER_TOKENS[self.policy_name]
        except KeyError as exc:
            raise D6ContractError(
                f"No compact folder token defined for policy: {self.policy_name!r}"
            ) from exc

    @property
    def policy_folder(self) -> str:
        if self.policy_realization_id:
            realization = self.policy_realization_id
            if realization.startswith("r"):
                return f"{self.policy_token}_{realization}"
            return f"{self.policy_token}_r{realization}"
        return self.policy_token

    def to_dict(self) -> dict[str, Any]:
        return {
            "silo": self.silo.value,
            "input_lag": self.input_lag,
            "target_horizon": self.target_horizon,
            "lag_horizon_folder": self.lag_horizon_folder,
            "policy_name": self.policy_name,
            "policy_token": self.policy_token,
            "policy_realization_id": self.policy_realization_id,
            "policy_folder": self.policy_folder,
            "policy_parameters": dict(self.policy_parameters),
            "policy_contract": self.policy.to_dict(),
        }


@dataclass(frozen=True)
class HeatRepresentationConfig:
    """Internal-heat representation used by one final dataset realization."""

    representation: HeatInputRepresentation
    include_visible_lighting_in_qzir: bool = True

    @property
    def folder_name(self) -> str:
        if self.representation is HeatInputRepresentation.GROUPED:
            return (
                "grp_vrin"
                if self.include_visible_lighting_in_qzir
                else "grp_vrsep"
            )
        return "cmp"

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation.value,
            "include_visible_lighting_in_qzir": (
                self.include_visible_lighting_in_qzir
            ),
            "folder_name": self.folder_name,
        }


@dataclass(frozen=True)
class ZoneSignalAvailability:
    """D4-derived usable signal set for one aggregate zone.

    `available_disturbances` contains only disturbances that D7 is allowed to
    expose to the downstream thermal-model dataset after applying D4
    applicability/nullability rules.
    """

    aggregate_zone_id: str
    available_disturbances: tuple[str, ...]
    qac_available: bool = True

    def __post_init__(self) -> None:
        _validate_path_token(self.aggregate_zone_id, field_name="aggregate_zone_id")
        if len(self.available_disturbances) != len(set(self.available_disturbances)):
            raise D6ContractError("available_disturbances must be unique")
        forbidden = {STATE_SIGNAL, CONTROL_SIGNAL, COMMON_DISTURBANCE_SIGNAL}
        overlap = forbidden & set(self.available_disturbances)
        if overlap:
            raise D6ContractError(
                "available_disturbances cannot contain state/control/common "
                f"signals: {sorted(overlap)}"
            )


@dataclass(frozen=True)
class BaseColumn:
    """One physical column before temporal lag/target expansion."""

    name: str
    physical_role: PhysicalRole
    aggregate_zone_id: str | None
    base_signal: str
    units: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "physical_role": self.physical_role.value,
            "aggregate_zone_id": self.aggregate_zone_id,
            "base_signal": self.base_signal,
            "units": self.units,
        }


@dataclass(frozen=True)
class FinalColumn:
    """One materialized D6 final-data column."""

    name: str
    physical_role: PhysicalRole
    temporal_role: str
    aggregate_zone_id: str | None
    base_signal: str
    offset_steps: int | None
    units: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "physical_role": self.physical_role.value,
            "temporal_role": self.temporal_role,
            "aggregate_zone_id": self.aggregate_zone_id,
            "base_signal": self.base_signal,
            "offset_steps": self.offset_steps,
            "units": self.units,
        }


@dataclass(frozen=True)
class SiloProductContract:
    """Complete D6 contract for one final Parquet realization."""

    silo: ModelingSilo
    mode: PhaseDMode
    temporal: TemporalConfig
    heat: HeatRepresentationConfig
    current_zones: tuple[ZoneSignalAvailability, ...]
    dependent_2_source_zone: ZoneSignalAvailability | None = None

    def __post_init__(self) -> None:
        if self.temporal.silo is not self.silo:
            raise D6ContractError("temporal.silo must match contract silo")
        zone_ids = [zone.aggregate_zone_id for zone in self.current_zones]
        if not zone_ids:
            raise D6ContractError("current_zones cannot be empty")
        if len(zone_ids) != len(set(zone_ids)):
            raise D6ContractError("current_zones must have unique zone IDs")
        if self.mode is PhaseDMode.DEPENDENT2:
            if self.dependent_2_source_zone is None:
                raise D6ContractError(
                    "dependent2 requires a matched all-to-one disturbance source"
                )
        elif self.dependent_2_source_zone is not None:
            raise D6ContractError(
                "dependent_2_source_zone is valid only for dependent2"
            )

    @property
    def silo_folder_name(self) -> str:
        return SILO_FOLDER_TOKENS[self.silo]

    @property
    def product_folder_name(self) -> str:
        return MODE_FOLDER_TOKENS[self.mode]

    def base_columns(
        self,
        *,
        independent_zone_id: str | None = None,
    ) -> tuple[BaseColumn, ...]:
        """Return physical state/control/disturbance columns in stable order."""

        if self.mode is PhaseDMode.INDEPENDENT:
            if independent_zone_id is None:
                raise D6ContractError(
                    "independent base_columns requires independent_zone_id"
                )
            zone = self._zone(independent_zone_id)
            zones_for_state_control = (zone,)
            disturbance_zones = (zone,)
        elif self.mode is PhaseDMode.DEPENDENT1:
            zones_for_state_control = self.current_zones
            disturbance_zones = self.current_zones
        else:
            zones_for_state_control = self.current_zones
            disturbance_zones = (self.dependent_2_source_zone,)

        columns: list[BaseColumn] = [
            BaseColumn(
                name=COMMON_DISTURBANCE_SIGNAL,
                physical_role=PhysicalRole.DISTURBANCE,
                aggregate_zone_id=None,
                base_signal=COMMON_DISTURBANCE_SIGNAL,
                units="degC",
            )
        ]

        for zone in zones_for_state_control:
            columns.append(
                BaseColumn(
                    name=zone_signal_name(zone.aggregate_zone_id, STATE_SIGNAL),
                    physical_role=PhysicalRole.STATE,
                    aggregate_zone_id=zone.aggregate_zone_id,
                    base_signal=STATE_SIGNAL,
                    units="degC",
                )
            )
            if zone.qac_available:
                columns.append(
                    BaseColumn(
                        name=zone_signal_name(zone.aggregate_zone_id, CONTROL_SIGNAL),
                        physical_role=PhysicalRole.CONTROL_INPUT,
                        aggregate_zone_id=zone.aggregate_zone_id,
                        base_signal=CONTROL_SIGNAL,
                        units="W",
                    )
                )

        for zone in disturbance_zones:
            for signal in _selected_disturbances(zone, self.heat):
                columns.append(
                    BaseColumn(
                        name=zone_signal_name(zone.aggregate_zone_id, signal),
                        physical_role=PhysicalRole.DISTURBANCE,
                        aggregate_zone_id=zone.aggregate_zone_id,
                        base_signal=signal,
                        units="W",
                    )
                )

        names = [column.name for column in columns]
        if len(names) != len(set(names)):
            raise D6ContractError(f"duplicate base columns: {names}")
        return tuple(columns)

    def final_columns(
        self,
        *,
        independent_zone_id: str | None = None,
    ) -> tuple[FinalColumn, ...]:
        """Return exact final Parquet schema after lag/target expansion."""

        base = self.base_columns(independent_zone_id=independent_zone_id)
        columns: list[FinalColumn] = [
            FinalColumn(
                name="timestamp",
                physical_role=PhysicalRole.METADATA,
                temporal_role="anchor_timestamp",
                aggregate_zone_id=None,
                base_signal="timestamp",
                offset_steps=None,
                units=None,
            ),
            FinalColumn(
                name="included",
                physical_role=PhysicalRole.METADATA,
                temporal_role="selection",
                aggregate_zone_id=None,
                base_signal="included",
                offset_steps=None,
                units=None,
            ),
            FinalColumn(
                name="partition",
                physical_role=PhysicalRole.METADATA,
                temporal_role="partition",
                aggregate_zone_id=None,
                base_signal="partition",
                offset_steps=None,
                units=None,
            ),
            FinalColumn(
                name="window_id",
                physical_role=PhysicalRole.METADATA,
                temporal_role="selection_window",
                aggregate_zone_id=None,
                base_signal="window_id",
                offset_steps=None,
                units=None,
            ),
            FinalColumn(
                name="season",
                physical_role=PhysicalRole.METADATA,
                temporal_role="season",
                aggregate_zone_id=None,
                base_signal="season",
                offset_steps=None,
                units=None,
            ),
        ]

        for lag in range(self.temporal.input_lag):
            for item in base:
                columns.append(
                    FinalColumn(
                        name=f"{item.name}__lag_{lag}",
                        physical_role=item.physical_role,
                        temporal_role="model_input",
                        aggregate_zone_id=item.aggregate_zone_id,
                        base_signal=item.base_signal,
                        offset_steps=-lag,
                        units=item.units,
                    )
                )

        state_columns = [
            item for item in base if item.physical_role is PhysicalRole.STATE
        ]
        for horizon in range(1, self.temporal.target_horizon + 1):
            for item in state_columns:
                columns.append(
                    FinalColumn(
                        name=f"{item.name}__target_{horizon}",
                        physical_role=PhysicalRole.TARGET,
                        temporal_role="prediction_target",
                        aggregate_zone_id=item.aggregate_zone_id,
                        base_signal=item.base_signal,
                        offset_steps=horizon,
                        units=item.units,
                    )
                )

        names = [column.name for column in columns]
        if len(names) != len(set(names)):
            raise D6ContractError("final columns must be unique")
        return tuple(columns)

    def relative_output_dir(
        self,
        *,
        independent_zone_id: str | None = None,
    ) -> Path:
        """Relative directory below silos/<silo> for one final realization."""

        parts: list[str] = [self.product_folder_name]

        if self.mode is PhaseDMode.INDEPENDENT:
            if independent_zone_id is None:
                raise D6ContractError(
                    "independent output path requires independent_zone_id"
                )
            self._zone(independent_zone_id)
            parts.append(independent_zone_id)
        elif independent_zone_id is not None:
            raise D6ContractError(
                "independent_zone_id is valid only for independent mode"
            )

        parts.extend(
            (
                self.heat.folder_name,
                self.temporal.lag_horizon_folder,
                self.temporal.policy_folder,
            )
        )
        return Path(*parts)

    def expected_files(
        self,
        *,
        independent_zone_id: str | None = None,
    ) -> tuple[Path, Path]:
        output_dir = self.relative_output_dir(
            independent_zone_id=independent_zone_id
        )
        return output_dir / "data.parquet", output_dir / "manifest.json"

    def to_manifest_contract(
        self,
        *,
        independent_zone_id: str | None = None,
    ) -> dict[str, Any]:
        base = self.base_columns(independent_zone_id=independent_zone_id)
        final = self.final_columns(independent_zone_id=independent_zone_id)
        data_path, manifest_path = self.expected_files(
            independent_zone_id=independent_zone_id
        )
        return {
            "schema_version": D6_SCHEMA_VERSION,
            "silo": self.silo.value,
            "silo_folder_name": self.silo_folder_name,
            "mode": self.mode.value,
            "product_folder_name": self.product_folder_name,
            "independent_zone_id": independent_zone_id,
            "heat_representation": self.heat.to_dict(),
            "temporal_config": self.temporal.to_dict(),
            "physical_semantics": {
                "state": STATE_SIGNAL,
                "control_input": CONTROL_SIGNAL,
                "common_disturbance": COMMON_DISTURBANCE_SIGNAL,
                "other_heat_and_solar_inputs": "disturbance",
            },
            "policy_columns": list(COMMON_POLICY_COLUMNS),
            "base_columns": [item.to_dict() for item in base],
            "final_columns": [item.to_dict() for item in final],
            "current_zone_ids": [
                zone.aggregate_zone_id for zone in self.current_zones
            ],
            "dependent_2_source_zone_id": (
                self.dependent_2_source_zone.aggregate_zone_id
                if self.dependent_2_source_zone
                else None
            ),
            "storage_contract": {
                "one_parquet_per_realization": True,
                "split_files_forbidden": True,
                "data_path": str(data_path).replace("\\", "/"),
                "manifest_path": str(manifest_path).replace("\\", "/"),
            },
        }

    def _zone(self, zone_id: str) -> ZoneSignalAvailability:
        matches = [
            zone for zone in self.current_zones
            if zone.aggregate_zone_id == zone_id
        ]
        if len(matches) != 1:
            raise D6ContractError(
                f"Unknown or ambiguous current aggregate zone: {zone_id!r}"
            )
        return matches[0]


def zone_signal_name(aggregate_zone_id: str, signal: str) -> str:
    _validate_path_token(aggregate_zone_id, field_name="aggregate_zone_id")
    if "__" in signal:
        raise D6ContractError("base signal names cannot contain '__'")
    return f"{aggregate_zone_id}__{signal}"


def _selected_disturbances(
    zone: ZoneSignalAvailability,
    heat: HeatRepresentationConfig,
) -> tuple[str, ...]:
    available = set(zone.available_disturbances)
    selected: list[str] = []

    # Solar remains separately exposed when available.
    for solar in SOLAR_DISTURBANCES:
        if solar in available:
            selected.append(solar)

    if heat.representation is HeatInputRepresentation.GROUPED:
        if any(component in available for component in QZIC_COMPONENTS) or "zic" in available:
            selected.append("zic")
        if (
            any(component in available for component in QZIR_COMPONENTS)
            or "zir" in available
            or (
                heat.include_visible_lighting_in_qzir
                and VISIBLE_LIGHTING_COMPONENT in available
            )
        ):
            selected.append("zir")
        if (
            not heat.include_visible_lighting_in_qzir
            and VISIBLE_LIGHTING_COMPONENT in available
        ):
            selected.append(VISIBLE_LIGHTING_COMPONENT)
    else:
        for component in (*QZIC_COMPONENTS, *QZIR_COMPONENTS):
            if component in available:
                selected.append(component)
        if VISIBLE_LIGHTING_COMPONENT in available:
            selected.append(VISIBLE_LIGHTING_COMPONENT)

    return tuple(dict.fromkeys(selected))


def validate_partition_record(
    policy: PolicyContract,
    *,
    included: bool,
    partition: Partition | str,
    window_id: str | None,
    season: Season | str | None,
) -> None:
    """Validate one row's common policy metadata against the D6 policy."""

    partition_value = Partition(partition)
    season_value = None if season in (None, "") else Season(season)

    if partition_value not in policy.allowed_partition_values:
        raise D6ContractError(
            f"Partition {partition_value.value!r} is invalid for policy "
            f"{policy.name!r}"
        )

    if included and partition_value is Partition.EXCLUDED:
        raise D6ContractError("included rows cannot use partition='excluded'")
    if not included and partition_value is not Partition.EXCLUDED:
        raise D6ContractError(
            "non-included rows must use partition='excluded'"
        )

    if included and policy.requires_window_id_for_included:
        if window_id is None or not str(window_id).strip():
            raise D6ContractError(
                f"Policy {policy.name!r} requires window_id for included rows"
            )

    if included and policy.requires_season_for_included:
        if season_value is None:
            raise D6ContractError(
                f"Policy {policy.name!r} requires season for included rows"
            )


def _validate_path_token(value: str, *, field_name: str) -> None:
    if not value or not value.strip():
        raise D6ContractError(f"{field_name} cannot be empty")
    forbidden = {"/", "\\", "\0"}
    if any(token in value for token in forbidden):
        raise D6ContractError(
            f"{field_name} contains a forbidden path character: {value!r}"
        )
