from __future__ import annotations

"""Physical-parameter instances, master sharing, and value resolution."""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from .specification import (
    ParameterConfig,
    ParameterSharingRule,
    RCCompileError,
)


@dataclass(frozen=True)
class ParameterInstance:
    instance_id: str
    physical_type: str
    family: str
    units: str
    zone_scope: tuple[str, ...]
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.instance_id or not self.family:
            raise RCCompileError("Parameter instance identifiers cannot be empty")
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise RCCompileError(
                    f"Invalid bounds for {self.instance_id}: "
                    f"{self.lower_bound} > {self.upper_bound}"
                )

    @property
    def compatibility_key(self) -> tuple[str, str, str]:
        return (self.physical_type, self.units, self.family)


@dataclass(frozen=True)
class MasterParameter:
    master_id: str
    member_instance_ids: tuple[str, ...]
    physical_type: str
    family: str
    units: str
    lower_bound: float | None
    upper_bound: float | None
    config: ParameterConfig | None = None


@dataclass(frozen=True)
class ParameterRegistry:
    instances: tuple[ParameterInstance, ...]
    masters: tuple[MasterParameter, ...]
    instance_to_master: Mapping[str, str]

    def instance(self, instance_id: str) -> ParameterInstance:
        lookup = {item.instance_id: item for item in self.instances}
        try:
            return lookup[instance_id]
        except KeyError as exc:
            raise RCCompileError(f"Unknown parameter instance: {instance_id}") from exc

    def master(self, master_id: str) -> MasterParameter:
        lookup = {item.master_id: item for item in self.masters}
        try:
            return lookup[master_id]
        except KeyError as exc:
            raise RCCompileError(f"Unknown master parameter: {master_id}") from exc

    @property
    def master_ids(self) -> tuple[str, ...]:
        return tuple(item.master_id for item in self.masters)

    def initial_master_values(self) -> dict[str, float]:
        missing = [item.master_id for item in self.masters if item.config is None]
        if missing:
            raise RCCompileError(
                "Cannot construct initial master values; missing ParameterConfig for "
                + ", ".join(missing)
            )
        return {
            item.master_id: float(item.config.initial_value)  # type: ignore[union-attr]
            for item in self.masters
        }

    def resolve_instance_values(
        self,
        values: Mapping[str, float],
    ) -> dict[str, float]:
        """Resolve values supplied by master ID or compatible instance IDs.

        Master-key input is preferred.  For convenience, a master may instead be
        supplied through all of its member instance IDs, but those values must be
        numerically equal.
        """

        out: dict[str, float] = {}
        for master in self.masters:
            if master.master_id in values:
                value = float(values[master.master_id])
            else:
                member_values = [
                    float(values[item])
                    for item in master.member_instance_ids
                    if item in values
                ]
                if len(member_values) != len(master.member_instance_ids):
                    raise RCCompileError(
                        f"Missing numerical value for master parameter "
                        f"{master.master_id!r}; members={master.member_instance_ids}"
                    )
                value = member_values[0]
                if any(abs(other - value) > 1e-12 for other in member_values[1:]):
                    raise RCCompileError(
                        f"Shared master {master.master_id!r} received unequal member values"
                    )

            _validate_master_value(master, value)
            for instance_id in master.member_instance_ids:
                out[instance_id] = value
        return out


def _validate_master_value(master: MasterParameter, value: float) -> None:
    if not isfinite(value):
        raise RCCompileError(f"Non-finite value for {master.master_id}: {value}")
    if master.physical_type in {"capacitance", "resistance"} and value <= 0.0:
        raise RCCompileError(
            f"{master.master_id} must be strictly positive; got {value}"
        )
    if master.lower_bound is not None and value < master.lower_bound:
        raise RCCompileError(
            f"{master.master_id}={value} is below lower bound {master.lower_bound}"
        )
    if master.upper_bound is not None and value > master.upper_bound:
        raise RCCompileError(
            f"{master.master_id}={value} is above upper bound {master.upper_bound}"
        )


def make_capacitance_instance(zone_id: str, family: str) -> ParameterInstance:
    return ParameterInstance(
        instance_id=f"C|{zone_id}|{family}",
        physical_type="capacitance",
        family=family,
        units="J/K",
        zone_scope=(zone_id,),
        lower_bound=0.0,
    )


def make_resistance_instance(
    *,
    family: str,
    zone_scope: Sequence[str],
    discriminator: str,
) -> ParameterInstance:
    zones = tuple(zone_scope)
    return ParameterInstance(
        instance_id=f"R|{discriminator}|{family}",
        physical_type="resistance",
        family=family,
        units="K/W",
        zone_scope=zones,
        lower_bound=0.0,
    )


def make_routing_instance(zone_id: str, family: str) -> ParameterInstance:
    return ParameterInstance(
        instance_id=f"routing|{zone_id}|{family}",
        physical_type="routing",
        family=family,
        units="1",
        zone_scope=(zone_id,),
        lower_bound=0.0,
        upper_bound=1.0,
    )


def _selected_instance_ids(
    instances: Sequence[ParameterInstance],
    rule: ParameterSharingRule,
) -> tuple[str, ...]:
    by_id = {item.instance_id: item for item in instances}
    if rule.instance_ids:
        missing = set(rule.instance_ids) - set(by_id)
        if missing:
            raise RCCompileError(
                f"Sharing rule {rule.name!r} references unknown instances: "
                f"{sorted(missing)}"
            )
        return tuple(rule.instance_ids)

    assert rule.family is not None
    allowed_zones = set(rule.zone_ids)
    selected = []
    for item in instances:
        if item.family != rule.family:
            continue
        if allowed_zones and not set(item.zone_scope).issubset(allowed_zones):
            continue
        selected.append(item.instance_id)
    if not selected:
        raise RCCompileError(
            f"Sharing rule {rule.name!r} selected no instances for family "
            f"{rule.family!r}"
        )
    return tuple(selected)


def build_parameter_registry(
    instances: Sequence[ParameterInstance],
    sharing_rules: Sequence[ParameterSharingRule] = (),
    parameter_configs: Mapping[str, ParameterConfig] | None = None,
) -> ParameterRegistry:
    """Create independent-by-default masters then apply explicit sharing."""

    instance_tuple = tuple(instances)
    ids = [item.instance_id for item in instance_tuple]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise RCCompileError(f"Duplicate parameter instance IDs: {duplicates}")

    by_id = {item.instance_id: item for item in instance_tuple}
    assigned: dict[str, str] = {}
    groups: dict[str, list[str]] = {}

    for rule in sharing_rules:
        selected = _selected_instance_ids(instance_tuple, rule)
        overlap = [item for item in selected if item in assigned]
        if overlap:
            raise RCCompileError(
                f"Sharing rule {rule.name!r} overlaps prior sharing assignments: "
                f"{overlap}"
            )
        compatibility = {by_id[item].compatibility_key for item in selected}
        if len(compatibility) != 1:
            raise RCCompileError(
                f"Sharing rule {rule.name!r} mixes incompatible physical parameters: "
                f"{sorted(compatibility)}"
            )
        master_id = f"shared|{rule.name}"
        groups[master_id] = list(selected)
        for item in selected:
            assigned[item] = master_id

    for item in instance_tuple:
        if item.instance_id not in assigned:
            master_id = item.instance_id
            assigned[item.instance_id] = master_id
            groups[master_id] = [item.instance_id]

    configs = dict(parameter_configs or {})
    masters: list[MasterParameter] = []
    for master_id, member_ids in groups.items():
        members = [by_id[item] for item in member_ids]
        first = members[0]
        lowers = [x.lower_bound for x in members if x.lower_bound is not None]
        uppers = [x.upper_bound for x in members if x.upper_bound is not None]
        lower = max(lowers) if lowers else None
        upper = min(uppers) if uppers else None
        if lower is not None and upper is not None and lower > upper:
            raise RCCompileError(f"Shared master {master_id!r} has incompatible bounds")

        config = configs.get(master_id)
        # For independent masters, permit config keyed by the physical instance ID.
        if config is None and len(member_ids) == 1:
            config = configs.get(member_ids[0])
        if config is not None:
            if first.physical_type in {"capacitance", "resistance"} and config.initial_value <= 0.0:
                raise RCCompileError(
                    f"Initial value for {master_id!r} must be strictly positive"
                )
            if lower is not None and config.initial_value < lower:
                raise RCCompileError(
                    f"Initial value for {master_id!r} violates compiler lower bound"
                )
            if upper is not None and config.initial_value > upper:
                raise RCCompileError(
                    f"Initial value for {master_id!r} violates compiler upper bound"
                )

        masters.append(
            MasterParameter(
                master_id=master_id,
                member_instance_ids=tuple(member_ids),
                physical_type=first.physical_type,
                family=first.family,
                units=first.units,
                lower_bound=lower,
                upper_bound=upper,
                config=config,
            )
        )

    return ParameterRegistry(
        instances=instance_tuple,
        masters=tuple(masters),
        instance_to_master=dict(assigned),
    )
