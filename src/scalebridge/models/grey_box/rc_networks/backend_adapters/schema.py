from __future__ import annotations

"""Build the canonical E0-6 raw-coordinate plan from a compiled E0-3 model."""

from math import isclose
from typing import Iterable

from ..allocation import default_allocation_result, validate_allocation_spec
from ..compiler import CompiledRCModel
from ..specification import AllocationMode, ParameterStatus
from .contracts import (
    AllocationParameterPlan,
    BackendAdapterError,
    BackendParameterizationPlan,
    RawCoordinate,
    ScalarParameterPlan,
    ScalarTransformKind,
    SimplexParameterPlan,
)

_TOL = 1e-10


def _effective_bounds(master) -> tuple[float | None, float | None]:
    lower = master.lower_bound
    upper = master.upper_bound
    config = master.config
    if config is not None:
        if config.lower_bound is not None:
            lower = max(lower, config.lower_bound) if lower is not None else config.lower_bound
        if config.upper_bound is not None:
            upper = min(upper, config.upper_bound) if upper is not None else config.upper_bound
    if lower is not None and upper is not None and lower > upper:
        raise BackendAdapterError(
            f"Master {master.master_id!r} has empty effective bounds [{lower}, {upper}]"
        )
    return lower, upper


def _require_configs(model: CompiledRCModel) -> None:
    missing = [m.master_id for m in model.parameter_registry.masters if m.config is None]
    if missing:
        raise BackendAdapterError(
            "E0-6 requires ParameterConfig for every master parameter so the campaign "
            "fixed/trainable policy and baseline are explicit; missing=" + ", ".join(missing)
        )


def _routing_simplex_groups(model: CompiledRCModel) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    if model.flavour.routing_kind != "gamma_r_3way":
        return ()

    groups: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    seen_tuples: set[tuple[str, ...]] = set()
    member_to_tuple: dict[str, tuple[str, ...]] = {}
    labels = ("gamma_a_r", "gamma_e_r", "gamma_m_r")

    for zone in model.spec.zone_ids:
        master_ids = tuple(
            model.parameter_registry.instance_to_master[model.routing_parameter_ids[(zone, label)]]
            for label in labels
        )
        if len(set(master_ids)) != 3:
            raise BackendAdapterError(
                f"4R3C routing group for zone {zone!r} aliases members within one simplex; "
                "each simplex component must have a distinct master"
            )
        for master_id in master_ids:
            prior = member_to_tuple.get(master_id)
            if prior is not None and prior != master_ids:
                raise BackendAdapterError(
                    "Partial cross-zone sharing of a 4R3C routing simplex is ambiguous. "
                    "Share the complete gamma_a_r/gamma_e_r/gamma_m_r vector or keep "
                    f"the zones independent. Offending master={master_id!r}"
                )
            member_to_tuple[master_id] = master_ids
        if master_ids not in seen_tuples:
            seen_tuples.add(master_ids)
            groups.append((f"routing_simplex|{'|'.join(master_ids)}", master_ids, labels))
    return tuple(groups)


def _is_default_allocation_bound(spec, zone: str, weight: float) -> bool:
    lo = float(spec.lower_bounds.get(zone, 0.0))
    hi = float(spec.upper_bounds.get(zone, 1.0 / weight))
    return isclose(lo, 0.0, abs_tol=_TOL) and isclose(hi, 1.0 / weight, rel_tol=0.0, abs_tol=_TOL)


def build_parameterization_plan(model: CompiledRCModel) -> BackendParameterizationPlan:
    """Create deterministic dimensionless E0-6 raw coordinates.

    The plan consumes E0-3 master-parameter identities and campaign-supplied
    ``ParameterConfig`` objects. It never redefines flavour topology.
    """

    _require_configs(model)

    raw: list[RawCoordinate] = []
    scalar_plans: list[ScalarParameterPlan] = []
    simplex_plans: list[SimplexParameterPlan] = []
    allocation_plans: list[AllocationParameterPlan] = []

    simplex_groups = _routing_simplex_groups(model)
    simplex_master_ids = {m for _, mids, _ in simplex_groups for m in mids}

    masters = {m.master_id: m for m in model.parameter_registry.masters}
    master_order = tuple(m.master_id for m in model.parameter_registry.masters)

    # Scalar masters retain the deterministic E0-3 registry ordering.
    for master_id in master_order:
        if master_id in simplex_master_ids:
            continue
        master = masters[master_id]
        config = master.config
        assert config is not None
        baseline = float(config.initial_value)
        lower, upper = _effective_bounds(master)
        estimated = config.status is ParameterStatus.ESTIMATED

        if not estimated:
            transform = ScalarTransformKind.FIXED
            raw_index = None
        elif lower is not None and upper is not None:
            if not (lower < baseline < upper):
                raise BackendAdapterError(
                    f"Estimated bounded master {master_id!r} requires a baseline strictly "
                    f"inside ({lower}, {upper}); got {baseline}"
                )
            transform = ScalarTransformKind.BOUNDED_SIGMOID
            raw_index = len(raw)
            raw.append(RawCoordinate(raw_index, f"rho|{master_id}", "master", master_id))
        elif master.physical_type in {"capacitance", "resistance"}:
            effective_lower = 0.0 if lower is None else float(lower)
            if not baseline > effective_lower:
                raise BackendAdapterError(
                    f"Estimated positive master {master_id!r} requires baseline > lower "
                    f"bound {effective_lower}; got {baseline}"
                )
            transform = (
                ScalarTransformKind.POSITIVE_EXP
                if abs(effective_lower) <= _TOL
                else ScalarTransformKind.SHIFTED_EXP
            )
            lower = effective_lower
            raw_index = len(raw)
            raw.append(RawCoordinate(raw_index, f"rho|{master_id}", "master", master_id))
        else:
            raise BackendAdapterError(
                f"E0-6 has no ratified unconstrained transform for estimated master "
                f"{master_id!r} physical_type={master.physical_type!r}"
            )

        scalar_plans.append(
            ScalarParameterPlan(
                master_id=master_id,
                baseline=baseline,
                status=config.status.value,
                transform=transform,
                raw_index=raw_index,
                lower_bound=lower,
                upper_bound=upper,
                physical_type=master.physical_type,
                family=master.family,
                units=master.units,
            )
        )

    # 4R3C radiative routing is one support-restricted simplex, not three scalars.
    for group_id, member_ids, labels in simplex_groups:
        member_objs = [masters[mid] for mid in member_ids]
        baselines = tuple(float(m.config.initial_value) for m in member_objs)  # type: ignore[union-attr]
        if any(value < 0.0 for value in baselines) or abs(sum(baselines) - 1.0) > _TOL:
            raise BackendAdapterError(
                f"Routing simplex {group_id!r} baseline must be nonnegative and sum to one; "
                f"got {baselines}"
            )
        fixed_mask = tuple(m.config.status is ParameterStatus.FIXED for m in member_objs)  # type: ignore[union-attr]
        fixed_mass = sum(v for v, fixed in zip(baselines, fixed_mask) if fixed)
        residual = 1.0 - fixed_mass
        estimated_positions = [i for i, fixed in enumerate(fixed_mask) if not fixed]
        estimated_baselines = [baselines[i] for i in estimated_positions]
        if residual < -_TOL:
            raise BackendAdapterError(f"Routing simplex {group_id!r} fixed mass exceeds one")
        if estimated_positions:
            if residual <= _TOL:
                raise BackendAdapterError(
                    f"Routing simplex {group_id!r} has estimated members but zero residual"
                )
            if any(value <= 0.0 for value in estimated_baselines):
                raise BackendAdapterError(
                    f"Trainable routing simplex {group_id!r} requires strictly positive "
                    "baseline on every estimated component"
                )
            if abs(sum(estimated_baselines) - residual) > _TOL:
                raise BackendAdapterError(
                    f"Routing simplex {group_id!r} estimated baseline mass does not equal "
                    f"residual {residual}"
                )
            for m in member_objs:
                cfg = m.config
                assert cfg is not None
                # The ratified E0-6 simplex transform enforces nonnegative/unit-sum.
                # Extra per-component estimated bounds require a different constrained
                # simplex map and are therefore rejected rather than silently violated.
                if cfg.status is ParameterStatus.ESTIMATED:
                    if cfg.lower_bound not in (None, 0.0) or cfg.upper_bound not in (None, 1.0):
                        raise BackendAdapterError(
                            f"Estimated routing simplex member {m.master_id!r} specifies "
                            "additional component bounds not represented by the frozen "
                            "anchored-softmax E0-6 transform"
                        )
        elif abs(fixed_mass - 1.0) > _TOL:
            raise BackendAdapterError(
                f"Fully fixed routing simplex {group_id!r} must sum to one"
            )

        raw_indices: list[int] = []
        anchor_position = estimated_positions[0] if estimated_positions else None
        for pos in estimated_positions[1:]:
            idx = len(raw)
            raw.append(
                RawCoordinate(
                    idx,
                    f"rho|{group_id}|{labels[pos]}",
                    "routing_simplex",
                    group_id,
                    labels[pos],
                )
            )
            raw_indices.append(idx)
        simplex_plans.append(
            SimplexParameterPlan(
                group_id=group_id,
                master_ids=member_ids,
                labels=labels,
                baseline=baselines,
                fixed_mask=fixed_mask,
                raw_indices=tuple(raw_indices),
                anchor_position=anchor_position,
                residual=max(0.0, residual),
            )
        )

    # DEP2 trainable allocations are appended after physical masters.
    aggregate_signal_order: list[str] = []
    if model.spec.mode.value == "dep2":
        for family_name, spec in model.allocation_families.items():
            validate_allocation_spec(spec, model.spec.zone_ids)
            for signal in spec.signals:
                aggregate_signal_order.append(signal)

            zones = tuple(model.spec.zone_ids)
            weights = tuple(float(spec.weights[z]) for z in zones)
            if spec.mode is AllocationMode.ESTIMATED:
                for zone, weight in zip(zones, weights):
                    if not _is_default_allocation_bound(spec, zone, weight):
                        raise BackendAdapterError(
                            f"Estimated DEP2 family {family_name!r} uses additional bounds "
                            "not represented by the frozen contribution-simplex transform"
                        )

            baseline_result = default_allocation_result(spec, zones)
            baseline_p = tuple(float(baseline_result.p_by_zone[z]) for z in zones)
            fixed_lambdas: list[float | None] = []
            estimated_positions: list[int] = []
            if spec.mode is AllocationMode.ESTIMATED:
                explicit_fixed = dict(spec.fixed_lambdas)
                participants = set(spec.participating_zone_ids or zones)
                for i, zone in enumerate(zones):
                    if zone not in participants:
                        fixed_lambdas.append(0.0)
                    elif zone in explicit_fixed:
                        fixed_lambdas.append(float(explicit_fixed[zone]))
                    else:
                        fixed_lambdas.append(None)
                        estimated_positions.append(i)
            else:
                for zone in zones:
                    fixed_lambdas.append(float(baseline_result.lambda_by_zone[zone]))

            fixed_mass = sum(
                weight * lam
                for weight, lam in zip(weights, fixed_lambdas)
                if lam is not None
            )
            residual = max(0.0, 1.0 - fixed_mass)
            if estimated_positions:
                if residual <= _TOL:
                    raise BackendAdapterError(
                        f"Estimated DEP2 family {family_name!r} has zero residual"
                    )
                if any(baseline_p[pos] <= 0.0 for pos in estimated_positions):
                    raise BackendAdapterError(
                        f"Estimated DEP2 family {family_name!r} requires strictly positive "
                        "baseline contribution p on estimated support"
                    )
                if abs(sum(baseline_p[pos] for pos in estimated_positions) - residual) > _TOL:
                    raise BackendAdapterError(
                        f"DEP2 family {family_name!r} baseline estimated mass does not equal residual"
                    )

            raw_indices: list[int] = []
            anchor_position = estimated_positions[0] if estimated_positions else None
            for pos in estimated_positions[1:]:
                zone = zones[pos]
                idx = len(raw)
                raw.append(
                    RawCoordinate(
                        idx,
                        f"rho|allocation|{family_name}|{zone}",
                        "allocation",
                        family_name,
                        zone,
                    )
                )
                raw_indices.append(idx)

            allocation_plans.append(
                AllocationParameterPlan(
                    family_name=family_name,
                    signal_names=tuple(spec.signals),
                    zone_ids=zones,
                    weights=weights,
                    fixed_lambdas=tuple(fixed_lambdas),
                    baseline_p=baseline_p,
                    estimated_positions=tuple(estimated_positions),
                    raw_indices=tuple(raw_indices),
                    anchor_position=anchor_position,
                    residual=residual,
                )
            )

    return BackendParameterizationPlan(
        raw_coordinates=tuple(raw),
        scalar_parameters=tuple(scalar_plans),
        simplex_parameters=tuple(simplex_plans),
        allocation_parameters=tuple(allocation_plans),
        master_order=master_order,
        aggregate_signal_order=tuple(aggregate_signal_order),
    )


def build_physical_parameterization_plan(model: CompiledRCModel):
    """Build direct physical decision coordinates for CasADi/IPOPT.

    Unlike :func:`build_parameterization_plan`, this plan does not eliminate
    constrained degrees of freedom with exp/sigmoid/softmax transforms.
    Estimated physical quantities are decision variables directly, while
    routing/allocation conservation is represented by explicit linear
    constraints suitable for an NLP solver.
    """

    from .contracts import (
        PhysicalDecisionCoordinate,
        PhysicalLinearConstraint,
        PhysicalParameterizationPlan,
    )

    _require_configs(model)
    masters = {m.master_id: m for m in model.parameter_registry.masters}
    master_order = tuple(m.master_id for m in model.parameter_registry.masters)
    simplex_groups = _routing_simplex_groups(model)
    simplex_master_ids = {mid for _, mids, _ in simplex_groups for mid in mids}

    coordinates = []
    constraints = []
    fixed_master_values: dict[str, float] = {}
    master_decision_index: dict[str, int] = {}
    allocation_p_index: dict[tuple[str, str], int] = {}

    def add_coordinate(*, name, owner_kind, owner_id, component, baseline, lower, upper, units, physical_type):
        idx = len(coordinates)
        coordinates.append(
            PhysicalDecisionCoordinate(
                index=idx,
                name=name,
                owner_kind=owner_kind,
                owner_id=owner_id,
                component=component,
                baseline=float(baseline),
                lower_bound=None if lower is None else float(lower),
                upper_bound=None if upper is None else float(upper),
                units=units,
                physical_type=physical_type,
            )
        )
        return idx

    # Ordinary scalar masters: fixed values remain constants; estimated values
    # become physical decision variables with compiler/campaign bounds.
    for master_id in master_order:
        if master_id in simplex_master_ids:
            continue
        master = masters[master_id]
        cfg = master.config
        assert cfg is not None
        lower, upper = _effective_bounds(master)
        baseline = float(cfg.initial_value)
        if cfg.status is ParameterStatus.FIXED:
            fixed_master_values[master_id] = baseline
            continue
        if master.physical_type in {'capacitance', 'resistance'} and baseline <= 0.0:
            raise BackendAdapterError(
                f"Estimated physical master {master_id!r} must start strictly positive"
            )
        idx = add_coordinate(
            name=f"theta|{master_id}",
            owner_kind="master",
            owner_id=master_id,
            component=None,
            baseline=baseline,
            lower=lower,
            upper=upper,
            units=master.units,
            physical_type=master.physical_type,
        )
        master_decision_index[master_id] = idx

    # 4R3C radiative routing: every estimated component stays physical and an
    # explicit equality enforces the remaining unit-mass residual.
    for group_id, member_ids, labels in simplex_groups:
        fixed_mass = 0.0
        estimated_indices = []
        estimated_baseline_sum = 0.0
        for master_id, label in zip(member_ids, labels):
            master = masters[master_id]
            cfg = master.config
            assert cfg is not None
            baseline = float(cfg.initial_value)
            lower, upper = _effective_bounds(master)
            if cfg.status is ParameterStatus.FIXED:
                fixed_master_values[master_id] = baseline
                fixed_mass += baseline
            else:
                idx = add_coordinate(
                    name=f"theta|{group_id}|{label}",
                    owner_kind="routing_simplex",
                    owner_id=group_id,
                    component=label,
                    baseline=baseline,
                    lower=0.0 if lower is None else lower,
                    upper=1.0 if upper is None else upper,
                    units=master.units,
                    physical_type=master.physical_type,
                )
                master_decision_index[master_id] = idx
                estimated_indices.append(idx)
                estimated_baseline_sum += baseline
        residual = 1.0 - fixed_mass
        if residual < -_TOL:
            raise BackendAdapterError(f"Routing simplex {group_id!r} fixed mass exceeds one")
        if estimated_indices:
            if abs(estimated_baseline_sum - residual) > _TOL:
                raise BackendAdapterError(
                    f"Routing simplex {group_id!r} estimated baseline mass must equal {residual}"
                )
            constraints.append(
                PhysicalLinearConstraint(
                    constraint_id=f"sum|{group_id}",
                    indices=tuple(estimated_indices),
                    coefficients=tuple(1.0 for _ in estimated_indices),
                    lower_bound=float(residual),
                    upper_bound=float(residual),
                )
            )
        elif abs(fixed_mass - 1.0) > _TOL:
            raise BackendAdapterError(f"Fully fixed routing simplex {group_id!r} must sum to one")

    aggregate_signal_order: list[str] = []
    if model.spec.mode.value == 'dep2':
        for family_name, spec in model.allocation_families.items():
            validate_allocation_spec(spec, model.spec.zone_ids)
            aggregate_signal_order.extend(spec.signals)
            zones = tuple(model.spec.zone_ids)
            weights = {z: float(spec.weights[z]) for z in zones}
            baseline = default_allocation_result(spec, zones)
            participants = set(spec.participating_zone_ids or zones)

            fixed_lambdas: dict[str, float] = {}
            if spec.mode is AllocationMode.NEUTRAL_FIXED:
                fixed_lambdas = {z: 1.0 for z in zones}
            elif spec.mode is AllocationMode.FIXED:
                fixed_lambdas = {z: float(spec.fixed_lambdas[z]) for z in zones}
            elif spec.mode is AllocationMode.ESTIMATED:
                for z in zones:
                    if z not in participants:
                        fixed_lambdas[z] = 0.0
                    elif z in spec.fixed_lambdas:
                        fixed_lambdas[z] = float(spec.fixed_lambdas[z])
            else:  # pragma: no cover
                raise BackendAdapterError(f"Unsupported allocation mode {spec.mode!r}")

            fixed_mass = sum(weights[z] * lam for z, lam in fixed_lambdas.items())
            residual = 1.0 - fixed_mass
            estimated_indices = []
            estimated_baseline_sum = 0.0
            if spec.mode is AllocationMode.ESTIMATED:
                for z in zones:
                    if z in fixed_lambdas:
                        continue
                    p0 = float(baseline.p_by_zone[z])
                    lam_lo = float(spec.lower_bounds.get(z, 0.0))
                    lam_hi = float(spec.upper_bounds.get(z, 1.0 / weights[z]))
                    idx = add_coordinate(
                        name=f"theta|allocation|{family_name}|p|{z}",
                        owner_kind="allocation_p",
                        owner_id=family_name,
                        component=z,
                        baseline=p0,
                        lower=weights[z] * lam_lo,
                        upper=weights[z] * lam_hi,
                        units="1",
                        physical_type="allocation_contribution",
                    )
                    allocation_p_index[(family_name, z)] = idx
                    estimated_indices.append(idx)
                    estimated_baseline_sum += p0
                if estimated_indices:
                    if residual <= _TOL:
                        raise BackendAdapterError(
                            f"Estimated DEP2 family {family_name!r} has no positive residual"
                        )
                    if abs(estimated_baseline_sum - residual) > _TOL:
                        raise BackendAdapterError(
                            f"DEP2 family {family_name!r} estimated baseline p mass must equal residual {residual}"
                        )
                    constraints.append(
                        PhysicalLinearConstraint(
                            constraint_id=f"sum|allocation|{family_name}",
                            indices=tuple(estimated_indices),
                            coefficients=tuple(1.0 for _ in estimated_indices),
                            lower_bound=float(residual),
                            upper_bound=float(residual),
                        )
                    )

    return PhysicalParameterizationPlan(
        coordinates=tuple(coordinates),
        constraints=tuple(constraints),
        master_order=master_order,
        aggregate_signal_order=tuple(aggregate_signal_order),
        fixed_master_values=fixed_master_values,
        master_decision_index=master_decision_index,
        allocation_p_index=allocation_p_index,
    )
