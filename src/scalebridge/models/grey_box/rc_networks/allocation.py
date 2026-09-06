from __future__ import annotations

"""DEP2 all-to-one allocation mathematics from the frozen Step 4G/4H contract."""

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Mapping, Sequence

from .specification import AllocationFamilySpec, AllocationMode, RCCompileError


_ALLOCATION_TOL = 1e-10


@dataclass(frozen=True)
class AllocationResult:
    family_name: str
    lambda_by_zone: Mapping[str, float]
    p_by_zone: Mapping[str, float]
    residual: float
    ab_error: float
    logits: tuple[float, ...] = ()

    @property
    def max_consistency_error(self) -> float:
        return abs(float(self.ab_error))


def _ordered_weights(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> dict[str, float]:
    expected = tuple(zone_ids)
    if set(spec.weights) != set(expected):
        raise RCCompileError(
            f"Allocation family {spec.name!r} weights must cover exactly modeled zones; "
            f"expected={expected}, found={tuple(spec.weights)}"
        )
    weights = {zone: float(spec.weights[zone]) for zone in expected}
    if any(not isfinite(value) or value <= 0.0 for value in weights.values()):
        raise RCCompileError(
            f"Allocation family {spec.name!r} requires strictly positive finite weights"
        )
    total = sum(weights.values())
    if abs(total - 1.0) > _ALLOCATION_TOL:
        raise RCCompileError(
            f"Allocation family {spec.name!r} weights must sum to 1; got {total}"
        )
    return weights


def _participants(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> tuple[str, ...]:
    if spec.participating_zone_ids:
        participants = tuple(spec.participating_zone_ids)
    else:
        participants = tuple(zone_ids)
    if len(participants) != len(set(participants)):
        raise RCCompileError(
            f"Allocation family {spec.name!r} participating zones must be unique"
        )
    unknown = set(participants) - set(zone_ids)
    if unknown:
        raise RCCompileError(
            f"Allocation family {spec.name!r} has unknown participating zones: "
            f"{sorted(unknown)}"
        )
    if not participants:
        raise RCCompileError(
            f"Allocation family {spec.name!r} must participate in at least one zone"
        )
    return participants


def _merged_fixed_lambdas(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> dict[str, float]:
    participants = set(_participants(spec, zone_ids))
    fixed = {str(k): float(v) for k, v in spec.fixed_lambdas.items()}
    unknown = set(fixed) - set(zone_ids)
    if unknown:
        raise RCCompileError(
            f"Allocation family {spec.name!r} fixed_lambdas has unknown zones: "
            f"{sorted(unknown)}"
        )
    # Structurally non-participating zones cannot receive this family.
    for zone in zone_ids:
        if zone not in participants:
            if zone in fixed and abs(fixed[zone]) > _ALLOCATION_TOL:
                raise RCCompileError(
                    f"Non-participating zone {zone!r} must have lambda=0 in allocation "
                    f"family {spec.name!r}"
                )
            fixed[zone] = 0.0
    if any(not isfinite(v) or v < 0.0 for v in fixed.values()):
        raise RCCompileError(
            f"Allocation family {spec.name!r} fixed lambdas must be finite/nonnegative"
        )
    return fixed


def _validate_optional_bounds(
    spec: AllocationFamilySpec,
    weights: Mapping[str, float],
    zone_ids: Sequence[str],
) -> None:
    unknown = (set(spec.lower_bounds) | set(spec.upper_bounds)) - set(zone_ids)
    if unknown:
        raise RCCompileError(
            f"Allocation family {spec.name!r} bounds reference unknown zones: "
            f"{sorted(unknown)}"
        )
    lower = {z: float(spec.lower_bounds.get(z, 0.0)) for z in zone_ids}
    upper = {
        z: float(spec.upper_bounds.get(z, 1.0 / weights[z]))
        for z in zone_ids
    }
    for zone in zone_ids:
        if lower[zone] < 0.0 or upper[zone] < lower[zone]:
            raise RCCompileError(
                f"Invalid allocation bounds for {spec.name!r}/{zone!r}"
            )
    low_mass = sum(weights[z] * lower[z] for z in zone_ids)
    high_mass = sum(weights[z] * upper[z] for z in zone_ids)
    if low_mass > 1.0 + _ALLOCATION_TOL or high_mass < 1.0 - _ALLOCATION_TOL:
        raise RCCompileError(
            f"Allocation family {spec.name!r} bounds define an empty feasible set"
        )


def validate_allocation_spec(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> None:
    weights = _ordered_weights(spec, zone_ids)
    participants = set(_participants(spec, zone_ids))
    fixed = _merged_fixed_lambdas(spec, zone_ids)
    _validate_optional_bounds(spec, weights, zone_ids)

    if spec.mode is AllocationMode.NEUTRAL_FIXED:
        if fixed:
            # Nonparticipant zero-lambdas are inserted above; neutral_fixed is
            # therefore only legal when every modeled zone participates.
            if participants != set(zone_ids):
                raise RCCompileError(
                    f"neutral_fixed family {spec.name!r} requires every modeled zone "
                    "to participate because neutral lambda is identically one"
                )
            if spec.fixed_lambdas:
                raise RCCompileError(
                    f"neutral_fixed family {spec.name!r} cannot also specify fixed_lambdas"
                )
        return

    if spec.mode is AllocationMode.FIXED:
        if set(spec.fixed_lambdas) != set(zone_ids):
            raise RCCompileError(
                f"fixed allocation family {spec.name!r} requires one lambda per zone"
            )
        _validate_lambda_vector(spec, zone_ids, spec.fixed_lambdas)
        return

    if spec.mode is not AllocationMode.ESTIMATED:
        raise RCCompileError(f"Unsupported allocation mode: {spec.mode!r}")

    fixed_mass = sum(weights[z] * fixed[z] for z in fixed)
    residual = 1.0 - fixed_mass
    if residual < -_ALLOCATION_TOL:
        raise RCCompileError(
            f"Allocation family {spec.name!r} has negative residual {residual}"
        )
    estimated = [z for z in zone_ids if z not in fixed]
    if residual > _ALLOCATION_TOL and not estimated:
        raise RCCompileError(
            f"Allocation family {spec.name!r} leaves positive residual but no "
            "estimated zones"
        )


def _validate_lambda_vector(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
    lambdas: Mapping[str, float],
) -> AllocationResult:
    weights = _ordered_weights(spec, zone_ids)
    if set(lambdas) != set(zone_ids):
        raise RCCompileError(
            f"Allocation family {spec.name!r} lambda vector must cover all zones"
        )
    lam = {z: float(lambdas[z]) for z in zone_ids}
    if any(not isfinite(v) or v < 0.0 for v in lam.values()):
        raise RCCompileError(
            f"Allocation family {spec.name!r} lambda values must be finite/nonnegative"
        )

    for zone in zone_ids:
        lo = float(spec.lower_bounds.get(zone, 0.0))
        hi = float(spec.upper_bounds.get(zone, 1.0 / weights[zone]))
        if lam[zone] < lo - _ALLOCATION_TOL or lam[zone] > hi + _ALLOCATION_TOL:
            raise RCCompileError(
                f"Allocation family {spec.name!r}/{zone!r} lambda={lam[zone]} "
                f"violates [{lo}, {hi}]"
            )

    p = {z: weights[z] * lam[z] for z in zone_ids}
    mass = sum(p.values())
    error = abs(mass - 1.0)
    if error > _ALLOCATION_TOL:
        raise RCCompileError(
            f"Allocation family {spec.name!r} violates A_g B_g = 1; "
            f"weighted mass={mass}"
        )
    return AllocationResult(
        family_name=spec.name,
        lambda_by_zone=lam,
        p_by_zone=p,
        residual=0.0,
        ab_error=error,
    )


def neutral_allocation_result(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> AllocationResult:
    validate_allocation_spec(spec, zone_ids)
    if spec.mode is not AllocationMode.NEUTRAL_FIXED:
        raise RCCompileError("neutral_allocation_result requires neutral_fixed mode")
    lambdas = {z: 1.0 for z in zone_ids}
    return _validate_lambda_vector(spec, zone_ids, lambdas)


def fixed_allocation_result(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> AllocationResult:
    validate_allocation_spec(spec, zone_ids)
    if spec.mode is not AllocationMode.FIXED:
        raise RCCompileError("fixed_allocation_result requires fixed mode")
    return _validate_lambda_vector(spec, zone_ids, spec.fixed_lambdas)


def estimated_zone_ids(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> tuple[str, ...]:
    validate_allocation_spec(spec, zone_ids)
    if spec.mode is not AllocationMode.ESTIMATED:
        return ()
    fixed = _merged_fixed_lambdas(spec, zone_ids)
    return tuple(z for z in zone_ids if z not in fixed)


def allocation_degrees_of_freedom(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> int:
    """Return independent trainable allocation DOF after fixed constraints."""

    estimated = estimated_zone_ids(spec, zone_ids)
    if not estimated:
        return 0
    weights = _ordered_weights(spec, zone_ids)
    fixed = _merged_fixed_lambdas(spec, zone_ids)
    residual = 1.0 - sum(weights[z] * fixed[z] for z in fixed)
    if residual <= _ALLOCATION_TOL:
        return 0
    return max(len(estimated) - 1, 0)


def initial_reference_logits(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> tuple[float, ...]:
    """Return residual-neutral N_est-1 logits with the last estimated zone as reference.

    With no fixed lambdas this is exactly alpha_i=log(w_i/w_ref), which yields
    p=w and lambda=1. With partial fixed allocation, the same relative weighting
    gives one common lambda over the remaining estimated zones while satisfying
    the residual constraint exactly.
    """

    validate_allocation_spec(spec, zone_ids)
    if spec.mode is not AllocationMode.ESTIMATED:
        raise RCCompileError("initial_reference_logits requires estimated mode")
    estimated = estimated_zone_ids(spec, zone_ids)
    if len(estimated) <= 1:
        return ()
    weights = _ordered_weights(spec, zone_ids)
    ref = weights[estimated[-1]]
    return tuple(log(weights[z] / ref) for z in estimated[:-1])


def _softmax_with_reference(
    logits: Sequence[float],
    n_estimated: int,
) -> tuple[float, ...]:
    if n_estimated == 0:
        if logits:
            raise RCCompileError("No estimated zones; logits must be empty")
        return ()
    if n_estimated == 1:
        if logits:
            raise RCCompileError("One estimated zone has zero allocation DOF")
        return (1.0,)
    if len(logits) != n_estimated - 1:
        raise RCCompileError(
            f"Expected {n_estimated - 1} reference logits, got {len(logits)}"
        )
    raw = tuple(float(x) for x in logits) + (0.0,)
    if any(not isfinite(x) for x in raw):
        raise RCCompileError("Allocation logits must be finite")
    shift = max(raw)
    exps = [exp(x - shift) for x in raw]
    denom = sum(exps)
    return tuple(value / denom for value in exps)


def estimated_allocation_result(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
    logits: Sequence[float],
) -> AllocationResult:
    """Map N_est-1 unconstrained logits to a feasible full-zone lambda vector."""

    validate_allocation_spec(spec, zone_ids)
    if spec.mode is not AllocationMode.ESTIMATED:
        raise RCCompileError("estimated_allocation_result requires estimated mode")

    weights = _ordered_weights(spec, zone_ids)
    fixed = _merged_fixed_lambdas(spec, zone_ids)
    estimated = tuple(z for z in zone_ids if z not in fixed)
    fixed_mass = sum(weights[z] * fixed[z] for z in fixed)
    residual = max(0.0, 1.0 - fixed_mass)

    lambdas = {z: float(fixed[z]) for z in fixed}
    if residual <= _ALLOCATION_TOL:
        if logits:
            raise RCCompileError("Zero residual allocation has no trainable logits")
        for zone in estimated:
            lambdas[zone] = 0.0
    else:
        shares = _softmax_with_reference(logits, len(estimated))
        for zone, share in zip(estimated, shares):
            lambdas[zone] = residual * share / weights[zone]

    result = _validate_lambda_vector(spec, zone_ids, lambdas)
    return AllocationResult(
        family_name=result.family_name,
        lambda_by_zone=result.lambda_by_zone,
        p_by_zone=result.p_by_zone,
        residual=residual,
        ab_error=result.ab_error,
        logits=tuple(float(x) for x in logits),
    )


def initial_estimated_allocation_result(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> AllocationResult:
    return estimated_allocation_result(
        spec,
        zone_ids,
        initial_reference_logits(spec, zone_ids),
    )


def default_allocation_result(
    spec: AllocationFamilySpec,
    zone_ids: Sequence[str],
) -> AllocationResult:
    """Explicitly construct the mode's deterministic starting/fixed result."""

    if spec.mode is AllocationMode.NEUTRAL_FIXED:
        return neutral_allocation_result(spec, zone_ids)
    if spec.mode is AllocationMode.FIXED:
        return fixed_allocation_result(spec, zone_ids)
    if spec.mode is AllocationMode.ESTIMATED:
        return initial_estimated_allocation_result(spec, zone_ids)
    raise RCCompileError(f"Unsupported allocation mode: {spec.mode!r}")
