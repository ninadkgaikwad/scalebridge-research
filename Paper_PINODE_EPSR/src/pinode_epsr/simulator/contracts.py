from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CommandMode(str, Enum):
    """Whether this zone is under external physical control or native EnergyPlus."""

    OVERRIDE = "override"
    NATIVE = "native"


class EffectiveControlMode(str, Enum):
    """What the simulator actually applies after the feasibility supervisor."""

    OVERRIDE = "override"
    NATIVE_REQUESTED = "native_requested"
    NATIVE_FALLBACK = "native_fallback"


@dataclass(frozen=True)
class PhysicalZoneCommand:
    """
    Physical controller/MPC command for one zone.

    All four RestaurantFastFood regimes are commandable:
      Dining heating
      Dining cooling
      Kitchen heating
      Kitchen cooling

    Feasibility is evaluated numerically and symmetrically per zone.
    """

    mode: CommandMode
    mass_flow_kg_s: Optional[float] = None
    supply_air_temperature_c: Optional[float] = None

    @classmethod
    def override(
        cls,
        mass_flow_kg_s: float,
        supply_air_temperature_c: float,
    ) -> "PhysicalZoneCommand":
        return cls(
            mode=CommandMode.OVERRIDE,
            mass_flow_kg_s=float(mass_flow_kg_s),
            supply_air_temperature_c=float(supply_air_temperature_c),
        )

    @classmethod
    def native(cls) -> "PhysicalZoneCommand":
        return cls(mode=CommandMode.NATIVE)

    def validate(self) -> None:
        if self.mode == CommandMode.NATIVE:
            return
        if self.mass_flow_kg_s is None:
            raise ValueError("OVERRIDE requires mass_flow_kg_s.")
        if self.supply_air_temperature_c is None:
            raise ValueError("OVERRIDE requires supply_air_temperature_c.")
        if self.mass_flow_kg_s <= 0:
            raise ValueError("mass_flow_kg_s must be positive.")


@dataclass(frozen=True)
class RestaurantFastFoodCommand:
    """
    Explicit four-command interface:

      u* = [m_dot_D*, T_sa,D*, m_dot_K*, T_sa,K*]^T

    Either zone can also be explicitly released by sending native().
    """

    dining: PhysicalZoneCommand
    kitchen: PhysicalZoneCommand

    @classmethod
    def four_physical_commands(
        cls,
        *,
        dining_mass_flow_kg_s: float,
        dining_supply_air_temperature_c: float,
        kitchen_mass_flow_kg_s: float,
        kitchen_supply_air_temperature_c: float,
    ) -> "RestaurantFastFoodCommand":
        return cls(
            dining=PhysicalZoneCommand.override(
                dining_mass_flow_kg_s,
                dining_supply_air_temperature_c,
            ),
            kitchen=PhysicalZoneCommand.override(
                kitchen_mass_flow_kg_s,
                kitchen_supply_air_temperature_c,
            ),
        )

    def as_zone_mapping(self):
        return {
            "DINING": self.dining,
            "KITCHEN": self.kitchen,
        }


@dataclass(frozen=True)
class FeasibilityEnvelope:
    """
    Generic symmetric per-zone actuator envelope.

    The SAME test is applied to Dining/Kitchen and heating/cooling. The sign of
    delta-T is not restricted.

    These defaults reflect the moderate experimentally exercised actuator
    region. They are configurable and may be refined when the final MPC
    contract is frozen.
    """

    min_flow_fraction_of_design: float = 0.50
    max_flow_fraction_of_design: float = 0.80
    min_abs_supply_minus_zone_temperature_c: float = 4.0
    max_abs_supply_minus_zone_temperature_c: float = 8.0

    def __post_init__(self):
        if not (
            0.0
            < self.min_flow_fraction_of_design
            <= self.max_flow_fraction_of_design
        ):
            raise ValueError("Invalid flow-fraction feasibility envelope.")
        if not (
            0.0
            <= self.min_abs_supply_minus_zone_temperature_c
            <= self.max_abs_supply_minus_zone_temperature_c
        ):
            raise ValueError("Invalid delta-T feasibility envelope.")


@dataclass(frozen=True)
class FeasibilityDecision:
    zone_token: str
    received_mode: str
    effective_control_mode: str
    feasible: bool
    fallback_applied: bool
    reason: str
    flow_fraction_of_design: Optional[float]
    delta_t_star_c: Optional[float]


class FeasibilitySupervisor:
    """
    Generic per-zone supervisor.

    - Explicit NATIVE request -> native requested.
    - OVERRIDE inside envelope -> override.
    - OVERRIDE outside envelope -> native fallback for that zone only.

    There are no zone-specific or mode-specific exclusions.
    """

    def __init__(
        self,
        envelope: FeasibilityEnvelope = FeasibilityEnvelope(),
    ) -> None:
        self.envelope = envelope

    def evaluate(
        self,
        *,
        zone_token: str,
        command: PhysicalZoneCommand,
        current_zone_temperature_c: float,
        design_max_mass_flow_kg_s: float,
    ) -> FeasibilityDecision:
        command.validate()

        if command.mode == CommandMode.NATIVE:
            return FeasibilityDecision(
                zone_token=zone_token,
                received_mode=command.mode.value,
                effective_control_mode=(
                    EffectiveControlMode.NATIVE_REQUESTED.value
                ),
                feasible=True,
                fallback_applied=False,
                reason="native_requested",
                flow_fraction_of_design=None,
                delta_t_star_c=None,
            )

        if design_max_mass_flow_kg_s <= 0:
            raise ValueError(
                f"{zone_token}: design_max_mass_flow_kg_s must be positive."
            )

        assert command.mass_flow_kg_s is not None
        assert command.supply_air_temperature_c is not None

        flow_fraction = (
            command.mass_flow_kg_s
            / design_max_mass_flow_kg_s
        )
        delta_t = (
            command.supply_air_temperature_c
            - current_zone_temperature_c
        )
        abs_delta_t = abs(delta_t)

        e = self.envelope
        reasons = []

        if not (
            e.min_flow_fraction_of_design
            <= flow_fraction
            <= e.max_flow_fraction_of_design
        ):
            reasons.append(
                "flow_fraction_of_design="
                f"{flow_fraction:.6g} outside "
                f"[{e.min_flow_fraction_of_design}, "
                f"{e.max_flow_fraction_of_design}]"
            )

        if not (
            e.min_abs_supply_minus_zone_temperature_c
            <= abs_delta_t
            <= e.max_abs_supply_minus_zone_temperature_c
        ):
            reasons.append(
                "|T_sa*-T_zone|="
                f"{abs_delta_t:.6g} C outside "
                f"[{e.min_abs_supply_minus_zone_temperature_c}, "
                f"{e.max_abs_supply_minus_zone_temperature_c}] C"
            )

        if reasons:
            return FeasibilityDecision(
                zone_token=zone_token,
                received_mode=command.mode.value,
                effective_control_mode=(
                    EffectiveControlMode.NATIVE_FALLBACK.value
                ),
                feasible=False,
                fallback_applied=True,
                reason="; ".join(reasons),
                flow_fraction_of_design=flow_fraction,
                delta_t_star_c=delta_t,
            )

        return FeasibilityDecision(
            zone_token=zone_token,
            received_mode=command.mode.value,
            effective_control_mode=EffectiveControlMode.OVERRIDE.value,
            feasible=True,
            fallback_applied=False,
            reason="inside_configured_feasibility_envelope",
            flow_fraction_of_design=flow_fraction,
            delta_t_star_c=delta_t,
        )


@dataclass(frozen=True)
class TransformedZoneCommand:
    zone_token: str
    received_mode: str
    effective_control_mode: str
    feasible: bool
    fallback_applied: bool
    feasibility_reason: str
    flow_fraction_of_design: Optional[float]
    received_mass_flow_kg_s: Optional[float]
    received_supply_air_temperature_c: Optional[float]
    transform_zone_temperature_c: Optional[float]
    delta_t_star_c: Optional[float]
    fan_actuator_command_kg_s: Optional[float]
    sensible_load_request_w: Optional[float]


@dataclass(frozen=True)
class ActuatorTransform:
    """
    Same physical->EnergyPlus transformation for Dining and Kitchen:

        a_fan = 0.5 * m_dot_star
        a_q   = m_dot_star * cp * (T_sa_star - T_zone)

    The supervisor decides whether an OVERRIDE is applied. This transform never
    contains zone-specific or heating/cooling-specific logic.
    """

    cp_air_j_kgk: float = 1006.0
    fan_command_factor: float = 0.5
    control_interval_seconds: int = 300

    def transform(
        self,
        *,
        zone_token: str,
        command: PhysicalZoneCommand,
        current_zone_temperature_c: float,
        decision: FeasibilityDecision,
    ) -> TransformedZoneCommand:
        command.validate()

        if (
            command.mode == CommandMode.NATIVE
            or decision.fallback_applied
        ):
            return TransformedZoneCommand(
                zone_token=zone_token,
                received_mode=command.mode.value,
                effective_control_mode=decision.effective_control_mode,
                feasible=decision.feasible,
                fallback_applied=decision.fallback_applied,
                feasibility_reason=decision.reason,
                flow_fraction_of_design=decision.flow_fraction_of_design,
                received_mass_flow_kg_s=command.mass_flow_kg_s,
                received_supply_air_temperature_c=(
                    command.supply_air_temperature_c
                ),
                transform_zone_temperature_c=current_zone_temperature_c,
                delta_t_star_c=decision.delta_t_star_c,
                fan_actuator_command_kg_s=None,
                sensible_load_request_w=None,
            )

        assert command.mass_flow_kg_s is not None
        assert command.supply_air_temperature_c is not None

        delta_t = (
            command.supply_air_temperature_c
            - current_zone_temperature_c
        )

        return TransformedZoneCommand(
            zone_token=zone_token,
            received_mode=command.mode.value,
            effective_control_mode=decision.effective_control_mode,
            feasible=decision.feasible,
            fallback_applied=decision.fallback_applied,
            feasibility_reason=decision.reason,
            flow_fraction_of_design=decision.flow_fraction_of_design,
            received_mass_flow_kg_s=command.mass_flow_kg_s,
            received_supply_air_temperature_c=command.supply_air_temperature_c,
            transform_zone_temperature_c=current_zone_temperature_c,
            delta_t_star_c=delta_t,
            fan_actuator_command_kg_s=(
                self.fan_command_factor
                * command.mass_flow_kg_s
            ),
            sensible_load_request_w=(
                command.mass_flow_kg_s
                * self.cp_air_j_kgk
                * delta_t
            ),
        )
