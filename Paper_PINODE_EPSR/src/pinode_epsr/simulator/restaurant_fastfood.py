from __future__ import annotations

from typing import Dict, List

from .signals import (
    SignalSpec,
    ZoneRuntimeSpec,
    environment_signal_specs,
    zone_signal_specs,
)


def restaurant_fastfood_zone_specs() -> Dict[str, ZoneRuntimeSpec]:
    return {
        "DINING": ZoneRuntimeSpec(
            zone_token="DINING",
            zone_name="Dining",
            unitary_name="PSZ-AC_1:1",
            fan_name="PSZ-AC_1:1_addAQ Fan",
            heating_coil_name="PSZ-AC_1:1_HeatC",
            cooling_coil_name="PSZ-AC_1:1_CoolC DXCoil",
            return_node="PSZ-AC_1:1 Supply Equipment Inlet Node",
            mixed_node="PSZ-AC_1:1_OA-PSZ-AC_1:1_COOLCNODE",
            cool_out_node="PSZ-AC_1:1_COOLC-PSZ-AC_1:1_HEATCNODE",
            heat_out_node="PSZ-AC_1:1_HEATC-PSZ-AC_1:1 FANNODE",
            supply_outlet_node="PSZ-AC_1:1 Supply Equipment Outlet Node",
            zone_supply_node="PSZ-AC_1:1 Zone Equipment Inlet Node",
        ),
        "KITCHEN": ZoneRuntimeSpec(
            zone_token="KITCHEN",
            zone_name="Kitchen",
            unitary_name="PSZ-AC_2:2",
            fan_name="PSZ-AC_2:2_addAQ Fan",
            heating_coil_name="PSZ-AC_2:2_HeatC",
            cooling_coil_name="PSZ-AC_2:2_CoolC DXCoil",
            return_node="PSZ-AC_2:2 Supply Equipment Inlet Node",
            mixed_node="PSZ-AC_2:2_OA-PSZ-AC_2:2_COOLCNODE",
            cool_out_node="PSZ-AC_2:2_COOLC-PSZ-AC_2:2_HEATCNODE",
            heat_out_node="PSZ-AC_2:2_HEATC-PSZ-AC_2:2 FANNODE",
            supply_outlet_node="PSZ-AC_2:2 Supply Equipment Outlet Node",
            zone_supply_node="PSZ-AC_2:2 Zone Equipment Inlet Node",
        ),
    }


def restaurant_fastfood_signal_specs() -> Dict[str, List[SignalSpec]]:
    zones = restaurant_fastfood_zone_specs()
    return {
        "ENVIRONMENT": environment_signal_specs(),
        **{
            token: zone_signal_specs(spec)
            for token, spec in zones.items()
        },
    }
