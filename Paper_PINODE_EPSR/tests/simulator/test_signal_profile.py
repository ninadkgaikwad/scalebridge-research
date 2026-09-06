import unittest

from pinode_epsr.simulator.restaurant_fastfood import (
    restaurant_fastfood_signal_specs,
)


class SignalProfileTests(unittest.TestCase):
    def test_full_air_path_present_both_zones(self):
        specs = restaurant_fastfood_signal_specs()

        for zone in ["DINING", "KITCHEN"]:
            aliases = {s.alias for s in specs[zone]}
            for prefix in [
                "return",
                "mixed",
                "cool_out",
                "heat_out",
                "supply_outlet",
                "zone_supply",
            ]:
                self.assertIn(
                    f"{prefix}_temperature_c",
                    aliases,
                )
                self.assertIn(
                    f"{prefix}_mass_flow_kg_s",
                    aliases,
                )
                self.assertIn(
                    f"{prefix}_humidity_ratio",
                    aliases,
                )

    def test_command_intermediate_hvac_signals_present(self):
        specs = restaurant_fastfood_signal_specs()
        aliases = {s.alias for s in specs["DINING"]}

        for required in [
            "fan_mass_flow_kg_s",
            "fan_electric_power_w",
            "heating_coil_rate_w",
            "cooling_coil_total_rate_w",
            "zone_temperature_c",
        ]:
            self.assertIn(required, aliases)


if __name__ == "__main__":
    unittest.main()
