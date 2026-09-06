import unittest

from pinode_epsr.simulator.contracts import (
    ActuatorTransform,
    CommandMode,
    EffectiveControlMode,
    FeasibilityEnvelope,
    FeasibilitySupervisor,
    PhysicalZoneCommand,
    RestaurantFastFoodCommand,
)


class ActuatorTransformTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = FeasibilitySupervisor(
            FeasibilityEnvelope(
                min_flow_fraction_of_design=0.50,
                max_flow_fraction_of_design=0.80,
                min_abs_supply_minus_zone_temperature_c=4.0,
                max_abs_supply_minus_zone_temperature_c=8.0,
            )
        )
        self.transform = ActuatorTransform()

    def decision(self, zone, command, tzone=22.0, max_flow=1.0):
        return self.supervisor.evaluate(
            zone_token=zone,
            command=command,
            current_zone_temperature_c=tzone,
            design_max_mass_flow_kg_s=max_flow,
        )

    def test_four_command_interface(self):
        c = RestaurantFastFoodCommand.four_physical_commands(
            dining_mass_flow_kg_s=0.65,
            dining_supply_air_temperature_c=16.0,
            kitchen_mass_flow_kg_s=0.60,
            kitchen_supply_air_temperature_c=28.0,
        )
        self.assertEqual(c.dining.mode, CommandMode.OVERRIDE)
        self.assertEqual(c.kitchen.mode, CommandMode.OVERRIDE)

    def test_all_four_zone_modes_allowed_inside_envelope(self):
        cases = [
            ("DINING", 16.0),   # cooling
            ("DINING", 28.0),   # heating
            ("KITCHEN", 16.0),  # cooling
            ("KITCHEN", 28.0),  # heating
        ]

        for zone, tsa in cases:
            with self.subTest(zone=zone, tsa=tsa):
                cmd = PhysicalZoneCommand.override(0.65, tsa)
                d = self.decision(zone, cmd)
                self.assertTrue(d.feasible)
                self.assertFalse(d.fallback_applied)
                self.assertEqual(
                    d.effective_control_mode,
                    EffectiveControlMode.OVERRIDE.value,
                )

    def test_identical_transform_dining_and_kitchen_heating(self):
        cmd = PhysicalZoneCommand.override(0.65, 28.0)

        dd = self.decision("DINING", cmd)
        kd = self.decision("KITCHEN", cmd)

        d = self.transform.transform(
            zone_token="DINING",
            command=cmd,
            current_zone_temperature_c=22.0,
            decision=dd,
        )
        k = self.transform.transform(
            zone_token="KITCHEN",
            command=cmd,
            current_zone_temperature_c=22.0,
            decision=kd,
        )

        self.assertAlmostEqual(
            d.fan_actuator_command_kg_s,
            k.fan_actuator_command_kg_s,
        )
        self.assertAlmostEqual(
            d.sensible_load_request_w,
            k.sensible_load_request_w,
        )
        self.assertGreater(k.sensible_load_request_w, 0.0)

    def test_outside_flow_falls_back_symmetrically(self):
        for zone in ["DINING", "KITCHEN"]:
            for tsa in [16.0, 28.0]:
                with self.subTest(zone=zone, tsa=tsa):
                    cmd = PhysicalZoneCommand.override(0.90, tsa)
                    d = self.decision(zone, cmd)
                    self.assertFalse(d.feasible)
                    self.assertTrue(d.fallback_applied)
                    self.assertEqual(
                        d.effective_control_mode,
                        EffectiveControlMode.NATIVE_FALLBACK.value,
                    )

                    x = self.transform.transform(
                        zone_token=zone,
                        command=cmd,
                        current_zone_temperature_c=22.0,
                        decision=d,
                    )
                    self.assertIsNone(
                        x.fan_actuator_command_kg_s
                    )
                    self.assertIsNone(
                        x.sensible_load_request_w
                    )

    def test_outside_delta_t_falls_back_symmetrically(self):
        for zone in ["DINING", "KITCHEN"]:
            # +10 heating and -10 cooling
            for tsa in [12.0, 32.0]:
                with self.subTest(zone=zone, tsa=tsa):
                    cmd = PhysicalZoneCommand.override(0.65, tsa)
                    d = self.decision(zone, cmd)
                    self.assertFalse(d.feasible)
                    self.assertTrue(d.fallback_applied)

    def test_direct_physics_inside_envelope(self):
        cmd = PhysicalZoneCommand.override(0.65, 16.0)
        d = self.decision("DINING", cmd)
        x = self.transform.transform(
            zone_token="DINING",
            command=cmd,
            current_zone_temperature_c=22.0,
            decision=d,
        )
        self.assertAlmostEqual(
            x.fan_actuator_command_kg_s,
            0.325,
        )
        self.assertAlmostEqual(
            x.sensible_load_request_w,
            0.65 * 1006.0 * -6.0,
        )

    def test_explicit_native_is_not_called_fallback(self):
        cmd = PhysicalZoneCommand.native()
        d = self.decision("KITCHEN", cmd)
        self.assertTrue(d.feasible)
        self.assertFalse(d.fallback_applied)
        self.assertEqual(
            d.effective_control_mode,
            EffectiveControlMode.NATIVE_REQUESTED.value,
        )


if __name__ == "__main__":
    unittest.main()
