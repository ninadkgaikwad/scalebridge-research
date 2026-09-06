import csv
import json
import tempfile
import unittest
from pathlib import Path

from pinode_epsr.simulator.history import BroadSimulatorHistory


class HistorySchemaTests(unittest.TestCase):
    def test_history_writes_all_layers(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            h = BroadSimulatorHistory(
                run_dir=run_dir,
                zone_tokens=["DINING", "KITCHEN"],
                environment_aliases=["outdoor_drybulb_c"],
                zone_signal_aliases=[
                    "zone_temperature_c",
                    "zone_supply_temperature_c",
                    "zone_supply_mass_flow_kg_s",
                    "fan_electric_power_w",
                    "heating_coil_rate_w",
                    "cooling_coil_total_rate_w",
                ],
            )

            meta = {
                "year": 2001,
                "month": 8,
                "day": 3,
                "current_time_hour": 17.0,
                "current_sim_time_hour": 1.0,
            }
            h.begin_control_step(
                control_step_index=0,
                meta=meta,
                decision_meta={
                    "DINING": {
                        "received_mode": "override",
                        "effective_control_mode": "override",
                        "feasible": True,
                        "fallback_applied": False,
                        "feasibility_reason": "inside_configured_feasibility_envelope",
                    },
                    "KITCHEN": {
                        "received_mode": "override",
                        "effective_control_mode": "native_fallback",
                        "feasible": False,
                        "fallback_applied": True,
                        "feasibility_reason": "example_outside",
                    },
                },
            )

            h.record_received_command({
                "control_step_index": 0,
                **meta,
                "zone_token": "DINING",
                "received_mode": "override",
                "effective_control_mode": "override",
                "feasible": True,
                "fallback_applied": False,
                "feasibility_reason": "inside_configured_feasibility_envelope",
                "flow_fraction_of_design": 0.65,
                "delta_t_star_c": -6.0,
                "received_mass_flow_kg_s": 1.0,
                "received_supply_air_temperature_c": 16.0,
            })

            for zone in ["DINING", "KITCHEN"]:
                h.record_system_row(row={
                    "control_step_index": 0,
                    "system_substep_index": 1,
                    **meta,
                    "system_timestep_seconds": 300.0,
                    "zone_token": zone,
                    "received_mode": "override",
                    "effective_control_mode": "override",
                    "feasible": True,
                    "fallback_applied": False,
                    "feasibility_reason": "inside_configured_feasibility_envelope",
                    "flow_fraction_of_design": 0.65,
                    "received_mass_flow_kg_s": 1.0,
                    "received_supply_air_temperature_c": 16.0,
                    "transform_zone_temperature_c": 22.0,
                    "delta_t_star_c": -6.0,
                    "transformed_fan_command_kg_s": 0.5,
                    "transformed_sensible_load_request_w": -6036.0,
                    "fan_override_active": True,
                    "load_override_active": True,
                    "fan_actuator_readback_kg_s": 0.5,
                    "load_actuator_readback_w": -6036.0,
                    "fan_actuator_api_value_raw": 0.5,
                    "load_actuator_api_value_raw": -6036.0,
                    "fan_design_max_mass_flow_kg_s": 1.5,
                    "unitary_design_heating_capacity_w": 50000.0,
                    "unitary_design_cooling_capacity_w": 35000.0,
                    "outdoor_drybulb_c": 30.0,
                    "zone_temperature_c": 22.0,
                    "zone_supply_temperature_c": 16.0,
                    "zone_supply_mass_flow_kg_s": 1.0,
                    "fan_electric_power_w": 500.0,
                    "heating_coil_rate_w": 0.0,
                    "cooling_coil_total_rate_w": 7000.0,
                    "q_zone_interface_w": -6036.0,
                })

            summaries = h.finalize_control_step(
                control_step_index=0,
                nested_payload={"example": True},
                end_boundary_meta={
                    "year": 2001,
                    "month": 8,
                    "day": 3,
                    "current_time_hour": 17.0833333333,
                    "current_sim_time_hour": 1.0833333333,
                },
                nominal_control_interval_seconds=300.0,
            )
            h.write_signal_catalog({"ok": True})
            h.write_api_registry([
                {
                    "what": "OutputVariable",
                    "name": "Zone Air Temperature",
                    "key": "Dining",
                }
            ])
            h.close()

            self.assertIn("DINING", summaries)
            self.assertAlmostEqual(
                summaries["DINING"][
                    "mean__fan_design_max_mass_flow_kg_s"
                ],
                1.5,
            )
            self.assertAlmostEqual(
                summaries["DINING"]["nominal_control_interval_seconds"],
                300.0,
            )
            self.assertGreater(
                summaries["DINING"]["end_sim_time_hour"],
                summaries["DINING"]["start_sim_time_hour"],
            )

            expected = [
                "system_timestep_zone_history.csv",
                "control_step_zone_history.csv",
                "received_command_history.csv",
                "control_steps.jsonl",
                "signal_catalog.json",
                "api_exchange_registry.csv",
            ]
            for name in expected:
                self.assertTrue(
                    (run_dir / "history" / name).exists(),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
