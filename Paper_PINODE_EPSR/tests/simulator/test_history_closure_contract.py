import inspect
import unittest

from pinode_epsr.simulator.energyplus_simulator import EnergyPlusSimulator
from pinode_epsr.simulator.history import COMMAND_COLUMNS


class HistoryClosureContractTests(unittest.TestCase):
    def test_effective_and_raw_actuator_history_are_distinct(self):
        self.assertIn("fan_actuator_readback_kg_s", COMMAND_COLUMNS)
        self.assertIn("load_actuator_readback_w", COMMAND_COLUMNS)
        self.assertIn("fan_actuator_api_value_raw", COMMAND_COLUMNS)
        self.assertIn("load_actuator_api_value_raw", COMMAND_COLUMNS)
        self.assertIn("fan_override_active", COMMAND_COLUMNS)
        self.assertIn("load_override_active", COMMAND_COLUMNS)

    def test_simulator_has_control_boundary_synchronization(self):
        source = inspect.getsource(EnergyPlusSimulator)
        self.assertIn("_control_boundary_synchronized", source)
        self.assertIn("if not self._control_boundary_synchronized", source)


if __name__ == "__main__":
    unittest.main()
