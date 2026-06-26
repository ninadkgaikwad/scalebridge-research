"""EnergyPlus simulation execution package.

The runner in this package executes one prepared IDF/EPW pair and reports
process-level diagnostics. Higher-level generation code will later prepare the
IDF, manage run manifests, and parse canonical outputs around this boundary.
"""

from scalebridge.integration.energyplus.manifests.models import GenerationResult
from scalebridge.integration.energyplus.simulation.runner import (
    EnergyPlusExecutionError,
    EnergyPlusInputError,
    EnergyPlusRunResult,
    EnergyPlusRunner,
    EnergyPlusRunnerError,
    OpyplusNotInstalledError,
    normalize_simulation_status,
    parse_energyplus_error_summary,
)

__all__ = [
    "EnergyPlusExecutionError",
    "EnergyPlusInputError",
    "EnergyPlusRunResult",
    "EnergyPlusRunner",
    "EnergyPlusRunnerError",
    "GenerationResult",
    "OpyplusNotInstalledError",
    "normalize_simulation_status",
    "parse_energyplus_error_summary",
]
