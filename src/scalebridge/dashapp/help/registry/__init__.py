"""Central contextual-help registry."""

from .shell_help import HELP_ENTRIES, get_help_entry
from .settings_help import SETTINGS_HELP_ENTRIES
from .profile_manager_help import PROFILE_MANAGER_HELP_ENTRIES
from .generation_help import GENERATION_HELP_ENTRIES
from .aggregation_help import AGGREGATION_HELP_ENTRIES
from .heat_input_help import HEAT_INPUT_HELP_ENTRIES
from .phase_d_help import PHASE_D_HELP_ENTRIES

HELP_ENTRIES.update(SETTINGS_HELP_ENTRIES)
HELP_ENTRIES.update(PROFILE_MANAGER_HELP_ENTRIES)
HELP_ENTRIES.update(GENERATION_HELP_ENTRIES)
HELP_ENTRIES.update(AGGREGATION_HELP_ENTRIES)
HELP_ENTRIES.update(HEAT_INPUT_HELP_ENTRIES)
HELP_ENTRIES.update(PHASE_D_HELP_ENTRIES)

__all__ = [
    "HELP_ENTRIES",
    "PROFILE_MANAGER_HELP_ENTRIES",
    "SETTINGS_HELP_ENTRIES",
    "GENERATION_HELP_ENTRIES",
    "AGGREGATION_HELP_ENTRIES",
    "HEAT_INPUT_HELP_ENTRIES",
    "PHASE_D_HELP_ENTRIES",
    "get_help_entry",
]
