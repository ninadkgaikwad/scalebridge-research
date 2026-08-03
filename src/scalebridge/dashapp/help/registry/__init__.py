"""Central contextual-help registry."""

from .shell_help import HELP_ENTRIES, get_help_entry
from .settings_help import SETTINGS_HELP_ENTRIES
from .profile_manager_help import PROFILE_MANAGER_HELP_ENTRIES
from .generation_help import GENERATION_HELP_ENTRIES

HELP_ENTRIES.update(SETTINGS_HELP_ENTRIES)
HELP_ENTRIES.update(PROFILE_MANAGER_HELP_ENTRIES)
HELP_ENTRIES.update(GENERATION_HELP_ENTRIES)

__all__ = [
    "HELP_ENTRIES",
    "PROFILE_MANAGER_HELP_ENTRIES",
    "SETTINGS_HELP_ENTRIES",
    "GENERATION_HELP_ENTRIES",
    "get_help_entry",
]
