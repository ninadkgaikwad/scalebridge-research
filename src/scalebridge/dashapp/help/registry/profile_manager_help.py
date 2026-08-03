"""Machine-profile manager help entries."""

PROFILE_MANAGER_HELP_ENTRIES = {
    "settings.machine.active_profile": {
        "title": "Active Saved Profile",
        "summary": "Machine profile selected for persisted BGIRS configuration.",
        "details": "Activation updates config/active_machine.json. Runtime environment variables continue to override profile values until changed or the process restarts.",
    },
    "settings.machine.profile_count": {
        "title": "Saved Profiles",
        "summary": "Number of active machine profiles stored under config/machines.",
        "details": "Archived profiles are stored separately and are not counted in the active profile selector.",
    },
}
