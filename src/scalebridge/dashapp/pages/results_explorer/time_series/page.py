"""Layout for Time-Series Explorer."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Time-Series Explorer shell layout."""
    return build_module_placeholder(
        page_id="results_explorer",
        subpage_id="time_series",
        title="Time-Series Explorer",
        description='The Time-Series Explorer workspace within Results Explorer. This shell module is ready for integration with the authoritative scientific implementation.',
    )
