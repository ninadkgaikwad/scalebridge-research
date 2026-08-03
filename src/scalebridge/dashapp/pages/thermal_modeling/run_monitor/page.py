"""Layout for Run Monitor."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Run Monitor shell layout."""
    return build_module_placeholder(
        page_id="thermal_modeling",
        subpage_id="run_monitor",
        title="Run Monitor",
        description='The Run Monitor workspace within Thermal Modeling. This shell module is ready for integration with the authoritative scientific implementation.',
    )
