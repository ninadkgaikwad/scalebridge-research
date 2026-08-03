"""Layout for Smart Community Simulator."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Smart Community Simulator shell layout."""
    return build_module_placeholder(
        page_id="simulators",
        subpage_id="smartcommunitysim",
        title="Smart Community Simulator",
        description='The Smart Community Simulator workspace within Simulators. This shell module is ready for integration with the authoritative scientific implementation.',
    )
