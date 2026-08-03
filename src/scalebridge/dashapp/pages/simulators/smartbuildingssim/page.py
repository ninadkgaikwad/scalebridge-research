"""Layout for Smart Buildings Simulator."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Smart Buildings Simulator shell layout."""
    return build_module_placeholder(
        page_id="simulators",
        subpage_id="smartbuildingssim",
        title="Smart Buildings Simulator",
        description='The Smart Buildings Simulator workspace within Simulators. This shell module is ready for integration with the authoritative scientific implementation.',
    )
