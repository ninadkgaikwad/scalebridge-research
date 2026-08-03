"""Layout for OpenDSS Simulator."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the OpenDSS Simulator shell layout."""
    return build_module_placeholder(
        page_id="simulators",
        subpage_id="opendsssim",
        title="OpenDSS Simulator",
        description='The OpenDSS Simulator workspace within Simulators. This shell module is ready for integration with the authoritative scientific implementation.',
    )
