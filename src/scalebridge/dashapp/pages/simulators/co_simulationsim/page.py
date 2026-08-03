"""Layout for Co-Simulation Platform."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Co-Simulation Platform shell layout."""
    return build_module_placeholder(
        page_id="simulators",
        subpage_id="co_simulationsim",
        title="Co-Simulation Platform",
        description='The Co-Simulation Platform workspace within Simulators. This shell module is ready for integration with the authoritative scientific implementation.',
    )
