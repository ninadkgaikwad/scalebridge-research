"""Layout for Simulator Results."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Simulator Results shell layout."""
    return build_module_placeholder(
        page_id="results_explorer",
        subpage_id="simulator_results",
        title="Simulator Results",
        description='The Simulator Results workspace within Results Explorer. This shell module is ready for integration with the authoritative scientific implementation.',
    )
