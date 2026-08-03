"""Layout for Metrics Explorer."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Metrics Explorer shell layout."""
    return build_module_placeholder(
        page_id="results_explorer",
        subpage_id="metrics",
        title="Metrics Explorer",
        description='The Metrics Explorer workspace within Results Explorer. This shell module is ready for integration with the authoritative scientific implementation.',
    )
