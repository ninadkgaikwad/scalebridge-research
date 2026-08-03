"""Layout for Diagnostics Explorer."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Diagnostics Explorer shell layout."""
    return build_module_placeholder(
        page_id="results_explorer",
        subpage_id="diagnostics",
        title="Diagnostics Explorer",
        description='The Diagnostics Explorer workspace within Results Explorer. This shell module is ready for integration with the authoritative scientific implementation.',
    )
