"""Layout for Parameter Explorer."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Parameter Explorer shell layout."""
    return build_module_placeholder(
        page_id="results_explorer",
        subpage_id="parameters",
        title="Parameter Explorer",
        description='The Parameter Explorer workspace within Results Explorer. This shell module is ready for integration with the authoritative scientific implementation.',
    )
