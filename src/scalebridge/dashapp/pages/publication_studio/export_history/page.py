"""Layout for Export History."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Export History shell layout."""
    return build_module_placeholder(
        page_id="publication_studio",
        subpage_id="export_history",
        title="Export History",
        description='The Export History workspace within Publication Studio. This shell module is ready for integration with the authoritative scientific implementation.',
    )
