"""Layout for Table Builder."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Table Builder shell layout."""
    return build_module_placeholder(
        page_id="publication_studio",
        subpage_id="table_builder",
        title="Table Builder",
        description='The Table Builder workspace within Publication Studio. This shell module is ready for integration with the authoritative scientific implementation.',
    )
