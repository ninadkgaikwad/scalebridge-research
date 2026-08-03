"""Layout for Matrix Preview."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Matrix Preview shell layout."""
    return build_module_placeholder(
        page_id="campaigns",
        subpage_id="matrix_preview",
        title="Matrix Preview",
        description='The Matrix Preview workspace within Campaigns. This shell module is ready for integration with the authoritative scientific implementation.',
    )
