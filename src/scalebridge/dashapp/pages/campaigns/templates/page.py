"""Layout for Campaign Templates."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Campaign Templates shell layout."""
    return build_module_placeholder(
        page_id="campaigns",
        subpage_id="templates",
        title="Campaign Templates",
        description='The Campaign Templates workspace within Campaigns. This shell module is ready for integration with the authoritative scientific implementation.',
    )
