"""Layout for Campaign Monitor."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Campaign Monitor shell layout."""
    return build_module_placeholder(
        page_id="campaigns",
        subpage_id="monitor",
        title="Campaign Monitor",
        description='The Campaign Monitor workspace within Campaigns. This shell module is ready for integration with the authoritative scientific implementation.',
    )
