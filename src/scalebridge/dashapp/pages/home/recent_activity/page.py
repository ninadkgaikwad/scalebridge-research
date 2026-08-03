"""Layout for Recent Activity."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Recent Activity shell layout."""
    return build_module_placeholder(
        page_id="home",
        subpage_id="recent_activity",
        title="Recent Activity",
        description='The Recent Activity workspace within Home. This shell module is ready for integration with the authoritative scientific implementation.',
    )
