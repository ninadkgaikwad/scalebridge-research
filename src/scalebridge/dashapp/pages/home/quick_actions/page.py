"""Layout for Quick Actions."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Quick Actions shell layout."""
    return build_module_placeholder(
        page_id="home",
        subpage_id="quick_actions",
        title="Quick Actions",
        description='The Quick Actions workspace within Home. This shell module is ready for integration with the authoritative scientific implementation.',
    )
