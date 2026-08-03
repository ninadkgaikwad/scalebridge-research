"""Layout for System Status."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the System Status shell layout."""
    return build_module_placeholder(
        page_id="home",
        subpage_id="system_status",
        title="System Status",
        description='The System Status workspace within Home. This shell module is ready for integration with the authoritative scientific implementation.',
    )
