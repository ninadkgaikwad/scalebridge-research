"""Layout for Overview."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Overview shell layout."""
    return build_module_placeholder(
        page_id="home",
        subpage_id="overview",
        title="Overview",
        description='The Overview workspace within Home. This shell module is ready for integration with the authoritative scientific implementation.',
    )
