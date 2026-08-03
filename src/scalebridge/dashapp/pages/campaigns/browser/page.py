"""Layout for Campaign Browser."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Campaign Browser shell layout."""
    return build_module_placeholder(
        page_id="campaigns",
        subpage_id="browser",
        title="Campaign Browser",
        description='The Campaign Browser workspace within Campaigns. This shell module is ready for integration with the authoritative scientific implementation.',
    )
