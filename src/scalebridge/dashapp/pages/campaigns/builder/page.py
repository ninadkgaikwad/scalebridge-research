"""Layout for Campaign Builder."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Campaign Builder shell layout."""
    return build_module_placeholder(
        page_id="campaigns",
        subpage_id="builder",
        title="Campaign Builder",
        description='The Campaign Builder workspace within Campaigns. This shell module is ready for integration with the authoritative scientific implementation.',
    )
