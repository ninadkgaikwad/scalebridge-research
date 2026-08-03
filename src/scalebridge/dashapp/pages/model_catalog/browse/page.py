"""Layout for Browse Models."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Browse Models shell layout."""
    return build_module_placeholder(
        page_id="model_catalog",
        subpage_id="browse",
        title="Browse Models",
        description='The Browse Models workspace within Model Catalog. This shell module is ready for integration with the authoritative scientific implementation.',
    )
