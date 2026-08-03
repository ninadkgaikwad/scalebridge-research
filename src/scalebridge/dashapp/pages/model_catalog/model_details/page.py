"""Layout for Model Details."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Model Details shell layout."""
    return build_module_placeholder(
        page_id="model_catalog",
        subpage_id="model_details",
        title="Model Details",
        description='The Model Details workspace within Model Catalog. This shell module is ready for integration with the authoritative scientific implementation.',
    )
