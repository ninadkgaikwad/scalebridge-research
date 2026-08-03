"""Layout for Model Validation."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Model Validation shell layout."""
    return build_module_placeholder(
        page_id="model_catalog",
        subpage_id="validation",
        title="Model Validation",
        description='The Model Validation workspace within Model Catalog. This shell module is ready for integration with the authoritative scientific implementation.',
    )
