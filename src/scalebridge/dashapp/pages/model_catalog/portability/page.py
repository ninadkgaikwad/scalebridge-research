"""Layout for Model Portability."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Model Portability shell layout."""
    return build_module_placeholder(
        page_id="model_catalog",
        subpage_id="portability",
        title="Model Portability",
        description='The Model Portability workspace within Model Catalog. This shell module is ready for integration with the authoritative scientific implementation.',
    )
