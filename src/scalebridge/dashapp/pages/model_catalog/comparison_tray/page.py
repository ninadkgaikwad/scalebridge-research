"""Layout for Comparison Tray."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Comparison Tray shell layout."""
    return build_module_placeholder(
        page_id="model_catalog",
        subpage_id="comparison_tray",
        title="Comparison Tray",
        description='The Comparison Tray workspace within Model Catalog. This shell module is ready for integration with the authoritative scientific implementation.',
    )
