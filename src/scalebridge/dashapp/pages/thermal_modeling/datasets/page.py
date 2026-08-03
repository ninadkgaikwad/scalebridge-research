"""Layout for Thermal-Model Datasets."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Thermal-Model Datasets shell layout."""
    return build_module_placeholder(
        page_id="thermal_modeling",
        subpage_id="datasets",
        title="Thermal-Model Datasets",
        description='The Thermal-Model Datasets workspace within Thermal Modeling. This shell module is ready for integration with the authoritative scientific implementation.',
    )
