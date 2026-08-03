"""Layout for Scientific Machine Learning."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Scientific Machine Learning shell layout."""
    return build_module_placeholder(
        page_id="thermal_modeling",
        subpage_id="scientific_ml",
        title="Scientific Machine Learning",
        description='The Scientific Machine Learning workspace within Thermal Modeling. This shell module is ready for integration with the authoritative scientific implementation.',
    )
