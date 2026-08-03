"""Layout for Optimization-Based Modeling."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Optimization-Based Modeling shell layout."""
    return build_module_placeholder(
        page_id="thermal_modeling",
        subpage_id="optimization",
        title="Optimization-Based Modeling",
        description='The Optimization-Based Modeling workspace within Thermal Modeling. This shell module is ready for integration with the authoritative scientific implementation.',
    )
