"""Layout for ANN / Sequence Models."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the ANN / Sequence Models shell layout."""
    return build_module_placeholder(
        page_id="thermal_modeling",
        subpage_id="ann_sequence_models",
        title="ANN / Sequence Models",
        description='The ANN / Sequence Models workspace within Thermal Modeling. This shell module is ready for integration with the authoritative scientific implementation.',
    )
