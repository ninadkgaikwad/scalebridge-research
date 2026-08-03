"""Layout for Bayesian Inference."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Bayesian Inference shell layout."""
    return build_module_placeholder(
        page_id="thermal_modeling",
        subpage_id="bayesian_inference",
        title="Bayesian Inference",
        description='The Bayesian Inference workspace within Thermal Modeling. This shell module is ready for integration with the authoritative scientific implementation.',
    )
