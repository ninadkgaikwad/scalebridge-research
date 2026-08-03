"""Layout for Single Experiment."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Single Experiment shell layout."""
    return build_module_placeholder(
        page_id="results_explorer",
        subpage_id="single_experiment",
        title="Single Experiment",
        description='The Single Experiment workspace within Results Explorer. This shell module is ready for integration with the authoritative scientific implementation.',
    )
