"""Layout for Phase B: Aggregation."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Phase B: Aggregation shell layout."""
    return build_module_placeholder(
        page_id="data_pipeline",
        subpage_id="phase_b_aggregation",
        title="Phase B: Aggregation",
        description='The Phase B: Aggregation workspace within Data Pipeline. This shell module is ready for integration with the authoritative scientific implementation.',
    )
