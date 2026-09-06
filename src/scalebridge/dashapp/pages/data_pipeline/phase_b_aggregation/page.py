"""Phase B Aggregation workspace with the locked three-tab structure."""

from dash import dcc, html

from ....components.help import help_button
from .campaign_builder import build_layout as build_builder
from .execution import build_layout as build_execution
from .results import build_layout as build_results


_TABS = {
    "campaign_builder": build_builder,
    "execution": build_execution,
    "results": build_results,
}


def get_tab_builder(value: str):
    """Return the builder for one Phase B workspace tab."""
    return _TABS.get(value, build_builder)


def build_layout():
    """Build the Phase B Aggregation workspace."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "Phase B: Aggregation",
                                className="page-subtitle",
                            ),
                            help_button(
                                "subpage.data_pipeline.phase_b_aggregation",
                                compact=False,
                            ),
                        ],
                        className="title-with-help",
                    ),
                    html.P(
                        "Build, execute, and inspect general Aggregation campaigns "
                        "derived from successful Phase A Generation campaigns.",
                        className="page-description",
                    ),
                ],
                className="aggregation-workspace-heading",
            ),
            dcc.Tabs(
                id="aggregation-workspace-tabs",
                value="campaign_builder",
                persistence=True,
                persistence_type="session",
                children=[
                    dcc.Tab(label="Campaign Builder", value="campaign_builder"),
                    dcc.Tab(label="Execution", value="execution"),
                    dcc.Tab(label="Results", value="results"),
                ],
            ),
            html.Div(
                build_builder(),
                id="aggregation-workspace-content",
            ),
        ],
        className="aggregation-workspace",
    )
