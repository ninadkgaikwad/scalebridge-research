"""Phase D Thermal-Model Data workspace with exactly three tabs."""
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
    return _TABS.get(value, build_builder)


def build_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Phase D: Thermal-Model Data", className="page-subtitle"),
                            help_button("subpage.data_pipeline.phase_d_thermal_model_data", compact=False),
                        ],
                        className="title-with-help",
                    ),
                    html.P(
                        "Configure the general Phase D campaign runner, execute saved definitions, and inspect the time-series thermal-model datasets it creates.",
                        className="page-description",
                    ),
                ],
                className="phase-d-workspace-heading",
            ),
            dcc.Tabs(
                id="phase-d-workspace-tabs",
                value="campaign_builder",
                persistence=True,
                persistence_type="session",
                children=[
                    dcc.Tab(label="Campaign Builder", value="campaign_builder"),
                    dcc.Tab(label="Execution", value="execution"),
                    dcc.Tab(label="Results", value="results"),
                ],
            ),
            html.Div(build_builder(), id="phase-d-workspace-content"),
        ],
        className="phase-d-workspace",
    )
