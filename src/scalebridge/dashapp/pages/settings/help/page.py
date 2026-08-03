"""Settings Help and About page."""

from dash import html
import dash_bootstrap_components as dbc

from ....constants import APP_NAME, APP_SUBTITLE
from ..components import settings_heading
from ..live_components import card


def build_layout():
    return html.Div(
        [
            settings_heading(
                "Help and About",
                "Application identity, author information, and guidance for the Settings workspace.",
                "bi-question-circle",
                "subpage.settings.help",
            ),
            card(
                "About BGIRS",
                html.Div(
                    [
                        html.P(
                            "Building-Grid Intelligence Research Studio (BGIRS) is the interactive research interface for running, monitoring, and exploring the ScaleBridge workflow."
                        ),
                        html.Ul(
                            [
                                html.Li("Data Pipeline pages focus on workflow-specific execution and outputs."),
                                html.Li("Results Explorer is the main place to compare experiments and create plots/tables."),
                                html.Li("Settings describes the current machine and application environment."),
                                html.Li("Visualization stores browser-level display preferences."),
                            ]
                        ),
                    ]
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            "Application",
                            html.Div(
                                [
                                    html.P([html.Strong("Name: "), APP_NAME]),
                                    html.P([html.Strong("Subtitle: "), APP_SUBTITLE]),
                                    html.P([html.Strong("Mode: "), "Read-only current-machine Settings"]),
                                ]
                            ),
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        card(
                            "Author and Links",
                            html.Div(
                                [
                                    html.P("Ninad Kiran Gaikwad — PhD Candidate, SCALE Lab, EECS, WSU"),
                                    html.Ul(
                                        [
                                            html.Li(html.A("WSU EECS", href="https://school.eecs.wsu.edu/", target="_blank")),
                                            html.Li(html.A("Washington State University", href="https://wsu.edu/", target="_blank")),
                                            html.Li(html.A("Advisor: Dr. Anamika Dubey", href="https://anamika-dubey.github.io/", target="_blank")),
                                            html.Li(html.A("Ninad Gaikwad Website", href="https://ninadkgaikwad.github.io/", target="_blank")),
                                        ]
                                    ),
                                ]
                            ),
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),
            card(
                "Settings Guidance",
                html.Ul(
                    [
                        html.Li("Paths shows the active repository and Data/ScaleBridge paths detected on this machine."),
                        html.Li("Current Machine shows host, OS, Python, GPU/CUDA, Git, and external tools visible to the current process."),
                        html.Li("Environment shows installed package versions, import visibility, and known environment variables."),
                        html.Li("MLflow shows current MLflow variables and derived local/export paths."),
                        html.Li("Visualization controls affect only browser-persisted display preferences."),
                    ]
                ),
            ),
        ],
        className="page-content-container settings-page",
    )
