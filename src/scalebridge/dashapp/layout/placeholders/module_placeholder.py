"""Structured placeholder for not-yet-integrated modules."""

from dash import html
import dash_bootstrap_components as dbc

from ...components.help import help_button


def build_module_placeholder(
    *,
    page_id: str,
    subpage_id: str,
    title: str,
    description: str,
    status: str = "Shell Implemented · Scientific Integration Pending",
):
    """Build a professional placeholder for a modular subpage."""
    help_id = f"subpage.{page_id}.{subpage_id}"
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        html.I(className="bi bi-braces-asterisk"),
                                        className="placeholder-icon",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H2(title, className="page-subtitle"),
                                                    help_button(help_id),
                                                ],
                                                className="title-with-help",
                                            ),
                                            html.P(description, className="page-description"),
                                        ]
                                    ),
                                ],
                                className="placeholder-heading",
                            )
                        ],
                        width=12,
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Implementation Status"),
                                dbc.CardBody(
                                    [
                                        dbc.Badge(
                                            status,
                                            color="info",
                                            className="status-badge",
                                        ),
                                        html.P(
                                            "The navigation, routing, theme, help, and modular "
                                            "page contract are available. Module-specific services, "
                                            "adapters, schemas, results, and execution controls will "
                                            "be connected after the authoritative scientific workflow "
                                            "is finalized.",
                                            className="mt-3 mb-0",
                                        ),
                                    ]
                                ),
                            ],
                            className="studio-card",
                        ),
                        lg=4,
                        md=6,
                        xs=12,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Expected Integration"),
                                dbc.CardBody(
                                    html.Ul(
                                        [
                                            html.Li("Authoritative module service"),
                                            html.Li("Artifact and result adapter"),
                                            html.Li("Validated data schemas"),
                                            html.Li("Interactive plots and tables"),
                                            html.Li("Execution and monitoring controls"),
                                            html.Li("Complete contextual-help coverage"),
                                        ],
                                        className="mb-0",
                                    )
                                ),
                            ],
                            className="studio-card",
                        ),
                        lg=4,
                        md=6,
                        xs=12,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Completion Standard"),
                                dbc.CardBody(
                                    html.P(
                                        "This subpage is complete only after scientific behavior, "
                                        "validation, provenance, tests, and contextual help for every "
                                        "significant input, output, plot, table, status, warning, and "
                                        "error have been implemented.",
                                        className="mb-0",
                                    )
                                ),
                            ],
                            className="studio-card",
                        ),
                        lg=4,
                        md=12,
                        xs=12,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="page-content-container",
    )
