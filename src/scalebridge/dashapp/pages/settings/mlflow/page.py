"""Current MLflow information."""
from dash import html
import dash_bootstrap_components as dbc

from ....services.system import mlflow_snapshot, mlflow_variables
from ..components import settings_heading
from ..live_components import table, card, notice


def build_layout():
    s = mlflow_snapshot()
    ui_address = s.get("Local UI Address", "Not set")
    has_http_ui = isinstance(ui_address, str) and ui_address.startswith(("http://", "https://"))

    actions = html.Div(
        [
            dbc.Button(
                "Open MLflow UI",
                href=ui_address if has_http_ui else None,
                external_link=has_http_ui,
                target="_blank" if has_http_ui else None,
                disabled=not has_http_ui,
                color="primary",
                className="me-2",
            ),
            html.Small(
                (
                    "A web tracking URI was not detected in the current process."
                    if not has_http_ui
                    else f"Launches {ui_address} in a new tab."
                ),
                className="text-muted",
            ),
        ],
        className="d-flex flex-wrap align-items-center gap-2",
    )

    return html.Div(
        [
            settings_heading(
                "MLflow",
                "Current MLflow package, variables, and derived artifact/export paths.",
                "bi-activity",
                "subpage.settings.mlflow",
            ),
            notice(s["Generated At"]),
            card("MLflow Web UI", actions),
            card("Resolved MLflow Information", table(s)),
            card(
                "MLflow Environment Variables",
                table(mlflow_variables(), ("name", "value")),
                "Known and discovered MLFLOW_* variables are shown; missing values show Not set.",
            ),
            card(
                "Operational Policy",
                html.P(
                    "This page does not start, stop, modify, or validate an MLflow server. It only reports what is visible to the running Dash process.",
                    className="mb-0",
                ),
            ),
        ],
        className="page-content-container settings-page",
    )
