"""Current Python environment, packages, and environment variables."""
from dash import html
from ....services.system import python_snapshot, package_snapshot, environment_variables
from ..components import settings_heading
from ..live_components import table, card, notice

def build_layout():
    p = python_snapshot()
    groups = [
        html.Div(
            card(
                f"{g['title']} · {g['classification']}",
                table(g["packages"], ("component", "installed_version", "importable")),
            ),
            className="settings-package-card-wrap",
        )
        for g in package_snapshot()
    ]

    return html.Div(
        [
            settings_heading(
                "Environment",
                "Live Python interpreter, package inventory, numerical backend, and known environment variables.",
                "bi-boxes",
                "subpage.settings.environments",
            ),
            notice(p["Generated At"]),
            card("Current Python Environment", table(p)),
            card(
                "Known Environment Variables",
                table(environment_variables(), ("name", "value", "description")),
                "Known ScaleBridge, MLflow, Python, GPU, Git, OS, and SLURM variables are listed. Missing values show Not set.",
            ),
            html.Div(groups, className="settings-package-sections"),
        ],
        className="page-content-container settings-page",
    )
