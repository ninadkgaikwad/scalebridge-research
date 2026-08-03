"""Reusable contextual-help button."""

from dash import html
import dash_bootstrap_components as dbc

from ...help.registry import get_help_entry


def help_button(help_id: str, *, compact: bool = True):
    """Build a tooltip-enabled help button linked to the shared help modal."""
    entry = get_help_entry(help_id)
    button_id = {"type": "context-help-button", "help_id": help_id}

    button = dbc.Button(
        html.I(className="bi bi-question-circle", **{"aria-hidden": "true"}),
        id=button_id,
        color="link",
        className="context-help-button",
        size="sm" if compact else None,
        n_clicks=0,
        title=f"Help: {entry['title']}",
    )

    tooltip = dbc.Tooltip(
        entry["summary"],
        target=button_id,
        placement="top",
    )

    return html.Span(
        [button, tooltip],
        className="context-help-wrapper",
    )
