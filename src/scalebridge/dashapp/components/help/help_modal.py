"""Shared contextual-help modal and callbacks."""

from dash import ALL, Input, Output, State, callback, ctx, html
import dash_bootstrap_components as dbc

from ...help.registry import get_help_entry


def build_help_modal():
    """Return the shared contextual-help modal."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(id="context-help-title")),
            dbc.ModalBody(id="context-help-body"),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="context-help-close",
                    color="secondary",
                    className="ms-auto",
                )
            ),
        ],
        id="context-help-modal",
        is_open=False,
        centered=True,
        scrollable=True,
        size="lg",
    )


def register_help_modal_callbacks() -> None:
    """Register one callback serving all contextual-help buttons."""

    @callback(
        Output("context-help-modal", "is_open"),
        Output("context-help-title", "children"),
        Output("context-help-body", "children"),
        Input({"type": "context-help-button", "help_id": ALL}, "n_clicks"),
        Input("context-help-close", "n_clicks"),
        State("context-help-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_help_modal(_help_clicks, _close_clicks, is_open):
        trigger = ctx.triggered_id
        if trigger == "context-help-close":
            return False, "", ""

        if isinstance(trigger, dict):
            help_id = trigger.get("help_id", "")
            entry = get_help_entry(help_id)
            body = html.Div(
                [
                    html.P(entry.get("summary", ""), className="help-summary"),
                    html.P(entry.get("details", "")),
                    html.Hr(),
                    html.Small(
                        f"Help Registry ID: {help_id}",
                        className="text-muted",
                    ),
                ]
            )
            return True, entry.get("title", "Help"), body

        return is_open, "", ""
