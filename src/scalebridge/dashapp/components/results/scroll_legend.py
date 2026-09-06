"""Shared clickable scroll-legend helpers for BGIRS result graphs."""
from __future__ import annotations

from dash import html


def _button_style(visible: bool) -> dict[str, object]:
    return {
        "width": "100%",
        "display": "flex",
        "alignItems": "flex-start",
        "gap": "0.55rem",
        "padding": "0.55rem 0.6rem",
        "marginBottom": "0.4rem",
        "border": "1px solid rgba(120,120,120,0.25)",
        "borderRadius": "0.4rem",
        "background": (
            "rgba(255,255,255,0.92)" if visible else "rgba(230,230,230,0.55)"
        ),
        "opacity": 1.0 if visible else 0.48,
        "cursor": "pointer",
        "textAlign": "left",
        "whiteSpace": "normal",
    }


def build_scroll_legend(items, *, toggle_type: str, empty_text: str = "No plotted traces."):
    """Render the locked 75/25 external clickable legend pattern."""
    if not items:
        return html.Div(empty_text, className="text-muted small")
    visible_count = sum(bool(item.get("visible", True)) for item in items)
    children = [
        html.Div(
            f"{visible_count} of {len(items)} traces visible",
            className="small text-muted mb-2",
        )
    ]
    for item in items:
        visible = bool(item.get("visible", True))
        color = str(item.get("color") or "#666")
        children.append(
            html.Button(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "1rem",
                            "minWidth": "1rem",
                            "height": "0.28rem",
                            "marginTop": "0.45rem",
                            "borderRadius": "0.2rem",
                            "backgroundColor": color,
                        }
                    ),
                    html.Span(
                        [
                            html.Div(
                                item.get("primary_label") or item.get("name") or "trace",
                                style={"fontWeight": 600},
                            ),
                            html.Div(
                                item.get("secondary_label") or "",
                                className="small text-muted",
                            ),
                        ],
                        style={"minWidth": 0, "overflowWrap": "anywhere"},
                    ),
                ],
                id={"type": toggle_type, "index": int(item["index"])},
                n_clicks=0,
                title=item.get("name") or "",
                style=_button_style(visible),
            )
        )
    return children


def toggle_trace_visibility(figure, items, index: int):
    """Return copied figure/items after toggling one external-legend trace."""
    if not figure or not items or index < 0:
        return figure, items
    data = list(figure.get("data", []))
    if index >= len(items) or index >= len(data):
        return figure, items
    updated_items = [dict(item) for item in items]
    updated_items[index]["visible"] = not bool(updated_items[index].get("visible", True))
    updated_figure = dict(figure)
    updated_figure["data"] = [dict(trace) for trace in data]
    updated_figure["data"][index]["visible"] = (
        True if updated_items[index]["visible"] else False
    )
    return updated_figure, updated_items
