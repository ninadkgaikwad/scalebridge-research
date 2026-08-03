"""Responsive left-sidebar navigation."""

from dash import html
import dash_bootstrap_components as dbc

from ...constants import (
    APP_NAME,
    APP_SHORT_NAME,
    AUTHOR_NAME,
    AUTHOR_ROLE,
    AUTHOR_LAB,
    EXTERNAL_LINKS,
)
from ...pages.registry import MAJOR_PAGES
from ...components.help import help_button


def _nav_item(page: dict):
    """Build one major-page sidebar navigation item and tooltip."""
    nav_id = f"sidebar-nav-{page['id']}"

    nav_link = dbc.NavLink(
        [
            html.I(
                className=f"bi {page['icon']} sidebar-nav-icon",
                **{"aria-hidden": "true"},
            ),
            html.Span(
                page["label"],
                className="sidebar-nav-label",
            ),
        ],
        id=nav_id,
        href=page["path"],
        active="exact",
        className="sidebar-nav-link",
    )

    tooltip = dbc.Tooltip(
        page["label"],
        target=nav_id,
        placement="right",
        delay={"show": 350, "hide": 100},
    )

    return html.Div(
        [nav_link, tooltip],
        className="sidebar-nav-item",
    )


def build_sidebar():
    """Build the major-page navigation sidebar."""
    return html.Aside(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                html.I(
                                    className="bi bi-diagram-3-fill",
                                    **{"aria-hidden": "true"},
                                ),
                                className="brand-mark",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        APP_SHORT_NAME,
                                        className="brand-short",
                                    ),
                                    html.Div(
                                        APP_NAME,
                                        className="brand-full",
                                    ),
                                ],
                                className="brand-text",
                            ),
                        ],
                        className="brand-row",
                    ),
                    help_button("app.identity"),
                ],
                className="sidebar-brand",
            ),
            html.Div(
                [
                    html.Span(
                        "Research Navigation",
                        className="sidebar-section-label",
                    ),
                    help_button("navigation.sidebar"),
                ],
                className="sidebar-section-header",
            ),
            dbc.Nav(
                [_nav_item(page) for page in MAJOR_PAGES],
                vertical=True,
                pills=True,
                className="sidebar-nav",
            ),
            html.Div(className="sidebar-spacer"),
            html.Div(
                [
                    html.Div(
                        [
                            html.A(
                                AUTHOR_NAME,
                                href=EXTERNAL_LINKS["author"],
                                target="_blank",
                                rel="noopener noreferrer",
                                className="sidebar-author-name",
                            ),
                            html.Div(
                                AUTHOR_ROLE,
                                className="sidebar-author-role",
                            ),
                            html.Div(
                                [
                                    html.A(
                                        AUTHOR_LAB,
                                        href=EXTERNAL_LINKS["scale_lab"],
                                        target="_blank",
                                        rel="noopener noreferrer",
                                    ),
                                    html.Span(" · "),
                                    html.A(
                                        "EECS",
                                        href=EXTERNAL_LINKS["eecs"],
                                        target="_blank",
                                        rel="noopener noreferrer",
                                    ),
                                    html.Span(" · "),
                                    html.A(
                                        "WSU",
                                        href=EXTERNAL_LINKS["wsu"],
                                        target="_blank",
                                        rel="noopener noreferrer",
                                    ),
                                ],
                                className="sidebar-author-links",
                            ),
                        ],
                        className="sidebar-attribution-text",
                    ),
                    dbc.Button(
                        html.I(
                            className="bi bi-info-circle",
                            **{"aria-hidden": "true"},
                        ),
                        id="open-about-modal",
                        color="link",
                        className="sidebar-about-button",
                        title="About This Application",
                    ),
                ],
                className="sidebar-attribution",
            ),
        ],
        id="app-sidebar",
        className="app-sidebar",
    )
