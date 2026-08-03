"""URL routing and horizontal-tab content callbacks."""

from dash import Input, Output, callback, html
import dash_bootstrap_components as dbc

from ...pages.home import build_page as build_home_page
from ...pages.campaigns import build_page as build_campaigns_page
from ...pages.data_pipeline import build_page as build_data_pipeline_page
from ...pages.thermal_modeling import build_page as build_thermal_modeling_page
from ...pages.model_catalog import build_page as build_model_catalog_page
from ...pages.simulators import build_page as build_simulators_page
from ...pages.results_explorer import build_page as build_results_explorer_page
from ...pages.publication_studio import build_page as build_publication_studio_page
from ...pages.settings import build_page as build_settings_page


_ROUTES = {
    "/": ("home", build_home_page),
    "/campaigns": ("campaigns", build_campaigns_page),
    "/data-pipeline": ("data_pipeline", build_data_pipeline_page),
    "/thermal-modeling": ("thermal_modeling", build_thermal_modeling_page),
    "/model-catalog": ("model_catalog", build_model_catalog_page),
    "/simulators": ("simulators", build_simulators_page),
    "/results-explorer": ("results_explorer", build_results_explorer_page),
    "/publication-studio": (
        "publication_studio",
        build_publication_studio_page,
    ),
    "/settings": ("settings", build_settings_page),
}


def _not_found(pathname: str):
    return dbc.Container(
        [
            html.Div(
                html.I(className="bi bi-signpost-2"),
                className="not-found-icon",
            ),
            html.H1("Page Not Found"),
            html.P(f"No application page is registered for: {pathname}"),
            dbc.Button("Return Home", href="/", color="primary"),
        ],
        fluid=True,
        className="not-found-page",
    )


def register_routing_callbacks() -> None:
    """Register URL routing and modular tab callbacks."""

    @callback(
        Output("page-content", "children"),
        Input("app-location", "pathname"),
    )
    def render_route(pathname):
        normalized = pathname or "/"
        route = _ROUTES.get(normalized)
        if route is None:
            return _not_found(normalized)
        _page_id, builder = route
        return builder()

    @callback(
        Output("home-subpage-content", "children"),
        Input(
            {"type": "major-page-tabs", "page_id": "home"},
            "value",
        ),
        prevent_initial_call=True,
    )
    def render_home_tab(tab_id):
        from ...pages.home.page import get_subpage_builder

        return get_subpage_builder(tab_id)()

    @callback(
        Output("campaigns-subpage-content", "children"),
        Input(
            {"type": "major-page-tabs", "page_id": "campaigns"},
            "value",
        ),
        prevent_initial_call=True,
    )
    def render_campaigns_tab(tab_id):
        from ...pages.campaigns.page import get_subpage_builder

        return get_subpage_builder(tab_id)()

    @callback(
        Output("data_pipeline-subpage-content", "children"),
        Input(
            {"type": "major-page-tabs", "page_id": "data_pipeline"},
            "value",
        ),
        prevent_initial_call=True,
    )
    def render_data_pipeline_tab(tab_id):
        from ...pages.data_pipeline.page import get_subpage_builder

        return get_subpage_builder(tab_id)()

    @callback(
        Output("thermal_modeling-subpage-content", "children"),
        Input(
            {"type": "major-page-tabs", "page_id": "thermal_modeling"},
            "value",
        ),
        prevent_initial_call=True,
    )
    def render_thermal_modeling_tab(tab_id):
        from ...pages.thermal_modeling.page import get_subpage_builder

        return get_subpage_builder(tab_id)()

    @callback(
        Output("model_catalog-subpage-content", "children"),
        Input(
            {"type": "major-page-tabs", "page_id": "model_catalog"},
            "value",
        ),
        prevent_initial_call=True,
    )
    def render_model_catalog_tab(tab_id):
        from ...pages.model_catalog.page import get_subpage_builder

        return get_subpage_builder(tab_id)()

    @callback(
        Output("simulators-subpage-content", "children"),
        Input(
            {"type": "major-page-tabs", "page_id": "simulators"},
            "value",
        ),
        prevent_initial_call=True,
    )
    def render_simulators_tab(tab_id):
        from ...pages.simulators.page import get_subpage_builder

        return get_subpage_builder(tab_id)()

    @callback(
        Output("results_explorer-subpage-content", "children"),
        Input(
            {"type": "major-page-tabs", "page_id": "results_explorer"},
            "value",
        ),
        prevent_initial_call=True,
    )
    def render_results_explorer_tab(tab_id):
        from ...pages.results_explorer.page import get_subpage_builder

        return get_subpage_builder(tab_id)()

    @callback(
        Output("publication_studio-subpage-content", "children"),
        Input(
            {
                "type": "major-page-tabs",
                "page_id": "publication_studio",
            },
            "value",
        ),
        prevent_initial_call=True,
    )
    def render_publication_studio_tab(tab_id):
        from ...pages.publication_studio.page import get_subpage_builder

        return get_subpage_builder(tab_id)()

    @callback(
        Output("settings-subpage-content", "children"),
        Input(
            {"type": "major-page-tabs", "page_id": "settings"},
            "value",
        ),
        prevent_initial_call=True,
    )
    def render_settings_tab(tab_id):
        from ...pages.settings.page import get_subpage_builder

        return get_subpage_builder(tab_id)()
