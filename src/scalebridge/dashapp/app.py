"""Dash application factory."""
from pathlib import Path
from dash import Dash
import dash_bootstrap_components as dbc
from .config import DashAppConfig
from .constants import APP_DESCRIPTION, APP_NAME
from .components.help import register_help_modal_callbacks
from .layout.routing import register_routing_callbacks
from .layout.shell import build_app_shell, register_shell_callbacks
from .pages.data_pipeline.phase_a_generation.callbacks import register_generation_callbacks

def create_app(config: DashAppConfig | None = None) -> Dash:
    resolved=config or DashAppConfig.from_environment()
    app=Dash(__name__,title=APP_NAME,update_title=None,external_stylesheets=[dbc.themes.BOOTSTRAP,dbc.icons.BOOTSTRAP],assets_folder=str(Path(__file__).resolve().parent/'assets'),use_pages=False,suppress_callback_exceptions=resolved.suppress_callback_exceptions,url_base_pathname=resolved.url_base_pathname,meta_tags=[{'name':'description','content':APP_DESCRIPTION},{'name':'viewport','content':'width=device-width, initial-scale=1'},{'name':'theme-color','content':'#0F6B78'}])
    app.layout=build_app_shell(); register_shell_callbacks(); register_routing_callbacks(); register_help_modal_callbacks(); register_generation_callbacks(); return app
