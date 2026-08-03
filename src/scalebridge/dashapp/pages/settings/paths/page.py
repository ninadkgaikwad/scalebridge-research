"""Current-machine paths."""
from dash import html
from ....services.system import path_snapshot
from ..components import settings_heading
from ..live_components import table, card, notice

def build_layout():
    s=path_snapshot()
    return html.Div([
        settings_heading("Paths","Live paths for the current repository and Data/ScaleBridge.","bi-folder2-open","subpage.settings.paths"),
        notice(s["generated_at"]),
        card("Authoritative Roots",table(s["roots"]),"Primary Data is resolved from <repository-root>/../../Data. Generated ScaleBridge data is Data/ScaleBridge."),
        card("Important ScaleBridge Paths",table(s["paths"],("name","path","exists","type"))),
        card("First-Level Entries under Data/ScaleBridge",table(s["children"],("name","path","type")) if s["children"] else html.P("No first-level entries were found.",className="mb-0")),
    ],className="page-content-container settings-page")
