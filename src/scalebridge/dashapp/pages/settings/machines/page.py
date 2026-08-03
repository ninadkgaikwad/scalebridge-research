"""Current machine information."""
from dash import html
from ....services.system import machine_snapshot,gpu_snapshot,external_snapshot
from ..components import settings_heading
from ..live_components import table,card,notice

def build_layout():
    m=machine_snapshot()
    return html.Div([
        settings_heading("Current Machine","Identity, operating system, compute hardware, and external tools for the machine running BGIRS.","bi-pc-display","subpage.settings.machines"),
        notice(m["Generated At"]),
        card("Machine and Process",table(m)),
        card("GPU and CUDA",table(gpu_snapshot())),
        card("External Programs",table(external_snapshot()),"External tools are checked independently of Python package metadata. Missing tools are reported as Not found."),
    ],className="page-content-container settings-page")
