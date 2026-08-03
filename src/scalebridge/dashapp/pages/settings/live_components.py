"""Simple read-only components for live Settings pages."""
from collections.abc import Mapping, Sequence
from dash import html
import dash_bootstrap_components as dbc

def table(rows, columns=None):
    if isinstance(rows, Mapping):
        headers=("Setting","Value"); data=[{"Setting":k,"Value":v} for k,v in rows.items()]
    else:
        data=list(rows); headers=columns or tuple(data[0].keys() if data else ())
    return html.Div(
        dbc.Table([
            html.Thead(html.Tr([html.Th(h.replace("_"," ").title()) for h in headers])),
            html.Tbody([html.Tr([html.Td(_value(r.get(h,""))) for h in headers]) for r in data])
        ], bordered=False, hover=True, responsive=True, size="sm", className="settings-live-table mb-0"),
        className="settings-live-table-wrap"
    )

def card(title, children, description=None):
    body=[]
    if description: body.append(html.P(description,className="settings-live-description"))
    body.append(children)
    return dbc.Card([dbc.CardHeader(title),dbc.CardBody(body)],className="studio-card settings-live-card")

def notice(timestamp):
    return dbc.Alert([
        html.Strong("Current-machine information. "),
        html.Span("Values are computed from the running Dash process, repository location, installed distributions, and current environment variables. Nothing here is editable."),
        html.Br(),html.Small(f"Snapshot generated: {timestamp}")
    ],color="info")

def _value(v):
    if v is None or v=="": return "Not set"
    s=str(v)
    if any(x in s for x in ("\\","/","https://","http://","sqlite:")): return html.Code(s)
    return s
