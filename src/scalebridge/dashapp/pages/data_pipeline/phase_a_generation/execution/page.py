from dash import dcc,html
import dash_bootstrap_components as dbc
from scalebridge.dashapp.services.generation import list_definitions

def build_layout():
    defs=list_definitions(); options=[{'label':f"{x['campaign_id']} ({x['case_count']} cases)",'value':x['campaign_id']} for x in defs]
    return html.Div([html.H3('Execution'),html.P('Select a saved campaign, launch the definition-driven runner, view its live console, or stop the full process tree.'),dcc.Dropdown(id='generation-execution-campaign',options=options,placeholder='Select a saved campaign'),html.Pre(id='generation-execution-definition',className='generation-command-preview mt-3'),dbc.Button('Start Execution',id='generation-execution-start',color='success',disabled=not bool(options)),dbc.Button('Stop Execution',id='generation-execution-stop',color='danger',className='ms-2',disabled=True),html.Div(id='generation-execution-status',className='mt-3'),html.Pre(id='generation-execution-console',className='generation-live-console'),dcc.Interval(id='generation-execution-poll',interval=1500,n_intervals=0)],className='page-content-container generation-page')
