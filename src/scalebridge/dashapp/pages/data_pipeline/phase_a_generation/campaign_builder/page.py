from dash import dcc,html
import dash_bootstrap_components as dbc

def build_layout():
    return html.Div([
      html.H3('Campaign Builder'),
      dbc.Row([dbc.Col([html.Label('Campaign ID'),dbc.Input(id='generation-builder-campaign-id',placeholder='generation_campaign_v1')],md=6),dbc.Col([html.Label('Machine ID'),dbc.Input(id='generation-builder-machine-id',value='laptop')],md=3),dbc.Col([html.Label('Source mode'),dcc.RadioItems(id='generation-builder-source-mode',options=[{'label':'ASHRAE Prototype Library','value':'ashrae_library'},{'label':'Uploaded ZIP','value':'uploaded_zip'}],value='ashrae_library')],md=3)],className='g-3'),
      html.Hr(),
      html.Div([dbc.Row([dbc.Col([html.Label('ASHRAE year'),dcc.Dropdown(id='generation-builder-year',options=[{'label':str(y),'value':y} for y in (2013,2016,2019)],value=2013,clearable=False)],md=3),dbc.Col([html.Label('Building IDFs'),dcc.Dropdown(id='generation-builder-buildings',multi=True,placeholder='Select one or more IDF files')],md=4),dbc.Col([html.Label('Weather EPWs'),dcc.Dropdown(id='generation-builder-weather',multi=True,placeholder='Select one or more weather files')],md=5)],className='g-3')],id='generation-builder-library-panel'),
      html.Div([dcc.Upload(id='generation-builder-upload',children=html.Div(['Drag and drop or select a ZIP containing ',html.Code('idf/*.idf'), ' and ',html.Code('epw/*.epw')]),className='generation-upload',multiple=False),html.Div(id='generation-builder-upload-status')],id='generation-builder-upload-panel',style={'display':'none'}),
      html.Hr(),
      dbc.Row([dbc.Col([html.Label('Case limit (optional)'),dbc.Input(id='generation-builder-case-limit',type='number',min=1)],md=2),dbc.Col([html.Label('Variable limit (testing)'),dbc.Input(id='generation-builder-variable-limit',type='number',min=1)],md=2),dbc.Col([html.Label('Parallel variable workers'),dbc.Input(id='generation-builder-workers',type='number',min=1,value=1)],md=2),dbc.Col([dbc.Checkbox(id='generation-builder-pickles',value=True,label='Write legacy pickles'),dbc.Checkbox(id='generation-builder-rerun',value=False,label='Rerun completed')],md=3),dbc.Col([dbc.Checkbox(id='generation-builder-mlflow',value=True,label='Enable MLflow'),dbc.Checkbox(id='generation-builder-mlflow-strict',value=False,label='Strict MLflow')],md=3)],className='g-3'),
      dbc.Row([dbc.Col([html.Label('MLflow tracking URI'),dbc.Input(id='generation-builder-mlflow-uri',value='http://127.0.0.1:5000')],md=6),dbc.Col([html.Label('MLflow experiment name'),dbc.Input(id='generation-builder-mlflow-experiment',placeholder='<campaign-id>_generation')],md=6)],className='g-3 mt-2'),
      html.Div(id='generation-builder-summary',className='mt-3'),
      dbc.Button('Save Campaign Definition',id='generation-builder-save',color='primary',className='mt-3'),html.Div(id='generation-builder-save-status',className='mt-2'),
      dcc.Store(id='generation-builder-upload-contents'),dcc.Store(id='generation-builder-catalog-cache')
    ],className='page-content-container generation-page')
