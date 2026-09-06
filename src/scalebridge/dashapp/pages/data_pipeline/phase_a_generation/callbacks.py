from __future__ import annotations
from dash import ALL, Input, Output, State, callback, ctx, no_update, html, dcc
import dash_bootstrap_components as dbc
import json
import plotly.express as px
from scalebridge.integration.energyplus.generation.source_catalog import discover_ashrae_buildings,discover_commercial_weather
from scalebridge.integration.energyplus.generation.campaign_definition import GenerationCampaignDefinition,SourceFileRef
from scalebridge.dashapp.services.generation.definition_store import save_definition,load_definition
from scalebridge.dashapp.services.generation.upload_import import import_zip
from scalebridge.dashapp.services.generation.execution import MANAGER
from scalebridge.dashapp.services.generation.results_data import (
    build_selected_data_export,
    campaign_index,
    discover_key_values,
    filter_index_rows,
    load_series,
)
from .page import get_tab_builder

def _compact_weather_label(value: object) -> str:
 """Return a concise weather label for the native Plotly legend."""
 text=str(value or "")
 if text.startswith("USA_"):
  parts=text.split("_")
  if len(parts)>=3:
   state=parts[1]
   location=parts[2].split(".")[0].split("-")[0]
   return f"{location}, {state}"
 return text


def _apply_custom_results_plot_layout(fig):
 """Use the full graph panel for data; legend lives in a separate Dash panel."""
 fig.update_layout(
  showlegend=False,
  margin=dict(l=70,r=25,t=45,b=65),
  hovermode="x unified",
  autosize=True,
 )
 return fig


def _legend_button_style(color: str, visible: bool) -> dict[str, object]:
 """Return compact clickable styling for one custom legend item."""
 return {
  "width":"100%",
  "display":"flex",
  "alignItems":"flex-start",
  "gap":"0.55rem",
  "padding":"0.55rem 0.6rem",
  "marginBottom":"0.4rem",
  "border":"1px solid rgba(120,120,120,0.25)",
  "borderRadius":"0.4rem",
  "background":"rgba(255,255,255,0.92)" if visible else "rgba(230,230,230,0.55)",
  "opacity":1.0 if visible else 0.48,
  "cursor":"pointer",
  "textAlign":"left",
  "whiteSpace":"normal",
 }


def _build_custom_legend(items):
 """Build the independently scrollable, clickable legend contents."""
 if not items:
  return html.Div("Plot signals to populate the legend.",className="text-muted small")
 visible_count=sum(1 for item in items if item.get("visible",True))
 children=[html.Div(f"{visible_count} of {len(items)} traces visible",className="small text-muted mb-2")]
 for item in items:
  visible=bool(item.get("visible",True))
  color=str(item.get("color") or "#666")
  children.append(
   html.Button(
    [
     html.Span(
      style={
       "display":"inline-block","width":"1rem","minWidth":"1rem","height":"0.28rem",
       "marginTop":"0.45rem","borderRadius":"0.2rem","backgroundColor":color,
      }
     ),
     html.Span(
      [
       html.Div(item.get("primary_label",item.get("display_name","")),style={"fontWeight":600}),
       html.Div(item.get("variable_name",""),className="small"),
       html.Div(f"Key: {item.get('key_value','')}",className="small text-muted"),
      ],
      style={"minWidth":0,"overflowWrap":"anywhere"},
     ),
    ],
    id={"type":"generation-results-legend-toggle","index":int(item["index"])},
    n_clicks=0,
    title=item.get("full_name",item.get("display_name","")),
    style=_legend_button_style(color,visible),
   )
  )
 return children


def register_generation_callbacks():
 @callback(Output('generation-workspace-content','children'),Input('generation-workspace-tabs','value'),prevent_initial_call=True)
 def tab(v): return get_tab_builder(v)()

 @callback(Output('generation-builder-library-panel','style'),Output('generation-builder-upload-panel','style'),Input('generation-builder-source-mode','value'))
 def mode(v): return ({'display':'block'},{'display':'none'}) if v=='ashrae_library' else ({'display':'none'},{'display':'block'})

 @callback(Output('generation-builder-buildings','options'),Output('generation-builder-weather','options'),Output('generation-builder-catalog-cache','data'),Input('generation-builder-year','value'))
 def catalogs(year):
  try:
   bs=discover_ashrae_buildings(standard_year=int(year)); ws=discover_commercial_weather()
   bopts=[{'label':f"{b.building_type} — {b.source_location or 'unlabeled'} — {b.idf_path.name}",'value':b.source_id} for b in bs]
   wopts=[{'label':f"{w.city or w.name} — {w.epw_path.name}",'value':w.source_id} for w in ws]
   data={'buildings':[{'source_id':b.source_id,'name':b.name,'path':str(b.idf_path),'sha256':b.idf_sha256,'building_type':b.building_type,'source_location':b.source_location,'standard_year':b.standard_year} for b in bs],'weather':[{'source_id':w.source_id,'name':w.name,'path':str(w.epw_path),'sha256':w.epw_sha256} for w in ws]}
   return bopts,wopts,data
  except Exception as e: return [],[],{'error':str(e),'buildings':[],'weather':[]}

 @callback(Output('generation-builder-upload-contents','data'),Output('generation-builder-upload-status','children'),Input('generation-builder-upload','contents'),State('generation-builder-upload','filename'),prevent_initial_call=True)
 def upload(contents,name): return contents,dbc.Alert(f'ZIP selected: {name}. It will be securely imported when the campaign is saved.',color='info')

 @callback(Output('generation-builder-summary','children'),Input('generation-builder-buildings','value'),Input('generation-builder-weather','value'),Input('generation-builder-case-limit','value'))
 def summary(b,w,limit):
  n=len(b or [])*len(w or []); effective=min(n,int(limit)) if limit and n else n
  return dbc.Alert(f'Selected matrix: {len(b or [])} IDF(s) × {len(w or [])} EPW(s) = {n} case(s); effective case count: {effective}. Fixed signal profile: p1_generation_fixed_v1 (35 requested variables).',color='secondary')

 @callback(Output('generation-builder-save-status','children'),Input('generation-builder-save','n_clicks'),State('generation-builder-campaign-id','value'),State('generation-builder-machine-id','value'),State('generation-builder-source-mode','value'),State('generation-builder-year','value'),State('generation-builder-buildings','value'),State('generation-builder-weather','value'),State('generation-builder-catalog-cache','data'),State('generation-builder-upload-contents','data'),State('generation-builder-case-limit','value'),State('generation-builder-variable-limit','value'),State('generation-builder-workers','value'),State('generation-builder-pickles','value'),State('generation-builder-rerun','value'),State('generation-builder-mlflow','value'),State('generation-builder-mlflow-uri','value'),State('generation-builder-mlflow-experiment','value'),State('generation-builder-mlflow-strict','value'),prevent_initial_call=True)
 def save(n,cid,machine,mode,year,bsel,wsel,cache,upload,case_limit,var_limit,workers,pickles,rerun,mlflow,uri,exp,strict):
  try:
   if not cid or not machine: raise ValueError('Campaign ID and machine ID are required')
   if mode=='uploaded_zip': buildings,weather,_=import_zip(upload,cid)
   else:
    bmap={x['source_id']:x for x in (cache or {}).get('buildings',[])}; wmap={x['source_id']:x for x in (cache or {}).get('weather',[])}
    buildings=tuple(SourceFileRef(**bmap[x]) for x in (bsel or [])); weather=tuple(SourceFileRef(**wmap[x]) for x in (wsel or []))
   definition=GenerationCampaignDefinition(campaign_id=cid,machine_id=machine,source_mode=mode,ashrae_year=int(year) if mode=='ashrae_library' else None,buildings=buildings,weather_files=weather,case_limit=int(case_limit) if case_limit else None,variable_limit=int(var_limit) if var_limit else None,parallel_variable_workers=int(workers or 1),write_legacy_pickles=bool(pickles),rerun_completed=bool(rerun),mlflow_enabled=bool(mlflow),mlflow_tracking_uri=uri or 'http://127.0.0.1:5000',mlflow_experiment_name=exp or None,mlflow_strict=bool(strict))
   path=save_definition(definition); return dbc.Alert(f'Saved {definition.case_count}-case definition: {path}',color='success')
  except Exception as e: return dbc.Alert(f'{type(e).__name__}: {e}',color='danger')

 @callback(Output('generation-execution-definition','children'),Input('generation-execution-campaign','value'))
 def show_def(cid):
  if not cid:return ''
  try:return json.dumps(load_definition(cid).model_dump(mode='json'),indent=2)
  except Exception as e:return str(e)

 @callback(Output('generation-execution-status','children',allow_duplicate=True),Input('generation-execution-start','n_clicks'),State('generation-execution-campaign','value'),prevent_initial_call=True)
 def start(n,cid):
  if not cid:return dbc.Alert('Select a campaign.',color='warning')
  try:MANAGER.start(cid);return dbc.Alert(f'Started {cid}.',color='success')
  except Exception as e:return dbc.Alert(str(e),color='danger')

 @callback(Output('generation-execution-status','children',allow_duplicate=True),Input('generation-execution-stop','n_clicks'),prevent_initial_call=True)
 def stop(n): MANAGER.stop(); return dbc.Alert('Stop requested for the complete process tree.',color='warning')

 @callback(Output('generation-execution-console','children'),Output('generation-execution-status','children'),Output('generation-execution-start','disabled'),Output('generation-execution-stop','disabled'),Input('generation-execution-poll','n_intervals'))
 def poll(_):
  s=MANAGER.snapshot(); active=s['status'] in {'running','stop_requested'}; details=f"Status: {s['status']} | PID: {s['pid']} | Started: {s['started_at']} | Return code: {s['return_code']}"
  return s['console'],dbc.Alert(details,color='info' if active else 'secondary'),active,not active

 @callback(Output('generation-results-index','data'),Output('generation-results-metadata','children'),Input('generation-results-campaign','value'))
 def idx(cid):
  if not cid:return [],''
  rows=campaign_index(cid); return rows,dbc.Alert(f'Loaded metadata for {cid}: {len({r["case_id"] for r in rows})} cases, {len(rows)} generated variable artifacts.',color='info')

 def options(rows,key,filters):
  for k,vals in filters:
   if vals: rows=[r for r in rows if r[k] in vals]
  vals=sorted({r[key] for r in rows if r.get(key)}); return [{'label':x,'value':x} for x in vals]
 @callback(Output('generation-results-building','options'),Input('generation-results-index','data'))
 def bopts(rows): return options(rows or [],'building_type',[])
 @callback(Output('generation-results-weather','options'),Input('generation-results-index','data'),Input('generation-results-building','value'))
 def wopts(rows,b): return options(rows or [],'weather_location',[('building_type',b)])
 @callback(Output('generation-results-case','options'),Input('generation-results-index','data'),Input('generation-results-building','value'),Input('generation-results-weather','value'))
 def copts(rows,b,w): return options(rows or [],'case_id',[('building_type',b),('weather_location',w)])
 @callback(Output('generation-results-run','options'),Input('generation-results-index','data'),Input('generation-results-building','value'),Input('generation-results-weather','value'),Input('generation-results-case','value'))
 def ropts(rows,b,w,c): return options(rows or [],'run_id',[('building_type',b),('weather_location',w),('case_id',c)])
 @callback(Output('generation-results-variable','options'),Input('generation-results-index','data'),Input('generation-results-building','value'),Input('generation-results-weather','value'),Input('generation-results-case','value'),Input('generation-results-run','value'))
 def vopts(rows,b,w,c,r): return options(rows or [],'variable_name',[('building_type',b),('weather_location',w),('case_id',c),('run_id',r)])
 @callback(Output('generation-results-key','options'),Output('generation-results-key','value'),Input('generation-results-index','data'),Input('generation-results-building','value'),Input('generation-results-weather','value'),Input('generation-results-case','value'),Input('generation-results-run','value'),Input('generation-results-variable','value'),State('generation-results-key','value'))
 def kopts(rows,b,w,c,r,v,current):
  selected=filter_index_rows(rows or [],building_types=b,weather_locations=w,case_ids=c,run_ids=r,variable_names=v)
  keys=discover_key_values(selected) if v else []
  opts=[{'label':key,'value':key} for key in keys]
  current_valid=[key for key in (current or []) if key in keys]
  return opts,(current_valid if current_valid else keys)
 @callback(Output('generation-results-start','disabled'),Output('generation-results-end','disabled'),Input('generation-results-range-mode','value'))
 def range_mode(v): return (v!='custom',v!='custom')

 @callback(Output('generation-results-graph','figure'),Output('generation-results-custom-legend','children'),Output('generation-results-legend-state','data'),Output('generation-results-message','children'),Input('generation-results-plot-button','n_clicks'),State('generation-results-index','data'),State('generation-results-building','value'),State('generation-results-weather','value'),State('generation-results-case','value'),State('generation-results-run','value'),State('generation-results-variable','value'),State('generation-results-key','value'),State('generation-results-range-mode','value'),State('generation-results-start','value'),State('generation-results-end','value'),prevent_initial_call=True)
 def plot(n,rows,b,w,c,r,v,keys,mode,start,end):
  selected=filter_index_rows(rows or [],building_types=b,weather_locations=w,case_ids=c,run_ids=r,variable_names=v)
  if not selected:return no_update,no_update,no_update,dbc.Alert('Select at least one variable within the current filter context.',color='warning')
  if not keys:return no_update,no_update,no_update,dbc.Alert('Select at least one variable column / key.',color='warning')
  try:
   frame=load_series(selected,start if mode=='custom' else None,end if mode=='custom' else None,key_values=keys)
   if frame.empty:return no_update,no_update,no_update,dbc.Alert('No rows matched the selected variable columns / keys and datetime range.',color='warning')
   fig=px.line(frame,x='timestamp',y='value',color='series')

   row_by_prefix={}
   for row in selected:
    prefix=(
     f"{row['building_type']} | {row['weather_location']} | "
     f"{row['case_id']} | {row['run_id']} | {row['variable_name']}"
    )
    row_by_prefix[prefix]=row

   legend_items=[]
   for index,trace in enumerate(fig.data):
    full_name=str(trace.name)
    prefix,key_value=full_name.rsplit(' | ',1) if ' | ' in full_name else (full_name,'*')
    row=row_by_prefix.get(prefix,{})
    weather_short=_compact_weather_label(row.get('weather_location',''))
    trace.hovertemplate=(
     f"{full_name}<br>Datetime=%{{x}}<br>Value=%{{y}}<extra></extra>"
    )
    color=getattr(getattr(trace,'line',None),'color',None) or '#666'
    legend_items.append({
     'index':index,
     'visible':True,
     'color':color,
     'full_name':full_name,
     'primary_label':f"{row.get('building_type','')} | {weather_short}",
     'variable_name':row.get('variable_name',prefix),
     'key_value':key_value,
    })

   _apply_custom_results_plot_layout(fig)
   return fig,_build_custom_legend(legend_items),legend_items,dbc.Alert(f'Plotted {len(fig.data)} selected variable/key trace(s), {len(frame)} rows.',color='success')
  except Exception as e:return no_update,no_update,no_update,dbc.Alert(f'{type(e).__name__}: {e}',color='danger')

 @callback(Output('generation-results-graph','figure',allow_duplicate=True),Output('generation-results-custom-legend','children',allow_duplicate=True),Output('generation-results-legend-state','data',allow_duplicate=True),Input({'type':'generation-results-legend-toggle','index':ALL},'n_clicks'),State('generation-results-graph','figure'),State('generation-results-legend-state','data'),prevent_initial_call=True)
 def toggle_trace(_clicks,figure,items):
  triggered=ctx.triggered_id
  if not isinstance(triggered,dict) or triggered.get('type')!='generation-results-legend-toggle':
   return no_update,no_update,no_update
  if not figure or not items:
   return no_update,no_update,no_update
  index=int(triggered['index'])
  if index<0 or index>=len(items) or index>=len(figure.get('data',[])):
   return no_update,no_update,no_update
  items=[dict(item) for item in items]
  items[index]['visible']=not bool(items[index].get('visible',True))
  figure=dict(figure)
  figure['data']=[dict(trace) for trace in figure.get('data',[])]
  figure['data'][index]['visible']=True if items[index]['visible'] else False
  return figure,_build_custom_legend(items),items

 @callback(Output('generation-results-download','data'),Output('generation-results-download-message','children'),Input('generation-results-download-button','n_clicks'),State('generation-results-campaign','value'),State('generation-results-index','data'),State('generation-results-building','value'),State('generation-results-weather','value'),State('generation-results-case','value'),State('generation-results-run','value'),State('generation-results-variable','value'),State('generation-results-key','value'),State('generation-results-range-mode','value'),State('generation-results-start','value'),State('generation-results-end','value'),State('generation-results-export-format','value'),prevent_initial_call=True)
 def download_selected(n,cid,rows,b,w,c,r,v,keys,mode,start,end,export_format):
  if not cid:return no_update,dbc.Alert('Select a generated campaign before downloading data.',color='warning')
  selected=filter_index_rows(rows or [],building_types=b,weather_locations=w,case_ids=c,run_ids=r,variable_names=v)
  if not selected:return no_update,dbc.Alert('Select at least one variable within the current filter context.',color='warning')
  if not keys:return no_update,dbc.Alert('Select at least one variable column / key before downloading data.',color='warning')
  try:
   payload,filename=build_selected_data_export(campaign_id=cid,rows=selected,export_format=export_format or 'csv',range_mode=mode or 'full',start=start,end=end,key_values=keys)
   return dcc.send_bytes(payload,filename),dbc.Alert(f'Prepared only the selected variable/key series as {filename}.',color='success')
  except Exception as e:return no_update,dbc.Alert(f'{type(e).__name__}: {e}',color='danger')
