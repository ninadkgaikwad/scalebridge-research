from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


def campaign_options():
    root=resolve_generated_data_root()/'campaigns'
    if not root.is_dir(): return []
    return [{'label':p.name,'value':p.name} for p in sorted(root.iterdir()) if (p/'generation'/'cases').is_dir()]


def campaign_index(campaign_id):
    root=resolve_generated_data_root()/'campaigns'/campaign_id/'generation'/'cases'; rows=[]
    if not root.is_dir(): return rows
    for case_root in sorted(p for p in root.iterdir() if p.is_dir()):
        latest=case_root/'latest_run.json'
        if not latest.is_file(): continue
        try: lp=json.loads(latest.read_text(encoding='utf-8'))
        except Exception: continue
        run_id=str(lp.get('run_id','')); run_root=case_root/'runs'/run_id
        manifest=run_root/'run_manifest.json'; payload={}
        if manifest.is_file():
            try: payload=json.loads(manifest.read_text(encoding='utf-8'))
            except Exception: payload={}
        spec=payload.get('case_spec',{}); tags=spec.get('tags',{}); run_period=spec.get('run_period',{})
        year=run_period.get('calendar_year') or 2013
        variable_manifest=run_root/'canonical'/'variable_manifest.json'
        artifacts=[]
        if variable_manifest.is_file():
            try: artifacts=json.loads(variable_manifest.read_text(encoding='utf-8')).get('artifacts',[])
            except Exception: artifacts=[]
        if artifacts:
            candidates=[(a.get('variable_name') or a.get('variable_id'), Path(a.get('canonical_parquet_path',''))) for a in artifacts]
        else:
            var_root=run_root/'canonical'/'variables'
            candidates=[(pq.stem,pq) for pq in sorted(var_root.glob('*.parquet'))] if var_root.is_dir() else []
        for variable_name,pq in candidates:
            if not pq.is_absolute(): pq=(run_root/pq).resolve()
            rows.append({'building_type':spec.get('building_type') or tags.get('source_idf_name') or case_root.name,'weather_location':spec.get('weather_location') or tags.get('source_weather_name') or '', 'case_id':case_root.name,'run_id':run_id,'variable_name':str(variable_name),'parquet_path':str(pq),'status':lp.get('status',''),'calendar_year':int(year)})
    return rows


def _parse_timestamp_raw(series, year):
    text=series.astype(str).str.strip().str.replace(r'\s+',' ',regex=True)
    # EnergyPlus commonly represents midnight as 24:00:00. Parse the date at
    # 00:00 and add one day for those rows.
    is_24=text.str.contains(r'\s24:',regex=True)
    normalized=text.str.replace(r'\s24:', ' 00:',regex=True)
    parsed=pd.to_datetime(str(year)+'/'+normalized,format='%Y/%m/%d %H:%M:%S',errors='coerce')
    parsed.loc[is_24 & parsed.notna()]=parsed.loc[is_24 & parsed.notna()]+pd.Timedelta(days=1)
    return parsed


def _time_value_columns(df):
    if 'timestamp_raw' in df.columns: return 'timestamp_raw','value' if 'value' in df.columns else None
    time_candidates=['timestamp','datetime','date_time','Date/Time','time']
    t=next((c for c in time_candidates if c in df.columns),None)
    if t is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]): t=c; break
    numeric=[c for c in df.columns if c!=t and pd.api.types.is_numeric_dtype(df[c])]
    v=next((c for c in ('value','Value','variable_value') if c in numeric),numeric[0] if numeric else None)
    if t is None or v is None: raise ValueError(f'Could not infer timestamp/value columns from {list(df.columns)}')
    return t,v


def load_series(rows,start=None,end=None):
    frames=[]
    for row in rows:
        df=pd.read_parquet(row['parquet_path']); t,v=_time_value_columns(df)
        timestamp=_parse_timestamp_raw(df[t],row.get('calendar_year',2013)) if t=='timestamp_raw' else pd.to_datetime(df[t],errors='coerce')
        frame=pd.DataFrame({'timestamp':timestamp,'value':pd.to_numeric(df[v],errors='coerce')}).dropna(subset=['timestamp'])
        if start: frame=frame[frame.timestamp>=pd.Timestamp(start)]
        if end: frame=frame[frame.timestamp<=pd.Timestamp(end)]
        frame['series']=f"{row['building_type']} | {row['weather_location']} | {row['case_id']} | {row['run_id']} | {row['variable_name']}"; frames.append(frame)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['timestamp','value','series'])
