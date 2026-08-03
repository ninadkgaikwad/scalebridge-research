from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
from threading import RLock,Thread
from collections import deque
from datetime import datetime,timezone
import subprocess,sys,os,signal
from .definition_store import definition_path

@dataclass
class State:
    status:str='not_started'; campaign_id:str|None=None; pid:int|None=None; command:list[str]=field(default_factory=list); started_at:str|None=None; return_code:int|None=None; stop_requested:bool=False; output:deque=field(default_factory=lambda:deque(maxlen=5000)); process:object|None=None

class GenerationProcessManager:
    def __init__(self): self._s=State(); self._lock=RLock()
    def snapshot(self):
        with self._lock:
            s=self._s; return {'status':s.status,'campaign_id':s.campaign_id,'pid':s.pid,'command':' '.join(s.command),'started_at':s.started_at,'return_code':s.return_code,'stop_requested':s.stop_requested,'console':'\n'.join(s.output)}
    def start(self,campaign_id):
        with self._lock:
            if self._s.process is not None and self._s.process.poll() is None: raise RuntimeError('A Generation campaign is already running')
            script=Path(__file__).resolve().parents[5]/'scripts'/'energyplus'/'run_generation_campaign.py'
            if not script.is_file():
                # package lives under repo/src; climb to repo
                script=Path(__file__).resolve().parents[6]/'scripts'/'energyplus'/'run_generation_campaign.py'
            cmd=[sys.executable,'-u',str(script),'--campaign-definition',str(definition_path(campaign_id))]
            kwargs={'stdout':subprocess.PIPE,'stderr':subprocess.STDOUT,'text':True,'bufsize':1,'cwd':str(script.parents[2])}
            if os.name=='nt': kwargs['creationflags']=subprocess.CREATE_NEW_PROCESS_GROUP
            else: kwargs['start_new_session']=True
            p=subprocess.Popen(cmd,**kwargs)
            self._s=State(status='running',campaign_id=campaign_id,pid=p.pid,command=cmd,started_at=datetime.now(timezone.utc).isoformat(),process=p)
            Thread(target=self._reader,args=(p,),daemon=True).start()
    def _reader(self,p):
        assert p.stdout is not None
        for line in p.stdout:
            with self._lock: self._s.output.append(line.rstrip())
        rc=p.wait()
        with self._lock:
            self._s.return_code=rc
            if self._s.stop_requested: self._s.status='stopped'
            else: self._s.status='completed' if rc==0 else 'failed'
    def stop(self):
        with self._lock:
            p=self._s.process
            if p is None or p.poll() is not None: return
            self._s.stop_requested=True; self._s.status='stop_requested'
            pid=p.pid
        if os.name=='nt': subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True,text=True)
        else:
            try: os.killpg(os.getpgid(pid),signal.SIGTERM)
            except ProcessLookupError: pass
MANAGER=GenerationProcessManager()
