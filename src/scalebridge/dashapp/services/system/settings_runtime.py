"""Read-only discovery and validation for detailed Settings pages."""
from __future__ import annotations
from importlib import metadata
import os, platform, socket, sys
from pathlib import Path
from urllib.parse import urlparse

KNOWN_MACHINE_IDS=("laptop","home-pc","lab-pc","kamiak")
def repository_root(): return Path(__file__).resolve().parents[5]
def detect_current_machine():
    configured=os.getenv("SCALEBRIDGE_MACHINE_ID","").strip().lower()
    mid=configured or "unknown"
    return {"machine_id":mid,"hostname":platform.node() or os.getenv("COMPUTERNAME","Unknown"),"operating_system":platform.system() or "Unknown","platform":platform.platform(),"status":"validated" if mid in KNOWN_MACHINE_IDS else "warning","source":"SCALEBRIDGE_MACHINE_ID" if configured else "Platform fallback"}
def _checks(path):
    exists=path.exists(); readable=os.access(path,os.R_OK) if exists else False
    target=path if exists else path.parent; writable=os.access(target,os.W_OK) if target.exists() else False
    return exists,readable,writable
def _row(key,label,value,status,source,description,kind="read_only"):
    rendered="Not Defined" if value in (None,"") else str(value)
    ex=rd=wr=None
    if isinstance(value,Path): ex,rd,wr=_checks(value)
    return {"key":key,"label":label,"value":rendered,"status":status,"source":source,"description":description,"path_kind":kind,"exists":ex,"readable":rd,"writable":wr}
def get_path_snapshot():
    repo=repository_root(); data=(repo.parents[1]/"Data").resolve(); generated=data/"ScaleBridge"; campaigns=generated/"campaigns"
    env_data=os.getenv("SCALEBRIDGE_EXTERNAL_DATA_ROOT"); env_generated=os.getenv("SCALEBRIDGE_GENERATED_DATA_ROOT"); mid=detect_current_machine()["machine_id"]
    data_ok=data.exists() and (not env_data or Path(env_data).resolve()==data); gen_ok=generated.exists() and (not env_generated or Path(env_generated).resolve()==generated)
    backend=(Path(os.getenv("LOCALAPPDATA",str(Path.home()/"AppData/Local")))/"ScaleBridge"/"mlflow") if mid=="laptop" else (Path("D:/ScaleBridge_MLflow/backend") if mid=="lab-pc" else None)
    backend_status="validated" if mid=="lab-pc" else ("known_machine_specific" if mid=="laptop" else "unknown")
    temp=os.getenv("TEMP") or os.getenv("TMPDIR") or os.getenv("TMP"); eplus=os.getenv("SCALEBRIDGE_EPLUS_WORK_ROOT")
    rows=[
      _row("repository_root","Repository Root",repo,"validated","Installed package location","Root of the active ScaleBridge checkout."),
      _row("data_root","Primary Data Root",data,"validated" if data_ok else "warning","Repository root + ../../Data; compared with SCALEBRIDGE_EXTERNAL_DATA_ROOT","Shared source-data root."),
      _row("generated_data_root","ScaleBridge Generated Data Root",generated,"validated" if gen_ok else "warning","Primary data root + ScaleBridge; compared with SCALEBRIDGE_GENERATED_DATA_ROOT","Generated research artifacts."),
      _row("campaigns_root","Campaigns Root",campaigns,"validated" if campaigns.exists() else "warning","Generated root + campaigns","Campaign directories and phase outputs."),
      _row("generation_template","Generation Output Template",campaigns/"<campaign_id>"/"generation","known_machine_specific","Campaign layout","Phase A output template."),
      _row("aggregation_template","Aggregation Output Template",campaigns/"<campaign_id>"/"aggregation","known_machine_specific","Campaign layout","Phase B output template."),
      _row("phase_c_template","Phase C Output Template",campaigns/"<campaign_id>"/"heat_input_regression","planned","Campaign-local convention","Planned canonical Phase C location."),
      _row("phase_d_template","Phase D Output Template",campaigns/"<campaign_id>"/"thermal_model_data","planned","Campaign-local convention","Planned canonical Phase D location."),
      _row("mlflow_backend_root","MLflow Backend Root",backend,backend_status,"Machine profile","MLflow metadata backend."),
      _row("mlflow_artifact_root","MLflow Artifact Root",generated/"mlflow_artifacts","known_machine_specific","Generated root + mlflow_artifacts","MLflow artifacts."),
      _row("mlflow_export_root","MLflow Machine Export Root",generated/"mlflow_exports"/mid,"validated" if (generated/"mlflow_exports"/mid).exists() else "warning","Generated root + mlflow_exports/<machine_id>","Machine export folder.","creatable"),
      _row("experiment_registry_root","Merged Experiment Registry",generated/"experiment_registry","validated" if (generated/"experiment_registry").exists() else "warning","Generated root + experiment_registry","Cross-machine registry.","creatable"),
      _row("energyplus_work_root","EnergyPlus Work Root",Path(eplus) if eplus else None,"known_machine_specific" if eplus else "unknown","SCALEBRIDGE_EPLUS_WORK_ROOT","Short local EnergyPlus workspace."),
      _row("temporary_root","Temporary Root",Path(temp) if temp else None,"known_machine_specific" if temp else "unknown","OS TEMP/TMPDIR","Machine-local temporary storage."),
      _row("publication_export_root","Publication Export Root",generated/"publication_exports","validated" if (generated/"publication_exports").exists() else "planned","Generated root + publication_exports","Publication-ready exports.","creatable"),
      _row("training_export_root","Training Export Root",generated/"training_exports","validated" if (generated/"training_exports").exists() else "planned","Generated root + training_exports","Curated training datasets.","creatable")]
    return rows
MACHINE_PROFILES=(
 {"id":"laptop","name":"Windows Laptop","os":"Windows","environment":"scalebridge-dev-gpu-laptop","gpu":"NVIDIA GeForce MX150","role":"Primary code, Dash, smoke-test, Phase C and Phase D development machine.","recommended":["Code editing and planning","Dash development","Controlled smoke campaigns","Phase C/Phase D validation"],"restricted":["No complete aggregation dataset locally","Avoid large GPU campaigns","Keep large Dropbox data online-only"],"storage":"Limited local capacity.","status":"validated"},
 {"id":"home-pc","name":"Home PC","os":"Windows","environment":"scalebridge-dev-gpu-homepc","gpu":"NVIDIA GeForce GTX 1050 Ti","role":"Secondary Windows compute and complete aggregation-storage machine.","recommended":["Complete aggregation storage","Moderate PyTorch experiments","Cross-machine validation"],"restricted":["Not preferred for primary full generation","GPU memory limits large batches"],"storage":"Sufficient for complete aggregation outputs.","status":"validated"},
 {"id":"lab-pc","name":"Lab PC","os":"Windows","environment":"scalebridge-dev-gpu-labpc","gpu":"NVIDIA RTX A4000","role":"Primary Windows production machine for full generation, aggregation, Phase D, and GPU campaigns.","recommended":["Full EnergyPlus campaigns","Full aggregation","Main Phase D P1/P2 campaign","Validated MLflow hosting"],"restricted":["Use D: for short work","Use F:/Dropbox for durable outputs"],"storage":"D: work/scratch and F: durable data.","status":"validated"},
 {"id":"kamiak","name":"Kamiak HPC","os":"Linux / SLURM","environment":"scalebridge-dev-gpu-kamiak","gpu":"NVIDIA A100-PCIE-40GB","role":"Scheduled HPC platform for ANN, sequence, and Scientific ML training.","recommended":["Large PyTorch training","ANN and sequence models","Scientific ML","SLURM compute-node validation"],"restricted":["No heavy login-node work","No unscheduled GPU work","No full uncurated campaign tree under /home","EnergyPlus not validated"],"storage":"Use curated exports and scheduler temporary storage.","status":"validated"})
BASE_WIN={"Environment":"pass","Python":"3.10.20","Conda":"24.11.3","Pip":"26.1.2","Dash":"4.0.0","Plotly":"6.8.0","NumPy":"1.26.3","SciPy":"1.15.2","Pandas":"2.3.3","PyArrow":"23.0.1","MLflow":"3.13.0","PyTorch":"2.5.1+cu118","CUDA Runtime":"11.8","CUDA Available":"True","CasADi":"3.7.2","IPOPT":"Available","Opyplus":"2.0.7","OpenBLAS":"0.3.32 pthreads"}
def _win(env,gpu,eplus,dbc_status="warning"):
 d={k:(v,"pass") for k,v in BASE_WIN.items() if k!="Environment"}; d["Environment"]=(env,"pass"); d["dash-bootstrap-components"]=("2.0.4",dbc_status); d["DuckDB"]=("Not validated","unavailable"); d["GPU"]=(gpu,"pass"); d["EnergyPlus"]=(eplus,"pass" if eplus=="9.0.1" else "warning"); return d
ENVIRONMENT_PROFILES={"laptop":_win("scalebridge-dev-gpu-laptop","NVIDIA GeForce MX150","9.0.1","pass"),"home-pc":_win("scalebridge-dev-gpu-homepc","NVIDIA GeForce GTX 1050 Ti","9.0.1 project version"),"lab-pc":_win("scalebridge-dev-gpu-labpc","NVIDIA RTX A4000","9.0.1"),"kamiak":{"Environment":("scalebridge-dev-gpu-kamiak","pass"),"Python":("3.10.12","pass"),"Conda":("23.3.1","pass"),"Pip":("26.1.2","pass"),"Dash":("Not revalidated","warning"),"dash-bootstrap-components":("Not validated","unavailable"),"Plotly":("Not revalidated","warning"),"NumPy":("Not revalidated","warning"),"SciPy":("Not revalidated","warning"),"Pandas":("Not revalidated","warning"),"PyArrow":("Not revalidated","warning"),"DuckDB":("Not validated","unavailable"),"MLflow":("3.13.0","pass"),"PyTorch":("2.5.1+cu118","pass"),"CUDA Runtime":("11.8","pass"),"CUDA Available":("True","pass"),"GPU":("NVIDIA A100-PCIE-40GB","pass"),"CasADi":("3.7.2","pass"),"IPOPT":("Available","pass"),"EnergyPlus":("Not validated","unavailable"),"Opyplus":("2.0.7","pass"),"OpenBLAS":("Not revalidated","warning")}}
def build_environment_rows(mid): return [{"component":k,"value":v,"status":s} for k,(v,s) in ENVIRONMENT_PROFILES.get(mid,{}).items()]
def _version(name):
    try:return metadata.version(name)
    except metadata.PackageNotFoundError:return "Unavailable"
def get_runtime_snapshot():
 m=detect_current_machine(); packages={n:_version(d) for n,d in {"Dash":"dash","dash-bootstrap-components":"dash-bootstrap-components","Plotly":"plotly","NumPy":"numpy","SciPy":"scipy","Pandas":"pandas","PyArrow":"pyarrow","DuckDB":"duckdb","MLflow":"mlflow","PyTorch":"torch","CasADi":"casadi","Opyplus":"opyplus"}.items()}; return {**m,"environment":os.getenv("CONDA_DEFAULT_ENV","Not detected"),"python":platform.python_version(),"python_executable":sys.executable,"packages":packages}
def _reachable(uri,timeout=.2):
 p=urlparse(uri)
 if p.scheme not in {"http","https"} or not p.hostname:return False
 try:
  with socket.create_connection((p.hostname,p.port or (443 if p.scheme=="https" else 80)),timeout=timeout):return True
 except OSError:return False
def get_mlflow_snapshot():
 mid=detect_current_machine()["machine_id"]; uri=os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000"); gen=repository_root().parents[1]/"Data"/"ScaleBridge"; ex=gen/"mlflow_exports"/mid; reg=gen/"experiment_registry"; art=gen/"mlflow_artifacts"
 return {"machine_id":mid,"tracking_uri":uri,"reachable":_reachable(uri),"backend_store_uri":"sqlite:///D:/ScaleBridge_MLflow/backend/mlflow_labpc.sqlite" if mid=="lab-pc" else "Not explicitly validated for this machine","artifact_root":str(art),"artifact_root_exists":art.exists(),"export_root":str(ex),"export_root_exists":ex.exists(),"registry_root":str(reg),"registry_root_exists":reg.exists(),"server_control_mode":"Inspect Only","generation_experiment":"p1_compact_4b4c_labpc_1w_v1_generation","aggregation_test_experiment":"ScaleBridge_P1_Aggregation_Test","aggregation_experiment":"ScaleBridge_P1_Aggregation_4b4c_1w"}
VISUALIZATION_DEFAULTS={"theme":"system","display_units":"selectable","publication_units":"si","interactive_width":"responsive","png_dpi":300,"vector_formats":["svg","pdf"],"table_precision":3,"table_page_size":25}
