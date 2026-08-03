"""Read-only live Settings data for the machine running Dash."""
from __future__ import annotations
from collections import OrderedDict
from datetime import datetime, timezone
import importlib.metadata as md
import importlib.util
import os, platform, shutil, socket, subprocess, sys, tempfile
from pathlib import Path

PACKAGE_GROUPS = (
("Core Python","REQUIRED_CORE",(("python",""),("conda",""),("pip","pip"),("setuptools","setuptools"),("wheel","wheel"),("packaging","packaging"),("importlib-metadata","importlib_metadata"),("typing-extensions","typing_extensions"),("tomli","tomli"),("platformdirs","platformdirs"))),
("Scientific Computing","REQUIRED_CORE",(("numpy","numpy"),("scipy","scipy"),("pandas","pandas"),("pyarrow","pyarrow"),("scikit-learn","sklearn"),("joblib","joblib"),("threadpoolctl","threadpoolctl"),("numba","numba"),("llvmlite","llvmlite"),("statsmodels","statsmodels"))),
("Data and Configuration","REQUIRED_CORE / PLANNED_REQUIRED",(("sqlalchemy","sqlalchemy"),("alembic","alembic"),("pyyaml","yaml"),("pydantic","pydantic"),("pydantic-settings","pydantic_settings"),("pydantic-extra-types","pydantic_extra_types"),("python-dotenv","dotenv"),("jsonschema","jsonschema"),("duckdb","duckdb"))),
("Dash / BGIRS","REQUIRED_CORE",(("dash","dash"),("dash-bootstrap-components","dash_bootstrap_components"),("plotly","plotly"),("flask","flask"),("flask-cors","flask_cors"),("werkzeug","werkzeug"),("jinja2","jinja2"),("waitress","waitress"),("retrying","retrying"),("narwhals","narwhals"),("requests","requests"),("urllib3","urllib3"))),
("Publication","REQUIRED_BY_WORKFLOW",(("matplotlib","matplotlib"),("seaborn","seaborn"),("kaleido","kaleido"),("pillow","PIL"),("openpyxl","openpyxl"),("xlsxwriter","xlsxwriter"))),
("PyTorch and GPU","REQUIRED_CORE",(("torch","torch"),("torchvision","torchvision"),("torchaudio","torchaudio"))),
("Scientific ML","PLANNED_REQUIRED",(("neuromancer","neuromancer"),("torchdiffeq","torchdiffeq"),("lightning","lightning"),("pytorch-lightning","pytorch_lightning"))),
("Optimization","REQUIRED_CORE / PLANNED_REQUIRED",(("casadi","casadi"),("cvxpy","cvxpy"),("pyomo","pyomo"),("osqp","osqp"),("clarabel","clarabel"),("scs","scs"),("ecos","ecos"),("cvxpylayers","cvxpylayers"))),
("Tuning","REQUIRED_BY_WORKFLOW",(("optuna","optuna"),("ray","ray"))),
("MLflow","REQUIRED_CORE",(("mlflow","mlflow"),("mlflow-skinny","mlflow"),("mlflow-ui","mlflow"),("cloudpickle","cloudpickle"),("protobuf","google.protobuf"),("gitpython","git"))),
("RL / Gymnasium","PLANNED_REQUIRED",(("gymnasium","gymnasium"),("stable-baselines3","stable_baselines3"),("shimmy","shimmy"),("pettingzoo","pettingzoo"))),
("EnergyPlus","REQUIRED_BY_WORKFLOW",(("opyplus","opyplus"),("eppy","eppy"),("geomeppy","geomeppy"))),
("OpenDSS","PLANNED_REQUIRED",(("opendssdirect","opendssdirect"),("dss-python","dss"))),
("GIS","OPTIONAL / REQUIRED_BY_WORKFLOW",(("geopandas","geopandas"),("shapely","shapely"),("pyproj","pyproj"),("fiona","fiona"),("pyogrio","pyogrio"),("folium","folium"),("mapclassify","mapclassify"))),
("Utilities","REQUIRED_CORE",(("python-slugify","slugify"),("text-unidecode","text_unidecode"),("Unidecode","unidecode"),("tqdm","tqdm"),("rich","rich"),("psutil","psutil"),("filelock","filelock"),("fsspec","fsspec"))),
("Testing","REQUIRED_CORE / OPTIONAL",(("pytest","pytest"),("coverage","coverage"),("pytest-cov","pytest_cov"),("ruff","ruff"),("mypy","mypy"))),
("ScaleBridge","REQUIRED_CORE",(("scalebridge","scalebridge"),)),
)

KNOWN_ENV_VARS = (
("SCALEBRIDGE_MACHINE_ID","ScaleBridge machine identity"),("SCALEBRIDGE_REPOSITORY_ROOT","Repository override"),
("SCALEBRIDGE_EXTERNAL_DATA_ROOT","Primary Data root"),("SCALEBRIDGE_DATA_ROOT","Alternate Data root"),
("SCALEBRIDGE_GENERATED_DATA_ROOT","Generated data root"),("SCALEBRIDGE_CAMPAIGNS_ROOT","Campaign root"),
("SCALEBRIDGE_TRAINING_EXPORT_ROOT","Training exports"),("SCALEBRIDGE_PUBLICATION_EXPORT_ROOT","Publication exports"),
("SCALEBRIDGE_CACHE_ROOT","Cache root"),("SCALEBRIDGE_TEMP_ROOT","Temporary root"),
("SCALEBRIDGE_EPLUS_WORK_ROOT","EnergyPlus work root"),("SCALEBRIDGE_ENERGYPLUS_EXE","EnergyPlus executable"),
("ENERGYPLUS_HOME","EnergyPlus home"),("ENERGYPLUS_EXE","EnergyPlus executable"),
("MLFLOW_TRACKING_URI","MLflow tracking URI"),("MLFLOW_BACKEND_STORE_URI","MLflow backend store"),
("MLFLOW_DEFAULT_ARTIFACT_ROOT","MLflow artifact root"),("MLFLOW_ARTIFACT_ROOT","Alternate artifact root"),
("MLFLOW_EXPERIMENT_NAME","Default experiment"),("MLFLOW_RUN_ID","Current run ID"),
("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING","System metrics logging"),
("CONDA_DEFAULT_ENV","Active Conda environment"),("CONDA_PREFIX","Conda prefix"),("CONDA_EXE","Conda executable"),
("PYTHONPATH","Python search path"),("PYTHONHOME","Python home"),("PIP_CONFIG_FILE","pip config"),
("PIP_INDEX_URL","pip index"),("PIP_EXTRA_INDEX_URL","Additional pip index"),("VIRTUAL_ENV","Virtual environment"),
("CUDA_HOME","CUDA home"),("CUDA_PATH","CUDA path"),("CUDA_VISIBLE_DEVICES","Visible CUDA devices"),
("NVIDIA_VISIBLE_DEVICES","Visible NVIDIA devices"),("OMP_NUM_THREADS","OpenMP threads"),
("OPENBLAS_NUM_THREADS","OpenBLAS threads"),("MKL_NUM_THREADS","MKL threads"),
("NUMEXPR_NUM_THREADS","NumExpr threads"),("KMP_DUPLICATE_LIB_OK","Duplicate OpenMP override"),
("GIT_PYTHON_GIT_EXECUTABLE","Git executable"),("GIT_AUTHOR_NAME","Git author"),("GIT_AUTHOR_EMAIL","Git email"),
("COMPUTERNAME","Windows hostname"),("HOSTNAME","Unix hostname"),("USERNAME","Windows username"),("USER","Unix username"),
("USERPROFILE","Windows profile"),("HOME","Home directory"),("TEMP","Windows temp"),("TMP","Temporary directory"),
("SHELL","Unix shell"),("PSModulePath","PowerShell modules"),("SLURM_JOB_ID","SLURM job"),
("SLURM_JOB_NAME","SLURM job name"),("SLURM_JOB_PARTITION","SLURM partition"),
("SLURM_CPUS_PER_TASK","SLURM CPUs"),("SLURM_GPUS","SLURM GPUs"),("SLURM_SUBMIT_DIR","SLURM submit directory"),
)

def repository_root(): return Path(__file__).resolve().parents[5]
def data_root(): return repository_root().parents[1]/"Data"
def generated_root(): return data_root()/"ScaleBridge"
def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _cmd(args,cwd=None):
    if not args or not args[0]: return "Not set"
    if shutil.which(args[0]) is None and not Path(args[0]).exists(): return "Not found"
    try:
        p=subprocess.run(args,cwd=str(cwd) if cwd else None,capture_output=True,text=True,timeout=4,check=False)
        t=(p.stdout or p.stderr or "").strip()
        return t.splitlines()[0] if t else f"Exit code {p.returncode}"
    except Exception as e: return f"Unable to inspect: {type(e).__name__}"

def _ver(dist):
    if dist=="python": return platform.python_version()
    if dist=="conda": return _cmd(["conda","--version"])
    try: return md.version(dist)
    except md.PackageNotFoundError: return "Not installed"
    except Exception as e: return f"Unable to read: {type(e).__name__}"

def _importable(mod):
    if not mod: return "Not applicable"
    try: return "Yes" if importlib.util.find_spec(mod) else "No"
    except Exception: return "No"

def path_snapshot():
    r,d,g=repository_root(),data_root(),generated_root()
    mid=os.getenv("SCALEBRIDGE_MACHINE_ID","unidentified-machine")
    paths=OrderedDict([
      ("Repository Root",r),("Primary Data Root",d),("ScaleBridge Data Root",g),
      ("Campaigns",g/"campaigns"),("Generation",g/"generation"),("Aggregation",g/"aggregation"),
      ("Phase C",g/"phase_c"),("Phase D",g/"phase_d"),("Thermal Models",g/"thermal_models"),
      ("MLflow Artifacts",g/"mlflow_artifacts"),("MLflow Exports",g/"mlflow_exports"/mid),
      ("Experiment Registry",g/"experiment_registry"),("Training Exports",g/"training_exports"),
      ("Publication Exports",g/"publication_exports"),
      ("Cache",Path(os.getenv("SCALEBRIDGE_CACHE_ROOT",str(g/"cache")))),
      ("Temporary Work",Path(os.getenv("SCALEBRIDGE_TEMP_ROOT",tempfile.gettempdir()))),
      ("EnergyPlus Work",Path(os.getenv("SCALEBRIDGE_EPLUS_WORK_ROOT",str(g/"energyplus_work")))),
    ])
    rows=[{"name":k,"path":str(v),"exists":"Yes" if v.exists() else "No","type":"Directory" if v.is_dir() else "File" if v.is_file() else "Not present"} for k,v in paths.items()]
    children=[]
    if g.is_dir():
        try: children=[{"name":p.name,"path":str(p),"type":"Directory" if p.is_dir() else "File"} for p in sorted(g.iterdir(),key=lambda x:x.name.lower())]
        except OSError: pass
    return {"generated_at":now(),"roots":{"Repository Root":str(r),"Primary Data Root":str(d),"ScaleBridge Data Root":str(g)},"paths":rows,"children":children}

def machine_snapshot():
    return OrderedDict([
      ("Machine ID",os.getenv("SCALEBRIDGE_MACHINE_ID","Not set")),("Hostname",socket.gethostname()),
      ("Fully Qualified Hostname",socket.getfqdn()),("Operating System",platform.system()),
      ("OS Release",platform.release()),("OS Version",platform.version()),("Platform",platform.platform()),
      ("Architecture",platform.machine()),("Processor",platform.processor() or "Not reported"),
      ("Logical CPUs",str(os.cpu_count() or "Not reported")),("Username",os.getenv("USERNAME") or os.getenv("USER") or "Not set"),
      ("Current Working Directory",str(Path.cwd())),("Repository Root",str(repository_root())),
      ("Python Executable",sys.executable),("Conda Environment",os.getenv("CONDA_DEFAULT_ENV","Not set")),
      ("Conda Prefix",os.getenv("CONDA_PREFIX","Not set")),("System Temp",tempfile.gettempdir()),("Generated At",now())
    ])

def python_snapshot():
    blas="Not reported"
    try:
        import numpy
        from threadpoolctl import threadpool_info
        info=threadpool_info()
        blas="; ".join(f"{x.get('internal_api') or x.get('prefix')} {x.get('version') or ''}, {x.get('num_threads')} threads" for x in info) or "Not reported"
    except Exception as e: blas=f"Unable to inspect: {type(e).__name__}"
    try:
        spec=importlib.util.find_spec("scalebridge"); origin=spec.origin if spec and spec.origin else "Not found"
    except Exception: origin="Not found"
    return OrderedDict([
      ("Python Version",platform.python_version()),("Implementation",platform.python_implementation()),
      ("Python Executable",sys.executable),("Python Prefix",sys.prefix),("Base Prefix",sys.base_prefix),
      ("Conda Environment",os.getenv("CONDA_DEFAULT_ENV","Not set")),("Conda Prefix",os.getenv("CONDA_PREFIX","Not set")),
      ("pip Version",_cmd([sys.executable,"-m","pip","--version"])),("Conda Version",_cmd(["conda","--version"])),
      ("ScaleBridge Version",_ver("scalebridge")),("ScaleBridge Import Path",origin),("BLAS / Thread Libraries",blas),("Generated At",now())
    ])

def package_snapshot():
    return [{"title":t,"classification":c,"packages":[{"component":d,"installed_version":_ver(d),"importable":_importable(m)} for d,m in ps]} for t,c,ps in PACKAGE_GROUPS]

def environment_variables():
    return [{"name":n,"value":os.getenv(n) or "Not set","description":d} for n,d in KNOWN_ENV_VARS]

def gpu_snapshot():
    out=OrderedDict([("PyTorch Version",_ver("torch")),("nvidia-smi",_cmd(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"]))])
    try:
        import torch
        out["PyTorch CUDA Runtime"]=str(torch.version.cuda or "Not available")
        out["CUDA Available"]="Yes" if torch.cuda.is_available() else "No"
        out["CUDA Device Count"]=str(torch.cuda.device_count())
        out["GPU Model"]=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available through PyTorch"
    except Exception as e:
        out["PyTorch CUDA Runtime"]=out["CUDA Available"]=out["CUDA Device Count"]=out["GPU Model"]=f"Unable to inspect: {type(e).__name__}"
    return out

def external_snapshot():
    ep=os.getenv("SCALEBRIDGE_ENERGYPLUS_EXE") or os.getenv("ENERGYPLUS_EXE") or "energyplus"
    return OrderedDict([
      ("Git",_cmd(["git","--version"])),("Git Branch",_cmd(["git","branch","--show-current"],repository_root())),
      ("Git Commit",_cmd(["git","rev-parse","--short","HEAD"],repository_root())),
      ("PowerShell",_cmd(["powershell","-NoProfile","-Command","$PSVersionTable.PSVersion.ToString()"])),
      ("EnergyPlus",_cmd([ep,"--version"])),("External IPOPT",_cmd(["ipopt","-v"])),
      ("SLURM srun",_cmd(["srun","--version"])),("SLURM sbatch",_cmd(["sbatch","--version"]))
    ])

def mlflow_snapshot():
    g=generated_root(); mid=os.getenv("SCALEBRIDGE_MACHINE_ID","unidentified-machine")
    uri=os.getenv("MLFLOW_TRACKING_URI","Not set")
    artifact=os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT") or os.getenv("MLFLOW_ARTIFACT_ROOT") or str(g/"mlflow_artifacts")
    return OrderedDict([
      ("Machine ID",os.getenv("SCALEBRIDGE_MACHINE_ID","Not set")),("MLflow Version",_ver("mlflow")),
      ("MLFLOW_TRACKING_URI",uri),("MLFLOW_BACKEND_STORE_URI",os.getenv("MLFLOW_BACKEND_STORE_URI","Not set")),
      ("Resolved Artifact Root",artifact),("Machine Export Root",str(g/"mlflow_exports"/mid)),
      ("Experiment Registry Root",str(g/"experiment_registry")),
      ("Local UI Address",uri if uri.startswith(("http://","https://")) else "Not set"),("Generated At",now())
    ])

def mlflow_variables():
    names={n for n,_ in KNOWN_ENV_VARS if n.startswith("MLFLOW_")}|{n for n in os.environ if n.startswith("MLFLOW_")}
    return [{"name":n,"value":os.getenv(n) or "Not set"} for n in sorted(names)]
