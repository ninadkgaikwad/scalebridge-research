#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executable Phase E.0 environment qualification for ScaleBridge.

This is intentionally stricter than environments/check_scalebridge_environment.py.
It verifies imports *and* small executable workflows needed by Phase E:
PyTorch/CUDA, Neuromancer, Optuna persistence, MLflow local tracking,
Pyomo + external IPOPT, CasADi + bundled IPOPT, PyArrow round-trip,
and an editable/repository-local ScaleBridge import.

Run from the repository root inside the intended ScaleBridge environment.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "phase_e0_environment_qualification_v1"

EXPECTED_ENVIRONMENTS = {
    "laptop": "scalebridge-dev-gpu-laptop",
    "home-pc": "scalebridge-dev-gpu-homepc",
    "lab-pc": "scalebridge-dev-gpu-labpc",
    "kamiak": "scalebridge-dev-gpu-kamiak",
}

REQUIRED_IMPORTS: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "numpy"),
    ("scipy", "scipy", "scipy"),
    ("pandas", "pandas", "pandas"),
    ("pyarrow", "pyarrow", "pyarrow"),
    ("sklearn", "sklearn", "scikit-learn"),
    ("torch", "torch", "torch"),
    ("neuromancer", "neuromancer", "neuromancer"),
    ("optuna", "optuna", "optuna"),
    ("mlflow", "mlflow", "mlflow"),
    ("pyomo", "pyomo", "pyomo"),
    ("casadi", "casadi", "casadi"),
    ("gymnasium", "gymnasium", "gymnasium"),
    ("scalebridge", "scalebridge", "scalebridge"),
)

OPTIONAL_IMPORTS: tuple[tuple[str, str, str], ...] = (
    ("torchvision", "torchvision", "torchvision"),
    ("torchaudio", "torchaudio", "torchaudio"),
    ("lightning", "lightning", "lightning"),
    ("stable_baselines3", "stable_baselines3", "stable-baselines3"),
    ("torchdiffeq", "torchdiffeq", "torchdiffeq"),
    ("torchsde", "torchsde", "torchsde"),
    ("opyplus", "opyplus", "opyplus"),
    ("duckdb", "duckdb", "duckdb"),
    ("opendssdirect", "opendssdirect", "opendssdirect.py"),
)

EXECUTABLE_CHECK_NAMES = (
    "python_version",
    "execution_context",
    "numpy_scipy_numeric",
    "pandas_pyarrow_roundtrip",
    "sklearn_fit_predict",
    "torch_autograd_device",
    "neuromancer_module",
    "optuna_sqlite_resume",
    "mlflow_local_tracking",
    "casadi_ipopt",
    "pyomo_external_ipopt",
    "phase_d_import",
    "phase_c_runtime_import",
    "scalebridge_repo_import",
    "pip_check",
)


@dataclass
class CheckResult:
    name: str
    category: str
    required: bool
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None


def _version(module: Any, distribution: str) -> str | None:
    value = getattr(module, "__version__", None)
    if value is not None:
        return str(value)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_check(label: str, module_name: str, distribution: str, *, required: bool) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
        return CheckResult(
            name=f"import:{label}",
            category="import",
            required=required,
            ok=True,
            details={
                "module": module_name,
                "distribution": distribution,
                "version": _version(module, distribution),
                "file": str(getattr(module, "__file__", "") or ""),
            },
        )
    except Exception as exc:
        return CheckResult(
            name=f"import:{label}",
            category="import",
            required=required,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def _run_check(
    name: str,
    func: Callable[[], dict[str, Any] | None],
    *,
    required: bool = True,
) -> CheckResult:
    try:
        details = func() or {}
        return CheckResult(
            name=name,
            category="executable",
            required=required,
            ok=True,
            details=details,
        )
    except Exception as exc:
        return CheckResult(
            name=name,
            category="executable",
            required=required,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def _check_python_version() -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python >=3.10 required; found {sys.version.split()[0]}")
    return {
        "version": sys.version,
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def _check_execution_context(machine_id: str) -> dict[str, Any]:
    expected_env = EXPECTED_ENVIRONMENTS[machine_id]
    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    details = {
        "machine_id": machine_id,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "expected_conda_env": expected_env,
        "conda_prefix": conda_prefix,
        "conda_default_env": active_env,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
    }

    if active_env != expected_env:
        raise RuntimeError(
            f"Wrong active Conda environment for {machine_id}: expected "
            f"{expected_env!r}, found {active_env!r}. Activate the authoritative "
            "machine-specific ScaleBridge environment before qualification or repair."
        )

    if machine_id == "kamiak" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Kamiak Phase E.0 qualification must run inside a SLURM compute job; "
            "SLURM_JOB_ID is not set."
        )
    return details


def _check_numpy_scipy() -> dict[str, Any]:
    import numpy as np
    from scipy.linalg import solve

    a = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=float)
    b = np.array([9.0, 8.0], dtype=float)
    x = solve(a, b)
    residual = float(np.linalg.norm(a @ x - b))
    if not np.isfinite(residual) or residual > 1.0e-10:
        raise RuntimeError(f"Unexpected SciPy solve residual: {residual}")
    return {"residual_norm": residual, "solution": x.tolist()}


def _check_parquet_roundtrip() -> dict[str, Any]:
    import pandas as pd
    import pyarrow  # noqa: F401

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "temperature": [20.0, 20.1, 20.2, 20.3],
            "qac": [0.0, -100.0, -200.0, 50.0],
        }
    )
    with tempfile.TemporaryDirectory(prefix="scalebridge_phasee0_parquet_") as td:
        path = Path(td) / "smoke.parquet"
        frame.to_parquet(path, index=False)
        loaded = pd.read_parquet(path)
    if list(loaded.columns) != list(frame.columns) or len(loaded) != len(frame):
        raise RuntimeError("Parquet round-trip changed schema or row count.")
    max_err = float((loaded["temperature"] - frame["temperature"]).abs().max())
    if max_err > 0.0:
        raise RuntimeError(f"Parquet round-trip changed numeric values: {max_err}")
    return {"rows": len(loaded), "columns": list(loaded.columns), "max_numeric_error": max_err}


def _check_sklearn() -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LinearRegression

    x = np.arange(8, dtype=float).reshape(-1, 1)
    y = 2.5 * x[:, 0] - 1.0
    model = LinearRegression().fit(x, y)
    prediction = float(model.predict([[10.0]])[0])
    if abs(prediction - 24.0) > 1.0e-10:
        raise RuntimeError(f"Unexpected scikit-learn prediction: {prediction}")
    return {
        "coefficient": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "prediction_at_10": prediction,
    }


def _check_torch(require_cuda: bool) -> dict[str, Any]:
    import torch

    cpu_x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    cpu_loss = (cpu_x.square()).sum()
    cpu_loss.backward()
    if cpu_x.grad is None:
        raise RuntimeError("PyTorch CPU autograd did not produce a gradient.")

    cuda_available = bool(torch.cuda.is_available())
    details: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cpu_grad": cpu_x.grad.detach().cpu().tolist(),
    }

    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for this Phase E.0 qualification but is unavailable.")

    if cuda_available:
        device = torch.device("cuda:0")
        gpu_x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        gpu_loss = (gpu_x.square()).sum()
        gpu_loss.backward()
        torch.cuda.synchronize()
        if gpu_x.grad is None:
            raise RuntimeError("PyTorch CUDA autograd did not produce a gradient.")
        details.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_grad": gpu_x.grad.detach().cpu().tolist(),
            }
        )
    return details


def _check_neuromancer() -> dict[str, Any]:
    import neuromancer as nm
    import torch

    # v1.5.x public API used in the upstream Neuromancer examples.
    model = nm.modules.blocks.MLP(
        insize=1,
        outsize=1,
        nonlin=torch.nn.ReLU,
        hsizes=[4],
    )
    x = torch.tensor([[0.25], [0.75]], dtype=torch.float32, requires_grad=True)
    y = model(x)
    if tuple(y.shape) != (2, 1):
        raise RuntimeError(f"Unexpected Neuromancer MLP output shape: {tuple(y.shape)}")
    y.sum().backward()
    if x.grad is None:
        raise RuntimeError("Neuromancer module did not preserve differentiability.")
    return {
        "neuromancer_version": _version(nm, "neuromancer"),
        "output_shape": list(y.shape),
        "input_grad_finite": bool(torch.isfinite(x.grad).all().item()),
    }


def _check_optuna_resume() -> dict[str, Any]:
    import optuna

    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        with tempfile.TemporaryDirectory(prefix="scalebridge_phasee0_optuna_") as td:
            db_path = (Path(td) / "study.sqlite3").resolve()
            storage = f"sqlite:///{db_path.as_posix()}"
            name = "phase_e0_resume_smoke"
            sampler = optuna.samplers.RandomSampler(seed=12345)

            study = optuna.create_study(
                study_name=name,
                direction="minimize",
                storage=storage,
                load_if_exists=True,
                sampler=sampler,
            )
            study.optimize(
                lambda trial: (trial.suggest_float("x", -2.0, 2.0) - 0.25) ** 2,
                n_trials=2,
            )
            reloaded = optuna.create_study(
                study_name=name,
                direction="minimize",
                storage=storage,
                load_if_exists=True,
            )
            if len(reloaded.trials) != 2:
                raise RuntimeError(
                    f"Optuna SQLite resume expected 2 trials; found {len(reloaded.trials)}."
                )
            return {
                "n_trials_after_reload": len(reloaded.trials),
                "best_value": float(reloaded.best_value),
                "best_params": dict(reloaded.best_params),
            }
    finally:
        optuna.logging.set_verbosity(previous_verbosity)


def _check_mlflow_local() -> dict[str, Any]:
    import mlflow
    from mlflow.tracking import MlflowClient

    previous_uri = mlflow.get_tracking_uri()
    try:
        with tempfile.TemporaryDirectory(prefix="scalebridge_phasee0_mlflow_") as td:
            tracking_root = (Path(td) / "mlruns").resolve()
            tracking_root.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_root.as_uri())

            experiment_name = "phase_e0_local_tracking_smoke"
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name="phase_e0_smoke") as run:
                mlflow.log_param("phase_e0", "true")
                mlflow.log_metric("objective", 1.25)
                run_id = run.info.run_id

            fetched = MlflowClient().get_run(run_id)
            if fetched.data.params.get("phase_e0") != "true":
                raise RuntimeError("MLflow local run did not preserve logged parameter.")
            metric = fetched.data.metrics.get("objective")
            if metric is None or abs(float(metric) - 1.25) > 1.0e-12:
                raise RuntimeError("MLflow local run did not preserve logged metric.")
            return {
                "run_id": run_id,
                "status": str(fetched.info.status),
                "tracking_mode": "temporary_file_store",
            }
    finally:
        mlflow.set_tracking_uri(previous_uri)


def _run_solver_subprocess(code: str, *, label: str) -> dict[str, Any]:
    """Run solver smoke tests out-of-process so native solver failures cannot kill the audit."""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} subprocess failed with exit code {completed.returncode}. "
            f"stdout={stdout!r}; stderr={stderr!r}"
        )
    payload_line = ""
    for line in reversed(stdout.splitlines()):
        if line.startswith("PHASE_E0_JSON="):
            payload_line = line.split("=", 1)[1]
            break
    if not payload_line:
        raise RuntimeError(
            f"{label} subprocess did not emit PHASE_E0_JSON payload. stdout={stdout!r}"
        )
    payload = json.loads(payload_line)
    payload["subprocess_returncode"] = completed.returncode
    return payload


def _check_casadi_ipopt() -> dict[str, Any]:
    code = r"""
import json
import casadi as ca
x = ca.MX.sym("x")
solver = ca.nlpsol(
    "phase_e0_casadi_ipopt",
    "ipopt",
    {"x": x, "f": (x - 3.0) ** 2},
    {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"},
)
solution = solver(x0=0.0)
x_value = float(solution["x"])
if abs(x_value - 3.0) > 1.0e-5:
    raise RuntimeError(f"Unexpected solution: {x_value}")
print("PHASE_E0_JSON=" + json.dumps({"solution": x_value, "solver": "casadi_ipopt"}))
"""
    return _run_solver_subprocess(code, label="CasADi IPOPT")


def _check_pyomo_ipopt() -> dict[str, Any]:
    code = r"""
import json
from pyomo.environ import ConcreteModel, Objective, SolverFactory, Var, value
solver = SolverFactory("ipopt")
available = bool(solver.available(exception_flag=False))
if not available:
    raise RuntimeError('External IPOPT is unavailable to Pyomo SolverFactory("ipopt").')
model = ConcreteModel()
model.x = Var(initialize=0.0)
model.obj = Objective(expr=(model.x - 3.0) ** 2)
result = solver.solve(model, tee=False)
x_value = float(value(model.x))
termination = str(result.solver.termination_condition)
if abs(x_value - 3.0) > 1.0e-5:
    raise RuntimeError(f"Unexpected solution: {x_value}")
try:
    executable = str(solver.executable() or "")
except Exception:
    executable = ""
print(
    "PHASE_E0_JSON="
    + json.dumps(
        {
            "solution": x_value,
            "termination_condition": termination,
            "solver_executable": executable,
        }
    )
)
"""
    return _run_solver_subprocess(code, label="Pyomo external IPOPT")


def _check_phase_d_import() -> dict[str, Any]:
    from scalebridge.data.thermal_modeling.constants import PHASE_D_SCHEMA_VERSION
    from scalebridge.data.thermal_modeling.models import ZoneSignalRecord

    return {
        "phase_d_schema_version": PHASE_D_SCHEMA_VERSION,
        "zone_signal_record": f"{ZoneSignalRecord.__module__}.{ZoneSignalRecord.__name__}",
    }


def _check_phase_c_runtime_import() -> dict[str, Any]:
    from scalebridge.models.heat_input_regression.base import HeatInputRegressionModel
    from scalebridge.models.heat_input_regression.serialization import (
        load_heat_input_regression_model,
    )

    return {
        "base_model": f"{HeatInputRegressionModel.__module__}.{HeatInputRegressionModel.__name__}",
        "loader": f"{load_heat_input_regression_model.__module__}.{load_heat_input_regression_model.__name__}",
    }


def _check_scalebridge_repo_import(repo_root: Path) -> dict[str, Any]:
    import scalebridge

    module_file = Path(scalebridge.__file__).resolve()
    expected_src = (repo_root / "src").resolve()
    try:
        module_file.relative_to(expected_src)
    except ValueError as exc:
        raise RuntimeError(
            "The active scalebridge import is not coming from this repository's src tree: "
            f"{module_file} not under {expected_src}. Activate/install the intended editable environment."
        ) from exc

    direct_url: dict[str, Any] | None = None
    try:
        dist = importlib.metadata.distribution("scalebridge")
        raw = dist.read_text("direct_url.json")
        if raw:
            direct_url = json.loads(raw)
    except Exception:
        direct_url = None

    return {
        "module_file": str(module_file),
        "expected_src": str(expected_src),
        "distribution_version": importlib.metadata.version("scalebridge"),
        "direct_url": direct_url,
    }


def _check_pip() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(
        part.strip()
        for part in (completed.stdout or "", completed.stderr or "")
        if part.strip()
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"pip check failed with exit code {completed.returncode}: {output or '<no output>'}"
        )
    return {"returncode": completed.returncode, "output": output or "No broken requirements found."}


def _write_text_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "ScaleBridge Phase E.0 Environment Qualification",
        "=" * 72,
        f"schema_version: {report['schema_version']}",
        f"generated_at: {report['generated_at']}",
        f"machine_id: {report['machine_id']}",
        f"repo_root: {report['repo_root']}",
        f"python: {report['python']['version']}",
        f"executable: {report['python']['executable']}",
        f"overall_ok: {report['overall_ok']}",
        f"required_failures: {report['summary']['required_failures']}",
        f"optional_failures: {report['summary']['optional_failures']}",
        "",
        "CHECKS",
        "-" * 72,
    ]
    for item in report["checks"]:
        status = "PASS" if item["ok"] else ("FAIL" if item["required"] else "WARN")
        req = "required" if item["required"] else "optional"
        lines.append(f"[{status:4}] {item['name']} ({req})")
        if item.get("error"):
            lines.append(f"       {item['error']}")
        details = item.get("details") or {}
        if details.get("version"):
            lines.append(f"       version={details['version']}")
        if item["name"] == "torch_autograd_device":
            lines.append(
                "       "
                f"cuda_available={details.get('cuda_available')} "
                f"cuda_build={details.get('cuda_build')} "
                f"gpu={details.get('gpu_name')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _list_checks() -> None:
    print("Required imports:")
    for label, module, _distribution in REQUIRED_IMPORTS:
        print(f"  import:{label} -> {module}")
    print("\nOptional imports:")
    for label, module, _distribution in OPTIONAL_IMPORTS:
        print(f"  import:{label} -> {module}")
    print("\nExecutable checks:")
    for name in EXECUTABLE_CHECK_NAMES:
        print(f"  {name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run executable ScaleBridge Phase E.0 environment qualification."
    )
    parser.add_argument(
        "--machine-id",
        choices=("laptop", "home-pc", "lab-pc", "kamiak"),
        help="ScaleBridge machine identity.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="ScaleBridge repository root; default is the current directory.",
    )
    parser.add_argument(
        "--output",
        help="JSON report path. A sibling .txt summary is also written.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail qualification if CUDA is unavailable.",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="List checks without importing Phase E dependencies.",
    )
    args = parser.parse_args()

    if args.list_checks:
        _list_checks()
        return 0

    if not args.machine_id:
        parser.error("--machine-id is required unless --list-checks is used.")
    if not args.output:
        parser.error("--output is required unless --list-checks is used.")

    repo_root = Path(args.repo_root).expanduser().resolve()
    if not (repo_root / "pyproject.toml").is_file() or not (repo_root / "src" / "scalebridge").is_dir():
        raise SystemExit(
            "Repository-root validation failed. Run from scalebridge-research or pass --repo-root."
        )

    checks: list[CheckResult] = []

    for label, module_name, distribution in REQUIRED_IMPORTS:
        checks.append(
            _import_check(label, module_name, distribution, required=True)
        )
    for label, module_name, distribution in OPTIONAL_IMPORTS:
        checks.append(
            _import_check(label, module_name, distribution, required=False)
        )

    checks.extend(
        [
            _run_check("python_version", _check_python_version),
            _run_check(
                "execution_context",
                lambda: _check_execution_context(args.machine_id),
            ),
            _run_check("numpy_scipy_numeric", _check_numpy_scipy),
            _run_check("pandas_pyarrow_roundtrip", _check_parquet_roundtrip),
            _run_check("sklearn_fit_predict", _check_sklearn),
            _run_check(
                "torch_autograd_device",
                lambda: _check_torch(args.require_cuda),
            ),
            _run_check("neuromancer_module", _check_neuromancer),
            _run_check("optuna_sqlite_resume", _check_optuna_resume),
            _run_check("mlflow_local_tracking", _check_mlflow_local),
            _run_check("casadi_ipopt", _check_casadi_ipopt),
            _run_check("pyomo_external_ipopt", _check_pyomo_ipopt),
            _run_check("phase_d_import", _check_phase_d_import),
            _run_check("phase_c_runtime_import", _check_phase_c_runtime_import),
            _run_check(
                "scalebridge_repo_import",
                lambda: _check_scalebridge_repo_import(repo_root),
            ),
            _run_check("pip_check", _check_pip),
        ]
    )

    required_failures = [c.name for c in checks if c.required and not c.ok]
    optional_failures = [c.name for c in checks if not c.required and not c.ok]

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine_id": args.machine_id,
        "repo_root": str(repo_root),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "environment": {
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "scale_bridge_machine_id": os.environ.get("SCALEBRIDGE_MACHINE_ID"),
            "scale_bridge_data_root": os.environ.get("SCALEBRIDGE_DATA_ROOT"),
            "scale_bridge_generated_data_root": os.environ.get(
                "SCALEBRIDGE_GENERATED_DATA_ROOT"
            ),
            "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "requirements": {
            "require_cuda": bool(args.require_cuda),
            "kamiak_requires_slurm_compute_job": True,
            "external_pyomo_ipopt_required": True,
            "casadi_ipopt_required": True,
        },
        "checks": [asdict(c) for c in checks],
        "summary": {
            "check_count": len(checks),
            "required_check_count": sum(1 for c in checks if c.required),
            "optional_check_count": sum(1 for c in checks if not c.required),
            "required_failures": required_failures,
            "optional_failures": optional_failures,
        },
        "overall_ok": not required_failures,
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    text_output = output.with_suffix(".txt")
    _write_text_summary(report, text_output)

    print("=" * 72)
    print("ScaleBridge Phase E.0 Environment Qualification")
    print("=" * 72)
    print(f"machine_id       : {args.machine_id}")
    print(f"json_report      : {output}")
    print(f"text_report      : {text_output}")
    print(f"required_failures: {len(required_failures)}")
    print(f"optional_failures: {len(optional_failures)}")
    print(f"overall_ok       : {report['overall_ok']}")
    if required_failures:
        print("Required failures:")
        for name in required_failures:
            print(f"  - {name}")
    if optional_failures:
        print("Optional warnings:")
        for name in optional_failures:
            print(f"  - {name}")

    return 0 if report["overall_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
