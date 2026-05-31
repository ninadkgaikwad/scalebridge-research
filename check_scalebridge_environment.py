"""
ScaleBridge environment smoke test.

Run from repository root after activating the target environment:

    python check_scalebridge_environment.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass


@dataclass
class ImportCheck:
    label: str
    module_name: str
    required: bool = True


CHECKS = [
    ImportCheck("numpy", "numpy"),
    ImportCheck("pandas", "pandas"),
    ImportCheck("scipy", "scipy"),
    ImportCheck("matplotlib", "matplotlib"),
    ImportCheck("sklearn", "sklearn"),
    ImportCheck("torch", "torch"),
    ImportCheck("neuromancer", "neuromancer"),
    ImportCheck("casadi", "casadi"),
    ImportCheck("pyomo", "pyomo"),
    ImportCheck("cvxpy", "cvxpy"),
    ImportCheck("stable_baselines3", "stable_baselines3"),
    ImportCheck("gymnasium", "gymnasium"),
    ImportCheck("mlflow", "mlflow"),
    ImportCheck("optuna", "optuna"),
    ImportCheck("ray", "ray"),
    ImportCheck("plotly", "plotly"),
    ImportCheck("dash", "dash"),
    ImportCheck("opyplus", "opyplus", required=False),
    ImportCheck("eppy", "eppy", required=False),
    ImportCheck("opendssdirect", "opendssdirect", required=False),
    ImportCheck("dss", "dss", required=False),
    ImportCheck("scalebridge", "scalebridge", required=False),
]


def check_import(item: ImportCheck) -> tuple[bool, str]:
    try:
        module = importlib.import_module(item.module_name)
        version = getattr(module, "__version__", "version unknown")
        return True, str(version)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print("=" * 72)
    print("SCALEBRIDGE ENVIRONMENT SMOKE TEST")
    print("=" * 72)
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version}")
    print(f"Platform:          {platform.platform()}")
    print("=" * 72)

    failures: list[str] = []

    for item in CHECKS:
        ok, detail = check_import(item)
        status = "PASS" if ok else ("FAIL" if item.required else "WARN")
        print(f"{status:>5} | {item.label:<22} | {detail}")

        if not ok and item.required:
            failures.append(item.label)

    print("=" * 72)

    try:
        import torch

        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.version.cuda:         {getattr(torch.version, 'cuda', None)}")
        if torch.cuda.is_available():
            print(f"torch.cuda.device_count():  {torch.cuda.device_count()}")
            print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"Could not run torch CUDA check: {type(exc).__name__}: {exc}")

    print("=" * 72)

    if failures:
        print("REQUIRED IMPORT FAILURES:")
        for name in failures:
            print(f"- {name}")
        return 1

    print("All required imports passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
