"""
Environment smoke test for ScaleBridge.

Run:
    conda activate scalebridge-dev
    python check_scalebridge_environment.py
"""

from __future__ import annotations

import importlib
import platform
import sys


def check_import(module_name: str, import_name: str | None = None) -> None:
    name = import_name or module_name
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "version unavailable")
        print(f"[OK] {module_name:<24} {version}")
    except Exception as exc:
        print(f"[FAIL] {module_name:<24} {type(exc).__name__}: {exc}")


def main() -> None:
    print("=" * 80)
    print("ScaleBridge environment smoke test")
    print("=" * 80)
    print(f"Python:   {sys.version}")
    print(f"Platform: {platform.platform()}")
    print("-" * 80)

    packages = [
        ("numpy", None),
        ("pandas", None),
        ("scipy", None),
        ("sklearn", "sklearn"),
        ("torch", None),
        ("torchvision", None),
        ("torchaudio", None),
        ("pytorch_lightning", "lightning"),
        ("neuromancer", None),
        ("torchdiffeq", None),
        ("torchsde", None),
        ("casadi", None),
        ("pyomo", None),
        ("cvxpy", None),
        ("cvxpylayers", None),
        ("stable_baselines3", None),
        ("gymnasium", None),
        ("mlflow", None),
        ("optuna", None),
        ("ray", None),
        ("opyplus", None),
        ("matplotlib", None),
        ("seaborn", None),
        ("plotly", None),
        ("geopandas", None),
        ("sqlalchemy", None),
    ]

    for label, import_name in packages:
        check_import(label, import_name)

    print("-" * 80)

    try:
        import torch

        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.version.cuda:          {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"CUDA device count:          {torch.cuda.device_count()}")
            print(f"CUDA device 0:              {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"[FAIL] CUDA check: {type(exc).__name__}: {exc}")

    print("=" * 80)


if __name__ == "__main__":
    main()
