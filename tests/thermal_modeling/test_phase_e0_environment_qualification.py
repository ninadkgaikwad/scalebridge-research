# -*- coding: utf-8 -*-
"""Structural tests for the Phase E0 environment qualification CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "thermal_modeling" / "qualify_phase_e0_environment.py"


def test_phase_e0_environment_qualifier_exists() -> None:
    assert SCRIPT.is_file()


def test_phase_e0_environment_qualifier_lists_checks_without_optional_imports() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-checks"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "neuromancer_module" in output
    assert "optuna_sqlite_resume" in output
    assert "mlflow_local_tracking" in output
    assert "casadi_ipopt" in output
    assert "pyomo_external_ipopt" in output
    assert "scalebridge_repo_import" in output
    assert "pip_check" in output


def test_phase_e0_environment_qualifier_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--machine-id" in completed.stdout
    assert "--require-cuda" in completed.stdout
