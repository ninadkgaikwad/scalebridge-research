"""Managed execution for BGIRS Phase C Heat-Input Regression.

Dash owns process lifecycle, runtime overrides, and console/status presentation.
Scientific work remains delegated to the authoritative generalized runner:
``scripts/heat_input_regression/run_phase_c_campaign.py``.

A saved :class:`HeatInputCampaignDefinition` contains the complete
``PhaseCCampaignConfig``.  Tab 2 may override only execution-time fields that
are already part of that public runner contract: phase_c_run_id, start_stage,
stop_stage, and overwrite_existing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from threading import RLock, Thread
from typing import Any

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig
from scalebridge.dashapp.services.system.settings_runtime import detect_current_machine

from .definition_store import (
    definition_path,
    definition_root,
    list_definitions,
    load_definition,
)


STAGES = tuple(f"C{i}" for i in range(1, 10))
ACTIVE_STATUSES = {"running", "dry_running", "stop_requested"}

_COMMAND_PROGRESS_RE = re.compile(
    r"^\[(?P<sequence>\d+)/(?P<total>\d+)\]\s+(?P<name>.+?)\s*$"
)
_STAGE_RE = re.compile(r"\b(C[1-9])\b")
_RUN_ID_RE = re.compile(r"\d{8}_\d{6}$")


@dataclass
class ExecutionState:
    """Mutable state owned by the single Phase C process manager."""

    status: str = "not_started"
    mode: str | None = None
    campaign_id: str | None = None
    phase_c_run_id: str | None = None
    pid: int | None = None
    command: list[str] = field(default_factory=list)
    effective_config_path: str | None = None
    started_at: str | None = None
    started_monotonic: float | None = None
    finished_at: str | None = None
    return_code: int | None = None
    stop_requested: bool = False
    stdout: deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    stderr: deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    console: deque[str] = field(default_factory=lambda: deque(maxlen=7000))
    process: object | None = None
    current_stage: str | None = None
    current_command: str | None = None
    command_sequence: int | None = None
    command_total: int | None = None
    stage_statuses: dict[str, str] = field(
        default_factory=lambda: {stage: "pending" for stage in STAGES}
    )


def _repo_root() -> Path:
    """Resolve the active scalebridge-research repository root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "scripts").is_dir():
            return parent
    raise FileNotFoundError(
        "Could not resolve scalebridge-research repository root from "
        f"{Path(__file__).resolve()}"
    )


def runner_script() -> Path:
    """Return the authoritative generalized Phase C runner."""
    path = _repo_root() / "scripts" / "heat_input_regression" / "run_phase_c_campaign.py"
    if not path.is_file():
        raise FileNotFoundError(f"Phase C runner not found: {path}")
    return path


def suggested_run_id(now: datetime | None = None) -> str:
    """Return a portable default Phase C execution/run ID."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"phase_c_{stamp}"


def execution_request_root() -> Path:
    """Return machine-local/generated storage for effective execution configs."""
    # definition_root() == <generated>/campaign_definitions/heat_input
    generated_root = definition_root().parents[1]
    root = generated_root / "execution_requests" / "heat_input"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    if not cleaned:
        raise ValueError("Execution token must not be empty")
    return cleaned


def execution_config_path(campaign_id: str, phase_c_run_id: str) -> Path:
    """Return deterministic runtime-config path used by command preview/start."""
    return (
        execution_request_root()
        / _safe_token(campaign_id)
        / _safe_token(phase_c_run_id)
        / "phase_c_effective_config.json"
    )


def effective_config(
    campaign_id: str,
    *,
    phase_c_run_id: str | None = None,
    start_stage: str | None = None,
    stop_stage: str | None = None,
    overwrite_existing: bool | None = None,
) -> PhaseCCampaignConfig:
    """Return the validated runner config after narrow Tab-2 runtime overrides."""
    definition = load_definition(campaign_id)
    config = definition.runner_config

    updates: dict[str, Any] = {}
    resolved_run_id = str(
        phase_c_run_id
        or config.phase_c_run_id
        or suggested_run_id()
    ).strip()
    if resolved_run_id:
        if not _RUN_ID_RE.search(resolved_run_id):
            raise ValueError("phase_c_run_id must end in YYYYMMDD_HHMMSS")
        updates["phase_c_run_id"] = resolved_run_id
    if start_stage:
        updates["start_stage"] = str(start_stage)
    if stop_stage:
        updates["stop_stage"] = str(stop_stage)
    if overwrite_existing is not None:
        updates["overwrite_existing"] = bool(overwrite_existing)

    return PhaseCCampaignConfig.model_validate(
        config.model_copy(update=updates).to_dict()
    )


def materialize_effective_config(
    campaign_id: str,
    *,
    phase_c_run_id: str | None = None,
    start_stage: str | None = None,
    stop_stage: str | None = None,
    overwrite_existing: bool | None = None,
) -> tuple[PhaseCCampaignConfig, Path]:
    """Persist the exact raw PhaseCCampaignConfig consumed by the runner."""
    config = effective_config(
        campaign_id,
        phase_c_run_id=phase_c_run_id,
        start_stage=start_stage,
        stop_stage=stop_stage,
        overwrite_existing=overwrite_existing,
    )
    if not config.phase_c_run_id:
        raise ValueError("phase_c_run_id must be resolved before execution")
    path = execution_config_path(campaign_id, config.phase_c_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config, path


def command_for(
    campaign_id: str,
    *,
    phase_c_run_id: str | None = None,
    start_stage: str | None = None,
    stop_stage: str | None = None,
    overwrite_existing: bool | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Return the exact argv list Tab 2 will launch.

    The effective config path is deterministic and may not exist until
    ``materialize_effective_config`` is called immediately before launch.
    """
    config = effective_config(
        campaign_id,
        phase_c_run_id=phase_c_run_id,
        start_stage=start_stage,
        stop_stage=stop_stage,
        overwrite_existing=overwrite_existing,
    )
    if not config.phase_c_run_id:
        raise ValueError("phase_c_run_id must be resolved before command construction")
    cmd = [
        sys.executable,
        "-u",
        str(runner_script()),
        "--config",
        str(execution_config_path(campaign_id, config.phase_c_run_id)),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def command_text(*args, **kwargs) -> str:
    """Return Windows-safe human-readable command preview."""
    return subprocess.list2cmdline(command_for(*args, **kwargs))


def effective_config_text(campaign_id: str, **overrides: Any) -> str:
    """Return normalized JSON displayed before execution."""
    config = effective_config(campaign_id, **overrides)
    return json.dumps(config.to_dict(), indent=2, sort_keys=True)


def definition_summary(campaign_id: str) -> dict[str, Any]:
    """Return execution-relevant saved-definition metadata."""
    definition = load_definition(campaign_id)
    config = definition.runner_config
    current_machine = detect_current_machine()
    return {
        "phase_c_campaign_id": definition.phase_c_campaign_id,
        "display_name": definition.display_name,
        "parent_aggregation_campaign_id": definition.parent_aggregation_campaign_id,
        "parent_generation_campaign_id": definition.parent_generation_campaign_id,
        "saved_machine_id": definition.machine_id,
        "current_machine_id": current_machine["machine_id"],
        "machine_match": definition.machine_id == current_machine["machine_id"],
        "matrix_run_id": config.matrix_run_id,
        "saved_phase_c_run_id": config.phase_c_run_id,
        "start_stage": config.start_stage,
        "stop_stage": config.stop_stage,
        "overwrite_existing": config.overwrite_existing,
        "validation_profile": config.validation_profile,
        "estimators": list(config.estimator_types),
        "pytorch_devices": list(config.pytorch_devices),
        "mlflow_enabled": config.mlflow_enabled,
        "definition_path": str(definition_path(campaign_id)),
    }


def runtime_warnings(
    campaign_id: str,
    *,
    phase_c_run_id: str | None = None,
    start_stage: str | None = None,
    stop_stage: str | None = None,
    overwrite_existing: bool | None = None,
) -> list[dict[str, str]]:
    """Return execution warnings and confirmation reasons."""
    definition = load_definition(campaign_id)
    config = effective_config(
        campaign_id,
        phase_c_run_id=phase_c_run_id,
        start_stage=start_stage,
        stop_stage=stop_stage,
        overwrite_existing=overwrite_existing,
    )
    current = detect_current_machine()
    rows: list[dict[str, str]] = []

    if definition.machine_id != current["machine_id"]:
        rows.append(
            {
                "code": "machine_mismatch",
                "severity": "warning",
                "message": (
                    f"Definition was saved for machine {definition.machine_id!r}, "
                    f"but the current machine is {current['machine_id']!r}."
                ),
            }
        )
    if config.overwrite_existing:
        rows.append(
            {
                "code": "overwrite",
                "severity": "danger",
                "message": (
                    "overwrite_existing=true: stages that implement replacement semantics "
                    "may replace existing C6/C8 artifacts."
                ),
            }
        )
    if config.start_stage != "C1" or config.stop_stage != "C9":
        rows.append(
            {
                "code": "recovery_range",
                "severity": "warning",
                "message": (
                    f"Recovery/partial stage range selected: "
                    f"{config.start_stage} → {config.stop_stage}."
                ),
            }
        )

    if config.phase_c_run_id and config.campaign_root:
        run_root = (
            Path(config.campaign_root)
            / "heat_input_regression"
            / "campaign_runs"
            / config.phase_c_run_id
        )
        if run_root.exists():
            rows.append(
                {
                    "code": "run_id_exists",
                    "severity": (
                        "warning"
                        if config.start_stage != "C1" or config.stop_stage != "C9"
                        else "danger"
                    ),
                    "message": (
                        f"Phase C run directory already exists: {run_root}. "
                        "Use a new run ID for a fresh campaign, or confirm a "
                        "deliberate recovery run."
                    ),
                }
            )

    explicit_cuda = (
        "cuda" in config.pytorch_devices
        or "cuda" in config.c5_pytorch_devices
        or "cuda" in config.evaluation_requested_devices
        or "cuda" in config.inference_requested_devices
    )
    if explicit_cuda:
        available: bool | None
        try:
            import torch

            available = bool(torch.cuda.is_available())
        except Exception:
            available = None
        if available is False:
            rows.append(
                {
                    "code": "cuda_unavailable",
                    "severity": "danger",
                    "message": (
                        "An explicit CUDA device is requested, but torch.cuda.is_available() "
                        "is false in the current Dash runtime."
                    ),
                }
            )
        elif available is None:
            rows.append(
                {
                    "code": "cuda_unverified",
                    "severity": "warning",
                    "message": (
                        "Explicit CUDA requested, but CUDA availability could "
                        "not be verified."
                    ),
                }
            )

    return rows


def confirmation_reasons(*args, **kwargs) -> list[dict[str, str]]:
    """Return warnings that require explicit Start confirmation."""
    return runtime_warnings(*args, **kwargs)


def list_execution_definitions() -> list[dict[str, Any]]:
    """Return saved Phase C definitions for the Execution dropdown."""
    return list_definitions()


def _initial_stage_statuses(config: PhaseCCampaignConfig) -> dict[str, str]:
    start = STAGES.index(config.start_stage)
    stop = STAGES.index(config.stop_stage)
    return {
        stage: ("pending" if start <= idx <= stop else "skipped")
        for idx, stage in enumerate(STAGES)
    }


class PhaseCProcessManager:
    """Single managed Phase C subprocess with stdout/stderr/progress capture."""

    ACTIVE_STATUSES = ACTIVE_STATUSES

    def __init__(self):
        self._s = ExecutionState()
        self._lock = RLock()

    def reset_for_tests(self) -> None:
        """Reset state only when no child process is active."""
        with self._lock:
            process = self._s.process
            if process is not None and process.poll() is None:
                raise RuntimeError("Cannot reset while a Phase C execution is running")
            self._s = ExecutionState()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._s
            runtime_seconds = None
            if state.started_monotonic is not None:
                if state.finished_at is None:
                    runtime_seconds = max(0.0, time.monotonic() - state.started_monotonic)
                else:
                    # Terminal snapshot keeps the last computed runtime if appended to state.
                    runtime_seconds = getattr(state, "_runtime_seconds", None)
            return {
                "status": state.status,
                "mode": state.mode,
                "campaign_id": state.campaign_id,
                "phase_c_run_id": state.phase_c_run_id,
                "pid": state.pid,
                "command": (
                    subprocess.list2cmdline(state.command)
                    if state.command
                    else ""
                ),
                "effective_config_path": state.effective_config_path,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "runtime_seconds": runtime_seconds,
                "return_code": state.return_code,
                "stop_requested": state.stop_requested,
                "stdout": "\n".join(state.stdout),
                "stderr": "\n".join(state.stderr),
                "console": "\n".join(state.console),
                "current_stage": state.current_stage,
                "current_command": state.current_command,
                "command_sequence": state.command_sequence,
                "command_total": state.command_total,
                "stage_statuses": dict(state.stage_statuses),
            }

    def start(
        self,
        campaign_id: str,
        *,
        phase_c_run_id: str | None = None,
        start_stage: str | None = None,
        stop_stage: str | None = None,
        overwrite_existing: bool | None = None,
        dry_run: bool = False,
    ) -> None:
        """Launch one generalized Phase C runner subprocess."""
        with self._lock:
            if self._s.process is not None and self._s.process.poll() is None:
                raise RuntimeError("A Phase C execution is already running")

            config, config_path = materialize_effective_config(
                campaign_id,
                phase_c_run_id=phase_c_run_id,
                start_stage=start_stage,
                stop_stage=stop_stage,
                overwrite_existing=overwrite_existing,
            )
            cmd = command_for(
                campaign_id,
                phase_c_run_id=config.phase_c_run_id,
                start_stage=config.start_stage,
                stop_stage=config.stop_stage,
                overwrite_existing=config.overwrite_existing,
                dry_run=dry_run,
            )
            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
            kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
                "cwd": str(_repo_root()),
                "env": env,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True

            process = subprocess.Popen(cmd, **kwargs)
            mode = "dry_run" if dry_run else "execute"
            self._s = ExecutionState(
                status="dry_running" if dry_run else "running",
                mode=mode,
                campaign_id=campaign_id,
                phase_c_run_id=config.phase_c_run_id,
                pid=process.pid,
                command=cmd,
                effective_config_path=str(config_path),
                started_at=datetime.now(timezone.utc).isoformat(),
                started_monotonic=time.monotonic(),
                process=process,
                stage_statuses=_initial_stage_statuses(config),
            )

            Thread(
                target=self._reader_stream,
                args=(process.stdout, "stdout"),
                daemon=True,
            ).start()
            Thread(
                target=self._reader_stream,
                args=(process.stderr, "stderr"),
                daemon=True,
            ).start()
            Thread(target=self._watcher, args=(process,), daemon=True).start()

    def _reader_stream(self, stream, stream_name: str) -> None:
        if stream is None:
            return
        for line in stream:
            text = line.rstrip()
            with self._lock:
                target = self._s.stdout if stream_name == "stdout" else self._s.stderr
                target.append(text)
                prefix = "" if stream_name == "stdout" else "[stderr] "
                self._s.console.append(prefix + text)
                if stream_name == "stdout":
                    self._update_progress(text)

    def _update_progress(self, line: str) -> None:
        match = _COMMAND_PROGRESS_RE.match(line)
        if not match:
            return
        name = match.group("name")
        stage_match = _STAGE_RE.search(name)
        if not stage_match:
            return
        stage = stage_match.group(1)
        previous = self._s.current_stage
        if previous and previous != stage and self._s.stage_statuses.get(previous) == "running":
            self._s.stage_statuses[previous] = "completed"
        self._s.current_stage = stage
        self._s.current_command = name
        self._s.command_sequence = int(match.group("sequence"))
        self._s.command_total = int(match.group("total"))
        self._s.stage_statuses[stage] = "running"

    def _watcher(self, process) -> None:
        rc = process.wait()
        finished = datetime.now(timezone.utc).isoformat()
        with self._lock:
            runtime = None
            if self._s.started_monotonic is not None:
                runtime = max(0.0, time.monotonic() - self._s.started_monotonic)
            setattr(self._s, "_runtime_seconds", runtime)
            self._s.return_code = rc
            self._s.finished_at = finished
            current = self._s.current_stage
            if self._s.stop_requested:
                self._s.status = "stopped"
                if current and self._s.stage_statuses.get(current) == "running":
                    self._s.stage_statuses[current] = "stopped"
            elif rc == 0:
                self._s.status = (
                    "dry_run_completed"
                    if self._s.mode == "dry_run"
                    else "completed"
                )
                if current and self._s.stage_statuses.get(current) == "running":
                    self._s.stage_statuses[current] = "completed"
            else:
                self._s.status = "failed"
                if current and self._s.stage_statuses.get(current) == "running":
                    self._s.stage_statuses[current] = "failed"

    def stop(self) -> None:
        """Request termination of the complete Phase C subprocess tree."""
        with self._lock:
            process = self._s.process
            if process is None or process.poll() is not None:
                return
            self._s.stop_requested = True
            self._s.status = "stop_requested"
            pid = process.pid

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass


MANAGER = PhaseCProcessManager()
