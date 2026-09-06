"""Managed subprocess execution for BGIRS Phase B Aggregation campaigns.

Dash owns only process lifecycle and console capture. Scientific execution is
delegated unchanged to scripts/aggregation/run_aggregation_campaign.py, which
loads the saved B1 AggregationCampaignDefinition and invokes the B2 runner.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import RLock, Thread

from .definition_store import definition_path, load_definition


@dataclass
class State:
    status: str = "not_started"
    campaign_id: str | None = None
    pid: int | None = None
    command: list[str] = field(default_factory=list)
    started_at: str | None = None
    return_code: int | None = None
    stop_requested: bool = False
    output: deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    process: object | None = None


def _repo_root() -> Path:
    """Resolve the repository root from this installed source file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "scripts").is_dir():
            return parent
    raise FileNotFoundError(
        "Could not resolve scalebridge-research repository root from "
        f"{Path(__file__).resolve()}"
    )


def runner_script() -> Path:
    path = _repo_root() / "scripts" / "aggregation" / "run_aggregation_campaign.py"
    if not path.is_file():
        raise FileNotFoundError(f"Aggregation campaign runner not found: {path}")
    return path


def command_for(campaign_id: str) -> list[str]:
    """Return the exact B2 CLI command used for a saved Aggregation definition."""
    script = runner_script()
    return [
        sys.executable,
        "-u",
        str(script),
        "--campaign-definition",
        str(definition_path(campaign_id)),
    ]


def command_text(campaign_id: str) -> str:
    """Human-readable command preview for the Execution tab."""
    return subprocess.list2cmdline(command_for(campaign_id))


def definition_summary(campaign_id: str) -> dict[str, object]:
    """Return execution-relevant metadata without creating plans or runs."""
    definition = load_definition(campaign_id)
    return {
        "aggregation_campaign_id": definition.aggregation_campaign_id,
        "parent_generation_campaign_id": definition.parent_generation_campaign_id,
        "machine_id": definition.machine_id,
        "selected_case_count": len(definition.case_ids),
        "case_limit": definition.case_limit,
        "plan_request_count": len(definition.plan_requests),
        "strategies": definition.requested_strategy_values,
        "weight_modes": definition.requested_weight_mode_values,
        "rule_sets": sorted({request.rule_set.value for request in definition.plan_requests}),
        "max_variables": definition.max_variables,
        "preview_rows": definition.preview_rows,
        "write_legacy_pickle": definition.write_legacy_pickle,
        "continue_on_error": definition.continue_on_error,
        "mlflow_enabled": definition.mlflow_enabled,
        "mlflow_tracking_uri": definition.mlflow_tracking_uri,
        "mlflow_experiment_name": definition.mlflow_experiment_name,
        "definition_path": str(definition_path(campaign_id)),
        "command": command_text(campaign_id),
    }


class AggregationProcessManager:
    """Single-process manager mirroring the proven Generation lifecycle."""

    ACTIVE_STATUSES = {"running", "stop_requested"}

    def __init__(self):
        self._s = State()
        self._lock = RLock()

    def reset_for_tests(self) -> None:
        """Reset state only when no child process is active."""
        with self._lock:
            if self._s.process is not None and self._s.process.poll() is None:
                raise RuntimeError("Cannot reset while an Aggregation campaign is running")
            self._s = State()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            s = self._s
            return {
                "status": s.status,
                "campaign_id": s.campaign_id,
                "pid": s.pid,
                "command": subprocess.list2cmdline(s.command) if s.command else "",
                "started_at": s.started_at,
                "return_code": s.return_code,
                "stop_requested": s.stop_requested,
                "console": "\n".join(s.output),
            }

    def start(self, campaign_id: str) -> None:
        """Launch the authoritative B2 CLI for one saved campaign definition."""
        with self._lock:
            if self._s.process is not None and self._s.process.poll() is None:
                raise RuntimeError("An Aggregation campaign is already running")

            # Validate the saved B1 definition before spawning.
            load_definition(campaign_id)

            script = runner_script()
            cmd = command_for(campaign_id)
            kwargs: dict[str, object] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
                "cwd": str(_repo_root()),
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True

            process = subprocess.Popen(cmd, **kwargs)
            self._s = State(
                status="running",
                campaign_id=campaign_id,
                pid=process.pid,
                command=cmd,
                started_at=datetime.now(timezone.utc).isoformat(),
                process=process,
            )
            Thread(target=self._reader, args=(process,), daemon=True).start()

    def _reader(self, process) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self._lock:
                self._s.output.append(line.rstrip())
        rc = process.wait()
        with self._lock:
            self._s.return_code = rc
            if self._s.stop_requested:
                self._s.status = "stopped"
            else:
                self._s.status = "completed" if rc == 0 else "failed"

    def stop(self) -> None:
        """Terminate the complete Aggregation subprocess tree."""
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


MANAGER = AggregationProcessManager()
