"""Managed execution for BGIRS Phase D Thermal-Model Data.

Tab 2 is intentionally thin.  A saved :class:`PhaseDCampaignDefinition` owns
all scientific configuration.  Execution may add only options that already
belong to the authoritative general runner's run-time interface:

* phase_d_run_id
* dry_run
* resume
* overwrite_existing
* continue_on_error

The subprocess launched here is exactly
``scripts/thermal_modeling/run_phase_d_campaign.py`` through the same
``command_for_definition`` compiler used by Campaign Builder.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from threading import RLock, Thread
from typing import Any

from scalebridge.data.thermal_modeling.campaign_runner import build_phase_d_run_id
from scalebridge.dashapp.services.system.settings_runtime import detect_current_machine

from .builder import command_for_definition, definition_summary
from .definition_store import definition_path, list_definitions, load_definition
from .upstream_phase_c import resolve_phase_c_context


ACTIVE_STATUSES = {"running", "dry_running", "stop_requested"}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")
_LOG_SEQUENCE_RE = re.compile(r"^(?P<sequence>\d+)_")
_MAX_CONSOLE_LINES = 7000
_MAX_LOG_TAIL_BYTES = 160_000


@dataclass
class ExecutionState:
    """Mutable state for the single managed Phase D campaign process."""

    status: str = "not_started"
    mode: str | None = None
    campaign_id: str | None = None
    phase_d_run_id: str | None = None
    pid: int | None = None
    command: list[str] = field(default_factory=list)
    campaign_run_root: str | None = None
    started_at: str | None = None
    started_monotonic: float | None = None
    finished_at: str | None = None
    return_code: int | None = None
    stop_requested: bool = False
    stdout: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_CONSOLE_LINES))
    stderr: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_CONSOLE_LINES))
    console: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_CONSOLE_LINES))
    process: object | None = None


def suggested_run_id() -> str:
    """Return the scientific runner's normal timestamped Phase D run ID."""
    return build_phase_d_run_id()


def _resolved_run_id(value: str | None) -> str:
    run_id = str(value or "").strip() or suggested_run_id()
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "Phase D Run ID must use only letters, numbers, '.', '_' or '-' "
            "and contain at least three characters"
        )
    return run_id


def list_execution_definitions() -> list[dict[str, object]]:
    """Return saved Campaign Builder definitions for the Execution dropdown."""
    return list_definitions()


def campaign_run_root(campaign_id: str, phase_d_run_id: str) -> Path:
    """Return the campaign-level Phase D run directory created by the runner."""
    definition = load_definition(campaign_id)
    config = definition.runner_config
    output_root = Path(config.output_root or config.campaign_root).expanduser().resolve()
    return output_root / "phase_d" / "campaign_runs" / _resolved_run_id(phase_d_run_id)


def execution_definition_summary(campaign_id: str) -> dict[str, Any]:
    """Return saved-definition information relevant to execution."""
    definition = load_definition(campaign_id)
    summary = definition_summary(definition)
    current = detect_current_machine()
    config = definition.runner_config
    return {
        **summary,
        "display_name": definition.display_name,
        "notes": definition.notes,
        "saved_machine_id": definition.machine_id,
        "current_machine_id": current["machine_id"],
        "machine_match": definition.machine_id == current["machine_id"],
        "phase_c_campaign_run_id": config.phase_c_campaign_run_id,
        "definition_path": str(definition_path(campaign_id)),
        "output_root": str(config.output_root or config.campaign_root),
    }


def command_for(
    campaign_id: str,
    *,
    phase_d_run_id: str | None = None,
    resume: bool = False,
    overwrite_existing: bool = False,
    continue_on_error: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Return the exact authoritative runner argv for one execution request."""
    definition = load_definition(campaign_id)
    return command_for_definition(
        definition,
        phase_d_run_id=_resolved_run_id(phase_d_run_id),
        resume=bool(resume),
        overwrite_existing=bool(overwrite_existing),
        continue_on_error=bool(continue_on_error),
        dry_run=bool(dry_run),
    )


def command_text(campaign_id: str, **kwargs: Any) -> str:
    """Return Windows-safe text for the exact command that will be launched."""
    return subprocess.list2cmdline(command_for(campaign_id, **kwargs))


def _selected_aggregation_rows(campaign_id: str) -> list[dict[str, Any]]:
    definition = load_definition(campaign_id)
    config = definition.runner_config
    context = resolve_phase_c_context(definition.parent_phase_c_run_key)
    cases = set(config.case_ids)
    aggregations = set(config.aggregation_ids)
    weights = set(config.weight_modes)
    rows = [
        dict(row)
        for row in (context.get("aggregation_rows") or [])
        if (not cases or row.get("case_id") in cases)
        and (not aggregations or row.get("aggregation_id") in aggregations)
        and (not weights or row.get("weight_mode") in weights)
    ]
    if config.max_aggregation_runs is not None:
        rows = rows[: max(0, int(config.max_aggregation_runs))]
    return rows


def _existing_selected_output_count(campaign_id: str) -> int:
    definition = load_definition(campaign_id)
    config = definition.runner_config
    output_root = Path(config.output_root or config.campaign_root).expanduser().resolve()
    count = 0
    for row in _selected_aggregation_rows(campaign_id):
        case_id = str(row.get("case_id") or "").strip()
        aggregation_run_id = str(row.get("aggregation_run_id") or "").strip()
        if not case_id or not aggregation_run_id:
            continue
        run_root = (
            output_root
            / "phase_d"
            / "cases"
            / case_id
            / "aggregation_runs"
            / aggregation_run_id
        )
        if run_root.exists():
            count += 1
    return count


def runtime_warnings(
    campaign_id: str,
    *,
    phase_d_run_id: str | None = None,
    resume: bool = False,
    overwrite_existing: bool = False,
    continue_on_error: bool = False,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    """Return execution-time warnings without changing the saved science."""
    if resume and overwrite_existing:
        raise ValueError("Resume and Overwrite Existing are mutually exclusive")

    run_id = _resolved_run_id(phase_d_run_id)
    definition = load_definition(campaign_id)
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

    run_root = campaign_run_root(campaign_id, run_id)
    if run_root.exists():
        rows.append(
            {
                "code": "campaign_run_id_exists",
                "severity": "danger",
                "message": (
                    f"Phase D campaign run directory already exists: {run_root}. "
                    "A new Run ID is safest unless reusing this ID is deliberate."
                ),
            }
        )

    existing = _existing_selected_output_count(campaign_id)
    if existing and not dry_run:
        if overwrite_existing:
            rows.append(
                {
                    "code": "overwrite_existing_outputs",
                    "severity": "danger",
                    "message": (
                        f"{existing} selected aggregation output director"
                        f"{'y' if existing == 1 else 'ies'} already exist. "
                        "Overwrite Existing allows the authoritative runner to remove "
                        "those selected outputs before rebuilding them."
                    ),
                }
            )
        elif resume:
            rows.append(
                {
                    "code": "resume_existing_outputs",
                    "severity": "warning",
                    "message": (
                        f"{existing} selected aggregation output director"
                        f"{'y' if existing == 1 else 'ies'} already exist. "
                        "Resume lets the scientific runner skip compatible completed "
                        "outputs and rebuild incomplete/incompatible outputs."
                    ),
                }
            )
        else:
            rows.append(
                {
                    "code": "existing_outputs_without_recovery_mode",
                    "severity": "danger",
                    "message": (
                        f"{existing} selected aggregation output director"
                        f"{'y' if existing == 1 else 'ies'} already exist. "
                        "A fresh run without Resume or Overwrite Existing will fail "
                        "when the runner encounters an existing selected output."
                    ),
                }
            )

    if continue_on_error:
        rows.append(
            {
                "code": "continue_on_error",
                "severity": "info",
                "message": (
                    "Continue on Error is enabled: the campaign will continue to later "
                    "aggregation runs after an individual aggregation failure."
                ),
            }
        )
    if dry_run:
        rows.append(
            {
                "code": "dry_run",
                "severity": "info",
                "message": (
                    "Dry Run is enabled: the runner will create its campaign plan/summary "
                    "but will not build per-aggregation Phase D datasets."
                ),
            }
        )
    return rows


def confirmation_reasons(campaign_id: str, **kwargs: Any) -> list[dict[str, str]]:
    """Return warnings that deserve explicit confirmation before execution."""
    return [
        row
        for row in runtime_warnings(campaign_id, **kwargs)
        if row.get("severity") in {"warning", "danger"}
    ]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_text(path: Path, *, max_bytes: int = _MAX_LOG_TAIL_BYTES) -> str:
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
                handle.readline()  # discard a possible partial first line
            raw = handle.read()
        return raw.decode("utf-8", errors="replace").rstrip()
    except Exception:
        return ""


def artifact_progress(campaign_id: str, phase_d_run_id: str) -> dict[str, Any]:
    """Read compact progress from the output artifacts the runner already writes."""
    root = campaign_run_root(campaign_id, phase_d_run_id)
    plan = _read_json(root / "phase_d_campaign_plan.json")
    registry = _read_csv_rows(root / "aggregation_run_registry.csv")
    datasets = _read_csv_rows(root / "dataset_registry.csv")
    failures = _read_csv_rows(root / "failures.csv")

    total = int(plan.get("selected_aggregation_run_count") or 0)
    if total <= 0:
        try:
            total = int(execution_definition_summary(campaign_id)["matched_aggregation_runs"])
        except Exception:
            total = 0

    completed = sum(1 for row in registry if row.get("status") == "completed")
    skipped = sum(1 for row in registry if row.get("status") == "skipped_completed")
    failed = sum(1 for row in registry if row.get("status") == "failed")
    planned = sum(1 for row in registry if row.get("status") == "planned")
    done = completed + skipped + failed + planned

    latest_log: Path | None = None
    log_root = root / "logs"
    if log_root.is_dir():
        logs = [path for path in log_root.glob("*.log") if path.is_file()]
        if logs:
            latest_log = max(logs, key=lambda path: path.stat().st_mtime)

    current_sequence = None
    current_aggregation_run_id = None
    latest_log_tail = ""
    latest_log_path = None
    if latest_log is not None:
        latest_log_path = str(latest_log)
        match = _LOG_SEQUENCE_RE.match(latest_log.name)
        if match:
            current_sequence = int(match.group("sequence"))
        stem = latest_log.stem
        current_aggregation_run_id = stem.split("_", 1)[1] if "_" in stem else stem
        latest_log_tail = _tail_text(latest_log)

    percent = (100.0 * done / total) if total else 0.0
    return {
        "campaign_run_root": str(root),
        "selected_aggregation_count": total,
        "completed_aggregation_count": completed,
        "skipped_aggregation_count": skipped,
        "failed_aggregation_count": failed,
        "planned_aggregation_count": planned,
        "finished_aggregation_count": done,
        "dataset_count": len(datasets),
        "failure_row_count": len(failures),
        "progress_percent": min(100.0, max(0.0, percent)),
        "current_sequence": current_sequence,
        "current_aggregation_run_id": current_aggregation_run_id,
        "latest_log_path": latest_log_path,
        "latest_log_tail": latest_log_tail,
    }


class PhaseDProcessManager:
    """Single managed Phase D subprocess with lightweight output-backed progress."""

    ACTIVE_STATUSES = ACTIVE_STATUSES

    def __init__(self):
        self._s = ExecutionState()
        self._lock = RLock()

    def reset_for_tests(self) -> None:
        with self._lock:
            process = self._s.process
            if process is not None and process.poll() is None:
                raise RuntimeError("Cannot reset while a Phase D execution is running")
            self._s = ExecutionState()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._s
            runtime_seconds = None
            if state.started_monotonic is not None:
                if state.finished_at is None:
                    runtime_seconds = max(0.0, time.monotonic() - state.started_monotonic)
                else:
                    runtime_seconds = getattr(state, "_runtime_seconds", None)
            snapshot = {
                "status": state.status,
                "mode": state.mode,
                "campaign_id": state.campaign_id,
                "phase_d_run_id": state.phase_d_run_id,
                "pid": state.pid,
                "command": subprocess.list2cmdline(state.command) if state.command else "",
                "campaign_run_root": state.campaign_run_root,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "runtime_seconds": runtime_seconds,
                "return_code": state.return_code,
                "stop_requested": state.stop_requested,
                "stdout": "\n".join(state.stdout),
                "stderr": "\n".join(state.stderr),
                "console": "\n".join(state.console),
            }

        progress: dict[str, Any] = {}
        if snapshot["campaign_id"] and snapshot["phase_d_run_id"]:
            try:
                progress = artifact_progress(
                    str(snapshot["campaign_id"]),
                    str(snapshot["phase_d_run_id"]),
                )
            except Exception:
                progress = {}
        snapshot.update(progress)
        return snapshot

    def start(
        self,
        campaign_id: str,
        *,
        phase_d_run_id: str | None = None,
        resume: bool = False,
        overwrite_existing: bool = False,
        continue_on_error: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Launch exactly one authoritative general Phase D runner subprocess."""
        with self._lock:
            if self._s.process is not None and self._s.process.poll() is None:
                raise RuntimeError("A Phase D execution is already running")

            run_id = _resolved_run_id(phase_d_run_id)
            cmd = command_for(
                campaign_id,
                phase_d_run_id=run_id,
                resume=resume,
                overwrite_existing=overwrite_existing,
                continue_on_error=continue_on_error,
                dry_run=dry_run,
            )
            run_root = campaign_run_root(campaign_id, run_id)

            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
            env["PYTHONUNBUFFERED"] = "1"

            kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
                "cwd": str(Path(__file__).resolve().parents[5]),
                "env": env,
            }
            # Resolve the repository root robustly rather than trusting parents[5].
            here = Path(__file__).resolve()
            for parent in here.parents:
                if (parent / "pyproject.toml").is_file() and (parent / "scripts").is_dir():
                    kwargs["cwd"] = str(parent)
                    break

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
                phase_d_run_id=run_id,
                pid=process.pid,
                command=cmd,
                campaign_run_root=str(run_root),
                started_at=datetime.now(timezone.utc).isoformat(),
                started_monotonic=time.monotonic(),
                process=process,
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
            if self._s.stop_requested:
                self._s.status = "stopped"
            elif rc == 0:
                self._s.status = (
                    "dry_run_completed" if self._s.mode == "dry_run" else "completed"
                )
            else:
                self._s.status = "failed"

    def stop(self) -> None:
        """Terminate the authoritative runner and its child aggregation process tree."""
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


MANAGER = PhaseDProcessManager()
