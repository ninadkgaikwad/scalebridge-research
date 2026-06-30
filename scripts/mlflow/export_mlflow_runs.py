from __future__ import annotations

import argparse
import csv
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


def _safe_machine_id() -> str:
    return (
        os.environ.get("SCALEBRIDGE_MACHINE_ID")
        or os.environ.get("COMPUTERNAME")
        or socket.gethostname()
    )


def _generated_data_root() -> Path:
    value = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT")
    if not value:
        raise RuntimeError("SCALEBRIDGE_GENERATED_DATA_ROOT is not configured.")
    return Path(value).resolve()


def _tracking_uri(default: str | None = None) -> str:
    uri = default or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(uri)
    return uri


def _ms_to_iso(ms: int | None) -> str:
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _flatten_run(
    *,
    machine_id: str,
    tracking_uri: str,
    experiment_name: str,
    experiment_id: str,
    run: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "machine_id": machine_id,
        "hostname": socket.gethostname(),
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "status": run.info.status,
        "lifecycle_stage": run.info.lifecycle_stage,
        "start_time_ms": run.info.start_time,
        "end_time_ms": run.info.end_time,
        "start_time_utc": _ms_to_iso(run.info.start_time),
        "end_time_utc": _ms_to_iso(run.info.end_time),
        "artifact_uri": run.info.artifact_uri,
    }

    for key, value in sorted(run.data.params.items()):
        row[f"param.{key}"] = value

    for key, value in sorted(run.data.metrics.items()):
        row[f"metric.{key}"] = value

    for key, value in sorted(run.data.tags.items()):
        row[f"tag.{key}"] = value

    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _search_runs_all_pages(
    client: MlflowClient,
    experiment_id: str,
    max_results_per_page: int,
) -> list[Any]:
    runs: list[Any] = []
    page_token: str | None = None

    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ALL,
            max_results=max_results_per_page,
            order_by=["attributes.start_time DESC"],
            page_token=page_token,
        )

        runs.extend(list(page))

        page_token = getattr(page, "token", None)
        if not page_token:
            break

    return runs


def export_mlflow_runs(
    *,
    tracking_uri: str | None,
    machine_id: str | None,
    output_dir: Path | None,
    experiment_names: list[str] | None,
    max_results_per_page: int,
) -> dict[str, Any]:
    resolved_tracking_uri = _tracking_uri(tracking_uri)
    resolved_machine_id = machine_id or _safe_machine_id()

    if output_dir is None:
        output_dir = (
            _generated_data_root()
            / "mlflow_exports"
            / resolved_machine_id
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=resolved_tracking_uri)

    all_experiments = list(client.search_experiments(view_type=ViewType.ALL))

    if experiment_names:
        wanted = set(experiment_names)
        experiments = [exp for exp in all_experiments if exp.name in wanted]
    else:
        experiments = all_experiments

    experiment_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for exp in experiments:
        experiment_rows.append(
            {
                "machine_id": resolved_machine_id,
                "tracking_uri": resolved_tracking_uri,
                "experiment_id": exp.experiment_id,
                "experiment_name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
            }
        )

        runs = _search_runs_all_pages(
            client=client,
            experiment_id=exp.experiment_id,
            max_results_per_page=max_results_per_page,
        )

        for run in runs:
            run_rows.append(
                _flatten_run(
                    machine_id=resolved_machine_id,
                    tracking_uri=resolved_tracking_uri,
                    experiment_name=exp.name,
                    experiment_id=exp.experiment_id,
                    run=run,
                )
            )

    export_timestamp = datetime.now(tz=timezone.utc).isoformat()

    experiments_csv = output_dir / "experiments.csv"
    runs_csv = output_dir / "runs.csv"
    runs_jsonl = output_dir / "runs.jsonl"
    manifest_json = output_dir / "export_manifest.json"

    _write_csv(experiments_csv, experiment_rows)
    _write_csv(runs_csv, run_rows)
    _write_jsonl(runs_jsonl, run_rows)

    manifest = {
        "export_timestamp_utc": export_timestamp,
        "machine_id": resolved_machine_id,
        "hostname": socket.gethostname(),
        "tracking_uri": resolved_tracking_uri,
        "output_dir": str(output_dir),
        "experiment_count": len(experiment_rows),
        "run_count": len(run_rows),
        "experiment_names": [row["experiment_name"] for row in experiment_rows],
        "files": {
            "experiments_csv": str(experiments_csv),
            "runs_csv": str(runs_csv),
            "runs_jsonl": str(runs_jsonl),
        },
    }

    manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export local MLflow experiments/runs to CSV and JSONL."
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or localhost.",
    )
    parser.add_argument(
        "--machine-id",
        default=None,
        help="Machine ID. Defaults to SCALEBRIDGE_MACHINE_ID or hostname.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "SCALEBRIDGE_GENERATED_DATA_ROOT/mlflow_exports/<machine_id>."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        action="append",
        default=None,
        help=(
            "Experiment name to export. Can be repeated. "
            "If omitted, all experiments are exported."
        ),
    )
    parser.add_argument(
        "--max-results-per-page",
        type=int,
        default=1000,
        help="MLflow search_runs page size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    manifest = export_mlflow_runs(
        tracking_uri=args.tracking_uri,
        machine_id=args.machine_id,
        output_dir=output_dir,
        experiment_names=args.experiment_name,
        max_results_per_page=args.max_results_per_page,
    )

    print("MLflow export completed")
    print(f"machine_id: {manifest['machine_id']}")
    print(f"tracking_uri: {manifest['tracking_uri']}")
    print(f"output_dir: {manifest['output_dir']}")
    print(f"experiment_count: {manifest['experiment_count']}")
    print(f"run_count: {manifest['run_count']}")


if __name__ == "__main__":
    main()