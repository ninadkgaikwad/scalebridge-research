from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _generated_data_root() -> Path:
    value = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT")
    if not value:
        raise RuntimeError("SCALEBRIDGE_GENERATED_DATA_ROOT is not configured.")
    return Path(value).resolve()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    if path.stat().st_size == 0:
        return []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    preferred_first = [
        "global_run_key",
        "machine_id",
        "experiment_name",
        "run_name",
        "run_id",
        "status",
        "start_time_utc",
        "end_time_utc",
        "tracking_uri",
        "artifact_uri",
        "source_export_dir",
        "source_runs_csv",
    ]

    for key in preferred_first:
        for row in rows:
            if key in row and key not in seen:
                seen.add(key)
                fieldnames.append(key)
                break

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


def _machine_from_export_dir(export_dir: Path) -> str:
    return export_dir.name


def _augment_run_row(
    row: dict[str, Any],
    *,
    export_dir: Path,
    runs_csv: Path,
) -> dict[str, Any]:
    out = dict(row)

    machine_id = out.get("machine_id") or _machine_from_export_dir(export_dir)
    run_id = out.get("run_id", "")

    out["machine_id"] = machine_id
    out["source_export_dir"] = str(export_dir)
    out["source_runs_csv"] = str(runs_csv)
    out["global_run_key"] = f"{machine_id}::{run_id}"

    return out


def _sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("experiment_name", "")),
        str(row.get("machine_id", "")),
        str(row.get("start_time_utc", "")),
        str(row.get("run_id", "")),
    )


def merge_mlflow_exports(
    *,
    exports_root: Path | None,
    output_dir: Path | None,
    allow_duplicate_global_run_keys: bool,
) -> dict[str, Any]:
    generated_root = _generated_data_root()

    if exports_root is None:
        exports_root = generated_root / "mlflow_exports"
    exports_root = exports_root.resolve()

    if output_dir is None:
        output_dir = generated_root / "experiment_registry"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    experiment_rows: list[dict[str, Any]] = []

    export_dirs = sorted(
        path for path in exports_root.iterdir()
        if path.is_dir()
    ) if exports_root.exists() else []

    included_exports: list[dict[str, Any]] = []
    missing_runs_csv: list[str] = []

    for export_dir in export_dirs:
        runs_csv = export_dir / "runs.csv"
        experiments_csv = export_dir / "experiments.csv"
        manifest_json = export_dir / "export_manifest.json"

        if not runs_csv.exists():
            missing_runs_csv.append(str(runs_csv))
            continue

        local_runs = _read_csv(runs_csv)
        local_experiments = _read_csv(experiments_csv)

        for row in local_runs:
            run_rows.append(
                _augment_run_row(
                    row,
                    export_dir=export_dir,
                    runs_csv=runs_csv,
                )
            )

        for row in local_experiments:
            exp_row = dict(row)
            exp_row["source_export_dir"] = str(export_dir)
            exp_row["source_experiments_csv"] = str(experiments_csv)
            experiment_rows.append(exp_row)

        included_exports.append(
            {
                "machine_id": _machine_from_export_dir(export_dir),
                "export_dir": str(export_dir),
                "runs_csv": str(runs_csv),
                "experiments_csv": str(experiments_csv),
                "manifest_json": str(manifest_json),
                "run_count": len(local_runs),
                "experiment_count": len(local_experiments),
            }
        )

    duplicate_keys: dict[str, int] = {}
    seen_keys: set[str] = set()
    deduped_rows: list[dict[str, Any]] = []

    for row in run_rows:
        key = row.get("global_run_key", "")
        if key in seen_keys:
            duplicate_keys[key] = duplicate_keys.get(key, 1) + 1
            if allow_duplicate_global_run_keys:
                deduped_rows.append(row)
        else:
            seen_keys.add(key)
            deduped_rows.append(row)

    if duplicate_keys and not allow_duplicate_global_run_keys:
        duplicates_preview = list(duplicate_keys.keys())[:10]
        raise RuntimeError(
            "Duplicate global_run_key values found. "
            f"Examples: {duplicates_preview}. "
            "Use --allow-duplicate-global-run-keys to keep duplicates."
        )

    deduped_rows = sorted(deduped_rows, key=_sort_key)
    experiment_rows = sorted(
        experiment_rows,
        key=lambda row: (
            str(row.get("experiment_name", "")),
            str(row.get("machine_id", "")),
            str(row.get("experiment_id", "")),
        ),
    )

    all_runs_csv = output_dir / "all_runs_merged.csv"
    all_runs_jsonl = output_dir / "all_runs_merged.jsonl"
    all_experiments_csv = output_dir / "all_experiments_merged.csv"
    manifest_json = output_dir / "merge_manifest.json"

    _write_csv(all_runs_csv, deduped_rows)
    _write_jsonl(all_runs_jsonl, deduped_rows)
    _write_csv(all_experiments_csv, experiment_rows)

    machine_ids = sorted({str(row.get("machine_id", "")) for row in deduped_rows})
    experiment_names = sorted(
        {str(row.get("experiment_name", "")) for row in deduped_rows}
    )

    manifest = {
        "merge_timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "exports_root": str(exports_root),
        "output_dir": str(output_dir),
        "included_export_count": len(included_exports),
        "included_exports": included_exports,
        "missing_runs_csv": missing_runs_csv,
        "run_count_raw": len(run_rows),
        "run_count_merged": len(deduped_rows),
        "experiment_row_count": len(experiment_rows),
        "machine_ids": machine_ids,
        "experiment_names": experiment_names,
        "duplicate_global_run_keys": duplicate_keys,
        "files": {
            "all_runs_csv": str(all_runs_csv),
            "all_runs_jsonl": str(all_runs_jsonl),
            "all_experiments_csv": str(all_experiments_csv),
            "manifest_json": str(manifest_json),
        },
    }

    manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge machine-local MLflow export CSVs into one registry."
    )
    parser.add_argument(
        "--exports-root",
        default=None,
        help=(
            "Root containing per-machine export folders. Defaults to "
            "SCALEBRIDGE_GENERATED_DATA_ROOT/mlflow_exports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output registry directory. Defaults to "
            "SCALEBRIDGE_GENERATED_DATA_ROOT/experiment_registry."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-global-run-keys",
        action="store_true",
        help="Keep duplicate machine_id::run_id rows instead of raising an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = merge_mlflow_exports(
        exports_root=Path(args.exports_root).resolve() if args.exports_root else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        allow_duplicate_global_run_keys=args.allow_duplicate_global_run_keys,
    )

    print("MLflow exports merged")
    print(f"exports_root: {manifest['exports_root']}")
    print(f"output_dir: {manifest['output_dir']}")
    print(f"included_export_count: {manifest['included_export_count']}")
    print(f"run_count_raw: {manifest['run_count_raw']}")
    print(f"run_count_merged: {manifest['run_count_merged']}")
    print(f"machine_ids: {manifest['machine_ids']}")
    print(f"experiment_names: {manifest['experiment_names']}")


if __name__ == "__main__":
    main()