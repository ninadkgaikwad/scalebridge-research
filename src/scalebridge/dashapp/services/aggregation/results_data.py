"""Read-only discovery and table loading for Phase B Aggregation results."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any
import zipfile

import pandas as pd

from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


MATRIX_MANIFEST = "aggregation_matrix_manifest.json"
CASE_RUNS_CSV = "aggregation_matrix_case_runs.csv"
OUTPUTS_CSV = "aggregation_matrix_outputs.csv"
SELECTED_PLANS_CSV = "selected_aggregation_plans.csv"
MISSING_ROWS_CSV = "missing_generation_rows.csv"


def campaigns_root() -> Path:
    return resolve_generated_data_root() / "campaigns"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _matrix_root(parent_generation_campaign_id: str) -> Path:
    return (
        campaigns_root()
        / parent_generation_campaign_id
        / "aggregation"
        / "matrix_runs"
    )


def discover_matrix_runs() -> list[dict[str, Any]]:
    """Discover every readable Aggregation matrix manifest below generated campaigns."""
    rows: list[dict[str, Any]] = []
    root = campaigns_root()
    if not root.is_dir():
        return rows

    for campaign_root in sorted(p for p in root.iterdir() if p.is_dir()):
        matrix_root = campaign_root / "aggregation" / "matrix_runs"
        if not matrix_root.is_dir():
            continue
        for run_root in sorted(
            (p for p in matrix_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        ):
            manifest_path = run_root / MATRIX_MANIFEST
            if not manifest_path.is_file():
                continue
            manifest = _read_json(manifest_path)
            if not manifest:
                continue

            parent_id = str(
                manifest.get("parent_generation_campaign_id")
                or manifest.get("campaign_id")
                or campaign_root.name
            )
            aggregation_id = str(
                manifest.get("aggregation_campaign_id")
                or f"legacy::{parent_id}"
            )
            rows.append(
                {
                    "aggregation_campaign_id": aggregation_id,
                    "parent_generation_campaign_id": parent_id,
                    "matrix_run_id": str(
                        manifest.get("matrix_run_id") or run_root.name
                    ),
                    "status": str(manifest.get("status") or ""),
                    "created_at_utc": str(manifest.get("created_at_utc") or ""),
                    "schema_version": str(manifest.get("schema_version") or ""),
                    "selected_plan_count": int(manifest.get("selected_plan_count") or 0),
                    "successful_plan_count": int(
                        manifest.get("successful_plan_count") or 0
                    ),
                    "failed_plan_count": int(manifest.get("failed_plan_count") or 0),
                    "selected_generation_case_count": int(
                        manifest.get("selected_generation_case_count") or 0
                    ),
                    "runtime_seconds": manifest.get("runtime_seconds"),
                    "plan_build_id": str(manifest.get("plan_build_id") or ""),
                    "matrix_root": str(run_root),
                    "manifest_path": str(manifest_path),
                }
            )
    return rows


def campaign_options() -> list[dict[str, str]]:
    """Dropdown options keyed by the true Aggregation campaign ID."""
    discovered = discover_matrix_runs()
    seen: set[str] = set()
    options: list[dict[str, str]] = []
    for row in discovered:
        campaign_id = row["aggregation_campaign_id"]
        if campaign_id in seen:
            continue
        seen.add(campaign_id)
        parent = row["parent_generation_campaign_id"]
        label = (
            f"{campaign_id} | parent={parent}"
            if not campaign_id.startswith("legacy::")
            else f"{parent} | legacy Aggregation matrices"
        )
        options.append({"label": label, "value": campaign_id})
    return options


def matrix_run_options(
    aggregation_campaign_id: str | None,
) -> list[dict[str, str]]:
    if not aggregation_campaign_id:
        return []
    rows = [
        row
        for row in discover_matrix_runs()
        if row["aggregation_campaign_id"] == aggregation_campaign_id
    ]
    return [
        {
            "label": (
                f"{row['matrix_run_id']} | {row['status'] or 'unknown'} | "
                f"{row['successful_plan_count']}/{row['selected_plan_count']} successful"
            ),
            "value": row["matrix_run_id"],
        }
        for row in rows
    ]


def _find_run(
    aggregation_campaign_id: str,
    matrix_run_id: str,
) -> dict[str, Any]:
    for row in discover_matrix_runs():
        if (
            row["aggregation_campaign_id"] == aggregation_campaign_id
            and row["matrix_run_id"] == matrix_run_id
        ):
            return row
    raise FileNotFoundError(
        f"Aggregation matrix run not found: "
        f"{aggregation_campaign_id} / {matrix_run_id}"
    )


def load_matrix_result(
    aggregation_campaign_id: str,
    matrix_run_id: str,
) -> dict[str, Any]:
    """Load one matrix result using only small JSON/CSV summary artifacts."""
    discovered = _find_run(aggregation_campaign_id, matrix_run_id)
    run_root = Path(discovered["matrix_root"])
    manifest = _read_json(run_root / MATRIX_MANIFEST)

    case_runs = _read_csv(run_root / CASE_RUNS_CSV)
    outputs = _read_csv(run_root / OUTPUTS_CSV)
    selected_plans = _read_csv(run_root / SELECTED_PLANS_CSV)
    missing_rows = _read_csv(run_root / MISSING_ROWS_CSV)

    error_rows = [
        row
        for row in case_runs
        if str(row.get("error_type") or "").strip()
        or str(row.get("error_message") or "").strip()
        or str(row.get("status") or "").casefold()
        not in {"completed", "planned", ""}
    ]

    artifact_paths = {
        "matrix_manifest": str(run_root / MATRIX_MANIFEST),
        "case_runs": str(run_root / CASE_RUNS_CSV),
        "outputs": str(run_root / OUTPUTS_CSV),
        "selected_plans": str(run_root / SELECTED_PLANS_CSV),
        "missing_generation_rows": (
            str(run_root / MISSING_ROWS_CSV)
            if (run_root / MISSING_ROWS_CSV).is_file()
            else ""
        ),
        "aggregation_campaign_definition": str(
            manifest.get("outputs", {}).get("aggregation_campaign_definition", "")
        ),
        "plan_build_summary": str(
            manifest.get("outputs", {}).get("plan_build_summary", "")
        ),
    }

    return {
        "manifest": manifest,
        "case_runs": case_runs,
        "outputs": outputs,
        "selected_plans": selected_plans,
        "missing_generation_rows": missing_rows,
        "error_rows": error_rows,
        "artifact_paths": artifact_paths,
        "discovery": discovered,
    }


def compact_case_run_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the B8 table projection; no scientific values are recomputed."""
    fields = (
        "case_id",
        "building_type",
        "weather_location",
        "climate_zone",
        "aggregation_id",
        "plan_strategy",
        "weight_mode",
        "rule_set",
        "source_zone_count",
        "aggregate_zone_count",
        "aggregation_compression_ratio",
        "aggregation_run_id",
        "status",
        "loaded_variable_count",
        "runtime_seconds",
        "error_type",
        "error_message",
    )
    return [{field: row.get(field, "") for field in fields} for row in rows]


def compact_missing_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    preferred = (
        "case_id",
        "building_type",
        "weather_location",
        "climate_zone",
        "status",
        "reason",
        "error_type",
        "error_message",
    )
    present = [name for name in preferred if any(name in row for row in rows)]
    if not present:
        present = list(rows[0].keys())
    return [{field: row.get(field, "") for field in present} for row in rows]


# ---------------------------------------------------------------------------
# Final Results Tab: Generation-parity signal discovery / plot / export
# ---------------------------------------------------------------------------

def _case_run_root(parent_campaign_id: str, case_id: str, aggregation_run_id: str) -> Path:
    """Reconstruct the authoritative local run path instead of trusting stored absolute paths."""
    return (
        campaigns_root()
        / parent_campaign_id
        / "aggregation"
        / "cases"
        / case_id
        / "runs"
        / aggregation_run_id
    )


def _calendar_year(run_root: Path) -> int:
    payload = _read_json(run_root / "inputs" / "source_run_manifest.json")
    try:
        return int(
            payload.get("case_spec", {})
            .get("run_period", {})
            .get("calendar_year")
            or payload.get("case_spec", {}).get("prototype_year")
            or 2013
        )
    except Exception:
        return 2013


def _selection_values(values):
    """Normalize one multi-select value list while preserving explicit emptiness."""
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def result_index(aggregation_campaign_ids) -> list[dict[str, Any]]:
    """Return completed Aggregation case/plan runs for the selected campaigns."""
    selected_campaigns = _selection_values(aggregation_campaign_ids)
    if not selected_campaigns:
        return []

    wanted_campaigns = set(selected_campaigns)
    output: list[dict[str, Any]] = []
    for matrix in discover_matrix_runs():
        campaign_id = str(matrix["aggregation_campaign_id"])
        if campaign_id not in wanted_campaigns:
            continue
        matrix_root = Path(matrix["matrix_root"])
        for row in _read_csv(matrix_root / CASE_RUNS_CSV):
            if str(row.get("status") or "").casefold() != "completed":
                continue
            case_id = str(row.get("case_id") or "")
            aggregation_run_id = str(row.get("aggregation_run_id") or "")
            if not case_id or not aggregation_run_id:
                continue
            parent_id = matrix["parent_generation_campaign_id"]
            run_root = _case_run_root(parent_id, case_id, aggregation_run_id)
            if not run_root.is_dir():
                continue
            output.append(
                {
                    **row,
                    "aggregation_campaign_id": campaign_id,
                    "parent_generation_campaign_id": parent_id,
                    "matrix_run_id": matrix["matrix_run_id"],
                    "strategy": str(
                        row.get("loaded_plan_strategy")
                        or row.get("plan_strategy")
                        or ""
                    ),
                    "weight_mode": str(
                        row.get("loaded_plan_weight_mode")
                        or row.get("weight_mode")
                        or ""
                    ),
                    "rule_set": str(
                        row.get("loaded_plan_rule_set")
                        or row.get("rule_set")
                        or ""
                    ),
                    "run_token": (
                        f"{campaign_id}::{matrix['matrix_run_id']}::{aggregation_run_id}"
                    ),
                    "run_root_local": str(run_root),
                    "calendar_year": _calendar_year(run_root),
                }
            )
    output.sort(
        key=lambda row: (
            str(row.get("aggregation_campaign_id") or ""),
            str(row.get("matrix_run_id") or ""),
            str(row.get("aggregation_run_id") or ""),
        ),
        reverse=True,
    )
    return output


def filter_result_index(
    rows,
    *,
    aggregation_campaign_ids=None,
    building_types=None,
    weather_locations=None,
    climate_zones=None,
    strategies=None,
    weight_modes=None,
    rule_sets=None,
    run_tokens=None,
):
    """Intersect every supplied multi-select filter.

    ``None`` means the caller has not constrained that dimension yet.
    An explicit empty list means the user selected nothing and therefore
    the result is empty. This keeps selector, plot, and export semantics aligned.
    """
    selected = list(rows or [])
    filters = (
        ("aggregation_campaign_id", aggregation_campaign_ids),
        ("building_type", building_types),
        ("weather_location", weather_locations),
        ("climate_zone", climate_zones),
        ("strategy", strategies),
        ("weight_mode", weight_modes),
        ("rule_set", rule_sets),
        ("run_token", run_tokens),
    )
    for key, values in filters:
        normalized = _selection_values(values)
        if normalized is None:
            continue
        if not normalized:
            return []
        wanted = set(normalized)
        selected = [
            row for row in selected if str(row.get(key) or "") in wanted
        ]
    return selected

def result_options(rows, key: str) -> list[dict[str, str]]:
    values = sorted(
        {str(row.get(key) or "") for row in (rows or []) if str(row.get(key) or "")},
        key=str.casefold,
    )
    return [{"label": value, "value": value} for value in values]


def run_options(rows) -> list[dict[str, str]]:
    options = []
    for row in rows or []:
        label = (
            f"{row.get('aggregation_campaign_id') or 'Unknown campaign'} | "
            f"{row.get('building_type') or 'Unknown building'} | "
            f"{row.get('weather_location') or 'Unknown weather'} | "
            f"{row.get('climate_zone') or 'Unknown climate'} | "
            f"{row.get('strategy') or ''} | {row.get('weight_mode') or ''} | "
            f"{row.get('aggregation_run_id') or ''}"
        )
        options.append({"label": label, "value": str(row["run_token"])})
    return options


def discover_zones(rows) -> list[str]:
    zones: set[str] = set()
    for row in rows or []:
        zone_root = Path(row["run_root_local"]) / "zones"
        if zone_root.is_dir():
            zones.update(p.name for p in zone_root.iterdir() if p.is_dir())
    return sorted(zones, key=str.casefold)


@lru_cache(maxsize=2048)
def _rule_catalog(path_text: str, mtime_ns: int) -> tuple[tuple[str, str, str], ...]:
    path = Path(path_text)
    rows = _read_csv(path)
    values: set[tuple[str, str, str]] = set()
    for row in rows:
        if str(row.get("status") or "").casefold() != "aggregated":
            continue
        zone = str(row.get("aggregate_zone_id") or "")
        source = str(row.get("source_variable_name") or "")
        output = str(row.get("output_variable_name") or "")
        if zone and source and output:
            values.add((zone, source, output))
    return tuple(sorted(values, key=lambda item: tuple(part.casefold() for part in item)))


def variable_catalog(rows, zones=None) -> list[dict[str, str]]:
    """Use rule_summary.csv as the lightweight authoritative variable→column catalog."""
    selected_zones = {str(value) for value in (zones or [])}
    values: set[tuple[str, str, str]] = set()
    for row in rows or []:
        path = Path(row["run_root_local"]) / "diagnostics" / "rule_summary.csv"
        if not path.is_file():
            continue
        for zone, source, output in _rule_catalog(
            str(path.resolve()), path.stat().st_mtime_ns
        ):
            if selected_zones and zone not in selected_zones:
                continue
            values.add((zone, source, output))
    return [
        {"aggregate_zone": zone, "variable": source, "variable_column": output}
        for zone, source, output in sorted(
            values, key=lambda item: tuple(part.casefold() for part in item)
        )
    ]


def variable_options(catalog) -> list[dict[str, str]]:
    values = sorted(
        {str(row["variable"]) for row in (catalog or []) if row.get("variable")},
        key=str.casefold,
    )
    return [{"label": value, "value": value} for value in values]


def variable_column_options(catalog, variables=None) -> list[dict[str, str]]:
    wanted = {str(value) for value in (variables or [])}
    values = sorted(
        {
            str(row["variable_column"])
            for row in (catalog or [])
            if row.get("variable_column")
            and (not wanted or str(row.get("variable")) in wanted)
        },
        key=str.casefold,
    )
    return [{"label": value, "value": value} for value in values]


def _parse_timestamp_raw(series: pd.Series, year: int) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    is_24 = text.str.contains(r"\s24:", regex=True)
    normalized = text.str.replace(r"\s24:", " 00:", regex=True)
    parsed = pd.to_datetime(
        str(year) + "/" + normalized,
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce",
    )
    parsed.loc[is_24 & parsed.notna()] = (
        parsed.loc[is_24 & parsed.notna()] + pd.Timedelta(days=1)
    )
    return parsed


def _read_selected_long(
    path: Path,
    *,
    variables,
    variable_columns,
) -> pd.DataFrame:
    columns = [
        "aggregate_zone_id",
        "timestamp_raw",
        "output_variable_name",
        "source_variable_name",
        "rule_family",
        "units",
        "semantic_role",
        "value",
    ]
    filters = []
    if variables:
        filters.append(("source_variable_name", "in", [str(x) for x in variables]))
    if variable_columns:
        filters.append(("output_variable_name", "in", [str(x) for x in variable_columns]))
    try:
        return pd.read_parquet(path, columns=columns, filters=filters or None)
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        if variables:
            frame = frame[
                frame["source_variable_name"].astype(str).isin(
                    {str(x) for x in variables}
                )
            ]
        if variable_columns:
            frame = frame[
                frame["output_variable_name"].astype(str).isin(
                    {str(x) for x in variable_columns}
                )
            ]
        return frame


def load_selected_signals(
    rows,
    *,
    zones,
    variables,
    variable_columns,
    start=None,
    end=None,
) -> pd.DataFrame:
    if not rows:
        raise ValueError("Select at least one Aggregation run.")
    if not zones:
        raise ValueError("Select at least one Aggregation zone.")
    if not variables:
        raise ValueError("Select at least one variable.")
    if not variable_columns:
        raise ValueError("Select at least one variable column.")

    frames = []
    wanted_zones = {str(value) for value in zones}
    wanted_variables = [str(value) for value in variables]
    wanted_columns = [str(value) for value in variable_columns]

    for row in rows:
        run_root = Path(row["run_root_local"])
        for zone in sorted(wanted_zones):
            path = run_root / "zones" / zone / "aggregated_timeseries_long.parquet"
            if not path.is_file():
                continue
            frame = _read_selected_long(
                path,
                variables=wanted_variables,
                variable_columns=wanted_columns,
            )
            if frame.empty:
                continue
            frame = frame.copy()
            frame["timestamp"] = _parse_timestamp_raw(
                frame["timestamp_raw"], int(row.get("calendar_year") or 2013)
            )
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.dropna(subset=["timestamp", "value"])
            if start:
                frame = frame[frame["timestamp"] >= pd.Timestamp(start)]
            if end:
                frame = frame[frame["timestamp"] <= pd.Timestamp(end)]
            if frame.empty:
                continue
            frame["aggregation_campaign_id"] = row.get(
                "aggregation_campaign_id", ""
            )
            frame["matrix_run_id"] = row.get("matrix_run_id", "")
            frame["aggregation_run_id"] = row.get("aggregation_run_id", "")
            frame["case_id"] = row.get("case_id", "")
            frame["building_type"] = row.get("building_type", "")
            frame["weather_location"] = row.get("weather_location", "")
            frame["climate_zone"] = row.get("climate_zone", "")
            frame["strategy"] = row.get("strategy", "")
            frame["weight_mode"] = row.get("weight_mode", "")
            frame["rule_set"] = row.get("rule_set", "")
            frame["series"] = frame.apply(
                lambda item: (
                    f"{row.get('aggregation_campaign_id','')} | "
                    f"{row.get('building_type','')} | "
                    f"{row.get('weather_location','')} | "
                    f"{row.get('climate_zone','')} | "
                    f"{zone} | "
                    f"{item['source_variable_name']} | "
                    f"{item['output_variable_name']} | "
                    f"{row.get('aggregation_run_id','')}"
                ),
                axis=1,
            )
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "value",
                "series",
                "aggregate_zone_id",
                "source_variable_name",
                "output_variable_name",
                "units",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _safe_component(value: object, *, max_length: int = 100) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._") or "unknown"
    return text[:max_length]


def build_selected_data_export(
    *,
    aggregation_campaign_ids,
    rows,
    zones,
    variables,
    variable_columns,
    export_format: str,
    range_mode: str,
    start=None,
    end=None,
):
    """Build a ZIP containing exactly the selected Aggregation signals and lineage."""
    if export_format not in {"csv", "parquet"}:
        raise ValueError("export_format must be 'csv' or 'parquet'")
    if range_mode not in {"full", "custom"}:
        raise ValueError("range_mode must be 'full' or 'custom'")
    effective_start = start if range_mode == "custom" else None
    effective_end = end if range_mode == "custom" else None
    if (
        effective_start
        and effective_end
        and pd.Timestamp(effective_end) < pd.Timestamp(effective_start)
    ):
        raise ValueError("Custom end datetime must not precede start datetime.")

    frame = load_selected_signals(
        rows,
        zones=zones,
        variables=variables,
        variable_columns=variable_columns,
        start=effective_start,
        end=effective_end,
    )
    if frame.empty:
        raise ValueError("No Aggregation signal rows matched the current selection.")

    archive_buffer = BytesIO()
    suffix = ".csv" if export_format == "csv" else ".parquet"
    signal_manifest = []

    with zipfile.ZipFile(
        archive_buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for index, (series, signal_frame) in enumerate(
            frame.groupby("series", sort=True), start=1
        ):
            signal_frame = signal_frame.reset_index(drop=True)
            export_columns = [
                "timestamp",
                "value",
                "units",
                "aggregation_campaign_id",
                "aggregate_zone_id",
                "source_variable_name",
                "output_variable_name",
                "matrix_run_id",
                "aggregation_run_id",
                "case_id",
                "building_type",
                "weather_location",
                "climate_zone",
                "strategy",
                "weight_mode",
                "rule_set",
            ]
            exported = signal_frame[export_columns]
            filename = (
                f"signals/{index:03d}__{_safe_component(series, max_length=180)}{suffix}"
            )
            if export_format == "csv":
                archive.writestr(filename, exported.to_csv(index=False))
            else:
                parquet_buffer = BytesIO()
                exported.to_parquet(parquet_buffer, index=False)
                archive.writestr(filename, parquet_buffer.getvalue())

            first = signal_frame.iloc[0]
            signal_manifest.append(
                {
                    "series": series,
                    "export_file": filename,
                    "aggregation_campaign_id": str(
                        first.get("aggregation_campaign_id") or ""
                    ),
                    "row_count": int(len(exported)),
                    "units": str(first.get("units") or ""),
                    "aggregate_zone": str(first.get("aggregate_zone_id") or ""),
                    "variable": str(first.get("source_variable_name") or ""),
                    "variable_column": str(first.get("output_variable_name") or ""),
                    "matrix_run_id": str(first.get("matrix_run_id") or ""),
                    "aggregation_run_id": str(first.get("aggregation_run_id") or ""),
                    "case_id": str(first.get("case_id") or ""),
                    "building_type": str(first.get("building_type") or ""),
                    "weather_location": str(first.get("weather_location") or ""),
                    "climate_zone": str(first.get("climate_zone") or ""),
                    "strategy": str(first.get("strategy") or ""),
                    "weight_mode": str(first.get("weight_mode") or ""),
                    "rule_set": str(first.get("rule_set") or ""),
                }
            )

        manifest = {
            "schema_version": "0.1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "aggregation_campaign_ids": [
                str(value) for value in (aggregation_campaign_ids or [])
            ],
            "selection": {
                "zones": [str(x) for x in zones],
                "variables": [str(x) for x in variables],
                "variable_columns": [str(x) for x in variable_columns],
                "range_mode": range_mode,
                "start": effective_start,
                "end": effective_end,
                "run_tokens": [str(row.get("run_token") or "") for row in rows],
                "building_types": sorted(
                    {str(row.get("building_type") or "") for row in rows}
                ),
                "weather_locations": sorted(
                    {str(row.get("weather_location") or "") for row in rows}
                ),
                "climate_zones": sorted(
                    {str(row.get("climate_zone") or "") for row in rows}
                ),
                "strategies": sorted(
                    {str(row.get("strategy") or "") for row in rows}
                ),
                "weight_modes": sorted(
                    {str(row.get("weight_mode") or "") for row in rows}
                ),
                "rule_sets": sorted(
                    {str(row.get("rule_set") or "") for row in rows}
                ),
            },
            "signals": signal_manifest,
        }
        archive.writestr(
            "selection_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    campaign_part = (
        _safe_component(aggregation_campaign_ids[0])
        if len(aggregation_campaign_ids or []) == 1
        else f"{len(aggregation_campaign_ids or [])}_campaigns"
    )
    filename = (
        f"aggregation_results__{campaign_part}"
        f"__selected_{export_format}.zip"
    )
    return archive_buffer.getvalue(), filename
