# -*- coding: utf-8 -*-
"""Robust diagnostic for ApartmentMidRise System Node Mass Flow Rate parquet failures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CAMPAIGN_ID = "p1_compact_4b4c_labpc_1w_v1"
DATA_ROOT = Path(
    r"F:\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge"
)

CAMPAIGN_ROOT = DATA_ROOT / "campaigns" / CAMPAIGN_ID
CASES_ROOT = CAMPAIGN_ROOT / "generation" / "cases"

TARGET_VARIABLE_FILE = "timestep_system_node_mass_flow_rate.parquet"


def load_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_text_recursive(payload, needle: str) -> bool:
    needle_l = needle.lower()

    if isinstance(payload, dict):
        return any(find_text_recursive(value, needle) for value in payload.values())

    if isinstance(payload, list):
        return any(find_text_recursive(value, needle) for value in payload)

    return needle_l in str(payload).lower()


def extract_first_value(payload, keys: list[str]) -> str:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload and payload[key] not in {None, ""}:
                return str(payload[key])

        for value in payload.values():
            found = extract_first_value(value, keys)
            if found:
                return found

    elif isinstance(payload, list):
        for value in payload:
            found = extract_first_value(value, keys)
            if found:
                return found

    return ""


def main() -> None:
    print("=" * 100)
    print("ApartmentMidRise mass-flow parquet diagnostic")
    print("=" * 100)
    print(f"campaign_root: {CAMPAIGN_ROOT}")
    print(f"cases_root   : {CASES_ROOT}")
    print(f"cases_exists : {CASES_ROOT.exists()}")
    print()

    case_dirs = sorted(path for path in CASES_ROOT.glob("epcase_*") if path.is_dir())
    print(f"case_dir_count: {len(case_dirs)}")
    print()

    apartment_case_count = 0
    inspected_run_count = 0

    for case_dir in case_dirs:
        json_paths = sorted(case_dir.rglob("*.json"))
        json_payloads = [(path, load_json_safe(path)) for path in json_paths]

        is_apartment = any(
            find_text_recursive(payload, "ApartmentMidRise")
            for _, payload in json_payloads
        )

        if not is_apartment:
            continue

        apartment_case_count += 1

        building_type = ""
        weather_location = ""
        climate_zone = ""

        for _, payload in json_payloads:
            building_type = building_type or extract_first_value(
                payload,
                ["building_type", "building", "prototype_building"],
            )
            weather_location = weather_location or extract_first_value(
                payload,
                ["weather_location", "weather_name", "city", "location"],
            )
            climate_zone = climate_zone or extract_first_value(
                payload,
                ["climate_zone", "ashrae_climate_zone"],
            )

        run_dirs = sorted(
            path for path in (case_dir / "runs").glob("*")
            if path.is_dir()
        )

        print("-" * 100)
        print(f"case_id          : {case_dir.name}")
        print(f"building_type    : {building_type}")
        print(f"weather_location : {weather_location}")
        print(f"climate_zone     : {climate_zone}")
        print(f"json_files_found : {len(json_paths)}")
        print(f"run_dir_count    : {len(run_dirs)}")

        for run_dir in run_dirs:
            inspected_run_count += 1
            parquet_path = run_dir / "canonical" / "variables" / TARGET_VARIABLE_FILE

            print()
            print(f"  run_dir      : {run_dir.name}")
            print(f"  parquet_path : {parquet_path}")
            print(f"  exists       : {parquet_path.exists()}")

            if parquet_path.exists():
                print(f"  size_bytes   : {parquet_path.stat().st_size}")

            if not parquet_path.exists():
                print("  READ_STATUS  : MISSING")
                continue

            if parquet_path.stat().st_size == 0:
                print("  READ_STATUS  : ZERO_BYTE_FILE")
                continue

            try:
                df = pd.read_parquet(parquet_path)
                print("  READ_STATUS  : OK")
                print(f"  shape        : {df.shape}")
                print(f"  columns      : {list(df.columns)}")
                print("  dtypes:")
                print(df.dtypes.to_string())

                print()
                print("  head:")
                print(df.head(5).to_string(index=False))

                candidate_cols = [
                    col for col in df.columns
                    if any(
                        token in col.lower()
                        for token in ["object", "key", "name", "node", "zone"]
                    )
                ]

                if candidate_cols:
                    print()
                    print("  candidate object/name/node columns:")
                    for col in candidate_cols:
                        values = (
                            df[col]
                            .dropna()
                            .astype(str)
                            .drop_duplicates()
                            .head(40)
                            .tolist()
                        )
                        print(f"    {col}: {values}")

            except Exception as exc:
                print("  READ_STATUS  : READ_FAILED")
                print(f"  error_type   : {type(exc).__name__}")
                print(f"  error_message: {exc}")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"apartment_case_count : {apartment_case_count}")
    print(f"inspected_run_count  : {inspected_run_count}")

    if apartment_case_count == 0:
        print()
        print("No ApartmentMidRise cases detected by scanning JSON contents.")
        print("This means building_type metadata is stored somewhere unexpected,")
        print("or the case JSON files do not include the string ApartmentMidRise.")


if __name__ == "__main__":
    main()