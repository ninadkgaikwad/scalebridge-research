from io import BytesIO
import json
import zipfile

import pandas as pd

from scalebridge.dashapp.services.generation.results_data import (
    build_selected_data_export,
    discover_key_values,
    load_series,
)


def _row(path):
    return {
        "campaign_id": "test_campaign",
        "building_type": "OfficeSmall",
        "weather_location": "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3",
        "case_id": "epcase_test",
        "run_id": "epvwr_test",
        "variable_name": "Zone Air Temperature",
        "variable_id": "zone_air_temperature",
        "units": "C",
        "parquet_path": str(path),
        "calendar_year": 2013,
    }


def _write_two_key_parquet(path):
    pd.DataFrame(
        {
            "timestamp_raw": [
                "01/01  00:05:00",
                "01/01  00:10:00",
                "01/01  00:05:00",
                "01/01  00:10:00",
            ],
            "key_value": ["CORE_ZN", "CORE_ZN", "PERIMETER_ZN", "PERIMETER_ZN"],
            "variable_name": ["Zone Air Temperature"] * 4,
            "units": ["C"] * 4,
            "value": [21.0, 21.1, 19.0, 19.1],
        }
    ).to_parquet(path, index=False)


def test_key_filter_discovers_columns_and_plots_only_selected_key(tmp_path):
    path = tmp_path / "zone_air_temperature.parquet"
    _write_two_key_parquet(path)
    row = _row(path)

    assert discover_key_values([row]) == ["CORE_ZN", "PERIMETER_ZN"]

    frame = load_series([row], key_values=["CORE_ZN"])
    assert set(frame["key_value"]) == {"CORE_ZN"}
    assert frame["series"].nunique() == 1
    assert frame["series"].iloc[0].endswith("| Zone Air Temperature | CORE_ZN")


def test_download_contains_only_selected_variable_key(tmp_path):
    path = tmp_path / "zone_air_temperature.parquet"
    _write_two_key_parquet(path)
    row = _row(path)

    payload, _ = build_selected_data_export(
        campaign_id="test_campaign",
        rows=[row],
        export_format="csv",
        range_mode="full",
        key_values=["PERIMETER_ZN"],
    )

    with zipfile.ZipFile(BytesIO(payload)) as archive:
        combined = pd.read_csv(archive.open("selected_signals_combined.csv"))
        assert set(combined["key_value"]) == {"PERIMETER_ZN"}
        signal_files = [name for name in archive.namelist() if name.startswith("signals/")]
        assert len(signal_files) == 1
        assert "PERIMETER_ZN" in signal_files[0]

        manifest = json.loads(archive.read("selection_manifest.json"))
        assert manifest["selected_key_values"] == ["PERIMETER_ZN"]
        assert manifest["signal_series_count"] == 1
        assert manifest["signals"][0]["key_value"] == "PERIMETER_ZN"
