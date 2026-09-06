from io import BytesIO
import json
import zipfile

import pandas as pd

from scalebridge.dashapp.services.generation.results_data import (
    build_selected_data_export,
    filter_index_rows,
)


def _row(path):
    return {
        "campaign_id": "bgirs_e2e_dropdown_2b2w_v1",
        "building_type": "RestaurantFastFood",
        "weather_location": "Seattle",
        "case_id": "epcase_abc123",
        "run_id": "epvwr_run123",
        "variable_name": "Zone Air Temperature",
        "variable_id": "zone_air_temperature",
        "units": "",
        "parquet_path": str(path),
        "calendar_year": 2013,
    }


def _write(path):
    pd.DataFrame(
        {
            "timestamp_raw": ["01/01  00:05:00", "01/01  00:10:00"],
            "key_value": ["CORE_ZN", "CORE_ZN"],
            "variable_name": ["Zone Air Temperature", "Zone Air Temperature"],
            "units": ["C", "C"],
            "value": [21.0, 21.2],
        }
    ).to_parquet(path, index=False)


def test_filter_index_rows_uses_generation_results_context(tmp_path):
    path = tmp_path / "x.parquet"
    _write(path)
    row = _row(path)
    other = dict(row, building_type="OfficeSmall")
    selected = filter_index_rows([row, other], building_types=["RestaurantFastFood"])
    assert selected == [row]


def test_csv_zip_export_preserves_variable_key_nomenclature_and_manifest(tmp_path):
    parquet_path = tmp_path / "zone_air_temperature.parquet"
    _write(parquet_path)

    payload, filename = build_selected_data_export(
        campaign_id="bgirs_e2e_dropdown_2b2w_v1",
        rows=[_row(parquet_path)],
        export_format="csv",
        range_mode="full",
        key_values=["CORE_ZN"],
    )

    assert filename == "bgirs_e2e_dropdown_2b2w_v1__generation_selected_data__csv.zip"
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = archive.namelist()
        assert "selection_manifest.json" in names
        assert "selected_signals_combined.csv" in names
        signal_files = [name for name in names if name.startswith("signals/")]
        assert len(signal_files) == 1
        assert signal_files[0].endswith("__Zone_Air_Temperature__CORE_ZN.csv")
        combined = pd.read_csv(archive.open("selected_signals_combined.csv"))
        assert set(combined["key_value"]) == {"CORE_ZN"}
        assert set(combined["units"]) == {"C"}
        manifest = json.loads(archive.read("selection_manifest.json"))
        assert manifest["signals"][0]["key_value"] == "CORE_ZN"


def test_custom_range_is_applied_to_download(tmp_path):
    parquet_path = tmp_path / "zone_air_temperature.parquet"
    _write(parquet_path)
    payload, _ = build_selected_data_export(
        campaign_id="bgirs_e2e_dropdown_2b2w_v1",
        rows=[_row(parquet_path)],
        export_format="csv",
        range_mode="custom",
        start="2013-01-01T00:10:00",
        end="2013-01-01T00:10:00",
        key_values=["CORE_ZN"],
    )
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        combined = pd.read_csv(archive.open("selected_signals_combined.csv"))
        assert len(combined) == 1
        assert float(combined.loc[0, "value"]) == 21.2
