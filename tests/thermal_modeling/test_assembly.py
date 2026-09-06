
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scalebridge.data.thermal_modeling.assembly import (
    AssemblyConfig,
    PhaseDAssemblyError,
    assemble_canonical_zone_table,
    required_phase_c_prediction_columns,
)


def _write_status_tables(tmp_path: Path):
    applicable = pd.DataFrame(
        [
            {
                "model_id": "QAC",
                "output_prediction_column": "predicted_QAC",
                "applicable": True,
                "reason_code": "applicable",
                "reason": "ok",
                "applicability_status": "applicable",
            },
            {
                "model_id": "QZic_P",
                "output_prediction_column": "predicted_QZic_P",
                "applicable": True,
                "reason_code": "applicable",
                "reason": "ok",
                "applicability_status": "applicable",
            },
            {
                "model_id": "QZir_P",
                "output_prediction_column": "predicted_QZir_P",
                "applicable": True,
                "reason_code": "applicable",
                "reason": "ok",
                "applicability_status": "applicable",
            },
            {
                "model_id": "QZivr_L",
                "output_prediction_column": "predicted_QZivr_L",
                "applicable": True,
                "reason_code": "applicable",
                "reason": "ok",
                "applicability_status": "applicable",
            },
            {
                "model_id": "PHVAC",
                "output_prediction_column": "predicted_PHVAC",
                "applicable": True,
                "reason_code": "applicable",
                "reason": "ok",
                "applicability_status": "applicable",
            },
        ]
    )
    unavailable_ids = [
        "QSol1", "QSol2", "QZic_L", "QZic_EE", "QZic_GE", "QZic_OE",
        "QZic_HWE", "QZic_SE", "QZir_L", "QZir_EE", "QZir_GE",
        "QZir_OE", "QZir_HWE", "QZir_SE",
    ]
    unavailable = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "output_prediction_column": f"predicted_{model_id}",
                "applicable": False,
                "reason_code": (
                    "invalid_all_zero_target" if model_id == "QSol1"
                    else "not_applicable_rdd_unavailable"
                ),
                "reason": "not usable",
                "applicability_status": "unavailable",
            }
            for model_id in unavailable_ids
        ]
    )
    app_path = tmp_path / "applicable.csv"
    un_path = tmp_path / "unavailable.csv"
    applicable.to_csv(app_path, index=False)
    unavailable.to_csv(un_path, index=False)
    return app_path, un_path


def _aligned() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2001-01-01 00:05", periods=3, freq="5min"),
            "zone_temperature": [20.0, 20.1, 20.2],
            "outdoor_temperature": [5.0, 5.1, 5.2],
            "predicted_QAC": [-10.0, -11.0, -12.0],
            "predicted_QZic_P": [100.0, 100.0, 100.0],
            "predicted_QZir_P": [20.0, 21.0, 22.0],
            "predicted_QZivr_L": [5.0, 5.0, 5.0],
            "predicted_PHVAC": [3.0, 4.0, 5.0],
            "predicted_PHVAC_oracle": [2.5, 3.5, 4.5],
            "split": ["train", "validation", "test"],
        }
    )


def test_assembly_retains_constant_nonzero_and_nulls_unavailable(tmp_path: Path):
    app, un = _write_status_tables(tmp_path)
    result = assemble_canonical_zone_table(
        _aligned(),
        applicable_models_path=app,
        unavailable_models_path=un,
    )
    assert result.table["qzic_p"].tolist() == [100.0, 100.0, 100.0]
    assert result.table["qsol1"].isna().all()
    records = {r.signal_name: r for r in result.signal_records}
    assert records["qzic_p"].phase_d_status.value == "constant_nonzero"
    assert records["qsol1"].phase_d_status.value == "nullable_complete_zero"


def test_grouping_uses_only_active_components(tmp_path: Path):
    app, un = _write_status_tables(tmp_path)
    result = assemble_canonical_zone_table(
        _aligned(),
        applicable_models_path=app,
        unavailable_models_path=un,
    )
    assert result.table["zic"].tolist() == [100.0, 100.0, 100.0]
    assert result.table["zir"].tolist() == [25.0, 26.0, 27.0]
    assert result.diagnostics.zic_active_components == ("qzic_p",)
    assert result.diagnostics.zir_active_components == ("qzir_p", "qzivr_l")


def test_visible_lighting_can_be_excluded_from_zir(tmp_path: Path):
    app, un = _write_status_tables(tmp_path)
    result = assemble_canonical_zone_table(
        _aligned(),
        applicable_models_path=app,
        unavailable_models_path=un,
        config=AssemblyConfig(include_visible_lighting_in_zir=False),
    )
    assert result.table["zir"].tolist() == [20.0, 21.0, 22.0]


def test_missing_applicable_prediction_column_fails(tmp_path: Path):
    app, un = _write_status_tables(tmp_path)
    frame = _aligned().drop(columns=["predicted_QAC"])
    with pytest.raises(PhaseDAssemblyError, match="missing prediction column"):
        assemble_canonical_zone_table(
            frame,
            applicable_models_path=app,
            unavailable_models_path=un,
        )


def test_missing_value_in_active_prediction_fails(tmp_path: Path):
    app, un = _write_status_tables(tmp_path)
    frame = _aligned()
    frame.loc[1, "predicted_QAC"] = pd.NA
    with pytest.raises(PhaseDAssemblyError, match="failed Phase D classification"):
        assemble_canonical_zone_table(
            frame,
            applicable_models_path=app,
            unavailable_models_path=un,
        )


def test_required_prediction_projection_excludes_unavailable_models(tmp_path: Path):
    app, un = _write_status_tables(tmp_path)
    columns = required_phase_c_prediction_columns(
        app,
        un,
        available_parquet_columns={
            "timestamp",
            "predicted_QAC",
            "predicted_QZic_P",
            "predicted_QZir_P",
            "predicted_QZivr_L",
            "predicted_PHVAC",
            "predicted_PHVAC_oracle",
            "predicted_QSol1",
        },
    )
    assert "predicted_QAC" in columns
    assert "predicted_PHVAC_oracle" in columns
    assert "predicted_QSol1" not in columns


def test_required_prediction_projection_rejects_missing_applicable_column(
    tmp_path: Path,
):
    app, un = _write_status_tables(tmp_path)
    with pytest.raises(
        PhaseDAssemblyError,
        match="absent from Parquet schema",
    ):
        required_phase_c_prediction_columns(
            app,
            un,
            available_parquet_columns={"timestamp"},
        )


def test_standard_manifest_uses_compact_signal_summary(tmp_path: Path):
    app, un = _write_status_tables(tmp_path)
    result = assemble_canonical_zone_table(
        _aligned(),
        applicable_models_path=app,
        unavailable_models_path=un,
    )
    payload = result.manifest_dict(
        aggregate_zone_id="ZoneA",
        include_signal_records=False,
    )
    assert "signal_records" not in payload
    assert payload["signal_summary"]["qzic_p"]["phase_d_status"] == (
        "constant_nonzero"
    )
