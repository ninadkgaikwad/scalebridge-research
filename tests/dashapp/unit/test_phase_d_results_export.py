from io import BytesIO
import json
from pathlib import Path
import zipfile

import pandas as pd

from scalebridge.dashapp.services.phase_d import results_data
from importlib.util import module_from_spec, spec_from_file_location


def _load_fixture():
    fixture_path = Path(__file__).with_name("test_phase_d_results_service.py")
    spec = spec_from_file_location("_phase_d_results_service_fixture", fixture_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Results fixture module: {fixture_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fixture


_fixture = _load_fixture()


def test_visible_plot_export_excludes_hidden_trace(monkeypatch, tmp_path):
    _, _, manifest_path = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref("demo::phase_d_demo")
    figure = {
        "data": [
            {"name": "visible", "x": ["2001-01-01"], "y": [1.0], "visible": True},
            {"name": "hidden", "x": ["2001-01-01"], "y": [9.0], "visible": False},
        ],
        "layout": {"meta": {"phase_d_plot_export": {"selection": {"partition": "__INCLUDED__"}}}},
    }
    payload, filename = results_data.build_visible_plot_export(
        figure,
        file_format="csv",
        run_ref=ref,
        manifest_key=str(manifest_path),
    )
    assert filename.endswith("__visible_plot_data_csv.zip")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        text = archive.read("data/plotted_data.csv").decode("utf-8")
        assert "visible" in text
        assert "hidden" not in text
        manifest = json.loads(archive.read("selection_manifest.json"))
        assert manifest["contract"] == "visible_plot_snapshot_equals_exported_data"
        assert manifest["visible_trace_names"] == ["visible"]
        assert manifest["hidden_trace_names"] == ["hidden"]
