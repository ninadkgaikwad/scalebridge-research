from __future__ import annotations

import sys
import types
from pathlib import Path

from scalebridge.tracking.mlflow.thermal_modeling import PhaseDMLflowTracker


class FakeRun:
    class Info:
        run_id = "mlflow_phase_d_test"
    info = Info()


class FakeMlflow(types.ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.params = {}
        self.metrics = {}
        self.tags = {}
        self.artifacts = []
        self.ended = None
    def start_run(self, run_name=None):
        self.run_name = run_name
        return FakeRun()
    def log_param(self, key, value): self.params[key] = value
    def log_metric(self, key, value): self.metrics[key] = value
    def set_tag(self, key, value): self.tags[key] = value
    def log_artifact(self, path, artifact_path=None): self.artifacts.append((Path(path).name, artifact_path))
    def end_run(self, status=None): self.ended = status


def test_phase_d_mlflow_parent_logs_params_metrics_and_compact_artifacts(tmp_path, monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    semantic = types.ModuleType("scalebridge.tracking.mlflow.semantic")
    semantic.get_or_create_semantic_experiment = lambda *a, **k: "exp"
    def tags(**kwargs):
        fake.tags.update({"campaign_id": kwargs.get("campaign_id"), **(kwargs.get("extra_tags") or {})})
    semantic.set_standard_tags = tags
    monkeypatch.setitem(sys.modules, "scalebridge.tracking.mlflow.semantic", semantic)

    tracker = PhaseDMLflowTracker(True, "ScaleBridge_P1_PhaseD", "run_1", True)
    run_id = tracker.start(
        campaign_id="campaign", matrix_run_id="matrix", phase_c_campaign_run_id="pc",
        phase_d_run_id="pd", selected_aggregation_run_count=240,
        configuration={"ml_input_lags": [1,3,6], "ml_policies": ["monthly_distributed_holdout"]},
        resume=False, overwrite_existing=False, continue_on_error=True, dry_run=False,
    )
    assert run_id == "mlflow_phase_d_test"
    assert fake.params["matrix_run_id"] == "matrix"
    assert fake.params["ml_input_lags"] == "1,3,6"
    assert fake.tags["pipeline_stage"] == "phase_d"

    tracker.log_summary({
        "status": "completed", "dataset_count": 100,
        "ml_dataset_count": 75, "opt_bayes_dataset_count": 25,
        "ind_dataset_count": 60, "dep1_dataset_count": 20, "dep2_dataset_count": 20,
        "selected_aggregation_run_count": 240, "completed_aggregation_run_count": 240,
        "skipped_completed_aggregation_run_count": 0, "failed_aggregation_run_count": 0,
        "runtime_seconds": 12.5,
    })
    assert fake.metrics["dataset_count"] == 100.0
    assert fake.metrics["dep2_dataset_count"] == 20.0

    for name in (
        "phase_d_campaign_plan.json", "phase_d_campaign_run_manifest.json",
        "aggregation_run_registry.csv", "dataset_registry.csv", "failures.csv",
    ):
        (tmp_path/name).write_text("x", encoding="utf-8")
    (tmp_path/"do_not_upload.log").write_text("large", encoding="utf-8")
    tracker.log_campaign_artifacts(tmp_path)
    assert {x[0] for x in fake.artifacts} == {
        "phase_d_campaign_plan.json", "phase_d_campaign_run_manifest.json",
        "aggregation_run_registry.csv", "dataset_registry.csv", "failures.csv",
    }
    tracker.finish(failed=False)
    assert fake.ended == "FINISHED"


def test_phase_d_mlflow_non_strict_does_not_fail_without_mlflow(monkeypatch):
    tracker = PhaseDMLflowTracker(enabled=True, strict=False)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert tracker.start(
        campaign_id="c", matrix_run_id="m", phase_c_campaign_run_id="pc", phase_d_run_id="pd",
        selected_aggregation_run_count=1, configuration={}, resume=False,
        overwrite_existing=False, continue_on_error=False, dry_run=False,
    ) is None
