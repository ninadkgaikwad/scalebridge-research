"""Tests for complete loop-safe EnergyPlus generation orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scalebridge.integration.energyplus import (
    EnergyPlusGenerationOrchestrator,
    RunStatus,
    TrackingMetadata,
    load_run_manifest,
)
from scalebridge.integration.energyplus.outputs import CanonicalExtractionError


class FakePreparer:
    """Create a prepared IDF or raise a configured preparation failure."""

    def __init__(self, *, failure: Exception | None = None) -> None:
        self._failure = failure

    def prepare(self, case_spec, destination_path: Path) -> SimpleNamespace:
        """Write the fake prepared IDF and return its path."""
        if self._failure is not None:
            raise self._failure
        destination_path.write_text("prepared\n", encoding="utf-8")
        return SimpleNamespace(prepared_idf_path=destination_path)


class FakeRunner:
    """Create representative raw files and return a successful run result."""

    def run(
        self,
        *,
        idf_path: Path,
        epw_path: Path,
        output_directory: Path,
    ) -> SimpleNamespace:
        """Return a successful result containing one EnergyPlus warning."""
        (output_directory / "eplusout.err").write_text(
            "EnergyPlus Completed Successfully-- 1 Warning; 0 Severe Errors\n",
            encoding="utf-8",
        )
        (output_directory / "eplusout.eso").write_text("eso\n", encoding="utf-8")
        (output_directory / "eplusout.eio").write_text("eio\n", encoding="utf-8")
        return SimpleNamespace(
            completed_successfully=True,
            output_directory=output_directory,
            warning_count=1,
            severe_count=0,
            fatal_count=0,
            failure_message=None,
        )


class FakeExtractor:
    """Create canonical and compatibility artifacts or raise a failure."""

    def __init__(self, *, failure: Exception | None = None) -> None:
        self._failure = failure

    def extract(
        self,
        *,
        case_spec,
        simulation_directory: Path,
        canonical_directory: Path,
        legacy_directory: Path,
    ) -> SimpleNamespace:
        """Write representative extraction artifacts and return counts."""
        if self._failure is not None:
            raise self._failure

        canonical_directory.mkdir(parents=True)
        legacy_directory.mkdir(parents=True)
        parquet_path = canonical_directory / "timeseries_timestep.parquet"
        metadata_path = canonical_directory / "metadata.json"
        eio_path = canonical_directory / "eio_tables.json"
        output_pickle = legacy_directory / "IDF_OutputVariables_DictDF.pickle"
        eio_pickle = legacy_directory / "Eio_OutputFile.pickle"
        for path in (
            parquet_path,
            metadata_path,
            eio_path,
            output_pickle,
            eio_pickle,
        ):
            path.write_bytes(b"artifact")

        return SimpleNamespace(
            produced_signal_count=3,
            missing_required_signals=(),
            timestep_count=576,
        )


class FakeMLflowTracker:
    """Capture optional MLflow lifecycle calls without importing MLflow."""

    def __init__(self) -> None:
        self.started = False
        self.finished = False

    def start(self, **kwargs):
        """Return a tracking handle compatible with the orchestrator."""
        self.started = True
        return SimpleNamespace(
            to_metadata=lambda: TrackingMetadata(
                mlflow_experiment="p1",
                mlflow_run_id="mlflow-run",
                mlflow_tracking_uri="http://laptop:5000",
            )
        )

    def finish(self, **kwargs) -> None:
        """Record that final metrics and manifest logging were requested."""
        self.finished = True


def _build_orchestrator(
    tmp_path: Path,
    *,
    preparer: FakePreparer | None = None,
    extractor: FakeExtractor | None = None,
) -> EnergyPlusGenerationOrchestrator:
    """Build an orchestrator with deterministic injected services."""
    return EnergyPlusGenerationOrchestrator(
        generated_data_root=tmp_path / "generated",
        machine_id="test-machine",
        idf_preparer=preparer or FakePreparer(),
        runner=FakeRunner(),
        output_extractor=extractor or FakeExtractor(),
    )


def test_orchestrator_persists_successful_attempt(
    tmp_path: Path,
    case_spec,
) -> None:
    """A successful run must return paths, validation, and a full manifest."""
    result = _build_orchestrator(tmp_path).generate(
        case_spec,
        campaign_id="p1_test",
        run_id="eprun_success",
    )

    assert result.status is RunStatus.COMPLETED_WITH_WARNINGS
    assert result.warning_count == 1
    assert result.produced_signal_count == 3
    assert result.timestep_count == 576
    assert result.manifest_path.is_file()
    assert "timeseries_timestep.parquet" in result.canonical_output_paths
    assert "IDF_OutputVariables_DictDF.pickle" in (
        result.compatibility_output_paths
    )

    manifest = load_run_manifest(result.manifest_path)
    assert manifest.run_id == "eprun_success"
    assert manifest.campaign_id == "p1_test"
    assert manifest.status is RunStatus.COMPLETED_WITH_WARNINGS
    assert manifest.execution.machine_id == "test-machine"
    assert manifest.validation.produced_signals == 3
    assert manifest.error is None

    latest_path = result.artifact_root.parents[1] / "latest_run.json"
    assert latest_path.is_file()


def test_orchestrator_contains_preparation_failure(
    tmp_path: Path,
    case_spec,
) -> None:
    """Preparation failures must return failed results so loops can continue."""
    orchestrator = _build_orchestrator(
        tmp_path,
        preparer=FakePreparer(failure=RuntimeError("preparation failed")),
    )

    result = orchestrator.generate(case_spec, run_id="eprun_prepare_failure")

    assert result.status is RunStatus.FAILED
    assert result.error_type == "RuntimeError"
    assert result.error_message == "preparation failed"
    manifest = load_run_manifest(result.manifest_path)
    assert manifest.error is not None
    assert manifest.error.traceback_path == Path("traceback.txt")
    assert (result.artifact_root / "traceback.txt").is_file()


def test_orchestrator_marks_extraction_failure_invalid(
    tmp_path: Path,
    case_spec,
) -> None:
    """A successful simulation with invalid outputs must be marked invalid."""
    orchestrator = _build_orchestrator(
        tmp_path,
        extractor=FakeExtractor(
            failure=CanonicalExtractionError("required signal missing")
        ),
    )

    result = orchestrator.generate(case_spec, run_id="eprun_invalid_output")

    assert result.status is RunStatus.INVALID
    assert result.warning_count == 1
    assert result.error_type == "CanonicalExtractionError"
    manifest = load_run_manifest(result.manifest_path)
    assert manifest.status is RunStatus.INVALID
    assert manifest.error is not None


def test_orchestrator_persists_optional_mlflow_references(
    tmp_path: Path,
    case_spec,
) -> None:
    """MLflow identifiers must be persisted without owning scientific data."""
    tracker = FakeMLflowTracker()
    orchestrator = EnergyPlusGenerationOrchestrator(
        generated_data_root=tmp_path / "generated",
        machine_id="test-machine",
        idf_preparer=FakePreparer(),
        runner=FakeRunner(),
        output_extractor=FakeExtractor(),
        mlflow_tracker=tracker,
    )

    result = orchestrator.generate(case_spec, run_id="eprun_mlflow")

    manifest = load_run_manifest(result.manifest_path)
    assert tracker.started
    assert tracker.finished
    assert manifest.tracking.mlflow_run_id == "mlflow-run"
