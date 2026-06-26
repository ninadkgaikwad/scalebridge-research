"""Tests for the opyplus-managed EnergyPlus simulation runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scalebridge.integration.energyplus import (
    EnergyPlusExecutionError,
    EnergyPlusInputError,
    EnergyPlusRunner,
)
from scalebridge.integration.energyplus.simulation.runner import (
    normalize_simulation_status,
    parse_energyplus_error_summary,
)


SUCCESSFUL_ERROR_FILE = """
Program Version,EnergyPlus, Version 22.1.0
EnergyPlus Warmup Error Summary. During Warmup: 1 Warning; 0 Severe Errors.
EnergyPlus Completed Successfully-- 3 Warning; 1 Severe Errors
"""

FAILED_ERROR_FILE = """
Program Version,EnergyPlus, Version 22.1.0
EnergyPlus Terminated-- 2 Warning; 4 Severe Errors; 1 Fatal Error
"""


class FakeErr:
    """Small stand-in for the opyplus ``Err`` output object."""

    def __init__(self, content: str) -> None:
        self._content = content

    def get_content(self) -> str:
        """Return the configured EnergyPlus ERR content."""
        return self._content


class FakeSimulation:
    """Stand-in for the opyplus ``Simulation`` object used by the runner."""

    def __init__(self, directory: Path, status: str, error_text: str) -> None:
        self._directory = directory
        self._status = status
        self._error_text = error_text

    def get_dir_path(self) -> str:
        """Return the fake simulation directory."""
        return str(self._directory)

    def get_status(self) -> str:
        """Return the configured opyplus status."""
        return self._status

    def get_out_err(self) -> FakeErr:
        """Return a fake opyplus ERR output object."""
        return FakeErr(self._error_text)

    def get_resource_path(self, reference: str) -> str:
        """Resolve the ERR resource used by the runner."""
        assert reference == "out_err"
        return str(self._directory / "eplusout.err")


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal IDF and EPW placeholder files for runner tests."""
    idf_path = tmp_path / "prepared.idf"
    epw_path = tmp_path / "weather.epw"
    idf_path.write_text("Version,22.1;\n", encoding="utf-8")
    epw_path.write_text("LOCATION,Test\n", encoding="utf-8")
    return idf_path, epw_path


def _fake_opyplus(
    simulation: FakeSimulation,
    captured_calls: list[dict[str, Any]],
) -> SimpleNamespace:
    """Build a fake opyplus module exposing the documented ``simulate`` API."""

    def simulate(
        idf_path: str,
        epw_path: str,
        base_dir_path: str,
        simulation_name: str | None = None,
        print_function=None,
        beat_freq: float | None = None,
    ) -> FakeSimulation:
        """Capture arguments, emit progress, and return the fake simulation."""
        captured_calls.append(
            {
                "idf_path": idf_path,
                "epw_path": epw_path,
                "base_dir_path": base_dir_path,
                "simulation_name": simulation_name,
                "beat_freq": beat_freq,
            }
        )
        if print_function is not None:
            print_function("EnergyPlus started")
            print_function("EnergyPlus finished")
        return simulation

    return SimpleNamespace(simulate=simulate)


def test_error_summary_parser_uses_completion_summary() -> None:
    """The final completion line must override earlier phase summaries."""
    assert parse_energyplus_error_summary(SUCCESSFUL_ERROR_FILE) == (3, 1, 0)
    assert parse_energyplus_error_summary(FAILED_ERROR_FILE) == (2, 4, 1)


def test_status_normalization_handles_documented_values() -> None:
    """Documented opyplus status spellings must normalize consistently."""
    assert normalize_simulation_status(" Finished ") == "finished"
    assert normalize_simulation_status("SUCCESS") == "success"
    assert normalize_simulation_status(None) == "unknown"


def test_runner_delegates_successful_execution_to_opyplus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful run must use opyplus and return normalized diagnostics."""
    idf_path, epw_path = _write_inputs(tmp_path)
    output_directory = tmp_path / "outputs"
    output_directory.mkdir()
    (output_directory / "eplusout.err").write_text(
        SUCCESSFUL_ERROR_FILE,
        encoding="utf-8",
    )

    simulation = FakeSimulation(
        output_directory,
        status="finished",
        error_text=SUCCESSFUL_ERROR_FILE,
    )
    captured_calls: list[dict[str, Any]] = []
    fake_module = _fake_opyplus(simulation, captured_calls)
    monkeypatch.setattr(EnergyPlusRunner, "_import_opyplus", lambda self: fake_module)

    callback_messages: list[str] = []
    result = EnergyPlusRunner(
        beat_frequency_seconds=2,
        progress_callback=callback_messages.append,
    ).run(
        idf_path=idf_path,
        epw_path=epw_path,
        output_directory=output_directory,
    )

    assert result.completed_successfully
    assert result.status == "finished"
    assert result.warning_count == 3
    assert result.severe_count == 1
    assert result.fatal_count == 0
    assert result.simulation is simulation
    assert captured_calls[0]["beat_freq"] == 2
    assert captured_calls[0]["simulation_name"] is None
    assert callback_messages == ["EnergyPlus started", "EnergyPlus finished"]
    assert result.progress_log_path.read_text(encoding="utf-8") == (
        "EnergyPlus started\nEnergyPlus finished\n"
    )


def test_runner_returns_failed_result_for_opyplus_error_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EnergyPlus runtime errors must be returned so a batch can continue."""
    idf_path, epw_path = _write_inputs(tmp_path)
    output_directory = tmp_path / "outputs"
    simulation = FakeSimulation(
        output_directory,
        status="error",
        error_text=FAILED_ERROR_FILE,
    )
    fake_module = _fake_opyplus(simulation, [])
    monkeypatch.setattr(EnergyPlusRunner, "_import_opyplus", lambda self: fake_module)

    result = EnergyPlusRunner().run(
        idf_path=idf_path,
        epw_path=epw_path,
        output_directory=output_directory,
    )

    assert not result.completed_successfully
    assert result.status == "error"
    assert result.severe_count == 4
    assert result.fatal_count == 1
    assert "status was 'error'" in result.failure_message
    assert "1 fatal error" in result.failure_message


def test_runner_wraps_opyplus_execution_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected opyplus execution failures must retain exception context."""
    idf_path, epw_path = _write_inputs(tmp_path)

    def fail_simulation(*args, **kwargs):
        """Raise the representative exception emitted by a failed API call."""
        raise RuntimeError("EnergyPlus executable unavailable")

    fake_module = SimpleNamespace(simulate=fail_simulation)
    monkeypatch.setattr(EnergyPlusRunner, "_import_opyplus", lambda self: fake_module)

    with pytest.raises(EnergyPlusExecutionError, match="could not execute"):
        EnergyPlusRunner().run(
            idf_path=idf_path,
            epw_path=epw_path,
            output_directory=tmp_path / "outputs",
        )


def test_runner_rejects_missing_input_before_opyplus_import(tmp_path: Path) -> None:
    """Missing prepared IDFs must fail before opyplus execution begins."""
    _, epw_path = _write_inputs(tmp_path)

    with pytest.raises(EnergyPlusInputError, match="IDF file does not exist"):
        EnergyPlusRunner().run(
            idf_path=tmp_path / "missing.idf",
            epw_path=epw_path,
            output_directory=tmp_path / "outputs",
        )
