"""Run prepared EnergyPlus cases through the opyplus 2.x simulation API.

Opyplus is ScaleBridge's single Python interface to EnergyPlus. This module
therefore delegates simulation creation, EnergyPlus process execution, status
inspection, and output-resource access to ``opyplus.simulate`` and the returned
``opyplus.Simulation`` object.

The wrapper exists to provide ScaleBridge-specific behavior around opyplus:

1. validate prepared IDF, EPW, and run-directory inputs;
2. capture opyplus progress messages in a stable log file;
3. normalize opyplus status values into a boolean completion result;
4. extract warning, severe, and fatal counts from ``Simulation.get_out_err``;
5. preserve the simulation directory and failure explanation; and
6. return a serializable result suitable for manifests and distributed loops.

The runner intentionally does not edit IDFs, parse canonical time-series data,
write manifests, or interact with MLflow. Those responsibilities remain in
their own modules.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from scalebridge.integration.energyplus.idf.backend import OpyplusNotInstalledError


# Opyplus documentation currently uses both "finished" and "success" in its
# examples/API descriptions. Accepting both isolates callers from that detail.
SUCCESS_STATUSES = frozenset({"finished", "success"})
PROGRESS_LOG_FILENAME = "opyplus_progress.log"


class EnergyPlusRunnerError(RuntimeError):
    """Base exception for failures in the opyplus simulation boundary."""


class EnergyPlusInputError(EnergyPlusRunnerError):
    """Raised when prepared IDF, EPW, or run-directory inputs are invalid."""


class EnergyPlusExecutionError(EnergyPlusRunnerError):
    """Raised when opyplus cannot create or execute an EnergyPlus simulation."""


class EnergyPlusRunResult(BaseModel):
    """Structured outcome of one opyplus-managed EnergyPlus simulation.

    Attributes
    ----------
    output_directory:
        Actual simulation directory reported by opyplus.
    status:
        Normalized lowercase value returned by ``Simulation.get_status``.
    completed_successfully:
        Whether status and fatal-error validation indicate success.
    warning_count, severe_count, fatal_count:
        Diagnostic counts parsed from the EnergyPlus ERR content.
    runtime_seconds:
        Wall-clock duration measured around ``opyplus.simulate``.
    progress_log_path:
        File containing messages emitted through opyplus ``print_function``.
    failure_message:
        Concise explanation when successful completion cannot be established.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    output_directory: Path
    status: str
    completed_successfully: bool = False
    warning_count: int = Field(default=0, ge=0)
    severe_count: int = Field(default=0, ge=0)
    fatal_count: int = Field(default=0, ge=0)
    runtime_seconds: float = Field(ge=0)
    progress_log_path: Path
    error_file_path: Path | None = None
    failure_message: str | None = None

    # The native object enables later ESO/EIO/ERR access without rediscovering
    # the simulation directory. It is excluded from JSON serialization.
    simulation: Any = Field(exclude=True, repr=False)


def parse_energyplus_error_summary(error_text: str) -> tuple[int, int, int]:
    """Extract aggregate warning, severe-error, and fatal-error counts.

    Parameters
    ----------
    error_text:
        Complete text returned by ``Simulation.get_out_err().get_content()``.

    Returns
    -------
    tuple[int, int, int]
        Warning, severe-error, and fatal-error counts in that order.

    Notes
    -----
    EnergyPlus summary wording has remained regular, but individual diagnostic
    lines can contain the same terms. The parser evaluates candidate summary
    lines from the end of the file and falls back to zero when a category is
    absent.
    """
    import re

    summary_lines = [
        line
        for line in error_text.splitlines()
        if "energyplus" in line.casefold()
        and ("completed" in line.casefold() or "terminated" in line.casefold())
    ]
    summary_text = summary_lines[-1] if summary_lines else error_text

    patterns = (
        r"(?P<count>\d+)\s+Warning(?:s)?",
        r"(?P<count>\d+)\s+Severe Error(?:s)?",
        r"(?P<count>\d+)\s+Fatal Error(?:s)?",
    )

    counts: list[int] = []
    for pattern in patterns:
        match = re.search(pattern, summary_text, flags=re.IGNORECASE)
        counts.append(int(match.group("count")) if match else 0)

    return counts[0], counts[1], counts[2]


def normalize_simulation_status(status: Any) -> str:
    """Normalize an opyplus simulation status for stable comparisons."""
    return str(status or "unknown").strip().casefold()


class EnergyPlusRunner:
    """Execute one prepared IDF/EPW pair exclusively through opyplus.

    Parameters
    ----------
    beat_frequency_seconds:
        Optional opyplus heartbeat interval forwarded as ``beat_freq``.
    progress_callback:
        Optional caller callback invoked for every opyplus progress message in
        addition to writing the persistent progress log.
    """

    def __init__(
        self,
        *,
        beat_frequency_seconds: float | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if beat_frequency_seconds is not None and beat_frequency_seconds <= 0:
            raise ValueError("beat_frequency_seconds must be greater than zero")

        self._beat_frequency_seconds = beat_frequency_seconds
        self._progress_callback = progress_callback

    def run(
        self,
        *,
        idf_path: str | Path,
        epw_path: str | Path,
        output_directory: str | Path,
    ) -> EnergyPlusRunResult:
        """Execute one prepared EnergyPlus case through ``opyplus.simulate``.

        Parameters
        ----------
        idf_path:
            Prepared IDF containing all requested output objects.
        epw_path:
            EPW weather file used by the simulation.
        output_directory:
            Dedicated directory passed to opyplus as ``base_dir_path``.

        Returns
        -------
        EnergyPlusRunResult
            Normalized status, diagnostics, native simulation object, and
            output locations.

        Raises
        ------
        EnergyPlusInputError
            If inputs are absent, use invalid extensions, or the output path
            conflicts with an existing file.
        OpyplusNotInstalledError
            If opyplus is unavailable in the active environment.
        EnergyPlusExecutionError
            If opyplus raises while creating or running the simulation.
        """
        # ------------------------------------------------------------------
        # Phase 1: Normalize and validate simulation filesystem inputs.
        # ------------------------------------------------------------------
        prepared_idf = Path(idf_path).expanduser().resolve()
        weather_file = Path(epw_path).expanduser().resolve()
        requested_output = Path(output_directory).expanduser().resolve()

        self._validate_input_file(prepared_idf, expected_suffix=".idf", label="IDF")
        self._validate_input_file(weather_file, expected_suffix=".epw", label="EPW")

        if requested_output.exists() and not requested_output.is_dir():
            raise EnergyPlusInputError(
                f"simulation output path is not a directory: {requested_output}"
            )
        requested_output.mkdir(parents=True, exist_ok=True)

        opyplus = self._import_opyplus()
        progress_messages: list[str] = []

        # Keep callback handling local to one execution so concurrent runner
        # instances do not share mutable progress state.
        def capture_progress(message: Any) -> None:
            """Capture one opyplus progress message and notify the caller."""
            normalized_message = str(message)
            progress_messages.append(normalized_message)
            if self._progress_callback is not None:
                self._progress_callback(normalized_message)

        # ------------------------------------------------------------------
        # Phase 2: Delegate all EnergyPlus execution to opyplus.
        # ------------------------------------------------------------------
        started_at = time.perf_counter()
        try:
            simulation = opyplus.simulate(
                str(prepared_idf),
                str(weather_file),
                str(requested_output),
                print_function=capture_progress,
                beat_freq=self._beat_frequency_seconds,
            )
        except Exception as exc:
            raise EnergyPlusExecutionError(
                f"opyplus could not execute EnergyPlus for IDF {prepared_idf}"
            ) from exc
        runtime_seconds = time.perf_counter() - started_at

        # ------------------------------------------------------------------
        # Phase 3: Resolve the actual simulation directory and persist logs.
        # ------------------------------------------------------------------
        simulation_directory = self._get_simulation_directory(
            simulation,
            fallback=requested_output,
        )
        simulation_directory.mkdir(parents=True, exist_ok=True)

        progress_log_path = simulation_directory / PROGRESS_LOG_FILENAME
        progress_text = "\n".join(progress_messages)
        if progress_text:
            progress_text += "\n"
        progress_log_path.write_text(progress_text, encoding="utf-8")

        # ------------------------------------------------------------------
        # Phase 4: Inspect opyplus status and EnergyPlus ERR diagnostics.
        # ------------------------------------------------------------------
        try:
            status = normalize_simulation_status(simulation.get_status())
        except Exception as exc:
            raise EnergyPlusExecutionError(
                "opyplus simulation did not expose a readable status"
            ) from exc

        error_text, error_file_path = self._read_error_output(
            simulation,
            simulation_directory,
        )
        warning_count, severe_count, fatal_count = parse_energyplus_error_summary(
            error_text
        )

        completed_successfully = status in SUCCESS_STATUSES and fatal_count == 0
        failure_message = None
        if not completed_successfully:
            failure_message = self._build_failure_message(
                status=status,
                severe_count=severe_count,
                fatal_count=fatal_count,
            )

        return EnergyPlusRunResult(
            output_directory=simulation_directory,
            status=status,
            completed_successfully=completed_successfully,
            warning_count=warning_count,
            severe_count=severe_count,
            fatal_count=fatal_count,
            runtime_seconds=runtime_seconds,
            progress_log_path=progress_log_path,
            error_file_path=error_file_path,
            failure_message=failure_message,
            simulation=simulation,
        )

    @staticmethod
    def _import_opyplus() -> Any:
        """Import opyplus lazily when EnergyPlus execution is requested."""
        try:
            import opyplus
        except ImportError as exc:
            raise OpyplusNotInstalledError(
                "opyplus 2.0.7 or newer is required for EnergyPlus execution"
            ) from exc
        return opyplus

    @staticmethod
    def _validate_input_file(path: Path, *, expected_suffix: str, label: str) -> None:
        """Validate one required EnergyPlus input file."""
        if not path.is_file():
            raise EnergyPlusInputError(f"{label} file does not exist: {path}")
        if path.suffix.casefold() != expected_suffix:
            raise EnergyPlusInputError(
                f"{label} file must use the {expected_suffix} extension: {path}"
            )

    @staticmethod
    def _get_simulation_directory(simulation: Any, *, fallback: Path) -> Path:
        """Return the directory reported by opyplus or a validated fallback."""
        try:
            directory = simulation.get_dir_path()
        except (AttributeError, TypeError):
            directory = None
        return Path(directory).expanduser().resolve() if directory else fallback

    @staticmethod
    def _read_error_output(
        simulation: Any,
        simulation_directory: Path,
    ) -> tuple[str, Path | None]:
        """Read EnergyPlus ERR content through the opyplus Simulation object."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*ChainedAssignmentError.*",
                    category=FutureWarning,
                    module=r"opyplus\.err",
                )
                error_output = simulation.get_out_err()
                error_text = str(error_output.get_content())
        except Exception:
            return "", None

        # Prefer opyplus resource discovery; retain the conventional path as a
        # fallback because Simulation.get_out_err already proved the resource.
        error_file_path: Path | None = None
        try:
            candidate = simulation.get_resource_path("out_err")
            if candidate:
                error_file_path = Path(candidate).expanduser().resolve()
        except Exception:
            pass

        if error_file_path is None:
            conventional_path = simulation_directory / "eplusout.err"
            if conventional_path.exists():
                error_file_path = conventional_path

        return error_text, error_file_path

    @staticmethod
    def _build_failure_message(
        *,
        status: str,
        severe_count: int,
        fatal_count: int,
    ) -> str:
        """Build a concise failure explanation from status and diagnostics."""
        reasons = [f"opyplus simulation status was {status!r}"]
        if severe_count:
            reasons.append(f"EnergyPlus reported {severe_count} severe error(s)")
        if fatal_count:
            reasons.append(f"EnergyPlus reported {fatal_count} fatal error(s)")
        return "; ".join(reasons)
