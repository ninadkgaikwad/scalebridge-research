"""
EnergyPlus RDD probe utilities.

The P1 requested output-variable list is a maximum desired vocabulary.
A specific building/case may not be able to produce every variable in that
maximum list. EnergyPlus exposes the case-specific available report variables
through eplusout.rdd when the IDF includes:

    Output:VariableDictionary,
        Regular;

This module creates a temporary probe IDF, runs EnergyPlus once, and returns
the produced eplusout.rdd path. The RDD can then be parsed by
scalebridge.integration.energyplus.generation.rdd.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


_OUTPUT_VARIABLE_DICTIONARY_RE = re.compile(
    r"^\s*Output:VariableDictionary\s*,",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class RddProbeResult:
    """
    Result of an EnergyPlus RDD probe run.
    """

    source_idf_path: Path
    probe_idf_path: Path
    weather_path: Path
    output_dir: Path
    rdd_path: Path
    err_path: Path
    return_code: int
    stdout_path: Path
    stderr_path: Path


def ensure_output_variable_dictionary_regular(idf_text: str) -> str:
    """
    Ensure the IDF requests an EnergyPlus report-variable dictionary.

    If an Output:VariableDictionary object already exists, the IDF is returned
    unchanged. Otherwise, this appends:

        Output:VariableDictionary,
            Regular;

    Parameters
    ----------
    idf_text:
        Raw IDF text.

    Returns
    -------
    str
        IDF text with Output:VariableDictionary enabled.
    """
    if _OUTPUT_VARIABLE_DICTIONARY_RE.search(idf_text):
        return idf_text

    suffix = (
        "\n\n"
        "! Added by ScaleBridge RDD probe.\n"
        "Output:VariableDictionary,\n"
        "    Regular;\n"
    )

    return idf_text.rstrip() + suffix + "\n"


def write_rdd_probe_idf(
    *,
    source_idf_path: Path | str,
    probe_idf_path: Path | str,
) -> Path:
    """
    Write a probe IDF that will produce eplusout.rdd.

    Parameters
    ----------
    source_idf_path:
        Normalized source IDF path.
    probe_idf_path:
        Destination probe IDF path.

    Returns
    -------
    Path
        Written probe IDF path.
    """
    source = Path(source_idf_path)
    target = Path(probe_idf_path)

    if not source.exists():
        raise FileNotFoundError(f"Source IDF does not exist: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)

    text = source.read_text(encoding="utf-8", errors="replace")
    text = ensure_output_variable_dictionary_regular(text)
    target.write_text(text, encoding="utf-8")

    return target


def resolve_energyplus_executable(
    energyplus_exe: Path | str | None = None,
) -> Path:
    """
    Resolve the EnergyPlus executable path.

    Resolution order:
      1. explicit energyplus_exe argument
      2. SCALEBRIDGE_ENERGYPLUS_EXE environment variable
      3. ENERGYPLUS_EXE environment variable
      4. energyplus found on PATH
      5. common Windows EnergyPlus 9.0.1 install path

    Returns
    -------
    Path
        Resolved EnergyPlus executable.

    Raises
    ------
    FileNotFoundError
        If no executable can be found.
    """
    candidates: list[str] = []

    if energyplus_exe is not None:
        candidates.append(str(energyplus_exe))

    for env_name in ("SCALEBRIDGE_ENERGYPLUS_EXE", "ENERGYPLUS_EXE"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(value)

    which_result = shutil.which("energyplus")
    if which_result:
        candidates.append(which_result)

    candidates.extend(
        [
            r"C:\EnergyPlusV9-0-1\energyplus.exe",
            r"C:\EnergyPlusV9-0-1\EnergyPlus.exe",
            r"C:\EnergyPlusV9-0-1\energyplus",
        ]
    )

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find EnergyPlus executable. Set one of:\n"
        "  SCALEBRIDGE_ENERGYPLUS_EXE\n"
        "  ENERGYPLUS_EXE\n"
        "or pass energyplus_exe explicitly."
    )


def run_energyplus_rdd_probe(
    *,
    source_idf_path: Path | str,
    weather_path: Path | str,
    output_dir: Path | str,
    energyplus_exe: Path | str | None = None,
    overwrite_output_dir: bool = True,
) -> RddProbeResult:
    """
    Run an EnergyPlus probe simulation to produce eplusout.rdd.

    Parameters
    ----------
    source_idf_path:
        Normalized IDF path for the case.
    weather_path:
        EPW weather path for the case.
    output_dir:
        Probe output directory. This should be a local scratch/work directory
        when possible.
    energyplus_exe:
        Optional explicit EnergyPlus executable path.
    overwrite_output_dir:
        If True, deletes an existing probe output directory before running.

    Returns
    -------
    RddProbeResult
        Probe result including eplusout.rdd path.

    Raises
    ------
    FileNotFoundError
        If input files or eplusout.rdd are missing.
    RuntimeError
        If EnergyPlus exits with a non-zero return code.
    """
    source_idf = Path(source_idf_path)
    weather = Path(weather_path)
    output = Path(output_dir)

    if not source_idf.exists():
        raise FileNotFoundError(f"Source IDF does not exist: {source_idf}")

    if not weather.exists():
        raise FileNotFoundError(f"Weather EPW does not exist: {weather}")

    exe = resolve_energyplus_executable(energyplus_exe)

    if output.exists() and overwrite_output_dir:
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)

    probe_idf = output / "rdd_probe.idf"
    stdout_path = output / "energyplus_stdout.txt"
    stderr_path = output / "energyplus_stderr.txt"

    write_rdd_probe_idf(
        source_idf_path=source_idf,
        probe_idf_path=probe_idf,
    )

    command = [
        str(exe),
        "-w",
        str(weather),
        "-d",
        str(output),
        str(probe_idf),
    ]

    completed = subprocess.run(
        command,
        cwd=str(output),
        text=True,
        capture_output=True,
        check=False,
    )

    stdout_path.write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8", errors="replace")

    rdd_path = output / "eplusout.rdd"
    err_path = output / "eplusout.err"

    result = RddProbeResult(
        source_idf_path=source_idf,
        probe_idf_path=probe_idf,
        weather_path=weather,
        output_dir=output,
        rdd_path=rdd_path,
        err_path=err_path,
        return_code=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    if completed.returncode != 0:
        err_tail = ""
        if err_path.exists():
            lines = err_path.read_text(encoding="utf-8", errors="replace").splitlines()
            err_tail = "\n".join(lines[-80:])

        raise RuntimeError(
            "EnergyPlus RDD probe failed.\n"
            f"return_code: {completed.returncode}\n"
            f"source_idf_path: {source_idf}\n"
            f"weather_path: {weather}\n"
            f"output_dir: {output}\n"
            f"energyplus_exe: {exe}\n"
            f"stderr_path: {stderr_path}\n"
            f"err_path: {err_path}\n"
            f"err_tail:\n{err_tail}"
        )

    if not rdd_path.exists():
        err_tail = ""
        if err_path.exists():
            lines = err_path.read_text(encoding="utf-8", errors="replace").splitlines()
            err_tail = "\n".join(lines[-80:])

        raise FileNotFoundError(
            "EnergyPlus RDD probe completed but did not produce eplusout.rdd.\n"
            f"source_idf_path: {source_idf}\n"
            f"weather_path: {weather}\n"
            f"output_dir: {output}\n"
            f"energyplus_exe: {exe}\n"
            f"err_path: {err_path}\n"
            f"err_tail:\n{err_tail}"
        )

    return result