"""Atomic JSON persistence for EnergyPlus case and run manifests.

Manifest files are first written beside their destination under a temporary
name and then atomically replaced. Readers therefore see either the previous
complete manifest or the new complete manifest, rather than partially written
JSON after an interrupted process.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from scalebridge.integration.energyplus.manifests.models import CaseSpec, RunManifest


def _atomic_write_json(model: CaseSpec | RunManifest, path: str | Path) -> Path:
    """Serialize a supported manifest model with an atomic file replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")

    try:
        temporary.write_text(
            model.model_dump_json(
                indent=2,
                exclude_none=True,
                exclude_computed_fields=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination


def write_case_spec(case_spec: CaseSpec, path: str | Path) -> Path:
    """Write a case specification atomically as UTF-8 JSON.

    Computed fields such as ``case_id`` are omitted because they are derived
    from the persisted scientific configuration during validation.
    """
    return _atomic_write_json(case_spec, path)


def load_case_spec(path: str | Path) -> CaseSpec:
    """Load and validate a case specification from UTF-8 JSON."""
    return CaseSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_run_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Write a run manifest atomically as UTF-8 JSON."""
    return _atomic_write_json(manifest, path)


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load and validate a run manifest from UTF-8 JSON."""
    return RunManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
