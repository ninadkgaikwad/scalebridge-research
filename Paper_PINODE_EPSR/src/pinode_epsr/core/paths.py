from __future__ import annotations

import os
from pathlib import Path

PAPER_DATA_ENV = "SCALEBRIDGE_PINODE_EPSR_DATA_ROOT"
GENERATED_DATA_ENV = "SCALEBRIDGE_GENERATED_DATA_ROOT"


def paper_repository_root() -> Path:
    """Return the standalone-capable ``Paper_PINODE_EPSR`` project root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "pinode_epsr").is_dir():
            return parent
    # Editable/source-layout fallback.
    return here.parents[3]


def scalebridge_repository_root() -> Path | None:
    """Return the embedding main ScaleBridge repo when this package lives inside it."""
    paper = paper_repository_root()
    parent = paper.parent
    if (parent / "src" / "scalebridge").is_dir() and (parent / "pyproject.toml").is_file():
        return parent
    return None


def repository_root() -> Path:
    """Compatibility name: prefer the embedding ScaleBridge repo, else paper repo."""
    return scalebridge_repository_root() or paper_repository_root()


def default_paper_data_root() -> Path:
    generated = os.environ.get(GENERATED_DATA_ENV)
    if generated:
        return Path(generated).expanduser().resolve() / "Paper_PINODE_EPSR"

    scale_repo = scalebridge_repository_root()
    if scale_repo is not None:
        # .../BuildingModelingProject_Condensed/NewOrg/scalebridge-research
        # -> .../BuildingModelingProject_Condensed/Data/ScaleBridge/Paper_PINODE_EPSR
        project = scale_repo.parent.parent
        return project / "Data" / "ScaleBridge" / "Paper_PINODE_EPSR"

    # Standalone-repository fallback. Explicit/env override is preferred for real runs.
    return paper_repository_root() / "external_data" / "Paper_PINODE_EPSR"


def resolve_paper_data_root(explicit: str | Path | None = None, *, create: bool = False) -> Path:
    if explicit is not None:
        root = Path(explicit).expanduser()
    elif os.environ.get(PAPER_DATA_ENV):
        root = Path(os.environ[PAPER_DATA_ENV]).expanduser()
    else:
        root = default_paper_data_root()
    root = root.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def validation_run_dir(run_id: str, explicit: str | Path | None = None) -> Path:
    root = resolve_paper_data_root(explicit, create=True) / "validation" / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root
