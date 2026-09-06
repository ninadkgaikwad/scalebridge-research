from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os


@dataclass(frozen=True)
class EPSRProjectLayout:
    """
    Permanent PINODE/EPSR code/data separation.

    Clean repo root:
      .../NewOrg/scalebridge-research/Paper_PINODE_EPSR

    Scientific data root:
      .../Data/ScaleBridge/Paper_PINODE_EPSR
    """

    repo_root: Path
    data_root: Path

    @classmethod
    def from_repo_root(
        cls,
        repo_root: str | Path,
        data_root: str | Path | None = None,
    ) -> "EPSRProjectLayout":
        repo = Path(repo_root).expanduser().resolve()

        if data_root is None:
            # .../BuildingModelingProject_Condensed/NewOrg/scalebridge-research/Paper_PINODE_EPSR
            project_condensed = repo.parents[2]
            data = (
                project_condensed
                / "Data"
                / "ScaleBridge"
                / "Paper_PINODE_EPSR"
            ).resolve()
        else:
            data = Path(data_root).expanduser().resolve()

        layout = cls(repo_root=repo, data_root=data)
        layout.assert_separation()
        return layout

    @classmethod
    def from_environment_or_source(cls) -> "EPSRProjectLayout":
        repo_env = os.environ.get("PINODE_EPSR_REPO_ROOT")
        data_env = os.environ.get("PINODE_EPSR_DATA_ROOT")

        if repo_env:
            return cls.from_repo_root(repo_env, data_env)

        here = Path(__file__).resolve()
        for parent in here.parents:
            if parent.name == "Paper_PINODE_EPSR":
                return cls.from_repo_root(parent, data_env)

        raise RuntimeError(
            "Could not locate Paper_PINODE_EPSR. "
            "Set PINODE_EPSR_REPO_ROOT."
        )

    def assert_separation(self) -> None:
        repo = self.repo_root.resolve()
        data = self.data_root.resolve()

        if repo == data:
            raise RuntimeError("Repo root and data root cannot be identical.")
        if repo in data.parents:
            raise RuntimeError("Scientific data root cannot live inside repo.")
        if data in repo.parents:
            raise RuntimeError("Code repo cannot live inside scientific data root.")

    def new_simulator_run_dir(
        self,
        *,
        label: str,
        timestamp: str | None = None,
    ) -> Path:
        stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        clean = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in label.strip()
        ).strip("_") or "simulator_run"

        return (
            self.data_root
            / "05_closed_loop_mpc_runs"
            / f"{clean}_{stamp}"
        )
