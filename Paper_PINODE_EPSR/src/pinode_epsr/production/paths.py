from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

from ..core.config import PaperConfig
from ..core.paths import GENERATED_DATA_ENV, resolve_paper_data_root, scalebridge_repository_root


def resolve_production_config() -> PaperConfig:
    """Resolve the controlled paper config without requiring a machine-local path.

    Priority is the explicit ``SCALEBRIDGE_GENERATED_DATA_ROOT`` override.
    Otherwise, when embedded in the ScaleBridge repository, the authoritative
    data root is derived from the locked layout::

        <Project>/NewOrg/scalebridge-research
        <Project>/Data/ScaleBridge

    This keeps machine portability while preserving the shared campaign root as
    read-only scientific input.
    """
    explicit = os.environ.get(GENERATED_DATA_ENV)
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        repo = scalebridge_repository_root()
        if repo is None:
            raise RuntimeError(
                f"{GENERATED_DATA_ENV} is not set and the embedding ScaleBridge "
                "repository could not be resolved"
            )
        root = (repo.parent.parent / "Data" / "ScaleBridge").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ScaleBridge generated-data root is missing: {root}")
    return PaperConfig(generated_data_root=root)


@dataclass(frozen=True)
class ProductionLayout:
    generated_data_root: Path
    campaign_root: Path
    paper_data_root: Path
    hpo_root: Path
    training_root: Path
    checkpoint_root: Path
    offline_root: Path
    manifest_root: Path

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    def ensure(self) -> "ProductionLayout":
        # The campaign is scientific input and must already exist.  No mkdir is
        # ever performed beneath it by the production layer.
        if not self.campaign_root.is_dir():
            raise FileNotFoundError(f"Controlled campaign root is missing: {self.campaign_root}")
        if self.paper_data_root == self.campaign_root or self.campaign_root in self.paper_data_root.parents:
            raise RuntimeError("EPSR generated-data root must not be inside the read-only campaign root")
        for path in (
            self.paper_data_root,
            self.hpo_root,
            self.training_root,
            self.checkpoint_root,
            self.offline_root / "sim1",
            self.offline_root / "sim2",
            self.offline_root / "sim3",
            self.paper_data_root / "05_closed_loop_mpc_runs",
            self.paper_data_root / "06_tables",
            self.paper_data_root / "07_figures",
            self.manifest_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def resolve_production_layout(config: PaperConfig, *, paper_data_root: str | Path | None = None, create: bool = False) -> ProductionLayout:
    paper = resolve_paper_data_root(paper_data_root, create=create)
    layout = ProductionLayout(
        generated_data_root=config.generated_data_root.resolve(),
        campaign_root=config.campaign_root.resolve(),
        paper_data_root=paper.resolve(),
        hpo_root=paper / "01_hpo",
        training_root=paper / "02_training",
        checkpoint_root=paper / "03_checkpoints",
        offline_root=paper / "04_offline_results",
        manifest_root=paper / "08_manifests",
    )
    return layout.ensure() if create else layout
