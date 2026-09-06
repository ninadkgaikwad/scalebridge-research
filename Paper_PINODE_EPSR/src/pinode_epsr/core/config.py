from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

SpatialCase = Literal["all_to_one", "identity_ind", "identity_dep1", "identity_dep2"]

CAMPAIGN_ID = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
CASE_ID = "epcase_827ca4812c0199221d031e59"
ALL_TO_ONE_RUN = "aggr_20260715_114247_0001_a8695a44_smoke_l01_all_to_one_equal"
IDENTITY_RUN = "aggr_20260715_114401_0002_a8695a44_smoke_l05_identity_equal"
CONTROLLED_PHASE_C_RUN_ID = "phase_c_full_updated_test_laptop_20260802_172455"

EXPECTED_ROWS = 105_120
EXPECTED_PARTITIONS = {
    "train": 73_567,
    "validation": 15_763,
    "test": 15_754,
    "excluded": 36,
}
DT_SECONDS = 300.0


@dataclass(frozen=True)
class CaseSpec:
    name: SpatialCase
    aggregation_run_id: str
    dependency_mode: str
    zone_ids: tuple[str, ...]
    phase_d_paths: tuple[str, ...]
    all_to_one_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperConfig:
    generated_data_root: Path
    campaign_id: str = CAMPAIGN_ID
    case_id: str = CASE_ID
    dt_seconds: float = DT_SECONDS
    heat_routing: str = "phase_d_convective_radiative"
    # Locked heat semantics:
    #   convective -> QZIC + QSol1 + QAC
    #   radiative  -> QZIR + QSol2
    # 1R1C uses all channels at unit gain in its single balance.
    # 2R2C sends convective heat to air and splits radiative heat using eta_rad.
    eta_rad_2r2c: float = 1.0
    eta_rad_mode_2r2c: str = "mass_only"
    controlled_phase_c_run_id: str = CONTROLLED_PHASE_C_RUN_ID
    optuna_representative_max_windows: int = 256
    optuna_seed: int = 42

    @classmethod
    def from_environment(cls) -> "PaperConfig":
        root = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT")
        if not root:
            raise RuntimeError(
                "SCALEBRIDGE_GENERATED_DATA_ROOT is required. "
                "The paper code reads canonical ScaleBridge campaign artifacts in place."
            )
        return cls(generated_data_root=Path(root))

    @property
    def campaign_root(self) -> Path:
        return self.generated_data_root / "campaigns" / self.campaign_id

    @property
    def phase_d_case_root(self) -> Path:
        return self.campaign_root / "phase_d" / "cases" / self.case_id


def canonical_case_specs() -> dict[SpatialCase, CaseSpec]:
    # All-to-one IND/DEP1/DEP2 were audited as byte-identical.  We read IND once
    # and retain aliases in provenance rather than training duplicate models.
    return {
        "all_to_one": CaseSpec(
            name="all_to_one",
            aggregation_run_id=ALL_TO_ONE_RUN,
            dependency_mode="independent",
            zone_ids=("RestaurantFastFood_All",),
            phase_d_paths=(
                f"aggregation_runs/{ALL_TO_ONE_RUN}/silos/ml/ind/"
                "RestaurantFastFood_All/grp_vrin/l1_h1/mdh",
            ),
            all_to_one_aliases=("ind", "dep1", "dep2"),
        ),
        "identity_ind": CaseSpec(
            name="identity_ind",
            aggregation_run_id=IDENTITY_RUN,
            dependency_mode="independent",
            zone_ids=("Dining", "Kitchen"),
            phase_d_paths=(
                f"aggregation_runs/{IDENTITY_RUN}/silos/ml/ind/Dining/grp_vrin/l1_h1/mdh",
                f"aggregation_runs/{IDENTITY_RUN}/silos/ml/ind/Kitchen/grp_vrin/l1_h1/mdh",
            ),
        ),
        "identity_dep1": CaseSpec(
            name="identity_dep1",
            aggregation_run_id=IDENTITY_RUN,
            dependency_mode="dependent1",
            zone_ids=("Dining", "Kitchen"),
            phase_d_paths=(
                f"aggregation_runs/{IDENTITY_RUN}/silos/ml/dep1/grp_vrin/l1_h1/mdh",
            ),
        ),
        "identity_dep2": CaseSpec(
            name="identity_dep2",
            aggregation_run_id=IDENTITY_RUN,
            dependency_mode="dependent2",
            zone_ids=("Dining", "Kitchen"),
            phase_d_paths=(
                f"aggregation_runs/{IDENTITY_RUN}/silos/ml/dep2/grp_vrin/l1_h1/mdh",
            ),
        ),
    }
