# -*- coding: utf-8 -*-
"""Print and validate representative D6 contracts; writes JSON metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scalebridge.data.thermal_modeling.constants import ModelingSilo, PhaseDMode
from scalebridge.data.thermal_modeling.silo_contracts import (
    HeatInputRepresentation,
    HeatRepresentationConfig,
    SiloProductContract,
    TemporalConfig,
    ZoneSignalAvailability,
)


DINING = ZoneSignalAvailability(
    "Dining",
    (
        "qsol1", "qsol2",
        "qzic_p", "qzic_l", "qzic_ee",
        "qzir_p", "qzir_l", "qzivr_l",
    ),
)
KITCHEN = ZoneSignalAvailability(
    "Kitchen",
    (
        "qsol1",
        "qzic_p", "qzic_ee",
        "qzir_p", "qzir_ee", "qzivr_l",
    ),
)
ALL_ZONE = ZoneSignalAvailability(
    "RestaurantFastFood_All",
    (
        "qsol1", "qsol2",
        "qzic_p", "qzic_l", "qzic_ee",
        "qzir_p", "qzir_l", "qzir_ee", "qzivr_l",
    ),
)


def _heat() -> HeatRepresentationConfig:
    return HeatRepresentationConfig(
        HeatInputRepresentation.GROUPED,
        include_visible_lighting_in_qzir=True,
    )


def _contracts() -> list[tuple[str, SiloProductContract, str | None]]:
    ml = TemporalConfig(
        ModelingSilo.ML_SCIML,
        input_lag=12,
        target_horizon=6,
        policy_name="monthly_distributed_holdout",
    )
    opt = TemporalConfig(
        ModelingSilo.OPT_BAYES,
        input_lag=1,
        target_horizon=1,
        policy_name="seasonal_distributed",
    )

    return [
        (
            "ml_ind_dining",
            SiloProductContract(
                ModelingSilo.ML_SCIML,
                PhaseDMode.INDEPENDENT,
                ml,
                _heat(),
                (DINING, KITCHEN),
            ),
            "Dining",
        ),
        (
            "ml_dep1",
            SiloProductContract(
                ModelingSilo.ML_SCIML,
                PhaseDMode.DEPENDENT1,
                ml,
                _heat(),
                (DINING, KITCHEN),
            ),
            None,
        ),
        (
            "ml_dep2",
            SiloProductContract(
                ModelingSilo.ML_SCIML,
                PhaseDMode.DEPENDENT2,
                ml,
                _heat(),
                (DINING, KITCHEN),
                ALL_ZONE,
            ),
            None,
        ),
        (
            "opt_ind_dining",
            SiloProductContract(
                ModelingSilo.OPT_BAYES,
                PhaseDMode.INDEPENDENT,
                opt,
                _heat(),
                (DINING, KITCHEN),
            ),
            "Dining",
        ),
        (
            "opt_dep1",
            SiloProductContract(
                ModelingSilo.OPT_BAYES,
                PhaseDMode.DEPENDENT1,
                opt,
                _heat(),
                (DINING, KITCHEN),
            ),
            None,
        ),
        (
            "opt_dep2",
            SiloProductContract(
                ModelingSilo.OPT_BAYES,
                PhaseDMode.DEPENDENT2,
                opt,
                _heat(),
                (DINING, KITCHEN),
                ALL_ZONE,
            ),
            None,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    for name, contract, zone_id in _contracts():
        payload = contract.to_manifest_contract(
            independent_zone_id=zone_id
        )
        path = args.output_root / f"{name}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            f"{name}: mode={contract.mode.value} "
            f"silo={contract.silo.value} "
            f"base_columns={len(payload['base_columns'])} "
            f"final_columns={len(payload['final_columns'])} "
            f"path={payload['storage_contract']['data_path']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
