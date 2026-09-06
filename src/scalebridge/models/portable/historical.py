from __future__ import annotations

"""Historical Phase-D replay source for downstream Sim1/Sim2/Sim3 evaluators.

E0-7 resolves and validates authoritative Phase-D data; it intentionally does
not implement Sim1/Sim2/Sim3 reset/control policy itself.
"""

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from scalebridge.data.thermal_modeling.phase_e_adapter import (
    load_phase_e_data_contract,
    validate_materialized_columns,
    validate_partition_values,
)
from scalebridge.data.thermal_modeling.phase_e_contracts import PhaseEDataContract

from .contracts import DataLocator, PortableModelError
from .lineage import DataRootRegistry


@dataclass(frozen=True)
class HistoricalReplayDataset:
    contract: PhaseEDataContract
    frame: pd.DataFrame
    manifest_locator: DataLocator
    data_locator: DataLocator

    @classmethod
    def load(
        cls,
        *,
        registry: DataRootRegistry,
        manifest_locator: DataLocator,
        data_locator: DataLocator,
        verify_hashes: bool = False,
    ) -> "HistoricalReplayDataset":
        manifest_path = registry.resolve(
            manifest_locator,
            must_exist=True,
            verify_sha256=verify_hashes,
        )
        data_path = registry.resolve(
            data_locator,
            must_exist=True,
            verify_sha256=verify_hashes,
        )
        if not data_path.is_file():
            raise PortableModelError(f"Phase-D replay data is not a file: {data_path}")
        contract = load_phase_e_data_contract(manifest_path)
        frame = pd.read_parquet(data_path)
        validate_materialized_columns(contract, frame.columns)
        if "partition" in frame.columns:
            validate_partition_values(contract, frame["partition"].dropna().astype(str).unique())
        return cls(
            contract=contract,
            frame=frame,
            manifest_locator=manifest_locator,
            data_locator=data_locator,
        )

    def select(
        self,
        *,
        partitions: Iterable[str] | None = None,
        included_only: bool = True,
    ) -> pd.DataFrame:
        out = self.frame
        if included_only and "included" in out.columns:
            out = out.loc[out["included"].astype(bool)]
        if partitions is not None:
            requested = tuple(str(v) for v in partitions)
            validate_partition_values(self.contract, requested)
            if "partition" not in out.columns:
                raise PortableModelError("Phase-D replay table has no partition column")
            out = out.loc[out["partition"].astype(str).isin(requested)]
        return out.copy()
