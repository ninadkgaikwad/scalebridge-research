# -*- coding: utf-8 -*-
"""Deterministic solar predictors for heat-input regression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalebridge.data.heat_input_regression.signal_catalog import resolve_present_column

GHI_OUTPUT_COLUMN = "derived_GHI_W_per_m2"


def build_ghi(frame: pd.DataFrame) -> tuple[pd.Series, dict[str, object]]:
    """Build global horizontal irradiance from Stage B solar signals.

    Formula
    -------
    GHI = DNI * abs(sin(theta)) + DHI

    EnergyPlus solar altitude is supplied in degrees. The absolute sine keeps
    the legacy regression convention while avoiding negative direct components
    below the horizon.
    """
    columns = {str(column) for column in frame.columns}
    direct_column = resolve_present_column("site_direct_solar", columns)
    diffuse_column = resolve_present_column("site_diffuse_solar", columns)
    altitude_column = resolve_present_column("solar_altitude_angle", columns)

    missing = [
        name
        for name, column in (
            ("site_direct_solar", direct_column),
            ("site_diffuse_solar", diffuse_column),
            ("solar_altitude_angle", altitude_column),
        )
        if column is None
    ]
    if missing:
        raise KeyError(f"Cannot build GHI; missing semantic signals: {missing}")

    direct = pd.to_numeric(frame[direct_column], errors="coerce")
    diffuse = pd.to_numeric(frame[diffuse_column], errors="coerce")
    altitude_radians = np.deg2rad(
        pd.to_numeric(frame[altitude_column], errors="coerce")
    )
    ghi = direct * np.abs(np.sin(altitude_radians)) + diffuse
    ghi = pd.Series(ghi, index=frame.index, name=GHI_OUTPUT_COLUMN, dtype="float64")

    metadata = {
        "feature_name": GHI_OUTPUT_COLUMN,
        "feature_family": "solar",
        "formula": "DNI * abs(sin(radians(solar_altitude_angle))) + DHI",
        "units": "W/m2",
        "source_columns": [direct_column, diffuse_column, altitude_column],
    }
    return ghi, metadata
