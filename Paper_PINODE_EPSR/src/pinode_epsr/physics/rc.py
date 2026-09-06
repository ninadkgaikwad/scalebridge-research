from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

RadiativeParticipationMode = Literal["full", "zero", "fixed", "learnable"]
RadiativeAllocationMode = Literal["mass_only", "air_only", "fixed", "learnable"]


@dataclass(frozen=True)
class HeatInputs:
    """All Phase D thermal heat-input information for one zone.

    Locked Phase-C/D physical interpretation for this paper:
      * convective: QZIC, QSol1, QAC
      * radiative: QZIR, QSol2

    Every available channel is used by every RC structure. Missing keys map to
    zero only when the upstream Phase D contract says the signal is structurally
    unavailable for that zone.
    """

    qac: float
    zic: float
    zir: float = 0.0
    qsol1: float = 0.0
    qsol2: float = 0.0

    @property
    def convective(self) -> float:
        return self.qac + self.zic + self.qsol1

    @property
    def radiative(self) -> float:
        return self.zir + self.qsol2

    @property
    def total(self) -> float:
        return self.convective + self.radiative

    def as_dict(self) -> dict[str, float]:
        return {
            "qac": self.qac,
            "zic": self.zic,
            "zir": self.zir,
            "qsol1": self.qsol1,
            "qsol2": self.qsol2,
        }


@dataclass(frozen=True)
class RC1ZoneParams:
    """1R1C parameters with an effective radiative-participation parameter.

    Convective heat (QZIC + QSol1 + QAC) always enters the single lumped
    balance with unit gain. Because 1R1C has no separate surface/mass state,
    ``eta_rad`` represents the effective fraction of radiative heat
    (QZIR + QSol2) that participates in the lumped temperature balance.

    ``eta_rad_mode`` options:
      * ``full``: eta_rad = 1 (default/main model)
      * ``zero``: eta_rad = 0 (diagnostic ablation)
      * ``fixed``: use supplied bounded eta_rad
      * ``learnable``: use supplied eta_rad as bounded trainable initial value;
        optimization is implemented in a later training patch.
    """

    c_air: float
    r_out: float
    eta_rad: float = 1.0
    eta_rad_mode: RadiativeParticipationMode = "full"

    def resolved_eta_rad(self) -> float:
        if self.eta_rad_mode == "full":
            eta = 1.0
        elif self.eta_rad_mode == "zero":
            eta = 0.0
        elif self.eta_rad_mode in {"fixed", "learnable"}:
            eta = float(self.eta_rad)
        else:
            raise ValueError(f"Unknown eta_rad_mode: {self.eta_rad_mode!r}")

        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta_rad must lie in [0, 1], got {eta}")
        return eta


@dataclass(frozen=True)
class RC2ZoneParams:
    """2R2C parameters with a single physical radiative-allocation parameter.

    Convective heat (QZIC + QSol1 + QAC) always enters the zone-air node with
    unit gain. Radiative heat (QZIR + QSol2) is conserved and split using
    ``eta_rad``:

        mass receives eta_rad * Q_rad
        air  receives (1 - eta_rad) * Q_rad

    ``eta_rad_mode`` provides explicit paper options:
      * ``mass_only``: eta_rad = 1 (default strict physical prior)
      * ``air_only``: eta_rad = 0 (diagnostic/ablation)
      * ``fixed``: use the supplied bounded eta_rad
      * ``learnable``: use eta_rad as an initial bounded parameter; actual
        optimization is implemented in a later training patch.

    No separate gain is applied to QAC or any other heat channel.
    """

    c_air: float
    c_mass: float
    r_out: float
    r_mass: float
    eta_rad: float = 1.0
    eta_rad_mode: RadiativeAllocationMode = "mass_only"

    def resolved_eta_rad(self) -> float:
        if self.eta_rad_mode == "mass_only":
            eta = 1.0
        elif self.eta_rad_mode == "air_only":
            eta = 0.0
        elif self.eta_rad_mode in {"fixed", "learnable"}:
            eta = float(self.eta_rad)
        else:  # defensive for runtime strings outside the Literal contract
            raise ValueError(f"Unknown eta_rad_mode: {self.eta_rad_mode!r}")

        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta_rad must lie in [0, 1], got {eta}")
        return eta


def allocate_heat_2c(heat: HeatInputs, p: RC2ZoneParams) -> tuple[float, float]:
    """Allocate all available heat to 2R2C air/mass nodes conservatively.

    QZIC + QSol1 + QAC are convective and therefore always enter air.
    QZIR + QSol2 are radiative and are split by eta_rad without loss.
    """
    eta = p.resolved_eta_rad()
    air_q = heat.convective + (1.0 - eta) * heat.radiative
    mass_q = eta * heat.radiative
    return float(air_q), float(mass_q)


def effective_heat_1c(heat: HeatInputs, p: RC1ZoneParams) -> float:
    """Effective 1R1C heat using all convective/radiative information.

    Convective channels retain unit gain. Radiative channels enter through the
    bounded effective participation factor ``eta_rad`` because the single-state
    model has no explicit surface/mass state.
    """
    eta = p.resolved_eta_rad()
    return float(heat.convective + eta * heat.radiative)


def rhs_1r1c_single(
    tz: float,
    tout: float,
    heat: HeatInputs,
    p: RC1ZoneParams,
) -> float:
    """1R1C balance with unit convective gain and optional eta_rad."""
    q = (tout - tz) / p.r_out + effective_heat_1c(heat, p)
    return q / p.c_air


def rhs_2r2c_single(
    state: np.ndarray,
    tout: float,
    heat: HeatInputs,
    p: RC2ZoneParams,
) -> np.ndarray:
    """2R2C air/mass model with locked convective/radiative semantics."""
    tz, tm = np.asarray(state, dtype=float)
    heat_air, heat_mass = allocate_heat_2c(heat, p)
    air_q = (tout - tz) / p.r_out + (tm - tz) / p.r_mass + heat_air
    mass_q = (tz - tm) / p.r_mass + heat_mass
    return np.array([air_q / p.c_air, mass_q / p.c_mass], dtype=float)


def rhs_1r1c_coupled(
    tz: np.ndarray,
    tout: float,
    heats: tuple[HeatInputs, HeatInputs],
    params: tuple[RC1ZoneParams, RC1ZoneParams],
    r_dining_kitchen: float,
) -> np.ndarray:
    """Coupled Dining/Kitchen 1R1C balance using all heat channels per zone."""
    t_d, t_k = np.asarray(tz, dtype=float)
    h_d, h_k = heats
    p_d, p_k = params
    coupling_d = (t_k - t_d) / r_dining_kitchen
    coupling_k = (t_d - t_k) / r_dining_kitchen
    q_d = (tout - t_d) / p_d.r_out + coupling_d + effective_heat_1c(h_d, p_d)
    q_k = (tout - t_k) / p_k.r_out + coupling_k + effective_heat_1c(h_k, p_k)
    return np.array([q_d / p_d.c_air, q_k / p_k.c_air], dtype=float)


def rhs_2r2c_coupled(
    state: np.ndarray,
    tout: float,
    heats: tuple[HeatInputs, HeatInputs],
    params: tuple[RC2ZoneParams, RC2ZoneParams],
    r_dining_kitchen: float,
) -> np.ndarray:
    """Coupled Dining/Kitchen 2R2C model using all heat channels per zone."""
    t_d, tm_d, t_k, tm_k = np.asarray(state, dtype=float)
    h_d, h_k = heats
    p_d, p_k = params

    heat_air_d, heat_mass_d = allocate_heat_2c(h_d, p_d)
    heat_air_k, heat_mass_k = allocate_heat_2c(h_k, p_k)

    couple_d = (t_k - t_d) / r_dining_kitchen
    couple_k = (t_d - t_k) / r_dining_kitchen

    air_d = (
        (tout - t_d) / p_d.r_out
        + (tm_d - t_d) / p_d.r_mass
        + couple_d
        + heat_air_d
    )
    mass_d = (t_d - tm_d) / p_d.r_mass + heat_mass_d
    air_k = (
        (tout - t_k) / p_k.r_out
        + (tm_k - t_k) / p_k.r_mass
        + couple_k
        + heat_air_k
    )
    mass_k = (t_k - tm_k) / p_k.r_mass + heat_mass_k

    return np.array(
        [air_d / p_d.c_air, mass_d / p_d.c_mass, air_k / p_k.c_air, mass_k / p_k.c_mass],
        dtype=float,
    )


def heat_inputs_from_mapping(values: Mapping[str, float], zone: str) -> HeatInputs:
    """Read every available Phase D heat channel for a zone."""

    def get(signal: str) -> float:
        return float(values.get(f"{zone}__{signal}__lag_0", 0.0))

    return HeatInputs(
        qac=get("qac"),
        zic=get("zic"),
        zir=get("zir"),
        qsol1=get("qsol1"),
        qsol2=get("qsol2"),
    )
