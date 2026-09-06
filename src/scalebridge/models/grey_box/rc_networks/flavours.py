from __future__ import annotations

"""Ratified built-in elemental RC flavours for the generic compiler."""

from dataclasses import dataclass
from typing import Mapping

from .specification import HeatPortGroup, RCCompileError


DEFAULT_CORE_PORT_GROUPS: dict[str, HeatPortGroup] = {
    "qac": HeatPortGroup.CONVECTIVE,
    "zic": HeatPortGroup.CONVECTIVE,
    "qsol1": HeatPortGroup.CONVECTIVE,
    "zir": HeatPortGroup.RADIATIVE,
    "qsol2": HeatPortGroup.RADIATIVE,
}


@dataclass(frozen=True)
class ElementalEdgeTemplate:
    node_a: str
    node_b: str
    resistance_family: str


@dataclass(frozen=True)
class BoundaryEdgeTemplate:
    state_label: str
    boundary_label: str
    resistance_family: str


@dataclass(frozen=True)
class RCFlavor:
    name: str
    state_labels: tuple[str, ...]
    observed_state_labels: tuple[str, ...]
    capacitance_families: Mapping[str, str]
    state_edges: tuple[ElementalEdgeTemplate, ...]
    boundary_edges: tuple[BoundaryEdgeTemplate, ...]
    routing_kind: str

    @property
    def latent_state_labels(self) -> tuple[str, ...]:
        observed = set(self.observed_state_labels)
        return tuple(label for label in self.state_labels if label not in observed)


FLAVOURS: dict[str, RCFlavor] = {
    "1r1c": RCFlavor(
        name="1r1c",
        state_labels=("a",),
        observed_state_labels=("a",),
        capacitance_families={"a": "C_a"},
        state_edges=(),
        boundary_edges=(BoundaryEdgeTemplate("a", "outdoor_temperature", "R_ao"),),
        routing_kind="all_to_air",
    ),
    "2r2c": RCFlavor(
        name="2r2c",
        state_labels=("a", "m"),
        observed_state_labels=("a",),
        capacitance_families={"a": "C_a", "m": "C_m"},
        state_edges=(ElementalEdgeTemplate("a", "m", "R_am"),),
        boundary_edges=(BoundaryEdgeTemplate("a", "outdoor_temperature", "R_ao"),),
        routing_kind="eta_r",
    ),
    "3r2c": RCFlavor(
        name="3r2c",
        state_labels=("a", "m"),
        observed_state_labels=("a",),
        capacitance_families={"a": "C_a", "m": "C_m"},
        state_edges=(ElementalEdgeTemplate("a", "m", "R_am"),),
        boundary_edges=(
            BoundaryEdgeTemplate("a", "outdoor_temperature", "R_ao"),
            BoundaryEdgeTemplate("m", "outdoor_temperature", "R_om"),
        ),
        routing_kind="eta_r",
    ),
    "4r3c": RCFlavor(
        name="4r3c",
        state_labels=("a", "e", "m"),
        observed_state_labels=("a",),
        capacitance_families={"a": "C_a", "e": "C_e", "m": "C_m"},
        state_edges=(
            ElementalEdgeTemplate("a", "e", "R_ae"),
            ElementalEdgeTemplate("a", "m", "R_am"),
        ),
        boundary_edges=(
            BoundaryEdgeTemplate("a", "outdoor_temperature", "R_ao"),
            BoundaryEdgeTemplate("e", "outdoor_temperature", "R_eo"),
        ),
        routing_kind="gamma_r_3way",
    ),
}


def normalize_flavour_name(name: str) -> str:
    token = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "1r1c": "1r1c",
        "2r2c": "2r2c",
        "3r2c": "3r2c",
        "4r3c": "4r3c",
    }
    try:
        return aliases[token]
    except KeyError as exc:
        raise RCCompileError(
            f"Unknown RC flavour {name!r}; expected one of {sorted(FLAVOURS)}"
        ) from exc


def get_flavour(name: str) -> RCFlavor:
    return FLAVOURS[normalize_flavour_name(name)]
