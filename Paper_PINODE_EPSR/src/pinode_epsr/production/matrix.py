from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

METHODS = ("inverse_pinn", "neural_ode", "base_pinode", "ebp_pinode")
RC_ORDERS = (1, 2)
CASES = ("all_to_one", "identity_ind", "identity_dep1", "identity_dep2")


@dataclass(frozen=True, order=True)
class ExperimentSpec:
    priority: str
    case_name: str
    rc_order: int
    method: str

    @property
    def configuration_id(self) -> str:
        return f"{self.case_name}__{self.rc_order}C__{self.method}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"configuration_id": self.configuration_id}


def _priority(case_name: str) -> str:
    if case_name == "all_to_one":
        return "A"
    if case_name == "identity_ind":
        return "B"
    return "C"


def production_matrix(*, priorities: Iterable[str] = ("A", "B", "C")) -> tuple[ExperimentSpec, ...]:
    allowed = set(priorities)
    out = []
    for case_name in CASES:
        p = _priority(case_name)
        if p not in allowed:
            continue
        for rc_order in RC_ORDERS:
            for method in METHODS:
                out.append(ExperimentSpec(p, case_name, rc_order, method))
    return tuple(out)
