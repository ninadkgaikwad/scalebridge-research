from __future__ import annotations

"""Executable E0-7 fitted normalization boundary."""

from typing import Any, Mapping

from .contracts import NormalizationContract, PortableModelError


def normalize_named_inputs(
    values: Mapping[str, Any],
    contract: NormalizationContract,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        transform = contract.inputs.get(key)
        out[key] = value if transform is None else transform.normalize(value)
    if strict:
        missing = set(contract.inputs) - set(values)
        if missing:
            raise PortableModelError(
                f"Missing values required by fitted input normalization: {sorted(missing)}"
            )
    return out


def denormalize_named_outputs(
    values: Mapping[str, Any],
    contract: NormalizationContract,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        transform = contract.outputs.get(key)
        out[key] = value if transform is None else transform.denormalize(value)
    if strict:
        missing = set(contract.outputs) - set(values)
        if missing:
            raise PortableModelError(
                f"Missing values required by fitted output renormalization: {sorted(missing)}"
            )
    return out
