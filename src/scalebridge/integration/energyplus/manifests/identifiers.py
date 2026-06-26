"""Deterministic identifiers for EnergyPlus simulation cases.

A case identifier represents scientific simulation intent rather than a
particular execution. Machine names, absolute paths, campaigns, storage
preferences, and tracking identifiers are therefore excluded. This allows the
same case to retain its identity on Windows workstations and Linux HPC nodes.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scalebridge.integration.energyplus.manifests.models import CaseSpec


def canonical_case_payload(case_spec: CaseSpec) -> dict[str, Any]:
    """Return the canonical payload used to identify a simulation case.

    Parameters
    ----------
    case_spec:
        Validated scientific specification for one EnergyPlus case.

    Returns
    -------
    dict[str, Any]
        JSON-compatible content containing only identity-defining inputs.

    Notes
    -----
    Output request order is normalized because requesting the same variables
    in a different order does not change the intended simulation. Schedule
    operation order is retained because later operations may depend on earlier
    mutations.
    """
    requests = sorted(
        (
            {
                "variable_name": request.variable_name,
                "key_value": request.key_value,
                "reporting_frequency": request.reporting_frequency,
            }
            for request in case_spec.output_variables
        ),
        key=lambda item: (
            item["variable_name"].casefold(),
            item["key_value"].casefold(),
            item["reporting_frequency"],
        ),
    )

    return {
        "schema_version": case_spec.schema_version,
        "idf_sha256": case_spec.idf_sha256,
        "epw_sha256": case_spec.epw_sha256,
        "run_period": case_spec.run_period.model_dump(mode="json", exclude_none=True),
        "timestep_minutes": case_spec.timestep_minutes,
        "output_variables": requests,
        # Schedule operation order is retained because later operations may depend on earlier ones.
        "schedule_operations": [
            operation.model_dump(mode="json", exclude_none=True)
            for operation in case_spec.schedule_operations
        ],
        "request_variable_dictionary": case_spec.request_variable_dictionary,
        "energyplus_version": case_spec.energyplus_version,
    }


def build_case_id(case_spec: CaseSpec) -> str:
    """Build a stable case identifier from canonical scientific inputs.

    The identifier contains a readable ``epcase_`` prefix followed by the
    first 24 hexadecimal characters of a SHA-256 digest. The complete
    canonical payload remains available in the case manifest for provenance.
    """
    encoded = json.dumps(
        canonical_case_payload(case_spec),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"epcase_{hashlib.sha256(encoded).hexdigest()[:24]}"
