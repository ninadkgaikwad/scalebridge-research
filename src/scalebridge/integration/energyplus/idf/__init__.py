"""EnergyPlus IDF preparation services.

The package converts an immutable ``CaseSpec`` into a prepared IDF without
modifying the source model. It exposes a backend-independent preparation
service and an optional opyplus adapter for production use.
"""

from scalebridge.integration.energyplus.idf.backend import (
    IdfBackend,
    IdfBackendError,
    OpyplusIdfBackend,
    OpyplusNotInstalledError,
)
from scalebridge.integration.energyplus.idf.preparer import (
    IdfPreparationError,
    IdfPreparer,
    PreparedIdfResult,
    prepare_idf,
)
from scalebridge.integration.energyplus.idf.opyplus_compat import (
    install_opyplus_207_idd_compatibility,
)

__all__ = [
    "IdfBackend",
    "IdfBackendError",
    "IdfPreparationError",
    "IdfPreparer",
    "OpyplusIdfBackend",
    "OpyplusNotInstalledError",
    "PreparedIdfResult",
    "prepare_idf",
    "install_opyplus_207_idd_compatibility",
]
