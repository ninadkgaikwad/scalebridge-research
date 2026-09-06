from .assembly import (
    AssemblyConfig,
    AssemblyDiagnostics,
    AssemblyResult,
    PhaseDAssemblyError,
    assemble_canonical_zone_table,
    required_phase_c_prediction_columns,
)
# -*- coding: utf-8 -*-
"""Phase D thermal-model data contracts and source discovery."""

from .constants import *
from .discovery import (
    PhaseDDiscoveryError,
    discover_phase_d_sources,
    resolve_aggregation_run,
    resolve_aggregation_zone,
    resolve_phase_c_child_runs,
    resolve_phase_c_zone,
)
from .identities import PhaseDDatasetIdentity, PhaseDSourceLineage
from .manifests import PhaseDDatasetManifest, PhaseDZoneManifest
from .models import ZoneSignalRecord
from .signals import (
    SignalClassification,
    SignalDefinition,
    build_signal_registry,
    classify_phase_c_signal,
)
from .source_refs import (
    AggregationRunRef,
    AggregationZoneRef,
    PhaseCChildRunRefs,
    PhaseCZoneRef,
    PhaseDDiscoveryResult,
)

__all__ = [
    "AssemblyConfig",
    "AssemblyDiagnostics",
    "AssemblyResult",
    "PhaseDAssemblyError",
    "assemble_canonical_zone_table",
    "required_phase_c_prediction_columns",
    "PhaseDDiscoveryError",
    "discover_phase_d_sources",
    "resolve_aggregation_run",
    "resolve_aggregation_zone",
    "resolve_phase_c_child_runs",
    "resolve_phase_c_zone",
    "AggregationRunRef",
    "AggregationZoneRef",
    "PhaseCChildRunRefs",
    "PhaseCZoneRef",
    "PhaseDDiscoveryResult",
    "PhaseDDatasetIdentity",
    "PhaseDSourceLineage",
    "PhaseDDatasetManifest",
    "PhaseDZoneManifest",
    "ZoneSignalRecord",
    "SignalClassification",
    "SignalDefinition",
    "build_signal_registry",
    "classify_phase_c_signal",
]



from .alignment import (
    AlignmentDiagnostics,
    PhaseDAlignmentError,
    TimestampNormalizationConfig,
    align_sources,
    clean_phase_b,
    load_and_align_paths,
    parse_energyplus_timestamp,
    rewrite_placeholder_year,
)
