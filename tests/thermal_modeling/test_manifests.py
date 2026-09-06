import pytest

from scalebridge.data.thermal_modeling.constants import (
    ModelingSilo,
    NullableReason,
    PhaseDMode,
    PhaseDSignalStatus,
)
from scalebridge.data.thermal_modeling.identities import (
    PhaseDDatasetIdentity,
    PhaseDSourceLineage,
)
from scalebridge.data.thermal_modeling.manifests import (
    PhaseDDatasetManifest,
    PhaseDZoneManifest,
)
from scalebridge.data.thermal_modeling.models import ZoneSignalRecord


def _identity():
    return PhaseDDatasetIdentity(
        phase_d_run_id="phase_d_test",
        dataset_id="dataset_1",
        mode=PhaseDMode.INDEPENDENT,
        silo=ModelingSilo.ML_SCIML,
        building_type="RestaurantFastFood",
        climate_zone="5A",
        weather_location="Buffalo",
        aggregate_zone_ids=("Kitchen",),
    )


def _lineage():
    return PhaseDSourceLineage(
        campaign_id="campaign",
        case_id="case",
        aggregation_matrix_run_id="matrix",
        aggregation_run_id="aggregation",
        aggregation_id="aggregation_id",
        phase_c_campaign_run_id="phase_c",
        phase_c_inference_run_id="inference",
        phase_c_split_run_id="splits",
    )


def test_nullable_record_requires_reason():
    with pytest.raises(ValueError, match="nullable_reason"):
        ZoneSignalRecord(
            signal_name="qsol1",
            source_phase="phase_c",
            source_name="predicted_QSol1",
            units="W",
            phase_d_status=PhaseDSignalStatus.NULLABLE_COMPLETE_ZERO,
            nullable=True,
            nullable_reason=NullableReason.NONE,
        )


def test_dataset_manifest_is_json_serializable_contract():
    signal = ZoneSignalRecord(
        signal_name="qsol1",
        source_phase="phase_c",
        source_name="predicted_QSol1",
        units="W",
        phase_d_status=PhaseDSignalStatus.NULLABLE_COMPLETE_ZERO,
        nullable=True,
        nullable_reason=NullableReason.COMPLETE_ZERO_SIGNAL,
        constant_value=0.0,
    )
    zone = PhaseDZoneManifest(
        aggregate_zone_id="Kitchen",
        row_count=105120,
        signal_records=(signal,),
    )
    manifest = PhaseDDatasetManifest(
        identity=_identity(),
        lineage=_lineage(),
        zone_manifests=(zone,),
        canonical_columns=("timestamp", "zone_temperature", "qsol1"),
        split_strategy="monthly_distributed_holdout",
    )
    payload = manifest.to_dict()
    assert payload["schema_version"] == "phase_d_d1_v1"
    assert payload["zone_manifests"][0]["signal_records"][0]["nullable_reason"] == "complete_zero_signal"
