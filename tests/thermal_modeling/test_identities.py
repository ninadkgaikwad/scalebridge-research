import pytest

from scalebridge.data.thermal_modeling.constants import ModelingSilo, PhaseDMode
from scalebridge.data.thermal_modeling.identities import PhaseDDatasetIdentity


def test_dataset_identity_serializes_enum_values():
    identity = PhaseDDatasetIdentity(
        phase_d_run_id="phase_d_test",
        dataset_id="dataset_1",
        mode=PhaseDMode.INDEPENDENT,
        silo=ModelingSilo.ML_SCIML,
        building_type="RestaurantFastFood",
        climate_zone="5A",
        weather_location="Buffalo",
        aggregate_zone_ids=("Dining",),
    )
    payload = identity.to_dict()
    assert payload["mode"] == "independent"
    assert payload["silo"] == "ml_sciml"
    assert payload["aggregate_zone_ids"] == ["Dining"]


def test_dataset_identity_rejects_duplicate_zones():
    with pytest.raises(ValueError, match="unique"):
        PhaseDDatasetIdentity(
            phase_d_run_id="run",
            dataset_id="dataset",
            mode=PhaseDMode.DEPENDENT1,
            silo=ModelingSilo.ML_SCIML,
            building_type="RestaurantFastFood",
            climate_zone="5A",
            weather_location="Buffalo",
            aggregate_zone_ids=("Dining", "Dining"),
        )
