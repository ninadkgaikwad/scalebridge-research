from __future__ import annotations

import pytest
from pydantic import ValidationError

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig


def test_defaults_match_validated_phase_c_runner_policy() -> None:
    config = PhaseCCampaignConfig()
    assert config.split_strategy == "monthly_distributed_holdout"
    assert (config.train_fraction, config.validation_fraction, config.test_fraction) == (0.70, 0.15, 0.15)
    assert config.estimator_types == ("pytorch_linear",)
    assert config.pytorch_devices == ("auto",)
    assert config.validation_profile == "full"
    assert config.mlflow_enabled is True
    assert config.write_full_predictions is True
    assert config.fit_intercept_override is None


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="must equal 1"):
        PhaseCCampaignConfig(train_fraction=0.8, validation_fraction=0.15, test_fraction=0.15)


def test_start_stage_must_precede_stop_stage() -> None:
    with pytest.raises(ValidationError, match="must not come after"):
        PhaseCCampaignConfig(start_stage="C8", stop_stage="C4")


def test_repeatable_selections_reject_duplicates() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        PhaseCCampaignConfig(estimator_types=("pytorch_linear", "pytorch_linear"))


def test_capability_manifest_covers_every_public_field() -> None:
    manifest = PhaseCCampaignConfig.capability_manifest()
    names = {item["name"] for item in manifest["fields"]}
    assert names == set(PhaseCCampaignConfig.model_fields) - {"schema_version"}
    by_name = {item["name"]: item for item in manifest["fields"]}
    assert by_name["fit_intercept_override"]["ui_visibility"] == "expert"
    assert by_name["split_strategy"]["phase_c_stages"] == ["C3"]


def test_downstream_zone_filter_cannot_broaden_upstream_zone() -> None:
    with pytest.raises(ValueError, match="cannot broaden"):
        PhaseCCampaignConfig(
            aggregate_zone_id="Zone_A",
            downstream_aggregate_zone_ids=("Zone_A", "Zone_B"),
        )
