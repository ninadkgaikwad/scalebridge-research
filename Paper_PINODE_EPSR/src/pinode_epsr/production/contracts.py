from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, Mapping

HPOObjective = Literal[
    "recursive_temperature_normalized",
    "recursive_temperature_rmse_C",
    "recursive_temperature_mae_C",
    "recursive_temperature_cvrmse",
]

_CONTROLLED_ZONES = ("RestaurantFastFood_All", "Dining", "Kitchen")
_ACTUATION_FIELDS = ("T_supply_C", "mdot_nominal_kg_s", "mdot_max_kg_s")


@dataclass(frozen=True)
class HPOConfig:
    """Fast, TRAIN-only HPO policy.

    ``train_percentage`` is applied independently to each calendar month's
    authoritative Phase-D TRAIN block. Selected material remains entirely
    inside TRAIN. ``holdout_percentage`` partitions only the selected HPO
    target rows; Phase-D VALIDATION and TEST never enter HPO.

    ``max_rollout_steps`` and ``max_encoder_history_steps`` are part of the
    scientific HPO protocol. Production defaults preserve the full search
    geometry (N_r<=12, L_e<=12). Tiny plumbing qualifications may explicitly
    reduce only the rollout geometry (for example N_r<=3) when the requested
    HPO percentage cannot support a 20% holdout with N_r=12.
    """

    train_percentage: float = 2.0
    holdout_percentage: float = 20.0
    objective: HPOObjective = "recursive_temperature_normalized"
    n_trials: int = 12
    max_epochs_per_trial: int = 25
    patience: int = 5
    seed: int = 42
    timeout_seconds: float | None = None
    max_batch_windows: int = 32
    sampling_blocks_per_month: int = 4
    max_rollout_steps: int = 12
    max_encoder_history_steps: int = 12

    def __post_init__(self) -> None:
        if not 0.0 < self.train_percentage <= 100.0:
            raise ValueError("train_percentage must be in (0, 100]")
        if not 0.0 < self.holdout_percentage < 100.0:
            raise ValueError("holdout_percentage must be in (0, 100)")
        if self.n_trials < 1 or self.max_epochs_per_trial < 1 or self.patience < 1:
            raise ValueError("n_trials/max_epochs_per_trial/patience must be positive")
        if self.max_batch_windows < 1 or self.sampling_blocks_per_month < 2:
            raise ValueError("max_batch_windows >=1 and sampling_blocks_per_month >=2 are required")
        if self.max_rollout_steps < 1 or self.max_encoder_history_steps < 1:
            raise ValueError("max_rollout_steps/max_encoder_history_steps must be >=1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerOverrideConfig:
    """User-overridable Sim3 controller calibration/actuation settings.

    Deadband values are **half-widths** around the active setpoint. Therefore
    ``deadband_half_width_C=1.0`` means the thermostat uses setpoint-1 C and
    setpoint+1 C as its ordinary on/off thresholds (2 C total band).

    Resolution precedence is:
      user override > observed TRAIN-derived value > documented fallback.
    Null/omitted values never replace data-derived defaults.
    """

    deadband_half_width_C: float | None = None
    heating_mode_deadband_half_width_C: float | None = None
    zones: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_path: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("deadband_half_width_C", self.deadband_half_width_C),
            ("heating_mode_deadband_half_width_C", self.heating_mode_deadband_half_width_C),
        ):
            if value is not None and (not float(value) >= 0.0):
                raise ValueError(f"{name} must be nonnegative or null")
        unknown = sorted(set(self.zones).difference(_CONTROLLED_ZONES))
        if unknown:
            raise ValueError(f"Unknown controlled controller zones: {unknown}")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        source_path: str | None = None,
    ) -> "ControllerOverrideConfig":
        raw = dict(mapping or {})
        zones_raw = dict(raw.get("zones") or {})
        zones: dict[str, dict[str, Any]] = {}
        for zone, zone_payload in zones_raw.items():
            if zone not in _CONTROLLED_ZONES:
                raise ValueError(f"Unknown controlled controller zone {zone!r}")
            zp = dict(zone_payload or {})
            normalized: dict[str, Any] = {}
            for key in ("deadband_half_width_C", "heating_mode_deadband_half_width_C"):
                value = zp.get(key)
                if value is not None:
                    value = float(value)
                    if value < 0:
                        raise ValueError(f"{zone}.{key} must be nonnegative")
                    normalized[key] = value
            for mode in ("cooling", "heating"):
                mode_raw = dict(zp.get(mode) or {})
                mode_out: dict[str, float] = {}
                for field_name in _ACTUATION_FIELDS:
                    value = mode_raw.get(field_name)
                    if value is None:
                        continue
                    value = float(value)
                    if field_name.startswith("mdot_") and value < 0:
                        raise ValueError(f"{zone}.{mode}.{field_name} must be nonnegative")
                    mode_out[field_name] = value
                if (
                    "mdot_nominal_kg_s" in mode_out
                    and "mdot_max_kg_s" in mode_out
                    and mode_out["mdot_max_kg_s"] + 1e-12 < mode_out["mdot_nominal_kg_s"]
                ):
                    raise ValueError(f"{zone}.{mode}: mdot_max_kg_s must be >= mdot_nominal_kg_s")
                if mode_out:
                    normalized[mode] = mode_out
            zones[zone] = normalized

        global_deadband = raw.get("deadband_half_width_C")
        global_mode_deadband = raw.get("heating_mode_deadband_half_width_C")
        return cls(
            deadband_half_width_C=None if global_deadband is None else float(global_deadband),
            heating_mode_deadband_half_width_C=(
                None if global_mode_deadband is None else float(global_mode_deadband)
            ),
            zones=zones,
            source_path=source_path,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "deadband_half_width_C": self.deadband_half_width_C,
            "heating_mode_deadband_half_width_C": self.heating_mode_deadband_half_width_C,
            "zones": self.zones,
            "source_path": self.source_path,
            "deadband_semantics": "half_width_about_setpoint",
            "parameter_priority": "user_override > observed_train_data > documented_fallback",
        }

    def deadband_overrides_C(self, zone_ids=_CONTROLLED_ZONES) -> dict[str, float]:
        out: dict[str, float] = {}
        for zone in zone_ids:
            local = self.zones.get(zone, {}).get("deadband_half_width_C")
            value = self.deadband_half_width_C if local is None else local
            if value is not None:
                out[zone] = float(value)
        return out

    def heating_mode_deadband_overrides_C(self, zone_ids=_CONTROLLED_ZONES) -> dict[str, float]:
        out: dict[str, float] = {}
        for zone in zone_ids:
            local = self.zones.get(zone, {}).get("heating_mode_deadband_half_width_C")
            value = self.heating_mode_deadband_half_width_C if local is None else local
            if value is not None:
                out[zone] = float(value)
        return out

    def actuation_overrides(self, zone_id: str) -> dict[str, dict[str, float]]:
        if zone_id not in _CONTROLLED_ZONES:
            raise KeyError(zone_id)
        zp = self.zones.get(zone_id, {})
        out: dict[str, dict[str, float]] = {}
        for mode in ("cooling", "heating"):
            raw = dict(zp.get(mode) or {})
            if raw:
                out[mode] = {k: float(v) for k, v in raw.items() if v is not None}
        return out


def default_production_yaml_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "production.yaml"


def load_controller_override_config(path: str | Path | None = None) -> ControllerOverrideConfig:
    """Load only the ``controller`` section of production YAML/JSON.

    The rest of Day-1 CLI defaults remain unchanged by this narrow R2 loader.
    Supplying no path reads ``Paper_PINODE_EPSR/configs/production.yaml``.
    """
    source = default_production_yaml_path() if path is None else Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Controller configuration file not found: {source}")
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read controller YAML overrides") from exc
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    controller = dict(payload.get("controller") or {})
    return ControllerOverrideConfig.from_mapping(controller, source_path=str(source))


@dataclass(frozen=True)
class ProductionTrainingConfig:
    max_epochs: int = 500
    patience: int = 50
    seed: int = 0
    max_batch_windows: int = 64
    validation_max_windows: int = 256
    gradient_clip_norm: float | None = 10.0
    continue_on_error: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProductionEvaluationConfig:
    sim1_max_points: int | None = None
    sim2_horizon_per_episode: int | None = None
    sim3_horizon_per_episode: int | None = None
    all_test_episodes: bool = True
    cooling_mdot_choice: str = "nominal"
    heating_mdot_choice: str = "nominal"
    unobserved_mode_policy: str = "fallback"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
