from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Any
import csv
import json
import math


COMMAND_COLUMNS = [
    "received_mode",
    "effective_control_mode",
    "feasible",
    "fallback_applied",
    "flow_fraction_of_design",
    "received_mass_flow_kg_s",
    "received_supply_air_temperature_c",
    "transform_zone_temperature_c",
    "delta_t_star_c",
    "transformed_fan_command_kg_s",
    "transformed_sensible_load_request_w",
    "fan_override_active",
    "load_override_active",
    "fan_actuator_readback_kg_s",
    "load_actuator_readback_w",
    "fan_actuator_api_value_raw",
    "load_actuator_api_value_raw",
]

INTERNAL_COLUMNS = [
    "fan_design_max_mass_flow_kg_s",
    "unitary_design_heating_capacity_w",
    "unitary_design_cooling_capacity_w",
]

DERIVED_COLUMNS = [
    "q_zone_interface_w",
    "q_return_to_mixed_w",
    "q_mixed_to_cool_out_w",
    "q_cool_out_to_heat_out_w",
    "q_heat_out_to_supply_outlet_w",
    "q_mixed_to_supply_outlet_w",
    "delta_t_return_to_mixed_c",
    "delta_t_mixed_to_cool_out_c",
    "delta_t_cool_out_to_heat_out_c",
    "delta_t_heat_out_to_supply_outlet_c",
    "delta_t_zone_supply_minus_zone_c",
]


class NumericAccumulator:
    def __init__(self):
        self.weight_seconds = 0.0
        self.sum = {}
        self.min = {}
        self.max = {}
        self.last = {}

    def add(self, row: Mapping[str, Any], weight_seconds: float) -> None:
        if weight_seconds <= 0:
            return

        self.weight_seconds += weight_seconds

        for key, value in row.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue

            self.sum[key] = self.sum.get(key, 0.0) + v * weight_seconds
            self.min[key] = v if key not in self.min else min(self.min[key], v)
            self.max[key] = v if key not in self.max else max(self.max[key], v)
            self.last[key] = v

    def mean(self, key: str):
        if self.weight_seconds <= 0 or key not in self.sum:
            return None
        return self.sum[key] / self.weight_seconds


class BroadSimulatorHistory:
    """
    Broad, visualization-ready simulator history.

    Files:
      history/system_timestep_zone_history.csv
      history/control_step_zone_history.csv
      history/received_command_history.csv
      history/control_steps.jsonl
      history/signal_catalog.json
      history/api_exchange_registry.csv
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        zone_tokens: Iterable[str],
        environment_aliases: List[str],
        zone_signal_aliases: List[str],
    ) -> None:
        self.run_dir = run_dir
        self.history_dir = run_dir / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.zone_tokens = list(zone_tokens)
        self.environment_aliases = list(environment_aliases)
        self.zone_signal_aliases = list(zone_signal_aliases)

        self.system_path = (
            self.history_dir / "system_timestep_zone_history.csv"
        )
        self.control_path = (
            self.history_dir / "control_step_zone_history.csv"
        )
        self.command_path = (
            self.history_dir / "received_command_history.csv"
        )
        self.control_jsonl_path = (
            self.history_dir / "control_steps.jsonl"
        )

        meta = [
            "control_step_index",
            "system_substep_index",
            "year",
            "month",
            "day",
            "current_time_hour",
            "current_sim_time_hour",
            "system_timestep_seconds",
            "zone_token",
        ]

        self.system_fields = (
            meta
            + ["feasibility_reason"]
            + COMMAND_COLUMNS
            + INTERNAL_COLUMNS
            + self.environment_aliases
            + self.zone_signal_aliases
            + DERIVED_COLUMNS
        )

        self.command_fields = [
            "control_step_index",
            "year",
            "month",
            "day",
            "current_time_hour",
            "current_sim_time_hour",
            "zone_token",
            "received_mode",
            "effective_control_mode",
            "feasible",
            "fallback_applied",
            "feasibility_reason",
            "flow_fraction_of_design",
            "delta_t_star_c",
            "received_mass_flow_kg_s",
            "received_supply_air_temperature_c",
        ]

        # One row per zone/control step. Means for all numeric signals plus
        # min/max/last for the main physical command/response quantities.
        summary_numeric = (
            COMMAND_COLUMNS
            + INTERNAL_COLUMNS
            + self.environment_aliases
            + self.zone_signal_aliases
            + DERIVED_COLUMNS
        )
        self.summary_numeric = summary_numeric

        self.control_fields = [
            "control_step_index",
            "zone_token",
            "received_mode",
            "effective_control_mode",
            "feasible",
            "fallback_applied",
            "feasibility_reason",
            "start_year",
            "start_month",
            "start_day",
            "start_time_hour",
            "start_sim_time_hour",
            "end_year",
            "end_month",
            "end_day",
            "end_time_hour",
            "end_sim_time_hour",
            "nominal_control_interval_seconds",
            "accumulated_system_seconds",
        ]
        for key in summary_numeric:
            self.control_fields.append(f"mean__{key}")

        for key in [
            "zone_temperature_c",
            "zone_supply_temperature_c",
            "zone_supply_mass_flow_kg_s",
            "q_zone_interface_w",
            "fan_electric_power_w",
            "heating_coil_rate_w",
            "cooling_coil_total_rate_w",
            "fan_actuator_readback_kg_s",
            "load_actuator_readback_w",
        ]:
            self.control_fields.extend([
                f"min__{key}",
                f"max__{key}",
                f"last__{key}",
            ])

        self._system_fh = self.system_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._system_writer = csv.DictWriter(
            self._system_fh,
            fieldnames=self.system_fields,
            extrasaction="ignore",
        )
        self._system_writer.writeheader()

        self._command_fh = self.command_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._command_writer = csv.DictWriter(
            self._command_fh,
            fieldnames=self.command_fields,
            extrasaction="ignore",
        )
        self._command_writer.writeheader()

        self._control_fh = self.control_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._control_writer = csv.DictWriter(
            self._control_fh,
            fieldnames=self.control_fields,
            extrasaction="ignore",
        )
        self._control_writer.writeheader()

        self._jsonl_fh = self.control_jsonl_path.open(
            "w", encoding="utf-8"
        )

        self._acc = {
            zone: NumericAccumulator()
            for zone in self.zone_tokens
        }
        self._start_meta = {}
        self._end_meta = {}
        self._decision_meta = {}

    def record_received_command(
        self,
        row: Mapping[str, Any],
    ) -> None:
        self._command_writer.writerow(row)
        self._command_fh.flush()

    def begin_control_step(
        self,
        *,
        control_step_index: int,
        meta: Mapping[str, Any],
        decision_meta: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._start_meta = dict(meta)
        self._end_meta = dict(meta)
        self._decision_meta = {
            str(zone): dict(values)
            for zone, values in (decision_meta or {}).items()
        }
        self._acc = {
            zone: NumericAccumulator()
            for zone in self.zone_tokens
        }

    def record_system_row(
        self,
        *,
        row: Mapping[str, Any],
    ) -> None:
        self._system_writer.writerow(row)
        self._system_fh.flush()

        zone = str(row["zone_token"])
        weight = float(row["system_timestep_seconds"])
        self._acc[zone].add(row, weight)

        self._end_meta = {
            "year": row["year"],
            "month": row["month"],
            "day": row["day"],
            "current_time_hour": row["current_time_hour"],
        }

    def finalize_control_step(
        self,
        *,
        control_step_index: int,
        nested_payload: Mapping[str, Any],
        end_boundary_meta: Mapping[str, Any] | None = None,
        nominal_control_interval_seconds: float = 300.0,
    ) -> Dict[str, Dict[str, Any]]:
        summaries = {}
        boundary_end = dict(end_boundary_meta or self._end_meta)

        for zone in self.zone_tokens:
            acc = self._acc[zone]
            decision = self._decision_meta.get(zone, {})
            row = {
                "control_step_index": control_step_index,
                "zone_token": zone,
                "received_mode": decision.get("received_mode"),
                "effective_control_mode": decision.get("effective_control_mode"),
                "feasible": decision.get("feasible"),
                "fallback_applied": decision.get("fallback_applied"),
                "feasibility_reason": decision.get("feasibility_reason"),
                "start_year": self._start_meta.get("year"),
                "start_month": self._start_meta.get("month"),
                "start_day": self._start_meta.get("day"),
                "start_time_hour": self._start_meta.get("current_time_hour"),
                "start_sim_time_hour": self._start_meta.get("current_sim_time_hour"),
                "end_year": boundary_end.get("year"),
                "end_month": boundary_end.get("month"),
                "end_day": boundary_end.get("day"),
                "end_time_hour": boundary_end.get("current_time_hour"),
                "end_sim_time_hour": boundary_end.get("current_sim_time_hour"),
                "nominal_control_interval_seconds": nominal_control_interval_seconds,
                "accumulated_system_seconds": acc.weight_seconds,
            }

            for key in self.summary_numeric:
                row[f"mean__{key}"] = acc.mean(key)

            for key in [
                "zone_temperature_c",
                "zone_supply_temperature_c",
                "zone_supply_mass_flow_kg_s",
                "q_zone_interface_w",
                "fan_electric_power_w",
                "heating_coil_rate_w",
                "cooling_coil_total_rate_w",
                "fan_actuator_readback_kg_s",
                "load_actuator_readback_w",
            ]:
                row[f"min__{key}"] = acc.min.get(key)
                row[f"max__{key}"] = acc.max.get(key)
                row[f"last__{key}"] = acc.last.get(key)

            self._control_writer.writerow(row)
            summaries[zone] = row

        self._control_fh.flush()

        payload = {
            "control_step_index": control_step_index,
            "zone_summaries": summaries,
            "step_payload": self._jsonable(nested_payload),
        }
        self._jsonl_fh.write(
            json.dumps(payload, default=str) + "\n"
        )
        self._jsonl_fh.flush()

        return summaries

    def write_signal_catalog(self, payload: Mapping[str, Any]) -> None:
        (
            self.history_dir / "signal_catalog.json"
        ).write_text(
            json.dumps(self._jsonable(payload), indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def write_api_registry(self, rows: List[Mapping[str, Any]]) -> None:
        path = self.history_dir / "api_exchange_registry.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            return

        fields = sorted(
            {key for row in rows for key in row.keys()}
        )
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    def close(self) -> None:
        for fh in [
            self._system_fh,
            self._command_fh,
            self._control_fh,
            self._jsonl_fh,
        ]:
            try:
                fh.close()
            except Exception:
                pass

    @classmethod
    def _jsonable(cls, obj):
        if is_dataclass(obj):
            return cls._jsonable(asdict(obj))
        if isinstance(obj, dict):
            return {
                str(k): cls._jsonable(v)
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return [cls._jsonable(x) for x in obj]
        return obj
