from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Thread
from typing import Dict, Mapping, Optional, Any, List
import csv
import datetime as dt
import json
import math
import sys
import traceback

from .contracts import (
    ActuatorTransform,
    CommandMode,
    EffectiveControlMode,
    FeasibilityDecision,
    FeasibilityEnvelope,
    FeasibilitySupervisor,
    PhysicalZoneCommand,
    RestaurantFastFoodCommand,
    TransformedZoneCommand,
)
from .history import BroadSimulatorHistory
from .paths import EPSRProjectLayout
from .runtime_idf import prepare_300s_runtime_idf, sha256_file
from .signals import SignalSpec, ZoneRuntimeSpec


WEATHER_FILE_RUN_PERIOD = 3


@dataclass(frozen=True)
class ControlWindow:
    start_month: int
    start_day: int
    start_hour: float
    end_month: Optional[int] = None
    end_day: Optional[int] = None
    end_hour: float = 24.0

    @staticmethod
    def _ordinal_minute(month: int, day: int, hour: float) -> float:
        doy = dt.datetime(2001, month, day).timetuple().tm_yday
        return (doy - 1) * 1440.0 + hour * 60.0

    @property
    def start_ordinal_minute(self) -> float:
        return self._ordinal_minute(
            self.start_month,
            self.start_day,
            self.start_hour,
        )

    @property
    def end_ordinal_minute(self) -> Optional[float]:
        if self.end_month is None or self.end_day is None:
            return None
        return self._ordinal_minute(
            self.end_month,
            self.end_day,
            self.end_hour,
        )

    def contains(self, ordinal_minute: float) -> bool:
        if ordinal_minute < self.start_ordinal_minute:
            return False
        end = self.end_ordinal_minute
        return end is None or ordinal_minute < end

    def is_after(self, ordinal_minute: float) -> bool:
        end = self.end_ordinal_minute
        return end is not None and ordinal_minute >= end


@dataclass(frozen=True)
class ZoneObservation:
    zone_token: str
    zone_temperature_c: float
    design_max_mass_flow_kg_s: float
    signals: Dict[str, Optional[float]]


@dataclass(frozen=True)
class SimulatorObservation:
    year: int
    month: int
    day: int
    current_time_hour: float
    current_sim_time_hour: float
    environment: Dict[str, Optional[float]]
    zones: Dict[str, ZoneObservation]


@dataclass(frozen=True)
class SimulatorStepResult:
    control_step_index: int
    observation: SimulatorObservation
    received_command: RestaurantFastFoodCommand
    transformed_commands: Dict[str, TransformedZoneCommand]
    zone_history_summary: Dict[str, Dict[str, Any]]
    terminated: bool = False


class EnergyPlusSimulator:
    """
    Generic synchronous EnergyPlus simulator for later MPC/closed-loop use.

    Control boundary:
      - accepts physical commands for Dining/Kitchen heating/cooling;
      - applies the SAME configurable feasibility envelope to both zones and
        both heating/cooling signs;
      - if one zone is outside the envelope, only that zone releases both
        overrides and uses native EnergyPlus HVAC for that 300-s step;
      - inside the envelope, both zones use the identical actuator transform;
      - records the complete received/decision/transform/readback/plant history.

    There is NO categorical Kitchen-heating exclusion.
    """

    def __init__(
        self,
        *,
        layout: EPSRProjectLayout,
        energyplus_root: str | Path,
        authoritative_idf: str | Path,
        weather_epw: str | Path,
        authoritative_idf_sha256: str,
        zone_specs: Mapping[str, ZoneRuntimeSpec],
        signal_specs: Mapping[str, List[SignalSpec]],
        control_window: ControlWindow,
        actuator_transform: ActuatorTransform = ActuatorTransform(),
        feasibility_supervisor: FeasibilitySupervisor = FeasibilitySupervisor(),
        run_label: str = "energyplus_simulator",
        queue_timeout_seconds: float = 900.0,
        capture_api_registry: bool = True,
    ) -> None:
        self.layout = layout
        self.layout.assert_separation()

        self.energyplus_root = Path(energyplus_root).resolve()
        self.authoritative_idf = Path(authoritative_idf).resolve()
        self.weather_epw = Path(weather_epw).resolve()
        self.authoritative_idf_sha256 = authoritative_idf_sha256.lower()

        self.zone_specs = {
            k.upper(): v
            for k, v in zone_specs.items()
        }
        self.signal_specs = signal_specs
        self.control_window = control_window
        self.transform = actuator_transform
        self.feasibility_supervisor = feasibility_supervisor
        self.run_label = run_label
        self.queue_timeout_seconds = float(queue_timeout_seconds)
        self.capture_api_registry = bool(capture_api_registry)

        if set(self.zone_specs) != {"DINING", "KITCHEN"}:
            raise ValueError(
                "RestaurantFastFood simulator expects DINING and KITCHEN."
            )
        if self.transform.control_interval_seconds != 300:
            raise ValueError("Current simulator contract requires 300 s.")

        self.run_dir: Optional[Path] = None
        self.runtime_idf: Optional[Path] = None
        self.energyplus_output_dir: Optional[Path] = None

        self._api = None
        self._state = None
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._command_queue: Queue[Any] = Queue(maxsize=1)
        self._boundary_queue: Queue[Any] = Queue(maxsize=3)

        self._resolved = False
        self._handles: Dict[str, Any] = {}
        self._signal_catalog = {}
        self._active_received: Dict[str, PhysicalZoneCommand] = {}
        self._active_decisions: Dict[str, FeasibilityDecision] = {}
        self._active_transformed: Dict[str, TransformedZoneCommand] = {}

        self._control_step_index = -1
        self._system_substep_index = 0
        self._history: Optional[BroadSimulatorHistory] = None
        self._last_observation: Optional[SimulatorObservation] = None
        self._last_summary: Dict[str, Dict[str, Any]] = {}

        self._simulation_exit_code: Optional[int] = None
        self._simulation_error: Optional[str] = None

        # The first EnergyPlus callback that crosses the requested control-window
        # threshold can occur partway through a zone timestep because of API
        # clock semantics. Skip that boundary and begin external control on the
        # next zone-timestep callback so every controller action owns a complete
        # 300-s interval.
        self._control_boundary_synchronized = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> SimulatorObservation:
        if self._thread is not None:
            raise RuntimeError("Simulator already started.")

        self._prepare_run()
        self._load_api()
        self._request_signals()
        self._register_callbacks()

        self._thread = Thread(
            target=self._run_energyplus,
            name="PINODE-EPSR-EnergyPlus-Simulator",
            daemon=True,
        )
        self._thread.start()

        packet = self._wait_boundary()
        if packet["kind"] != "boundary":
            self._raise_packet(packet)

        self._last_observation = packet["observation"]
        self._write_manifest("RUNNING")
        return self._last_observation

    def step(
        self,
        command: RestaurantFastFoodCommand,
    ) -> SimulatorStepResult:
        if self._thread is None or self._last_observation is None:
            raise RuntimeError("Call reset() before step().")

        for zone_command in command.as_zone_mapping().values():
            zone_command.validate()

        self._command_queue.put(
            {"kind": "command", "command": command},
            timeout=self.queue_timeout_seconds,
        )

        packet = self._wait_boundary()

        if packet["kind"] == "terminated":
            return SimulatorStepResult(
                control_step_index=self._control_step_index,
                observation=self._last_observation,
                received_command=command,
                transformed_commands=packet.get(
                    "transformed_commands", {}
                ),
                zone_history_summary=packet.get(
                    "zone_history_summary", {}
                ),
                terminated=True,
            )

        if packet["kind"] != "boundary":
            self._raise_packet(packet)

        self._last_observation = packet["observation"]

        return SimulatorStepResult(
            control_step_index=packet["control_step_index"],
            observation=packet["observation"],
            received_command=command,
            transformed_commands=packet["transformed_commands"],
            zone_history_summary=packet["zone_history_summary"],
            terminated=False,
        )

    def close(self) -> None:
        self._stop_event.set()

        try:
            if self._command_queue.empty():
                self._command_queue.put_nowait({"kind": "stop"})
        except Exception:
            pass

        if self._api is not None and self._state is not None:
            try:
                self._api.runtime.stop_simulation(self._state)
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=20.0)

        if self._history is not None:
            self._history.close()

        self._write_manifest(
            "FAILED" if self._simulation_error else "CLOSED"
        )
        self._thread = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _prepare_run(self) -> None:
        self.run_dir = self.layout.new_simulator_run_dir(
            label=self.run_label
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.runtime_idf = (
            self.run_dir / "runtime_inputs" / "model_300s.idf"
        )
        self.energyplus_output_dir = (
            self.run_dir / "energyplus_output"
        )
        self.energyplus_output_dir.mkdir(parents=True)

        prepare_300s_runtime_idf(
            authoritative_idf=self.authoritative_idf,
            runtime_idf=self.runtime_idf,
            expected_source_sha256=self.authoritative_idf_sha256,
        )

        env_aliases = [
            s.alias
            for s in self.signal_specs["ENVIRONMENT"]
        ]
        zone_aliases = []
        seen = set()
        for token in ["DINING", "KITCHEN"]:
            for spec in self.signal_specs[token]:
                if spec.alias not in seen:
                    seen.add(spec.alias)
                    zone_aliases.append(spec.alias)

        self._history = BroadSimulatorHistory(
            run_dir=self.run_dir,
            zone_tokens=["DINING", "KITCHEN"],
            environment_aliases=env_aliases,
            zone_signal_aliases=zone_aliases,
        )

    def _load_api(self) -> None:
        eplus = str(self.energyplus_root)
        if eplus not in sys.path:
            sys.path.insert(0, eplus)

        from pyenergyplus.api import EnergyPlusAPI

        self._api = EnergyPlusAPI()
        self._state = self._api.state_manager.new_state()
        self._api.runtime.set_console_output_status(
            self._state,
            False,
        )

    def _request_signals(self) -> None:
        ex = self._api.exchange

        for group in self.signal_specs.values():
            for spec in group:
                ex.request_variable(
                    self._state,
                    spec.variable_name,
                    spec.key,
                )

    def _register_callbacks(self) -> None:
        rt = self._api.runtime

        rt.callback_begin_zone_timestep_before_init_heat_balance(
            self._state,
            self._on_control_boundary,
        )
        rt.callback_after_predictor_after_hvac_managers(
            self._state,
            self._on_load_command,
        )
        rt.callback_inside_system_iteration_loop(
            self._state,
            self._on_fan_command,
        )
        rt.callback_end_system_timestep_after_hvac_reporting(
            self._state,
            self._on_system_timestep_end,
        )

    # ------------------------------------------------------------------
    # EnergyPlus thread/callbacks
    # ------------------------------------------------------------------

    def _run_energyplus(self) -> None:
        try:
            code = self._api.runtime.run_energyplus(
                self._state,
                [
                    "-w",
                    str(self.weather_epw),
                    "-d",
                    str(self.energyplus_output_dir),
                    str(self.runtime_idf),
                ],
            )
            self._simulation_exit_code = int(code)

            # If the control-window callback already emitted the final
            # terminated packet, active commands were cleared there.
            if self._active_received or self._control_step_index < 0:
                self._put_boundary({
                    "kind": "terminated",
                    "transformed_commands": dict(self._active_transformed),
                    "zone_history_summary": dict(self._last_summary),
                })

        except Exception:
            self._simulation_error = traceback.format_exc()
            self._put_boundary({
                "kind": "error",
                "message": self._simulation_error,
            })

    def _on_control_boundary(self, state) -> None:
        try:
            if not self._runtime_active(state):
                return

            ordinal = self._ordinal_minute(state)

            if self.control_window.is_after(ordinal):
                if self._control_step_index >= 0 and self._active_received:
                    self._last_summary = self._history.finalize_control_step(
                        control_step_index=self._control_step_index,
                        nested_payload={
                            "received_commands": self._active_received,
                            "feasibility_decisions": self._active_decisions,
                            "transformed_commands": self._active_transformed,
                            "termination_boundary": True,
                        },
                        end_boundary_meta=self._time_meta(state),
                        nominal_control_interval_seconds=(
                            self.transform.control_interval_seconds
                        ),
                    )
                    self._put_boundary({
                        "kind": "terminated",
                        "transformed_commands": dict(self._active_transformed),
                        "zone_history_summary": dict(self._last_summary),
                    })
                    self._active_received = {}
                    self._active_decisions = {}
                    self._active_transformed = {}
                self._api.runtime.stop_simulation(state)
                return

            if not self.control_window.contains(ordinal):
                return

            self._ensure_handles(state)

            if not self._control_boundary_synchronized:
                self._control_boundary_synchronized = True
                return

            boundary_meta = self._time_meta(state)

            # Finish previous control interval before exposing new observation.
            if self._control_step_index >= 0:
                self._last_summary = self._history.finalize_control_step(
                    control_step_index=self._control_step_index,
                    nested_payload={
                        "received_commands": self._active_received,
                        "transformed_commands": self._active_transformed,
                    },
                )

            observation = self._read_observation(state)

            self._put_boundary({
                "kind": "boundary",
                "control_step_index": self._control_step_index,
                "observation": observation,
                "transformed_commands": dict(self._active_transformed),
                "zone_history_summary": dict(self._last_summary),
            })

            while not self._stop_event.is_set():
                try:
                    packet = self._command_queue.get(timeout=0.5)
                except Empty:
                    continue

                if packet["kind"] == "stop":
                    self._api.runtime.stop_simulation(state)
                    return

                if packet["kind"] == "command":
                    command: RestaurantFastFoodCommand = packet["command"]
                    self._control_step_index += 1
                    self._system_substep_index = 0

                    self._active_received = command.as_zone_mapping()

                    meta = boundary_meta

                    # Generic symmetric per-zone feasibility decision at the
                    # 300-s boundary. The same envelope applies to Dining and
                    # Kitchen and to heating/cooling.
                    self._active_decisions = {}
                    for zone, cmd in self._active_received.items():
                        zobs = observation.zones[zone]
                        self._active_decisions[zone] = (
                            self.feasibility_supervisor.evaluate(
                                zone_token=zone,
                                command=cmd,
                                current_zone_temperature_c=(
                                    zobs.zone_temperature_c
                                ),
                                design_max_mass_flow_kg_s=(
                                    zobs.design_max_mass_flow_kg_s
                                ),
                            )
                        )

                    decision_meta = {
                        zone: {
                            "received_mode": decision.received_mode,
                            "effective_control_mode": (
                                decision.effective_control_mode
                            ),
                            "feasible": decision.feasible,
                            "fallback_applied": decision.fallback_applied,
                            "feasibility_reason": decision.reason,
                        }
                        for zone, decision in self._active_decisions.items()
                    }

                    self._history.begin_control_step(
                        control_step_index=self._control_step_index,
                        meta=meta,
                        decision_meta=decision_meta,
                    )

                    # Record exactly what arrived and exactly what the generic
                    # supervisor decided.
                    for zone, cmd in self._active_received.items():
                        decision = self._active_decisions[zone]
                        self._history.record_received_command({
                            "control_step_index": self._control_step_index,
                            **meta,
                            "zone_token": zone,
                            "received_mode": cmd.mode.value,
                            "effective_control_mode": (
                                decision.effective_control_mode
                            ),
                            "feasible": decision.feasible,
                            "fallback_applied": (
                                decision.fallback_applied
                            ),
                            "feasibility_reason": decision.reason,
                            "flow_fraction_of_design": (
                                decision.flow_fraction_of_design
                            ),
                            "delta_t_star_c": decision.delta_t_star_c,
                            "received_mass_flow_kg_s": cmd.mass_flow_kg_s,
                            "received_supply_air_temperature_c": (
                                cmd.supply_air_temperature_c
                            ),
                        })

                    # Initial transform uses boundary zone temperature. An
                    # infeasible command transforms to no low-level override.
                    self._active_transformed = {}
                    for zone, cmd in self._active_received.items():
                        ztemp = observation.zones[
                            zone
                        ].zone_temperature_c
                        self._active_transformed[zone] = (
                            self.transform.transform(
                                zone_token=zone,
                                command=cmd,
                                current_zone_temperature_c=ztemp,
                                decision=self._active_decisions[zone],
                            )
                        )
                    return

            self._api.runtime.stop_simulation(state)

        except Exception:
            self._fail_from_callback(state)

    def _on_load_command(self, state) -> None:
        try:
            if not self._actuation_active(state):
                return

            # Refresh thermal transform at the actual load-command call point
            # using the current EnergyPlus zone temperature.
            for zone, cmd in self._active_received.items():
                h = self._handles[zone]

                decision = self._active_decisions[zone]

                if (
                    cmd.mode == CommandMode.NATIVE
                    or decision.fallback_applied
                ):
                    # Per-zone native behavior: release BOTH override channels
                    # for this zone for the current 300-s step.
                    self._api.exchange.reset_actuator(
                        state,
                        h["load_actuator"],
                    )
                    continue

                ztemp = self._value(
                    state,
                    h["signals"]["zone_temperature_c"],
                )

                transformed = self.transform.transform(
                    zone_token=zone,
                    command=cmd,
                    current_zone_temperature_c=ztemp,
                    decision=decision,
                )
                self._active_transformed[zone] = transformed

                self._api.exchange.set_actuator_value(
                    state,
                    h["load_actuator"],
                    transformed.sensible_load_request_w,
                )

        except Exception:
            self._fail_from_callback(state)

    def _on_fan_command(self, state) -> None:
        try:
            if not self._actuation_active(state):
                return

            for zone, cmd in self._active_received.items():
                h = self._handles[zone]

                decision = self._active_decisions[zone]

                if (
                    cmd.mode == CommandMode.NATIVE
                    or decision.fallback_applied
                ):
                    self._api.exchange.reset_actuator(
                        state,
                        h["fan_actuator"],
                    )
                    continue

                transformed = self._active_transformed[zone]
                self._api.exchange.set_actuator_value(
                    state,
                    h["fan_actuator"],
                    transformed.fan_actuator_command_kg_s,
                )

        except Exception:
            self._fail_from_callback(state)

    def _on_system_timestep_end(self, state) -> None:
        try:
            if not self._actuation_active(state):
                return

            dt_seconds = (
                float(self._api.exchange.system_time_step(state))
                * 3600.0
            )
            self._system_substep_index += 1

            env_values = self._read_environment_signals(state)
            meta = self._time_meta(state)

            for zone in ["DINING", "KITCHEN"]:
                h = self._handles[zone]
                cmd = self._active_received[zone]
                transformed = self._active_transformed[zone]

                zsignals = self._read_zone_signals(state, zone)
                decision = self._active_decisions[zone]

                fan_api_raw = float(
                    self._api.exchange.get_actuator_value(
                        state,
                        h["fan_actuator"],
                    )
                )
                load_api_raw = float(
                    self._api.exchange.get_actuator_value(
                        state,
                        h["load_actuator"],
                    )
                )
                override_active = (
                    decision.effective_control_mode
                    == EffectiveControlMode.OVERRIDE.value
                )

                row = {
                    "control_step_index": self._control_step_index,
                    "system_substep_index": self._system_substep_index,
                    **meta,
                    "system_timestep_seconds": dt_seconds,
                    "zone_token": zone,

                    "received_mode": cmd.mode.value,
                    "effective_control_mode": (
                        decision.effective_control_mode
                    ),
                    "feasible": decision.feasible,
                    "fallback_applied": decision.fallback_applied,
                    "feasibility_reason": decision.reason,
                    "flow_fraction_of_design": (
                        decision.flow_fraction_of_design
                    ),
                    "received_mass_flow_kg_s": cmd.mass_flow_kg_s,
                    "received_supply_air_temperature_c": (
                        cmd.supply_air_temperature_c
                    ),

                    "transform_zone_temperature_c": (
                        transformed.transform_zone_temperature_c
                    ),
                    "delta_t_star_c": transformed.delta_t_star_c,
                    "transformed_fan_command_kg_s": (
                        transformed.fan_actuator_command_kg_s
                    ),
                    "transformed_sensible_load_request_w": (
                        transformed.sensible_load_request_w
                    ),

                    # "readback" means an ACTIVE external override. EnergyPlus
                    # get_actuator_value() can retain the previous written value
                    # after reset_actuator(), so the raw API value is stored
                    # separately and the effective readback is null in native
                    # requested/fallback modes.
                    "fan_override_active": override_active,
                    "load_override_active": override_active,
                    "fan_actuator_readback_kg_s": (
                        fan_api_raw if override_active else None
                    ),
                    "load_actuator_readback_w": (
                        load_api_raw if override_active else None
                    ),
                    "fan_actuator_api_value_raw": fan_api_raw,
                    "load_actuator_api_value_raw": load_api_raw,

                    "fan_design_max_mass_flow_kg_s": float(
                        self._api.exchange.get_internal_variable_value(
                            state,
                            h["fan_max"],
                        )
                    ),
                    "unitary_design_heating_capacity_w": float(
                        self._api.exchange.get_internal_variable_value(
                            state,
                            h["design_heating_capacity"],
                        )
                    ),
                    "unitary_design_cooling_capacity_w": float(
                        self._api.exchange.get_internal_variable_value(
                            state,
                            h["design_cooling_capacity"],
                        )
                    ),

                    **env_values,
                    **zsignals,
                }

                row.update(
                    self._derived_air_path_quantities(zsignals)
                )

                self._history.record_system_row(row=row)

        except Exception:
            self._fail_from_callback(state)

    # ------------------------------------------------------------------
    # Resolution / signals
    # ------------------------------------------------------------------

    def _ensure_handles(self, state) -> None:
        if self._resolved:
            return

        if not self._api.exchange.api_data_fully_ready(state):
            raise RuntimeError("EnergyPlus API data not ready.")

        ex = self._api.exchange

        if self.capture_api_registry:
            registry = []
            for p in ex.get_api_data(state):
                registry.append({
                    "what": str(getattr(p, "what", "")),
                    "name": str(getattr(p, "name", "")),
                    "key": str(getattr(p, "key", "")),
                    "type": str(getattr(p, "type", "")),
                    "unit": str(getattr(p, "unit", "")),
                })
            self._history.write_api_registry(registry)

        # Resolve environment signals.
        self._handles["ENVIRONMENT"] = {
            "signals": self._resolve_signal_group(
                state,
                self.signal_specs["ENVIRONMENT"],
            )
        }

        catalog = {
            "ENVIRONMENT": {},
            "DINING": {},
            "KITCHEN": {},
        }

        for spec in self.signal_specs["ENVIRONMENT"]:
            catalog["ENVIRONMENT"][spec.alias] = {
                **asdict(spec),
                "available": (
                    self._handles["ENVIRONMENT"]["signals"][
                        spec.alias
                    ] is not None
                ),
            }

        for token, zspec in self.zone_specs.items():
            fan_actuator = ex.get_actuator_handle(
                state,
                "Fan",
                "Fan Air Mass Flow Rate",
                zspec.fan_name,
            )
            load_actuator = ex.get_actuator_handle(
                state,
                "Unitary HVAC",
                "Sensible Load Request",
                zspec.unitary_name,
            )
            fan_max = ex.get_internal_variable_handle(
                state,
                "Fan Maximum Mass Flow Rate",
                zspec.fan_name,
            )
            design_heating_capacity = ex.get_internal_variable_handle(
                state,
                "Unitary HVAC Design Heating Capacity",
                zspec.unitary_name,
            )
            design_cooling_capacity = ex.get_internal_variable_handle(
                state,
                "Unitary HVAC Design Cooling Capacity",
                zspec.unitary_name,
            )

            if fan_actuator < 0:
                raise RuntimeError(
                    f"Missing fan actuator for {token}."
                )
            if load_actuator < 0:
                raise RuntimeError(
                    f"Missing sensible-load actuator for {token}."
                )
            if fan_max < 0:
                raise RuntimeError(
                    f"Missing fan max internal variable for {token}."
                )
            if design_heating_capacity < 0:
                raise RuntimeError(
                    f"Missing design heating capacity for {token}."
                )
            if design_cooling_capacity < 0:
                raise RuntimeError(
                    f"Missing design cooling capacity for {token}."
                )

            signal_handles = self._resolve_signal_group(
                state,
                self.signal_specs[token],
            )

            self._handles[token] = {
                "fan_actuator": fan_actuator,
                "load_actuator": load_actuator,
                "fan_max": fan_max,
                "design_heating_capacity": design_heating_capacity,
                "design_cooling_capacity": design_cooling_capacity,
                "signals": signal_handles,
            }

            for spec in self.signal_specs[token]:
                catalog[token][spec.alias] = {
                    **asdict(spec),
                    "available": (
                        signal_handles[spec.alias] is not None
                    ),
                }

        self._signal_catalog = catalog
        self._history.write_signal_catalog({
            "profile": "BROAD_HVAC_HISTORY_V1",
            "actuator_transform": {
                "fan": "a_fan = 0.5*m_dot_star",
                "thermal": (
                    "a_q = m_dot_star*cp*"
                    "(T_sa_star-T_zone_at_callback)"
                ),
                "cp_air_j_kgk": self.transform.cp_air_j_kgk,
            },
            "policy": {
                "type": "generic_symmetric_per_zone_native_fallback",
                "same_envelope_all_zones_modes": True,
                "categorical_kitchen_heating_exclusion": False,
                "feasibility_envelope": asdict(
                    self.feasibility_supervisor.envelope
                ),
                "fallback_action": (
                    "release fan and sensible-load overrides for only the "
                    "infeasible zone during that 300-s step"
                ),
            },
            "signals": catalog,
        })

        self._resolved = True

    def _resolve_signal_group(
        self,
        state,
        specs: List[SignalSpec],
    ) -> Dict[str, Optional[int]]:
        out = {}
        for spec in specs:
            handle = self._api.exchange.get_variable_handle(
                state,
                spec.variable_name,
                spec.key,
            )

            if handle < 0:
                if spec.required:
                    raise RuntimeError(
                        "Required signal unavailable: "
                        f"{spec.alias} -> "
                        f"{spec.variable_name} / {spec.key}"
                    )
                out[spec.alias] = None
            else:
                out[spec.alias] = handle

        return out

    def _read_environment_signals(self, state):
        out = {}
        for alias, handle in self._handles[
            "ENVIRONMENT"
        ]["signals"].items():
            out[alias] = (
                self._value(state, handle)
                if handle is not None
                else None
            )
        return out

    def _read_zone_signals(self, state, zone):
        out = {}
        for alias, handle in self._handles[
            zone
        ]["signals"].items():
            out[alias] = (
                self._value(state, handle)
                if handle is not None
                else None
            )
        return out

    def _read_observation(self, state) -> SimulatorObservation:
        environment = self._read_environment_signals(state)

        zones = {}
        for token in ["DINING", "KITCHEN"]:
            signals = self._read_zone_signals(state, token)
            zones[token] = ZoneObservation(
                zone_token=token,
                zone_temperature_c=float(
                    signals["zone_temperature_c"]
                ),
                design_max_mass_flow_kg_s=float(
                    self._api.exchange.get_internal_variable_value(
                        state,
                        self._handles[token]["fan_max"],
                    )
                ),
                signals=signals,
            )

        return SimulatorObservation(
            **self._time_meta(state),
            environment=environment,
            zones=zones,
        )

    # ------------------------------------------------------------------
    # Derived physics
    # ------------------------------------------------------------------

    def _derived_air_path_quantities(
        self,
        s: Mapping[str, Optional[float]],
    ) -> Dict[str, Optional[float]]:
        cp = self.transform.cp_air_j_kgk

        def val(name):
            v = s.get(name)
            return None if v is None else float(v)

        def q(mdot_name, t2_name, t1_name):
            m = val(mdot_name)
            t2 = val(t2_name)
            t1 = val(t1_name)
            if None in (m, t2, t1):
                return None
            return m * cp * (t2 - t1)

        def delta(t2_name, t1_name):
            t2 = val(t2_name)
            t1 = val(t1_name)
            if None in (t2, t1):
                return None
            return t2 - t1

        return {
            "q_zone_interface_w": q(
                "zone_supply_mass_flow_kg_s",
                "zone_supply_temperature_c",
                "zone_temperature_c",
            ),
            "q_return_to_mixed_w": q(
                "mixed_mass_flow_kg_s",
                "mixed_temperature_c",
                "return_temperature_c",
            ),
            "q_mixed_to_cool_out_w": q(
                "cool_out_mass_flow_kg_s",
                "cool_out_temperature_c",
                "mixed_temperature_c",
            ),
            "q_cool_out_to_heat_out_w": q(
                "heat_out_mass_flow_kg_s",
                "heat_out_temperature_c",
                "cool_out_temperature_c",
            ),
            "q_heat_out_to_supply_outlet_w": q(
                "supply_outlet_mass_flow_kg_s",
                "supply_outlet_temperature_c",
                "heat_out_temperature_c",
            ),
            "q_mixed_to_supply_outlet_w": q(
                "supply_outlet_mass_flow_kg_s",
                "supply_outlet_temperature_c",
                "mixed_temperature_c",
            ),
            "delta_t_return_to_mixed_c": delta(
                "mixed_temperature_c",
                "return_temperature_c",
            ),
            "delta_t_mixed_to_cool_out_c": delta(
                "cool_out_temperature_c",
                "mixed_temperature_c",
            ),
            "delta_t_cool_out_to_heat_out_c": delta(
                "heat_out_temperature_c",
                "cool_out_temperature_c",
            ),
            "delta_t_heat_out_to_supply_outlet_c": delta(
                "supply_outlet_temperature_c",
                "heat_out_temperature_c",
            ),
            "delta_t_zone_supply_minus_zone_c": delta(
                "zone_supply_temperature_c",
                "zone_temperature_c",
            ),
        }

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def _runtime_active(self, state) -> bool:
        ex = self._api.exchange
        if not ex.api_data_fully_ready(state):
            return False
        if bool(ex.warmup_flag(state)):
            return False
        return int(ex.kind_of_sim(state)) == WEATHER_FILE_RUN_PERIOD

    def _actuation_active(self, state) -> bool:
        if not self._runtime_active(state):
            return False
        if not self._active_received:
            return False
        return self.control_window.contains(
            self._ordinal_minute(state)
        )

    def _ordinal_minute(self, state) -> float:
        month = int(self._api.exchange.month(state))
        day = int(self._api.exchange.day_of_month(state))
        hour = float(self._api.exchange.current_time(state))
        doy = dt.datetime(
            2001, month, day
        ).timetuple().tm_yday
        return (doy - 1) * 1440.0 + hour * 60.0

    def _time_meta(self, state):
        return {
            "year": int(self._api.exchange.year(state)),
            "month": int(self._api.exchange.month(state)),
            "day": int(self._api.exchange.day_of_month(state)),
            "current_time_hour": float(
                self._api.exchange.current_time(state)
            ),
            "current_sim_time_hour": float(
                self._api.exchange.current_sim_time(state)
            ),
        }

    def _value(self, state, handle) -> float:
        return float(
            self._api.exchange.get_variable_value(
                state,
                handle,
            )
        )

    def _put_boundary(self, packet) -> None:
        try:
            self._boundary_queue.put_nowait(packet)
        except Exception:
            pass

    def _wait_boundary(self):
        try:
            return self._boundary_queue.get(
                timeout=self.queue_timeout_seconds
            )
        except Empty:
            if self._simulation_error:
                raise RuntimeError(self._simulation_error)
            raise TimeoutError(
                "Timed out waiting for EnergyPlus simulator boundary."
            )

    def _raise_packet(self, packet):
        raise RuntimeError(
            str(packet.get("message", packet))
        )

    def _fail_from_callback(self, state):
        self._simulation_error = traceback.format_exc()
        self._put_boundary({
            "kind": "error",
            "message": self._simulation_error,
        })
        self._api.runtime.stop_simulation(state)

    def _write_manifest(self, status):
        if self.run_dir is None:
            return

        payload = {
            "status": status,
            "simulator_contract_version": "20260829_v6_1",
            "simulator_policy": (
                "generic symmetric per-zone feasibility supervisor with "
                "native fallback; no categorical Kitchen-heating exclusion"
            ),
            "feasibility_envelope": asdict(
                self.feasibility_supervisor.envelope
            ),
            "code_repo_root": str(self.layout.repo_root),
            "data_root": str(self.layout.data_root),
            "run_dir": str(self.run_dir),
            "energyplus_root": str(self.energyplus_root),
            "authoritative_idf": str(self.authoritative_idf),
            "authoritative_idf_sha256_expected": (
                self.authoritative_idf_sha256
            ),
            "authoritative_idf_sha256_actual": (
                sha256_file(self.authoritative_idf)
                if self.authoritative_idf.exists()
                else None
            ),
            "weather_epw": str(self.weather_epw),
            "runtime_idf": (
                str(self.runtime_idf)
                if self.runtime_idf is not None
                else None
            ),
            "control_window": asdict(self.control_window),
            "actuator_transform": asdict(self.transform),
            "zones": {
                token: asdict(spec)
                for token, spec in self.zone_specs.items()
            },
            "history_profile": "BROAD_HVAC_HISTORY_V1",
            "simulation_exit_code": self._simulation_exit_code,
            "simulation_error": self._simulation_error,
        }

        (
            self.run_dir / "episode_manifest.json"
        ).write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
