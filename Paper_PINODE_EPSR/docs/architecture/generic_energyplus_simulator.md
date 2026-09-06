# PINODE/EPSR generic EnergyPlus simulator

## Purpose

This module is the plant/runtime layer used later by closed-loop MPC. It is
deliberately more generic than the final controller contract.

It has **no Sinergym dependency**.

It uses EnergyPlus 24.1 through `pyenergyplus.api`.

## Code/data ownership

Clean source, tests, configs, and documentation:

```text
.../NewOrg/scalebridge-research/Paper_PINODE_EPSR
```

Scientific runs and histories:

```text
.../Data/ScaleBridge/Paper_PINODE_EPSR
```

Runtime episodes are created under:

```text
<DataRoot>/05_closed_loop_mpc_runs/<label>_<timestamp>/
```

The simulator writes no scientific run products into the code repository.

## Two-zone physical interface

For RestaurantFastFood, the explicit controller-facing action is

\[
u^\star_k =
[\dot m_D^\star,\ T_{sa,D}^\star,\ \dot m_K^\star,\ T_{sa,K}^\star]^\top .
\]

Example:

```python
cmd = RestaurantFastFoodCommand.four_physical_commands(
    dining_mass_flow_kg_s=mdot_d,
    dining_supply_air_temperature_c=tsa_d,
    kitchen_mass_flow_kg_s=mdot_k,
    kitchen_supply_air_temperature_c=tsa_k,
)

result = simulator.step(cmd)
```

Both Dining and Kitchen may be commanded in heating or cooling.

The simulator does not categorically classify Kitchen heating as infeasible.
Kitchen heating is treated exactly like Dining heating and both cooling modes:
the numerical feasibility envelope is checked per zone at each 300-s control
boundary.

## Identical per-zone actuator transformation

For either zone \(i\in\{D,K\}\):

\[
a_{fan,i} = 0.5\dot m_i^\star
\]

and

\[
a_{q,i}
=
\dot m_i^\star c_p
(T_{sa,i}^\star-T_{z,i}),
\qquad c_p=1006~\mathrm{J/(kg\,K)}.
\]

The same transformation code is used for both zones.

Prior C2.6 Kitchen-heating behavior is preserved as scientific evidence, but
there is **no categorical Kitchen-heating exclusion**.

## Generic symmetric feasibility and fallback

Before each 300-s override interval, the simulator checks each zone
independently using the same configurable envelope:

\[
0.50 \le \dot m_i^\star/\dot m_{i,\max}\le0.80
\]

and

\[
4^\circ\mathrm C
\le |T_{sa,i}^\star-T_{z,i}|
\le 8^\circ\mathrm C.
\]

The sign of \(T_{sa}^\star-T_z\) is unrestricted, so this supports both heating
and cooling for both zones.

For each zone independently:

```text
physical command
      |
      v
generic feasibility check
      |
  +---+---+
  |       |
inside   outside
  |       |
  v       v
override release BOTH override channels
transform for that zone
  |       |
  v       v
EnergyPlus native EnergyPlus HVAC
```

If Dining is feasible and Kitchen is not, Dining remains under external
control while Kitchen alone falls back. The reverse is also true.

A controller can also explicitly send `PhysicalZoneCommand.native()`.

The exact numerical MPC bounds may be refined later after model/MPC
development. The reusable simulator mechanism is simply:
**inside configured envelope -> override; outside -> per-zone native
fallback**.

## Callback implementation

- fan command:
  `callback_inside_system_iteration_loop`
- unitary sensible-load request:
  `callback_after_predictor_after_hvac_managers`
- control boundary:
  `callback_begin_zone_timestep_before_init_heat_balance`
- high-resolution history:
  `callback_end_system_timestep_after_hvac_reporting`

EnergyPlus' API exposes these callback points, and the Data Exchange API
supports requested output variables, handles, actuator values, and variable
system timestep duration.

## Broad history contract

The simulator is intentionally over-instrumented relative to the eventual MPC
feature vector.

### Received command history

Every 300-s controller command is recorded before transformation:

- control step
- zone
- received mode
- received `m_dot*`
- received `T_sa*`

### Feasibility/supervisor history

For every received zone command the simulator records:

- effective control mode (`override`, `native_requested`, or
  `native_fallback`)
- whether the numerical feasibility test passed
- whether fallback was applied
- exact feasibility reason
- requested flow fraction of design
- requested \(T_{sa}^\star-T_z\)

This is preserved at both controller-command and system-timestep levels.

### Transformed/internal command history

At the system timestep we record:

- zone temperature used by the transform
- `DeltaT* = T_sa* - T_zone`
- transformed fan command
- transformed sensible-load command
- fan actuator readback
- unitary sensible-load actuator readback

This distinguishes:

```text
controller command
      ↓
received physical command
      ↓
internal transformation
      ↓
written EnergyPlus actuator command
      ↓
EnergyPlus actuator readback
      ↓
realized plant state
```

### Environment

The broad profile requests, where available:

- outdoor dry bulb
- outdoor wet bulb
- outdoor humidity ratio
- outdoor relative humidity
- barometric pressure
- wind speed
- direct solar
- diffuse solar

### Zone state / comfort / gains

For Dining and Kitchen:

- zone air temperature
- mean radiant temperature
- operative temperature
- relative humidity
- thermostat heating setpoint
- thermostat cooling setpoint
- people heat gain
- lighting heat gain
- electric-equipment heat gain
- zone air-system sensible heating/cooling rate

Optional signals are recorded when available and listed as unavailable rather
than causing the run to fail.

### Full known PSZ air path

For each zone:

```text
return
  ↓
mixed
  ↓
cooling-coil outlet
  ↓
heating-coil outlet
  ↓
supply-equipment outlet
  ↓
zone supply
```

For every node the default profile records:

- temperature
- air mass flow rate
- humidity ratio
- pressure when exposed

### HVAC components/intermediate state

Permanent EnergyPlus internal/design quantities:

- fan design maximum mass flow
- unitary design heating capacity
- unitary design cooling capacity

Dynamic HVAC quantities:

- fan mass flow
- fan electric power
- heating-coil heating rate
- cooling-coil total cooling rate
- cooling-coil sensible cooling rate when available
- unitary PLR when available
- unitary fan PLR when available
- DX speed ratio when available
- DX cycling ratio when available

### Derived air-side physics

At each system timestep the simulator derives:

- zone-interface sensible heat
- return-to-mixed sensible heat
- mixed-to-cooling-outlet sensible heat
- cooling-outlet-to-heating-outlet sensible heat
- heating-outlet-to-supply-outlet sensible heat
- mixed-to-supply-outlet sensible heat
- delta-T across each corresponding segment

## History files

Each run contains:

```text
episode_manifest.json

history/
    received_command_history.csv
    system_timestep_zone_history.csv
    control_step_zone_history.csv
    control_steps.jsonl
    signal_catalog.json
    api_exchange_registry.csv

runtime_inputs/
    model_300s.idf

energyplus_output/
    native EnergyPlus output files
```

### `system_timestep_zone_history.csv`

Highest-value forensic/visualization table.

One row per zone per EnergyPlus system timestep. It contains:

- received command
- internal transform
- actuator readback
- environment
- zone state
- all broad HVAC/node signals
- derived heat-flow quantities

This supports detailed plots of internal HVAC cycling and air-path behavior.

### `control_step_zone_history.csv`

One row per zone per 300-s control step.

It contains time-weighted means for all numeric history signals plus min/max/last
for key physical quantities. This is the primary table for closed-loop paper
plots.

### `control_steps.jsonl`

Nested lossless control-step metadata suitable for later MPC/controller/model
history augmentation.

### `signal_catalog.json`

Records every requested history signal, its EnergyPlus variable/key, units
hint, description, required/optional status, and whether it resolved.

### `api_exchange_registry.csv`

Snapshots the full EnergyPlus API exchange registry visible in the actual run.
This is intentionally retained so later development can determine what other
variables/actuators were available without reconstructing the API inventory.

## MPC contract intentionally deferred

The simulator does not yet decide:

- MPC observation vector
- learned-model feature adapter
- observer-state contract
- forecast contract
- MPC action bounds
- comfort constraints
- Kitchen-heating feasibility
- supervisory fallback policy
- objective logging schema

Those will be frozen only after the learned models and MPC are built.

The simulator history is intentionally broad enough that those future layers
can select the signals they need without rerunning plant-forensic development.


## Day-7 history closure semantics (v6.1)

The simulator synchronizes external control on the first complete EnergyPlus
zone-timestep boundary after entering the requested control window. Therefore
every completed controller action is expected to accumulate exactly 300 s of
EnergyPlus system-timestep history.

EnergyPlus `get_actuator_value()` can retain the previous written numeric value
after `reset_actuator()`. Consequently the broad history distinguishes:

- `fan_actuator_readback_kg_s` / `load_actuator_readback_w`: effective
  externally active override readback; null while native/fallback control is
  active;
- `fan_actuator_api_value_raw` / `load_actuator_api_value_raw`: raw EnergyPlus
  API value retained for forensic transparency, even when the external
  actuator has been released;
- `fan_override_active` / `load_override_active`: explicit external-override
  state.

This prevents a stale API value from being misinterpreted in visualization as
an active MPC command during native fallback.
