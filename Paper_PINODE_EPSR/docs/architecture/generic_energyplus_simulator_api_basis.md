# EnergyPlus API basis

The generic simulator uses the EnergyPlus 24.1 Runtime/Data Exchange APIs.

Key API capabilities used:

- zone timestep callback before init heat balance
- callback after predictor / after HVAC managers
- callback inside HVAC system iteration loop
- callback after HVAC reporting at system timestep end
- `request_variable()` before API-library simulation
- output-variable handle/value access
- internal-variable handle/value access
- actuator handle/set/reset/readback access
- `system_time_step()` for variable HVAC timestep duration
- `get_api_data()` to snapshot the exchange registry

These API mechanisms support the direct project-owned simulator wrapper without
Sinergym.
