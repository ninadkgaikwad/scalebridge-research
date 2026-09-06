# PINODE / EBP-PINODE EPSR Research Implementation

This folder is the reproducible paper implementation for the controlled EPSR study comparing **Inverse PINN-RC, Neural ODE, Base PINODE, and Energy-Balance-Projected PINODE (EBP-PINODE)**.

It intentionally remains **inside the main `scalebridge-research` repository** so it can reuse validated ScaleBridge Phase-C/Phase-D data products and environments, while its canonical code is organized as an independently installable package under:

```text
src/pinode_epsr/
```

## Canonical source layout

```text
src/pinode_epsr/
  core/
  physics/
  data/
  backends/
  methods/
  training/
  evaluation/
```

The four method implementations are under `src/pinode_epsr/methods/`. The differentiable EBP weighted energy projection is a first-class physics component under `src/pinode_epsr/physics/energy_projection.py`.

## Compatibility period

Historical imports such as:

```python
from Paper_PINODE_EPSR.ebp_pinode import EBPPINODEModel
```

remain valid through thin compatibility shims. New standalone-style code should use `pinode_epsr` after installing this folder in editable mode:

```powershell
pip install -e .\Paper_PINODE_EPSR --no-deps
```

`--no-deps` is recommended inside the already-qualified ScaleBridge environments so this paper package does not upgrade or downgrade shared dependencies.

## Scientific runtime contracts

- Neuromancer 1.5.6 provides MLP/Node/System/Problem/PenaltyLoss/DictDataset and fixed-step RK4.
- PyTorch provides tensors, autograd, optimization, physical-parameter transforms, and the differentiable EBP linear solve.
- No custom RK4 is carried in paper code.
- No direct TorchDiffEq integration is used by these four paper methods.
- `Q_AC` is thermal HVAC heat input; `P_HVAC` is electrical power and never enters the thermal balance.

## Validation

Patch 05R is a structural reorganization only. Before scientific development resumes, the complete historical Patch01-Patch05 contract suite and the previously exercised real-data validators are rerun. Patch05R preserved the pre-existing Kitchen no-TRAIN-heating condition as an equivalence check. Patch05B now separates TRAIN mode support from controller capability: both heating and cooling remain executable, with unobserved-mode actuation explicitly marked extrapolative.

Approved small validation evidence is in `validation/`. Large generated runs are stored outside git under the shared ScaleBridge data hierarchy.

## Development history

Patch-era notes and immutable pre-reorganization regression snapshots are retained under `development_history/`; they are not part of the installable runtime package.


## Day-6 finalized thermostat/evaluation contract (Patch05B)

Patch05B uses exact Phase-D `TRAIN + included` timestamp ownership for every
Phase-B thermostat calibration. EnergyPlus `24:00:00` aliases and sparse duplicate
physical timestamps are normalized/coalesced before partition filtering.

The controlled RestaurantFastFood audit established that Dining supports both
heating and cooling, while Kitchen has a valid QAC signal but no positive-net-QAC
heating regime in strict TRAIN. **This does not disable Kitchen heating.**

Sim3 controller capability is always bidirectional. For each zone and mode:

1. an explicit per-mode override, when deliberately supplied, has highest priority;
2. otherwise an observed same-mode TRAIN regime uses its data-derived constant
   supply temperature plus nominal/max mass flow;
3. otherwise the missing mode is resolved only from that same zone's observed
   opposite mode: the mass-flow-weighted active `|Ts-Tz|` is reflected around the
   zone medium setpoint and the observed opposite-mode nominal/max flow is reused.

The fallback does **not** inspect EnergyPlus equipment definitions and does **not**
search other weather/runs. A sign guard guarantees heating supply is above the high
setpoint and cooling supply is below the low setpoint. Unobserved-mode QAC generated
through the Phase-C surrogate is retained but explicitly flagged as extrapolative/OOD.
No requested action is suppressed because a mode was absent from TRAIN.

Cooling and heating active mass-flow choices remain independently selectable as
`nominal` or `max`; thermostat deadbands remain data-derived with explicit override
support. The low/medium/high schedule remains
`medium -> low -> high -> medium`, with the data-driven anti-collapse rule for
commercial-zone temperature plateaus.


Patch05B FINAL2 clarification: observed same-mode HVAC actuator Ts and
nominal/max mdot are preserved exactly from TRAIN. Paper experiment setpoints do
not reject or rewrite those values. Directional setpoint guards apply only to
synthesized unobserved-mode fallback parameters.
