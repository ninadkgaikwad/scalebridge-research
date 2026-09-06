# PINODE/EPSR Day-1 Production Infrastructure

## Root ownership

- Code: `NewOrg/scalebridge-research/Paper_PINODE_EPSR`
- Read-only scientific inputs: `Data/ScaleBridge/campaigns/<campaign>`
- All generated paper artifacts: `Data/ScaleBridge/Paper_PINODE_EPSR`

## Scientific matrix

32 configurations = 4 methods x 2 RC orders x 4 cases. Training seeds are configurable multi-start restarts, not additional scientific model configurations.

## HPO

HPO is strictly Phase-D TRAIN only. `--hpo-percentage` is applied independently to every monthly TRAIN block. Contiguous leakage-safe mini-blocks are selected from all twelve months. The selected HPO material is internally separated into fit and holdout portions; Phase-D VALIDATION and TEST remain untouched. The default objective is recursive normalized temperature error. Persistent SQLite studies resume in `01_hpo/studies`.

## Final training

Frozen HPO values are applied to the actual method configs. Final training uses the authoritative full TRAIN partition and Phase-D VALIDATION for best-epoch/restart selection.

## Controller and Phase-C chain

Thermostat transition calibration only accepts exact 300-s adjacent pairs. Sim3 uses `QHVAC_physics = 1000*1.005*mdot*(Ts-Tz)`, then the actual Phase-C QAC model to obtain the corrected/effective `QHVAC_phaseC`, then `abs(QHVAC_phaseC)` as input to the actual Phase-C PHVAC model. Both raw PHVAC regression output and nonnegative physical PHVAC are recorded.

## Evaluation

Sim1 covers TEST one-step prediction. Production Sim2 and Sim3 process all contiguous monthly TEST episodes. Sim3 resets at each TEST episode and applies medium -> low -> high -> medium setpoints within each episode.

## Commands

```powershell
python Paper_PINODE_EPSR\scripts\run_day1_production.py paths
python -m Paper_PINODE_EPSR.validate_day1_production
python Paper_PINODE_EPSR\scripts\run_day1_production.py micro32 --hpo-percentage 0.5
python Paper_PINODE_EPSR\scripts\run_day1_production.py campaign --scope priority-a --hpo-percentage 2 --hpo-trials 12 --seeds 0
```

Increase HPO percentage/trials/restart seeds only after measured runtime evidence justifies it.


## HPO protocol identity and resume
Persistent Optuna studies are keyed by scientific HPO protocol (configuration, TRAIN percentage, inner holdout percentage, objective, per-trial epoch/patience budget, sampler seed, batch/window policy). Increasing only the target trial count resumes the same SQLite study. Changing the sampled TRAIN percentage, objective, or other scientific protocol creates a distinct study so trials from different datasets/objectives are never mixed silently.

## Machine-portable data-root resolution
Production commands first honor `SCALEBRIDGE_GENERATED_DATA_ROOT`. If it is unset and the paper package is embedded in the ScaleBridge repository, the code resolves the locked sibling layout automatically: `<Project>/NewOrg/scalebridge-research` -> `<Project>/Data/ScaleBridge`. The shared `campaigns` subtree remains read-only; all generated paper artifacts are written beneath `Data/ScaleBridge/Paper_PINODE_EPSR` (or the explicit `SCALEBRIDGE_PINODE_EPSR_DATA_ROOT` override).

## Episode boundaries
Sim3 controller switch metrics are computed within each contiguous monthly TEST episode. Month-to-month gaps are never counted as 300-s thermostat/controller transitions.

## HPO percentage semantics (Patch01R1)

`hpo.train_percentage` is a hard **target-row** budget applied independently
to each authoritative monthly Phase-D TRAIN segment. The sampler never increases
the requested percentage to satisfy rollout/encoder geometry. For 2C models,
causal encoder history immediately preceding a selected target block may be read
from the same TRAIN segment as context-only support; those context rows are not
optimization/holdout targets and are not counted against the HPO percentage.
Fit target blocks are temporally earlier than the HPO-holdout block, so fit
windows never use future holdout observations as context. If a requested budget
is too small to provide at least one conservative fit and holdout rollout, the
sampler reports the minimum legal percentage and aborts that HPO request rather
than silently oversampling.

## Patch01R2 — strict floor budgets and controller overrides

Patch01R2 tightens two production contracts without changing model mathematics.

### HPO integer-row budget

For each authoritative monthly Phase-D TRAIN segment with `N_m` rows and requested
percentage `p`, selected HPO target rows are

`floor(N_m * p / 100)`.

The inner HPO holdout is also floored:

`floor(N_selected_m * holdout_percentage / 100)`.

Neither quantity is rounded up or inflated. If the selected target/holdout rows
cannot support the declared HPO rollout geometry, the sampler fails explicitly.
Production retains `N_r <= 12`, `L_e <= 12`. The `micro32` plumbing qualification
uses `N_r <= 3`, `L_e <= 12`, so the default 0.5% HPO budget can retain a strict
non-exceeding 20% holdout.

### Controller override contract

`configs/production.yaml` now contains a `controller` section. Null values preserve
the qualified data-derived defaults. `deadband_half_width_C` is a half-width about
the active setpoint: a value of `1.0` means thresholds at setpoint - 1 C and
setpoint + 1 C (2 C total band).

Per-zone heating/cooling `T_supply_C`, `mdot_nominal_kg_s`, and `mdot_max_kg_s`
may be overridden. Resolution precedence is:

`user_override > data_train_observed > documented same-zone opposite-mode fallback`.

Unobserved Kitchen heating remains explicitly extrapolative/OOD even when the user
supplies actuation parameters. Sim3 trajectory/provenance records the final source,
base source, overridden fields, and the full resolved actuation profile.
