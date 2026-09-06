# Day-6 QAC and thermostat calibration audit

This note records the controlled-data findings used to finalize Patch05B. It is
not a replacement for the external machine-generated audit artifact.

## Data scope

Calibration uses the controlled RestaurantFastFood/Buffalo Phase-B aggregation
and is restricted to the exact Phase-D `TRAIN + included` timestamp set.

Authoritative count per controlled zone:

- Phase-D TRAIN+included: **73,567**
- old calendar-key thermostat filter: **73,579**
- Patch05B exact canonical alignment: **73,567**

The 12-row excess in the old filter arose because EnergyPlus interval-ending
timestamps such as `24:00:00` can normalize onto the same physical instant as a
following-day `00:00:00` row. Patch05B normalizes onto the actual Phase-D year,
coalesces sparse duplicates, and then applies exact Phase-D timestamp ownership.

## QAC convention

For raw EnergyPlus sensible outputs,

\[
Q_{AC}^{raw}=Q_{heating}-Q_{cooling}.
\]

The Phase-D thermal-model input is the Phase-C annual-inference QAC signal.
The Phase-C HVAC proxy is

\[
Q_{HVAC,X}=1000(1.005)\dot m(T_s-T_z).
\]

## Controlled-zone findings

| Zone | Effective cooling TRAIN rows | Effective heating TRAIN rows | Raw net-QAC range |
|---|---:|---:|---:|
| RestaurantFastFood_All | 64,081 | 2,316 | -12.965 to +4.969 kW |
| Dining | 38,258 | 25,299 | -12.089 to +19.160 kW |
| Kitchen | 66,009 | 0 | -19.535 to 0 kW |

Kitchen still has a valid QAC signal and Phase-C QAC model. There are 43 raw
heating-rate-positive TRAIN rows, but all 43 are simultaneous with larger
cooling. Consequently Kitchen never exhibits a positive net-QAC heating
operating regime in strict TRAIN. This is recorded as missing same-mode support,
not as a controller restriction. Patch05B resolves Kitchen heating from the
locked same-zone cooling fallback described below.

## Aggregation consistency

The all-to-one controlled aggregate is internally exact to numerical precision:

- aggregate QAC = equal mean of Dining and Kitchen QAC,
- aggregate Tz = equal mean of Dining and Kitchen Tz,
- aggregate Ts = equal mean of Dining and Kitchen Ts,
- aggregate mass flow = Dining mass flow + Kitchen mass flow.

No Phase-B aggregation or Phase-C QAC contract is changed by Patch05B.

## Data-derived thermostat defaults

Approximate strict-TRAIN calibration values from the audit:

| Zone | Setpoints low / medium / high (C) | Ts cooling / heating (C) | mdot nominal cooling / heating (kg/s) | mdot max cooling / heating (kg/s) |
|---|---|---|---|---|
| RestaurantFastFood_All | 19.994 / 21.953 / 24.997 | 16.642 / 20.678 | 2.166 / 0.907 | 2.867 / 2.723 |
| Dining | 21.109 / 22.499 / 23.889 | 19.769 / 23.548 | 1.151 / 1.293 | 1.744 / 1.744 |
| Kitchen | 18.879 / 23.175 / 26.107 | 11.926 / unavailable | 0.980 / unavailable | 1.485 / unavailable |

Dining uses the Patch05B anti-collapse setpoint rule because its empirical P10
and P50 are essentially identical. Low/high remain P10/P90; medium becomes their
midpoint when the empirical median is not meaningfully separated.

## Sim3 bidirectional actuation rule

The exact MATLAB-style heating/cooling state machine can command **both heating
and cooling in every zone**, independent of whether that mode was observed in
TRAIN.

TRAIN mode availability is provenance and confidence information only.

For an observed mode, Sim3 uses the data-derived constant supply temperature and
the selected nominal/max mass flow. For an unobserved mode, the default
actuation profile is constructed from the same zone's observed opposite mode:

1. use the same zone's observed opposite-mode mass-flow-weighted active supply-air temperature offset
   \(|T_s-T_z|\),
2. mirror that offset around the zone's medium setpoint,
3. enforce a sign guard so heating supply is above the high setpoint and cooling
   supply is below the low setpoint,
4. reuse the same-zone opposite-mode nominal/max airflow.

Every resolved Ts and airflow can be explicitly overridden. Explicit overrides
have highest priority when deliberately supplied. The fallback never uses
EnergyPlus equipment-definition lookup and never searches other runs.

For Kitchen, heating therefore remains **controller-capable** even though it was
not observed as positive net QAC in TRAIN. Such heating actions are recorded as
`qac_extrapolation=True`, because the Kitchen Phase-C QAC surrogate is being
evaluated outside its observed heating support.

The strict diagnostic option is:

```text
unobserved_mode_policy = error
```

which raises on an unobserved-mode request. The default paper control run is:

```text
unobserved_mode_policy = fallback
```

No controller action is suppressed merely because the mode was absent from
training data.


### Observed actuator values versus experiment setpoints

Low/medium/high thermostat setpoints are experiment conditions, not validity
bounds on actuator values observed in TRAIN. Observed same-mode supply
temperature and nominal/max airflow are preserved exactly. The low/high
directional setpoint guard is applied only while synthesizing an unobserved-mode
fallback from that same zone's observed opposite mode.
