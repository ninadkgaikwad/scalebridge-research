# Validation evidence

Small approved regression/validation JSON records are retained here as reproducibility evidence.
Large generated experiment outputs remain outside git under the shared ScaleBridge data tree.

Historical Patch05 initially stopped at the Kitchen zero-positive-net-QAC heating condition. Patch05B treats TRAIN mode support as provenance rather than controller capability. Missing-mode actuator parameters use the locked same-zone opposite-mode deltaT/mdot fallback (or explicit override), and unobserved-mode Phase-C QAC is explicitly flagged extrapolative.


Patch05B is the first post-05R scientific Day-6 correction. Its authoritative
validation is generated outside git under the shared paper-data root. The
validator must pass exact TRAIN alignment, controlled Phase-C QAC/PHVAC loading,
all four methods × 1C/2C tiny Sim1/2/3 paths, and Kitchen unavailable-heating
suppression without inventing heating calibration values.

