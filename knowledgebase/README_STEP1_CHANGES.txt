ScaleBridge Phase C Step 1 update bundle

Implemented C1-C4 contract:
- QAC remains the signed sensible thermal HVAC model:
    x = 1000*c_a*mdot*(T_supply - T_zone)
    y = Q_heating - Q_cooling
    fit_intercept = False
- PHVAC is a separate model:
    x = abs(measured QAC target) for standalone/oracle training
    y = Facility_Total_HVAC_Electric_Demand_Power / aggregate_zone_count
    fit_intercept = True
- PHVAC metadata records:
    input_transform = absolute_value
    dependency_model_id = QAC
    target_allocation = equal_across_aggregate_zones
    model_role = hvac_electric_power
- All other heat-input models remain fit_intercept=False.
- Aggregate-zone count is propagated from the Stage B aggregation run into C1-C4 identity/metadata.
- The old facility_electric_demand option is removed from the QAC target choices to prevent conflating QAC and PHVAC.

Files included preserve repository-relative paths.

Next required batch for C5-C8:
- model implementations, factory/base/serialization
- C5 API validation
- C6 trainer and training script
- C7 evaluator and script
- C8 annual inference and script
- campaign runner
