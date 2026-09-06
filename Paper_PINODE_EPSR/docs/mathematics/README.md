# Authoritative mathematical contracts

The paper implementation was developed against five authoritative LaTeX
specifications. Exact copies are now stored in this repository under
`contracts/`:

1. `contracts/PINODE_EPSR_Part1_RC_Representations_v1.tex`
2. `contracts/PINODE_EPSR_Part2_Inverse_PINN_RC.tex`
3. `contracts/PINODE_EPSR_Part3_NeuralODE_Detailed.tex`
4. `contracts/PINODE_EPSR_Part4_Base_PINODE_Detailed.tex`
5. `contracts/PINODE_EPSR_Part5_EBP_PINODE_Detailed.tex`

These files are authoritative for the controlled EPSR PINODE paper workflow.
They are paper-specific scientific contracts, not automatically generic
ScaleBridge Phase-E contracts.

Generic ScaleBridge thermal-model mathematics is maintained separately under:

`docs/mathematics/thermal_modeling/`

That separation prevents paper-specific assumptions (for example the controlled
RestaurantFastFood/Buffalo scope, Dining/Kitchen identities, and paper-specific
1R1C/2R2C configurations) from silently becoming generic package behavior.

The generic Phase-E mathematical contracts may cite these paper contracts as
references and may deliberately generalize their scientifically reusable
mechanics.
