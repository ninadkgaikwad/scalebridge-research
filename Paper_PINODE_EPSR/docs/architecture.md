# Repository architecture

The paper implementation remains embedded inside the main `scalebridge-research` repository but is organized as an independently installable Python package.

- `src/pinode_epsr/core`: configuration, paths, shared numerical/data helpers
- `src/pinode_epsr/physics`: RC equations, aggregation relations, energy projection
- `src/pinode_epsr/data`: controlled Phase-D and Phase-C adapters
- `src/pinode_epsr/backends`: Neuromancer integration adapter
- `src/pinode_epsr/methods`: Inverse PINN-RC, NODE, Base PINODE, EBP-PINODE
- `src/pinode_epsr/training`: training/HPO orchestration
- `src/pinode_epsr/evaluation`: runtime, Sim scaffolding, thermostat, metrics

Historical `Paper_PINODE_EPSR/*.py` imports remain compatibility shims until the paper workflow is complete.
