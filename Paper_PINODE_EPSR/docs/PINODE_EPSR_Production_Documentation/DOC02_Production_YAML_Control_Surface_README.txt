PINODE/EPSR DOC02 - Production YAML Control Surface
=====================================================

Status: PRE-PATCH01R3 DESIGN CONTRACT
Date: 2026-09-02

This documentation-only package adds:
  PINODE_EPSR_Production_YAML_Control_Surface_Runtime_Effect_Contract.tex

It formalizes the accepted Patch01R3 target before implementation:
- production.yaml becomes authoritative for campaign/HPO/training/evaluation/controller policy;
- CLI options are explicit per-run overrides;
- resolved values and per-field provenance are persisted;
- evaluation mdot/unobserved-mode knobs must be real runtime consumers;
- method-specific HPO search spaces remain in training/trainer.py;
- Micro32 remains a separate qualification profile;
- current Patch01R2B remains the qualified runtime baseline until R3 passes validation.

This DOC02 package changes documentation only. It performs no Git operations and does not modify PINODE/EPSR production Python code.
