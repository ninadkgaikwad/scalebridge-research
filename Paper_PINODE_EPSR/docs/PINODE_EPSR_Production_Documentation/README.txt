PINODE/EPSR PRODUCTION DOCUMENTATION DOC01
=========================================

Install target (under repo root):
  PINODE_EPSR_Production_Documentation\

Primary files:
  PINODE_EPSR_Production_Campaigns_HPO_Technical_Reference.tex
  PINODE_EPSR_Production_Campaigns_HPO_Quick_Reference.tex
  PINODE_EPSR_Production_Source_of_Truth_Map.tex

Baseline documented:
  PINODE_EPSR_DAY1_PRODUCTION_HPO_PATCH01R2B_20260902_V5

Qualification evidence recorded:
  - 135/135 targeted real-data validator tests PASS
  - Micro32: 32/32 completed and accepted, 0 rejected, 0 failed, exit code 0

Important architecture note:
  The method HPO search spaces are in src/pinode_epsr/training/trainer.py.
  The campaign CLI constructs HPO/training config from CLI arguments.
  The production.yaml controller section is active, while its hpo/training/evaluation
  blocks are not currently loaded by the campaign CLI.

No Git operations are performed by this documentation installer.
