# Package Principles

1. Legacy code is reference logic only.
2. Production ML is PyTorch-first.
3. Neuromancer compatibility is a design requirement.
4. MLflow logging is part of the experiment contract.
5. Hyperparameter tuning is a first-class workflow.
6. `src/` contains reusable software.
7. `experiments/` contains paper-specific execution.
8. `outputs/` contains generated artifacts and should be reproducible.
9. `legacy_reference/` preserves old logic but is not imported by production code.
