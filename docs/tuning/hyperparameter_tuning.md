# Hyperparameter Tuning

Recommended default tuning stack:

- Optuna for local and medium-scale search
- Ray Tune for larger parallel sweeps
- MLflow for logging tuned runs and best artifacts

Search spaces should live under:

`src/scalebridge/tuning/search_spaces/`

Paper-specific tuning configs should live under:

`experiments/<paper>/configs/`
