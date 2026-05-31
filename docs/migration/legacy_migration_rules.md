# Legacy Migration Rules

Every legacy script must be treated as reference logic first.

For each script, identify:

1. Inputs
2. Outputs
3. Feature engineering
4. Model family
5. Training/evaluation logic
6. Paper relevance
7. New modular destination

No TensorFlow/Keras model should be migrated directly as production code.
Use old scripts to reconstruct behavior in PyTorch.
