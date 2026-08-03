# -*- coding: utf-8 -*-
"""Stage C6 heat-input regression training orchestration."""
from .trainer import (
    EstimatorTrainingConfig,
    ModelDatasetReference,
    TrainingResult,
    discover_model_datasets,
    train_model_dataset,
)
from .validation import validate_training_artifact

__all__ = [
    "EstimatorTrainingConfig",
    "ModelDatasetReference",
    "TrainingResult",
    "discover_model_datasets",
    "train_model_dataset",
    "validate_training_artifact",
]
