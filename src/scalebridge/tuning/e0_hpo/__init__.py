"""ScaleBridge E0-8 generic hyperparameter optimization framework."""

from .contracts import (
    CompletedTrialView,
    FrozenHyperparameters,
    HPOContractError,
    HPODataSelection,
    IncompatibleResumeError,
    ObjectiveDirection,
    ObjectiveSpec,
    RecoverableTrialError,
    StudySpec,
    TrialEvaluation,
    TrialTerminalState,
)
from .data_policy import create_train_only_selection, validate_train_only_selection
from .handoff import build_e07_hpo_provenance
from .mlflow_tracking import MLflowHPOConfig
from .provider import BaseHPOProvider, TrialContext, TrialSuggester
from .runner import HPOStudyConfig, StudyOutcome, run_hpo_study
from .seeding import derive_trial_seed

__all__ = [
    "BaseHPOProvider",
    "CompletedTrialView",
    "FrozenHyperparameters",
    "HPOContractError",
    "HPODataSelection",
    "HPOStudyConfig",
    "IncompatibleResumeError",
    "MLflowHPOConfig",
    "ObjectiveDirection",
    "ObjectiveSpec",
    "RecoverableTrialError",
    "StudyOutcome",
    "StudySpec",
    "TrialContext",
    "TrialEvaluation",
    "TrialSuggester",
    "TrialTerminalState",
    "build_e07_hpo_provenance",
    "create_train_only_selection",
    "derive_trial_seed",
    "run_hpo_study",
    "validate_train_only_selection",
]
