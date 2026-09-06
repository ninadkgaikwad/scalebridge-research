"""Production orchestration for the PINODE/EPSR paper campaign.

This layer owns experiment identity, month-balanced TRAIN-only HPO sampling,
persistent Optuna studies, production checkpoints, and standardized offline
Sim1/Sim2/Sim3 artifacts.  It deliberately reuses the validated scientific
method and evaluation implementations in the parent package.
"""

from .contracts import HPOConfig, ProductionTrainingConfig, ProductionEvaluationConfig, ControllerOverrideConfig, load_controller_override_config
from .matrix import ExperimentSpec, production_matrix
from .paths import ProductionLayout, resolve_production_config, resolve_production_layout
from .sampling import MonthBalancedHPOSample, select_month_balanced_hpo_sample

__all__ = [
    "HPOConfig", "ProductionTrainingConfig", "ProductionEvaluationConfig", "ControllerOverrideConfig", "load_controller_override_config",
    "ExperimentSpec", "production_matrix", "ProductionLayout",
    "resolve_production_config", "resolve_production_layout", "MonthBalancedHPOSample",
    "select_month_balanced_hpo_sample",
]
