from __future__ import annotations

"""Method-provider boundary for E0-8 generic HPO orchestration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import (
    CompletedTrialView,
    HPOContractError,
    HPODataSelection,
    ObjectiveSpec,
    TrialEvaluation,
)


class TrialSuggester:
    """Small method-neutral suggestion API backed by one Optuna Trial."""

    def __init__(self, optuna_trial: Any) -> None:
        self._trial = optuna_trial

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        return float(self._trial.suggest_float(name, low, high, step=step, log=log))

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        return int(self._trial.suggest_int(name, low, high, step=step, log=log))

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        return self._trial.suggest_categorical(name, list(choices))

    def suggest_bool(self, name: str) -> bool:
        return bool(self._trial.suggest_categorical(name, [False, True]))


@dataclass
class TrialContext:
    """Per-trial runtime context passed to a method-specific provider."""

    trial_number: int
    trial_seed: int
    data_selection: HPODataSelection
    pruning_allowed: bool
    objective_count: int
    _optuna_trial: Any

    def report_and_prune(self, value: float, step: int) -> None:
        """Report one intermediate scalar and prune if configured to do so."""
        if not self.pruning_allowed:
            raise HPOContractError("This provider did not authorize pruning")
        if self.objective_count != 1:
            raise HPOContractError(
                "E0-8 v1 intermediate pruning reports are supported only for single-objective studies"
            )
        self._optuna_trial.report(float(value), int(step))
        if self._optuna_trial.should_prune():
            import optuna

            raise optuna.TrialPruned(f"Pruned at provider-reported step {step}")

    def prune(self, reason: str = "provider-declared prune") -> None:
        if not self.pruning_allowed:
            raise HPOContractError("This provider did not authorize pruning")
        import optuna

        raise optuna.TrialPruned(str(reason))


class BaseHPOProvider(ABC):
    """Generic HPO-provider contract implemented later by E.1/E.2/E.3/E.4.

    The base class intentionally defines no model-family hyperparameter names.
    """

    @property
    @abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def method_family(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_version(self) -> str:
        raise NotImplementedError

    @property
    def pruning_supported(self) -> bool:
        return False

    def validate_hpo_configuration(
        self,
        *,
        sampler_name: str,
        pruner_name: str,
        objective_count: int,
    ) -> None:
        """Optionally restrict generic HPO capabilities for this method.

        E0-8 owns the generic registries, while E.x owns scientific
        compatibility.  The default contract rejects an active pruner unless
        the provider explicitly declares pruning support.
        """
        del sampler_name
        if int(objective_count) < 1:
            raise HPOContractError("At least one objective is required")
        if str(pruner_name).strip().lower() not in {"none", "nop", "disabled"}:
            if not self.pruning_supported:
                raise HPOContractError(
                    "This method provider does not authorize pruning; select pruner_name='none'"
                )

    @abstractmethod
    def search_space_snapshot(self) -> Mapping[str, Any]:
        """Return the provider-owned immutable search-space declaration."""
        raise NotImplementedError

    @abstractmethod
    def objective_specs(self) -> Sequence[ObjectiveSpec]:
        """Return provider-owned objective names/directions."""
        raise NotImplementedError

    @abstractmethod
    def data_selection(self, phase_d_contract: Any) -> HPODataSelection:
        """Return exact provider-owned HPO selection derived from Phase-D TRAIN."""
        raise NotImplementedError

    @abstractmethod
    def suggest_hyperparameters(self, suggester: TrialSuggester) -> Mapping[str, Any]:
        """Use the generic suggester to instantiate one conditional trial space."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_trial(
        self,
        hyperparameters: Mapping[str, Any],
        context: TrialContext,
    ) -> TrialEvaluation:
        """Perform method-specific fitting/scoring for one HPO trial."""
        raise NotImplementedError

    def select_final_multiobjective_trial(
        self,
        pareto_trials: Sequence[CompletedTrialView],
    ) -> int | None:
        """Optionally choose one Pareto trial.  Default: preserve Pareto set only."""
        return None
