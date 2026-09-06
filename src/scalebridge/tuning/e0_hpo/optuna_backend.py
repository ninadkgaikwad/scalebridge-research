from __future__ import annotations

"""Optuna-backed study creation and scientifically compatible resume."""

from dataclasses import dataclass
from typing import Any

from .contracts import HPOContractError, IncompatibleResumeError, StudySpec
from .seeding import derive_sampler_segment_seed


FINGERPRINT_ATTR = "scalebridge_e08_study_fingerprint"
SCHEMA_ATTR = "scalebridge_e08_schema_version"
STUDY_ID_ATTR = "scalebridge_e08_study_id"


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError("E0-8 requires Optuna. Install the ScaleBridge environment.") from exc
    return optuna


def build_sampler(name: str, *, seed: int, multi_objective: bool) -> Any:
    optuna = _require_optuna()
    key = str(name).strip().lower()
    if key == "auto":
        key = "nsga2" if multi_objective else "tpe"
    if key == "tpe":
        return optuna.samplers.TPESampler(seed=int(seed))
    if key == "random":
        return optuna.samplers.RandomSampler(seed=int(seed))
    if key == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=int(seed))
    raise HPOContractError(f"Unsupported E0-8 sampler {name!r}")


def build_pruner(name: str) -> Any:
    optuna = _require_optuna()
    key = str(name).strip().lower()
    if key in {"none", "nop", "disabled"}:
        return optuna.pruners.NopPruner()
    if key == "median":
        return optuna.pruners.MedianPruner()
    if key in {"successive_halving", "sha"}:
        return optuna.pruners.SuccessiveHalvingPruner()
    if key == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise HPOContractError(f"Unsupported E0-8 pruner {name!r}")


@dataclass(frozen=True)
class OptunaStudyConfig:
    n_trials: int = 50
    timeout_seconds: float | None = None
    storage_url: str | None = None
    resume: bool = False
    sampler_name: str = "auto"
    pruner_name: str = "none"

    def __post_init__(self) -> None:
        if int(self.n_trials) < 1:
            raise HPOContractError("n_trials must be >= 1")
        if self.timeout_seconds is not None and float(self.timeout_seconds) <= 0:
            raise HPOContractError("timeout_seconds must be positive when provided")
        if bool(self.resume) and not self.storage_url:
            raise HPOContractError(
                "E0-8 resume requires a persistent Optuna storage_url; in-memory studies cannot be resumed"
            )


def _existing_study_or_raise(optuna: Any, spec: StudySpec, config: OptunaStudyConfig) -> Any:
    if not config.storage_url:
        raise HPOContractError("E0-8 resume requires persistent Optuna storage")
    summaries = optuna.study.get_all_study_summaries(storage=config.storage_url)
    names = {str(item.study_name) for item in summaries}
    if spec.study_name not in names:
        raise IncompatibleResumeError(
            "E0-8 resume requested but the persistent Optuna study does not exist: "
            f"{spec.study_name!r}"
        )
    return optuna.load_study(study_name=spec.study_name, storage=config.storage_url)


def _assert_fingerprint_compatible(study: Any, spec: StudySpec) -> None:
    existing = study.user_attrs.get(FINGERPRINT_ATTR)
    if existing is None:
        if len(study.trials) > 0:
            raise IncompatibleResumeError(
                "Existing Optuna study has trials but no ScaleBridge E0-8 fingerprint"
            )
        return
    if str(existing) != spec.fingerprint:
        raise IncompatibleResumeError(
            "E0-8 study fingerprint mismatch; refusing scientifically incompatible resume. "
            f"stored={existing} requested={spec.fingerprint}"
        )


def _record_sampler_segment(study: Any, *, start_trial: int, seed: int, sampler_name: str) -> None:
    attr = "scalebridge_e08_sampler_segments"
    segments = list(study.user_attrs.get(attr, []))
    record = {
        "start_trial": int(start_trial),
        "seed": int(seed),
        "sampler_name": str(sampler_name),
    }
    if record not in segments:
        segments.append(record)
        study.set_user_attr(attr, segments)


def create_or_resume_study(spec: StudySpec, config: OptunaStudyConfig) -> Any:
    optuna = _require_optuna()
    directions = [item.direction.value for item in spec.objectives]
    multi = len(directions) > 1

    if config.resume:
        existing_study = _existing_study_or_raise(optuna, spec, config)
        _assert_fingerprint_compatible(existing_study, spec)
        start_trial = len(existing_study.trials)
        sampler_seed = derive_sampler_segment_seed(
            spec.study_seed, start_trial, spec.study_id
        )
    else:
        start_trial = 0
        sampler_seed = int(spec.study_seed)

    sampler = build_sampler(config.sampler_name, seed=sampler_seed, multi_objective=multi)
    pruner = build_pruner(config.pruner_name)

    kwargs: dict[str, Any] = {
        "study_name": spec.study_name,
        "storage": config.storage_url,
        "sampler": sampler,
        "pruner": pruner,
        "load_if_exists": bool(config.resume),
    }
    if multi:
        kwargs["directions"] = directions
    else:
        kwargs["direction"] = directions[0]

    study = optuna.create_study(**kwargs)
    existing = study.user_attrs.get(FINGERPRINT_ATTR)
    if existing is None:
        if config.resume and len(study.trials) > 0:
            raise IncompatibleResumeError(
                "Existing Optuna study has trials but no ScaleBridge E0-8 fingerprint"
            )
        study.set_user_attr(FINGERPRINT_ATTR, spec.fingerprint)
        study.set_user_attr(SCHEMA_ATTR, spec.schema_version)
        study.set_user_attr(STUDY_ID_ATTR, spec.study_id)
        study.set_user_attr("scalebridge_method_id", spec.method_id)
        study.set_user_attr("scalebridge_provider_version", spec.provider_version)
    elif str(existing) != spec.fingerprint:
        raise IncompatibleResumeError(
            "E0-8 study fingerprint mismatch; refusing scientifically incompatible resume. "
            f"stored={existing} requested={spec.fingerprint}"
        )

    _record_sampler_segment(
        study,
        start_trial=start_trial,
        seed=sampler_seed,
        sampler_name=config.sampler_name,
    )
    return study

