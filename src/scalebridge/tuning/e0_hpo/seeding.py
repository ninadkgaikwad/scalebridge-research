from __future__ import annotations

"""Deterministic E0-8 study/trial seed derivation."""

import hashlib


def derive_trial_seed(study_seed: int, trial_number: int, study_id: str) -> int:
    """Preserve the ratified/original E0-8 v1 absolute-trial seed contract."""
    raw = f"{int(study_seed)}::{int(trial_number)}::{study_id}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def derive_sampler_segment_seed(study_seed: int, start_trial_number: int, study_id: str) -> int:
    """Derive a deterministic sampler seed for one resumed study segment.

    A resumed Optuna study recreates its sampler object. Reusing ``study_seed``
    would replay the sampler's first pseudo-random segment for samplers whose
    RNG state is not persisted by storage. Binding the segment seed to the
    current trial count prevents deterministic replay while keeping the resumed
    study reproducible from immutable study identity + history length.

    Fresh-study behavior remains unchanged: the initial sampler still receives
    the original ``study_seed`` directly.
    """
    raw = (
        f"sampler_segment::{int(study_seed)}::{int(start_trial_number)}::{study_id}"
    ).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)
