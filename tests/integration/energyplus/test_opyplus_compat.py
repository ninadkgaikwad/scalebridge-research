"""Tests for narrowly scoped opyplus 2.0.7 compatibility behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scalebridge.integration.energyplus.idf.opyplus_compat import (
    MISSING_HEATPUMP_TABLE,
    _build_compatible_correct_idd,
)


def test_compatibility_suppresses_known_error_for_energyplus_901() -> None:
    """The misplaced 9.6 heat-pump repair must not block a 9.0.1 IDD."""

    def defective_correct_idd(idd: object) -> None:
        """Represent the final erroneous lookup in opyplus 2.0.7."""
        raise KeyError(MISSING_HEATPUMP_TABLE)

    wrapper = _build_compatible_correct_idd(defective_correct_idd)

    wrapper(SimpleNamespace(version=(9, 0, 1)))


def test_compatibility_preserves_known_error_for_energyplus_96() -> None:
    """A missing heat-pump table in a schema requiring it must still fail."""

    def defective_correct_idd(idd: object) -> None:
        """Represent a genuinely invalid EnergyPlus 9.6 IDD."""
        raise KeyError(MISSING_HEATPUMP_TABLE)

    wrapper = _build_compatible_correct_idd(defective_correct_idd)

    with pytest.raises(KeyError, match=MISSING_HEATPUMP_TABLE):
        wrapper(SimpleNamespace(version=(9, 6, 0)))


def test_compatibility_preserves_unrelated_key_errors() -> None:
    """The wrapper must never hide unrelated IDD corruption."""

    def defective_correct_idd(idd: object) -> None:
        """Raise an unrelated table lookup failure."""
        raise KeyError("unrelated_table")

    wrapper = _build_compatible_correct_idd(defective_correct_idd)

    with pytest.raises(KeyError, match="unrelated_table"):
        wrapper(SimpleNamespace(version=(9, 0, 1)))
