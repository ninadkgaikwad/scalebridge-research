"""Narrow compatibility fixes for known opyplus 2.0.7 defects.

Opyplus 2.0.7 contains an indentation defect in ``idd_debug.correct_idd``:
the repair for ``HeatPump:PlantLoop:EIR:Heating``, an object introduced in
newer EnergyPlus schemas, executes for every IDD version. Loading an
EnergyPlus 9.0.1 IDD therefore raises ``KeyError`` after all applicable
9.0.1 repairs have already completed.

ScaleBridge installs a process-local wrapper around that function. The wrapper
suppresses only the exact missing-table error on IDD versions below 9.6. All
other exceptions and versions retain the original opyplus behavior. No file in
the Python environment or EnergyPlus installation is modified.
"""

from __future__ import annotations

from typing import Any, Callable, Final


MISSING_HEATPUMP_TABLE: Final = "heatpump_plantloop_eir_heating"
HEATPUMP_REPAIR_MINIMUM_VERSION: Final = (9, 6, 0)
PATCH_MARKER: Final = "__scalebridge_opyplus_207_idd_patch__"


def install_opyplus_207_idd_compatibility() -> bool:
    """Install the EnergyPlus pre-9.6 IDD compatibility wrapper.

    Returns
    -------
    bool
        ``True`` when this call installs the wrapper and ``False`` when the
        wrapper was already installed.

    Notes
    -----
    ``opyplus.idd.idd`` imports ``correct_idd`` directly, so replacing the
    symbol in that module is required. The source module symbol is also
    replaced to keep subsequent imports and diagnostics consistent.
    """
    import opyplus
    from opyplus.idd import idd as idd_module
    from opyplus.idd import idd_debug

    if str(opyplus.__version__) != "2.0.7":
        return False

    current = idd_module.correct_idd
    if getattr(current, PATCH_MARKER, False):
        return False

    compatible = _build_compatible_correct_idd(current)
    idd_module.correct_idd = compatible
    idd_debug.correct_idd = compatible
    return True


def _build_compatible_correct_idd(
    original: Callable[[Any], None],
) -> Callable[[Any], None]:
    """Wrap opyplus IDD repair while preserving unrelated failures."""

    def compatible_correct_idd(idd: Any) -> None:
        """Run opyplus repairs and ignore only its known pre-9.6 lookup bug."""
        try:
            original(idd)
        except KeyError as exc:
            is_known_missing_table = exc.args == (MISSING_HEATPUMP_TABLE,)
            is_older_idd = tuple(idd.version) < HEATPUMP_REPAIR_MINIMUM_VERSION
            if is_known_missing_table and is_older_idd:
                return
            raise

    setattr(compatible_correct_idd, PATCH_MARKER, True)
    return compatible_correct_idd
