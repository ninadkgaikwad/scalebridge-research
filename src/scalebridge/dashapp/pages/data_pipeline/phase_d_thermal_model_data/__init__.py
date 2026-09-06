"""Phase D Thermal-Model Data Dash workspace."""

from .page import build_layout
from .callbacks import register_phase_d_callbacks

register_phase_d_callbacks()

__all__ = ["build_layout"]
