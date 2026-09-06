"""Phase C Heat-Input Regression Dash workspace."""

from .page import build_layout
from .callbacks import register_heat_input_callbacks

register_heat_input_callbacks()

__all__ = ["build_layout"]
