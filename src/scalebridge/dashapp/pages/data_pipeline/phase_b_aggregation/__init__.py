"""Phase B Aggregation Dash workspace."""

from .page import build_layout
from .callbacks import register_aggregation_callbacks

register_aggregation_callbacks()

__all__ = ["build_layout"]
