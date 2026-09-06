from __future__ import annotations

"""Backward-compatible import shim.

Canonical implementation: ``Paper_PINODE_EPSR/src/pinode_epsr/backends/neuromancer.py``.
New code should import ``pinode_epsr``.
"""

from pinode_epsr.backends.neuromancer import *  # noqa: F401,F403
