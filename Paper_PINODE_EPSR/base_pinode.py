from __future__ import annotations

"""Backward-compatible import shim.

Canonical implementation: ``Paper_PINODE_EPSR/src/pinode_epsr/methods/base_pinode.py``.
New code should import ``pinode_epsr``.
"""

from pinode_epsr.methods.base_pinode import *  # noqa: F401,F403
