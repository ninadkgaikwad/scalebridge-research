"""Compatibility package for the in-repository EPSR implementation.

Canonical source code is organized under ``Paper_PINODE_EPSR/src/pinode_epsr``.
This compatibility package exposes the canonical package while preserving all
historical ``Paper_PINODE_EPSR.*`` imports during the paper-development period.
"""
from pathlib import Path as _Path
import sys as _sys

_SRC = _Path(__file__).resolve().parent / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from pinode_epsr import *  # noqa: F401,F403,E402
