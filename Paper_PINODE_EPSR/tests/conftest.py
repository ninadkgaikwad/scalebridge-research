"""Pytest path bootstrap for the paper-local package.

When pytest is launched through the Windows ``pytest.exe`` console entry point,
the ScaleBridge repository root is not guaranteed to be present on
``sys.path``.  The paper package is intentionally not installed into the
environment, so tests add the repository root explicitly.

This file changes test discovery/import behavior only.  It does not modify
runtime, model, data, or scientific contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
