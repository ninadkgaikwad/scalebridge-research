# -*- coding: utf-8 -*-
"""Thin CLI wrapper for ScaleBridge P1 aggregation runner."""

from __future__ import annotations

from scalebridge.data.aggregation.engine import main


if __name__ == "__main__":
    raise SystemExit(main())