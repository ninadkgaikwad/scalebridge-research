from __future__ import annotations

from pathlib import Path
import sys

PAPER_ROOT = Path(__file__).resolve().parents[1]
SRC = PAPER_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinode_epsr.production.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
