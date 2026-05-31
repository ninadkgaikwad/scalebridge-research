from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    paper_id: str
    experiment_name: str
    artifact_dir: Path
    metadata: dict[str, Any]
