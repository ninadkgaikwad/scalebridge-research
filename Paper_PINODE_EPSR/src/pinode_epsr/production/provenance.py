from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_source_hash(paper_root: Path) -> str:
    """Hash current paper source/config text without depending on Git."""
    targets: list[Path] = []
    for base in (paper_root / "src" / "pinode_epsr", paper_root / "configs"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".toml"}:
                targets.append(path)
    h = hashlib.sha256()
    for path in sorted(targets, key=lambda p: str(p.relative_to(paper_root)).lower()):
        rel = path.relative_to(paper_root).as_posix().encode("utf-8")
        h.update(rel); h.update(b"\0"); h.update(path.read_bytes()); h.update(b"\0")
    return h.hexdigest()


def environment_manifest() -> dict[str, object]:
    out: dict[str, object] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "torch", "neuromancer", "optuna", "pyarrow"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            out[name] = f"IMPORT_FAILED: {exc}"
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = torch.version.cuda
        out["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return out


def stable_id(prefix: str, payload: dict[str, object], *, length: int = 12) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:length]
    return f"{prefix}__{digest}"
