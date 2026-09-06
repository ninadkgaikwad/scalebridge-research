from __future__ import annotations

"""E0-7 portable Phase-B/C/D lineage and historical artifact resolution."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Mapping

from .contracts import DataLocator, PortableModelError


DEFAULT_ROOT_ENVIRONMENT = {
    "generated_data": "SCALEBRIDGE_GENERATED_DATA_ROOT",
}


@dataclass(frozen=True)
class DataRootRegistry:
    roots: Mapping[str, Path]

    def __post_init__(self) -> None:
        normalized: dict[str, Path] = {}
        for alias, root in self.roots.items():
            token = str(alias).strip()
            if not token:
                raise PortableModelError("Data-root alias cannot be empty")
            normalized[token] = Path(root).expanduser().resolve()
        object.__setattr__(self, "roots", normalized)

    @classmethod
    def from_environment(
        cls,
        aliases: Mapping[str, str] | None = None,
        *,
        require_all: bool = False,
    ) -> "DataRootRegistry":
        roots: dict[str, Path] = {}
        for alias, env_name in dict(aliases or DEFAULT_ROOT_ENVIRONMENT).items():
            value = os.environ.get(env_name, "").strip()
            if not value:
                if require_all:
                    raise PortableModelError(
                        f"Required data-root environment variable {env_name!r} is not set"
                    )
                continue
            roots[str(alias)] = Path(value)
        return cls(roots)

    def resolve(
        self,
        locator: DataLocator,
        *,
        must_exist: bool = False,
        verify_sha256: bool = False,
    ) -> Path:
        try:
            root = self.roots[locator.root_alias]
        except KeyError as exc:
            raise PortableModelError(
                f"No registered root for alias {locator.root_alias!r}"
            ) from exc
        path = (root / Path(*locator.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PortableModelError("Resolved lineage path escapes registered data root") from exc
        if must_exist and not path.exists():
            raise PortableModelError(f"Resolved upstream artifact does not exist: {path}")
        if verify_sha256 and locator.sha256 is not None:
            if not path.is_file():
                raise PortableModelError("SHA256 verification requires a file locator")
            actual = sha256_file(path)
            if actual != locator.sha256:
                raise PortableModelError(
                    f"Upstream artifact SHA256 mismatch for {path}: {actual} != {locator.sha256}"
                )
        return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locator_from_path(
    path: str | Path,
    *,
    root_alias: str,
    root: str | Path,
    stage: str,
    artifact_kind: str,
    identifiers: Mapping[str, str] | None = None,
    include_sha256: bool = False,
    required_for_historical_replay: bool = False,
) -> DataLocator:
    root_path = Path(root).expanduser().resolve()
    artifact = Path(path).expanduser().resolve()
    try:
        relative = artifact.relative_to(root_path)
    except ValueError as exc:
        raise PortableModelError(
            f"Artifact {artifact} is outside registered root {root_path}"
        ) from exc
    digest = sha256_file(artifact) if include_sha256 and artifact.is_file() else None
    return DataLocator(
        stage=stage,
        root_alias=root_alias,
        relative_path=relative.as_posix(),
        artifact_kind=artifact_kind,
        identifiers=dict(identifiers or {}),
        sha256=digest,
        required_for_historical_replay=required_for_historical_replay,
    )
