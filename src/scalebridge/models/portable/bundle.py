from __future__ import annotations

"""Portable E0-7 bundle writer/loader with optional integrity validation."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Mapping

from .contracts import BundleFileRecord, PortableModelError, PortableModelManifest


BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_bundle_relative(path: str) -> PurePosixPath:
    rel = PurePosixPath(str(path).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise PortableModelError(f"Invalid bundle-relative path: {path!r}")
    if rel.as_posix() == BUNDLE_MANIFEST_FILENAME:
        raise PortableModelError("Embedded artifact cannot overwrite bundle_manifest.json")
    return rel


def _copy_source(source: Path, destination: Path) -> None:
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        raise PortableModelError(f"Embedded artifact source does not exist: {source}")


def _file_records(root: Path) -> tuple[BundleFileRecord, ...]:
    records: list[BundleFileRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == BUNDLE_MANIFEST_FILENAME:
            continue
        rel = path.relative_to(root).as_posix()
        records.append(
            BundleFileRecord(
                relative_path=rel,
                sha256=_sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(records)


def write_portable_model_bundle(
    output_dir: str | Path,
    manifest: PortableModelManifest,
    *,
    embedded_artifacts: Mapping[str, str | Path] | None = None,
    overwrite: bool = False,
) -> "PortableModelBundle":
    """Materialize an immutable bundle directory from static fitted artifacts."""

    root = Path(output_dir)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Portable model bundle already exists: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)

    try:
        for relative, source in dict(embedded_artifacts or {}).items():
            rel = _safe_bundle_relative(relative)
            _copy_source(Path(source), root / Path(*rel.parts))

        finalized = replace(manifest, files=_file_records(root))
        payload = finalized.to_dict()
        (root / BUNDLE_MANIFEST_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return PortableModelBundle.load(root, validate_integrity=True)


class PortableModelBundle:
    def __init__(self, root: Path, manifest: PortableModelManifest) -> None:
        self.root = root
        self.manifest = manifest

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        validate_integrity: bool = False,
    ) -> "PortableModelBundle":
        bundle_root = Path(root)
        path = bundle_root / BUNDLE_MANIFEST_FILENAME
        if not path.is_file():
            raise PortableModelError(f"Portable bundle manifest is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PortableModelError(f"Unable to read portable model manifest: {path}") from exc
        manifest = PortableModelManifest.from_dict(payload)
        bundle = cls(bundle_root.resolve(), manifest)
        if validate_integrity:
            bundle.validate_integrity()
        return bundle

    def path(self, relative_path: str) -> Path:
        rel = _safe_bundle_relative(relative_path)
        path = (self.root / Path(*rel.parts)).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise PortableModelError("Bundle path escapes bundle root") from exc
        return path

    def validate_integrity(self) -> None:
        expected = {item.relative_path: item for item in self.manifest.files}
        actual = {item.relative_path: item for item in _file_records(self.root)}
        if set(expected) != set(actual):
            raise PortableModelError(
                "Portable bundle file inventory mismatch: "
                f"missing={sorted(set(expected) - set(actual))}, "
                f"extra={sorted(set(actual) - set(expected))}"
            )
        for relative, wanted in expected.items():
            found = actual[relative]
            if found.sha256 != wanted.sha256 or found.size_bytes != wanted.size_bytes:
                raise PortableModelError(f"Portable bundle integrity mismatch: {relative}")
