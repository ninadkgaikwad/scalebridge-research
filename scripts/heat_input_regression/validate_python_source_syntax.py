#!/usr/bin/env python
"""Validate Python source syntax without creating or replacing .pyc files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _iter_python_files(paths: list[str]):
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".py":
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path
        elif path.is_dir():
            for candidate in sorted(path.rglob("*.py")):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", required=True)
    args = parser.parse_args()

    files = list(_iter_python_files(args.paths))
    failures: list[tuple[Path, BaseException]] = []

    print(f"python_source_file_count: {len(files)}")

    for index, path in enumerate(files, start=1):
        try:
            source = path.read_text(encoding="utf-8-sig")
            compile(source, str(path), "exec", dont_inherit=True)
            print(f"[{index}/{len(files)}] PASS {path}")
        except BaseException as exc:
            failures.append((path, exc))
            print(f"[{index}/{len(files)}] FAIL {path}: {exc}", file=sys.stderr)

    print("=" * 100)
    print("SOURCE SYNTAX CHECK SUMMARY")
    print("=" * 100)
    print(f"checked_file_count: {len(files)}")
    print(f"passed_file_count: {len(files) - len(failures)}")
    print(f"failed_file_count: {len(failures)}")

    if failures:
        for path, exc in failures:
            print(f"- {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
