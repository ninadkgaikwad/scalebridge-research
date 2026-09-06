from __future__ import annotations

from pathlib import Path
import hashlib
import re


TIMESTEP_RE = re.compile(
    r"(?ism)(^\s*Timestep\s*,\s*)(\d+)(\s*;)"
)


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_300s_runtime_idf(
    *,
    authoritative_idf: str | Path,
    runtime_idf: str | Path,
    expected_source_sha256: str,
) -> Path:
    """
    Copy the authoritative IDF into the DATA run directory and change only the
    scratch copy from Timestep=6 to Timestep=12.

    The source file is SHA-verified before and after the operation.
    """

    source = Path(authoritative_idf).resolve()
    target = Path(runtime_idf).resolve()

    actual = sha256_file(source)
    if actual.lower() != expected_source_sha256.lower():
        raise RuntimeError(
            "Authoritative IDF SHA256 mismatch.\n"
            f"Expected: {expected_source_sha256}\n"
            f"Actual:   {actual}"
        )

    text = source.read_text(encoding="utf-8", errors="replace")
    matches = list(TIMESTEP_RE.finditer(text))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Timestep object; found {len(matches)}."
        )

    old = int(matches[0].group(2))
    if old == 12:
        runtime_text = text
    elif old == 6:
        runtime_text, n = TIMESTEP_RE.subn(
            r"\g<1>12\g<3>",
            text,
            count=1,
        )
        if n != 1:
            raise RuntimeError("Could not create 300-s runtime IDF.")
    else:
        raise RuntimeError(
            f"Expected Timestep 6 or 12; found {old}."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(runtime_text, encoding="utf-8")

    after = sha256_file(source)
    if after.lower() != expected_source_sha256.lower():
        raise RuntimeError("Authoritative IDF changed unexpectedly.")

    return target
