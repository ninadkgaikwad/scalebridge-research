from __future__ import annotations

"""
SSL-safe MLflow CLI launcher for Windows environments where
ssl.create_default_context() fails while reading the Windows certificate store.

This patches ssl.create_default_context to use certifi's CA bundle by default,
then delegates to the normal MLflow CLI.
"""

import ssl

import certifi


_ORIGINAL_CREATE_DEFAULT_CONTEXT = ssl.create_default_context


def _create_default_context_with_certifi(
    purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
    *,
    cafile: str | None = None,
    capath: str | None = None,
    cadata: str | bytes | None = None,
) -> ssl.SSLContext:
    if cafile is None and capath is None and cadata is None:
        cafile = certifi.where()

    return _ORIGINAL_CREATE_DEFAULT_CONTEXT(
        purpose=purpose,
        cafile=cafile,
        capath=capath,
        cadata=cadata,
    )


ssl.create_default_context = _create_default_context_with_certifi

from mlflow.cli import cli  # noqa: E402


if __name__ == "__main__":
    cli()