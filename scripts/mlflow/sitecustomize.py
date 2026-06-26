from __future__ import annotations

"""
Optional ScaleBridge SSL patch.

When SCALEBRIDGE_PATCH_SSL_CERTIFI=1, this makes Python use certifi's CA bundle
for ssl.create_default_context() when no explicit cafile/capath/cadata is given.

This fixes Windows environments where reading the Windows certificate store fails
with:
    ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] not enough data
"""

import os
import ssl


if os.environ.get("SCALEBRIDGE_PATCH_SSL_CERTIFI") == "1":
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