"""Configuration model for the Dash application shell."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DashAppConfig:
    """Runtime configuration for the application shell."""

    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False
    url_base_pathname: str = "/"
    suppress_callback_exceptions: bool = True

    @classmethod
    def from_environment(cls) -> "DashAppConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("BGIRS_HOST", "127.0.0.1"),
            port=int(os.getenv("BGIRS_PORT", "8050")),
            debug=os.getenv("BGIRS_DEBUG", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            url_base_pathname=os.getenv("BGIRS_URL_BASE", "/"),
        )
