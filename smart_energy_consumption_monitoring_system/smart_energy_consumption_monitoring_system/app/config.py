"""Application configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Parse a truthy/falsey environment variable."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Base Flask configuration loaded from environment variables."""

    APP_ENV = os.environ.get("APP_ENV", os.environ.get("FLASK_ENV", "production")).strip().lower()
    IS_DEVELOPMENT = APP_ENV == "development"

    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me")
    DEBUG = _as_bool(os.environ.get("FLASK_DEBUG"), default=IS_DEVELOPMENT)
    HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
    PORT = int(os.environ.get("FLASK_PORT", "5000"))

    # Dev experience
    TEMPLATES_AUTO_RELOAD = _as_bool(
        os.environ.get("TEMPLATES_AUTO_RELOAD"),
        default=IS_DEVELOPMENT,
    )
    EXPLAIN_TEMPLATE_LOADING = _as_bool(
        os.environ.get("EXPLAIN_TEMPLATE_LOADING"),
        default=False,
    )
    # Disable static caching in development so CSS/JS changes reflect immediately.
    SEND_FILE_MAX_AGE_DEFAULT = 0 if IS_DEVELOPMENT else None

    # Session config
    SESSION_TYPE = os.environ.get("SESSION_TYPE", "filesystem")
    SESSION_PERMANENT = False
    # Keep disabled by default to avoid signer/cookie incompatibilities in this app setup.
    SESSION_USE_SIGNER = _as_bool(os.environ.get("SESSION_USE_SIGNER"), default=False)
    SESSION_FILE_DIR = os.environ.get(
        "SESSION_FILE_DIR",
        str(Path("flask_session").resolve()),
    )

    # Security cookies
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _as_bool(
        os.environ.get("SESSION_COOKIE_SECURE"),
        default=False,
    )

    # Upload guard
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "16")) * 1024 * 1024
