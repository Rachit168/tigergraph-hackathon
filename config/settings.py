"""TigerGraph connection settings from the environment. Secrets are never printed."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ingestion.paths import REPO_ROOT

SECRET_KEYS = (
    "TG_PASSWORD",
    "TG_API_TOKEN",
    "TG_JWT_TOKEN",
    "TG_SECRET",
    "TG_GSQL_SECRET",
)


@dataclass(frozen=True)
class TigerGraphSettings:
    host: str
    graphname: str
    username: str
    password: str
    api_token: str
    jwt_token: str
    secret: str
    restpp_port: str
    gs_port: str
    ssl_port: str
    tg_cloud: bool
    cert_path: str

    @property
    def configured(self) -> bool:
        return bool(self.host) and bool(self.graphname) and self.has_auth

    @property
    def has_auth(self) -> bool:
        if self.api_token or self.jwt_token or self.secret:
            return True
        return bool(self.username and self.password)

    @property
    def auth_method(self) -> str:
        if self.api_token:
            return "api_token"
        if self.jwt_token:
            return "jwt_token"
        if self.secret:
            return "gsql_secret"
        if self.username and self.password:
            return "password"
        return "none"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.host:
            errors.append("TG_HOST is missing")
        if not self.graphname:
            errors.append("TG_GRAPHNAME is missing")
        if not self.has_auth:
            errors.append("no TigerGraph auth: set TG_API_TOKEN, TG_JWT_TOKEN, TG_SECRET, or TG_USERNAME+TG_PASSWORD")
        if self.host and "://" not in self.host:
            errors.append("TG_HOST must include a URI scheme such as https://")
        return errors

    def redacted(self) -> dict[str, str | bool]:
        return {
            "host": _redact_host(self.host),
            "graphname": self.graphname,
            "username": self.username or "",
            "auth_method": self.auth_method,
            "restpp_port": self.restpp_port,
            "gs_port": self.gs_port,
            "ssl_port": self.ssl_port,
            "tg_cloud": self.tg_cloud,
            "cert_path_set": bool(self.cert_path),
            "configured": self.configured,
        }


def load_env_file(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings(path: str | Path | None = None) -> TigerGraphSettings:
    load_env_file(path)
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass
    cloud_raw = os.environ.get("TG_TGCLOUD", "").strip().casefold()
    return TigerGraphSettings(
        host=os.environ.get("TG_HOST", "").strip(),
        graphname=os.environ.get("TG_GRAPHNAME", "OlympicGraph").strip() or "OlympicGraph",
        username=os.environ.get("TG_USERNAME", "").strip(),
        password=os.environ.get("TG_PASSWORD", ""),
        api_token=os.environ.get("TG_API_TOKEN", "") or os.environ.get("TG_TOKEN", ""),
        jwt_token=os.environ.get("TG_JWT_TOKEN", ""),
        secret=os.environ.get("TG_SECRET", "") or os.environ.get("TG_GSQL_SECRET", ""),
        restpp_port=os.environ.get("TG_RESTPP_PORT", "9000").strip() or "9000",
        gs_port=os.environ.get("TG_GS_PORT", "14240").strip() or "14240",
        ssl_port=os.environ.get("TG_SSL_PORT", "443").strip() or "443",
        tg_cloud=cloud_raw in {"1", "true", "yes", "y"},
        cert_path=os.environ.get("TG_CERT_PATH", "").strip(),
    )


def _redact_host(host: str) -> str:
    if not host:
        return ""
    stripped = host.split("://", 1)[-1]
    name = stripped.split("/", 1)[0]
    if "@" in name:
        name = name.rsplit("@", 1)[-1]
    host_only = name.split(":", 1)[0]
    parts = host_only.split(".")
    if len(parts) <= 2:
        return "(set)"
    return f"{parts[0][:1]}***.{'.'.join(parts[-2:])}"
