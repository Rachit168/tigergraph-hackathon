"""Provider-neutral LLM settings. Secrets stay in the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

LLM_SECRET_KEYS = ("LLM_API_KEY", "OPENAI_API_KEY")
AUTH_HEADER_AUTHORIZATION = "Authorization"
AUTH_HEADER_CF_AIG = "cf-aig-authorization"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_S = 30.0

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_s: float
    max_tokens: int
    auth_header: str = ""

    @property
    def request_auth_header(self) -> str:
        return resolve_auth_header(self.auth_header, self.base_url)

    @property
    def configured(self) -> bool:
        if self.provider != "openai_compatible":
            return False
        return bool(self.base_url and self.model and self.api_key)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.provider in {"", "none", "deterministic"}:
            return errors
        if self.provider != "openai_compatible":
            errors.append(f"unsupported LLM_PROVIDER: {self.provider}")
            return errors
        if not self.base_url:
            errors.append("LLM_BASE_URL is missing")
        if not self.model:
            errors.append("LLM_MODEL is missing")
        if not self.api_key:
            errors.append("LLM_API_KEY is missing")
        return errors

    def redacted(self) -> dict[str, str | bool | float | int]:
        return {
            "provider": self.provider,
            "base_url": _redact_url(self.base_url),
            "model": self.model,
            "timeout_s": self.timeout_s,
            "max_tokens": self.max_tokens,
            "auth_header": self.request_auth_header,
            "api_key_set": bool(self.api_key),
            "configured": self.configured,
        }


def load_llm_settings(path: str | Path | None = None) -> LLMSettings:
    _load_env_file(path)
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env", override=False)
    except ImportError:
        pass
    provider = (os.environ.get("LLM_PROVIDER") or "none").strip().casefold()
    timeout_raw = (os.environ.get("LLM_TIMEOUT_S") or str(int(DEFAULT_TIMEOUT_S))).strip() or str(
        int(DEFAULT_TIMEOUT_S)
    )
    max_tokens_raw = (os.environ.get("LLM_MAX_TOKENS") or str(DEFAULT_MAX_TOKENS)).strip() or str(
        DEFAULT_MAX_TOKENS
    )
    try:
        timeout_s = float(timeout_raw)
    except ValueError:
        timeout_s = DEFAULT_TIMEOUT_S
    try:
        max_tokens = int(max_tokens_raw)
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS
    return LLMSettings(
        provider=provider,
        base_url=(os.environ.get("LLM_BASE_URL") or "").strip(),
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        model=(os.environ.get("LLM_MODEL") or "").strip(),
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        auth_header=(os.environ.get("LLM_AUTH_HEADER") or "").strip(),
    )


def _load_env_file(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else _REPO_ROOT / ".env"
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


def resolve_auth_header(explicit: str, base_url: str) -> str:
    """Select the HTTP auth header. Explicit LLM_AUTH_HEADER wins over URL inference."""

    folded = (explicit or "").strip().casefold().replace("_", "-")
    if folded in {"cf-aig-authorization", "cfaig-authorization"}:
        return AUTH_HEADER_CF_AIG
    if folded in {"authorization", "bearer"}:
        return AUTH_HEADER_AUTHORIZATION
    if (explicit or "").strip():
        return explicit.strip()
    parsed = urlparse(base_url or "")
    host = (parsed.hostname or "").casefold()
    path = (parsed.path or "").casefold()
    if host == "gateway.ai.cloudflare.com" or host.endswith(".gateway.ai.cloudflare.com"):
        normalized = path.rstrip("/")
        if normalized.endswith("/compat") or "/compat/" in path:
            return AUTH_HEADER_CF_AIG
    return AUTH_HEADER_AUTHORIZATION


def _redact_url(url: str) -> str:
    if not url:
        return ""
    stripped = url.split("://", 1)[-1]
    host = stripped.split("/", 1)[0]
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    host_only = host.split(":", 1)[0]
    parts = host_only.split(".")
    if len(parts) <= 2:
        return "(set)"
    return f"{parts[0][:1]}***.{'.'.join(parts[-2:])}"
