"""Provider-neutral embedding settings. Does not change the generation LLM."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config.llm import LLMSettings, load_llm_settings, resolve_auth_header
from ingestion.paths import REPO_ROOT

DEFAULT_EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMENSION = 384


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    dimension: int
    timeout_s: float
    auth_header: str
    batch_size: int

    @property
    def request_auth_header(self) -> str:
        return resolve_auth_header(self.auth_header, self.base_url)

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model and self.api_key and self.dimension > 0)

    def redacted(self) -> dict[str, str | bool | float | int]:
        return {
            "provider": self.provider,
            "base_url": _redact_url(self.base_url),
            "model": self.model,
            "dimension": self.dimension,
            "timeout_s": self.timeout_s,
            "auth_header": self.request_auth_header,
            "batch_size": self.batch_size,
            "api_key_set": bool(self.api_key),
            "configured": self.configured,
        }


def load_embedding_settings(
    path: str | Path | None = None,
    *,
    llm: LLMSettings | None = None,
) -> EmbeddingSettings:
    llm = llm or load_llm_settings(path)
    model = (os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL).strip()
    base_url = (os.environ.get("EMBEDDING_BASE_URL") or llm.base_url).strip()
    api_key = os.environ.get("EMBEDDING_API_KEY") or llm.api_key
    timeout_raw = (os.environ.get("EMBEDDING_TIMEOUT_S") or str(llm.timeout_s)).strip() or "60"
    dim_raw = (os.environ.get("EMBEDDING_DIMENSION") or str(DEFAULT_EMBEDDING_DIMENSION)).strip()
    batch_raw = (os.environ.get("EMBEDDING_BATCH_SIZE") or "16").strip() or "16"
    try:
        timeout_s = float(timeout_raw)
    except ValueError:
        timeout_s = 60.0
    try:
        dimension = int(dim_raw)
    except ValueError:
        dimension = DEFAULT_EMBEDDING_DIMENSION
    try:
        batch_size = max(1, int(batch_raw))
    except ValueError:
        batch_size = 16
    return EmbeddingSettings(
        provider="openai_compatible",
        base_url=base_url,
        api_key=api_key,
        model=model,
        dimension=dimension,
        timeout_s=timeout_s,
        auth_header=(os.environ.get("EMBEDDING_AUTH_HEADER") or llm.auth_header or "").strip(),
        batch_size=batch_size,
    )


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


def embedding_cache_path(model: str | None = None) -> Path:
    stem = "chunk_embeddings"
    if model:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model)
        stem = f"chunk_embeddings_{safe}"
    return REPO_ROOT / "_research" / "phase13_vector_ablation" / f"{stem}.jsonl"
