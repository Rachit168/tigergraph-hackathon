"""OpenAI-compatible embedding client. Isolated from the chat generation provider."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from answering.provider import ProviderError
from config.embeddings import EmbeddingSettings, load_embedding_settings

USER_AGENT = "tigergraph-graphrag/0.1"
MAX_EMBED_CHARS = 1800


class EmbeddingError(RuntimeError):
    """Fail-closed embedding failure. Callers must not substitute BM25."""


def build_embedding_request(model: str, inputs: str | list[str]) -> dict[str, Any]:
    if isinstance(inputs, str):
        payload_input: str | list[str] = inputs
    else:
        payload_input = list(inputs)
        if not payload_input:
            raise EmbeddingError("embedding input is empty")
    if not model:
        raise EmbeddingError("embedding model is missing")
    return {"model": model, "input": payload_input}


def normalize_embedding_text(text: str) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return " "
    if len(cleaned) <= MAX_EMBED_CHARS:
        return cleaned
    return cleaned[:MAX_EMBED_CHARS]


def parse_embedding_response(raw: Any, *, expected: int) -> list[list[float]]:
    if not isinstance(raw, dict):
        raise EmbeddingError("embedding response is not an object")
    rows = raw.get("data")
    if not isinstance(rows, list) or not rows:
        raise EmbeddingError("embedding response has no data")
    ordered = sorted(rows, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
    vectors: list[list[float]] = []
    for item in ordered:
        if not isinstance(item, dict):
            raise EmbeddingError("embedding row is not an object")
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError("embedding row is missing a vector")
        vectors.append([float(value) for value in vector])
    if len(vectors) != expected:
        raise EmbeddingError(f"embedding count mismatch: expected {expected}, got {len(vectors)}")
    return vectors


def text_digest(text: str) -> str:
    return hashlib.sha256(normalize_embedding_text(text).encode("utf-8")).hexdigest()


class FastEmbedEmbedder:
    """Local BGE-small encoder. Same 384-d COSINE space as the Cloudflare BGE model."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingError("fastembed is not installed") from exc
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self.settings = settings or load_embedding_settings()
        self._model = TextEmbedding(model_name=self.MODEL_NAME)
        self.backend = "fastembed"

    @property
    def model(self) -> str:
        return self.MODEL_NAME

    @property
    def dimension(self) -> int:
        return 384

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        values = [normalize_embedding_text(text) for text in texts]
        if not values:
            return []
        try:
            raw_vectors = self._model.embed(values, batch_size=256)
            vectors = []
            for vector in raw_vectors:
                if hasattr(vector, "tolist"):
                    vectors.append([float(value) for value in vector.tolist()])
                else:
                    vectors.append([float(value) for value in vector])
        except Exception as exc:
            raise EmbeddingError(f"fastembed_failed:{exc.__class__.__name__}") from exc
        if len(vectors) != len(values):
            raise EmbeddingError(f"fastembed count mismatch: expected {len(values)}, got {len(vectors)}")
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )
        return vectors


def build_embedder(settings: EmbeddingSettings | None = None):
    settings = settings or load_embedding_settings()
    try:
        return FastEmbedEmbedder(settings)
    except EmbeddingError:
        return OpenAICompatibleEmbedder(settings)


class OpenAICompatibleEmbedder:
    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self.settings = settings or load_embedding_settings()
        self.backend = "http"
        if not self.settings.configured:
            raise EmbeddingError(
                "embeddings require EMBEDDING_MODEL plus LLM/EMBEDDING base URL and API key"
            )

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def dimension(self) -> int:
        return self.settings.dimension

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        values = [normalize_embedding_text(text) for text in texts]
        if not values:
            return []
        output: list[list[float]] = []
        batch_size = max(1, self.settings.batch_size)
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            output.extend(self._embed_batch(batch))
            if start + batch_size < len(values):
                time.sleep(0.35)
        for vector in output:
            if len(vector) != self.settings.dimension:
                raise EmbeddingError(
                    f"embedding dimension mismatch: expected {self.settings.dimension}, got {len(vector)}"
                )
        return output

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(12):
            try:
                return self._embed_batch_once(texts)
            except EmbeddingError as exc:
                last_error = exc
                message = str(exc)
                if not any(token in message for token in ("429", "500", "502", "503", "timeout")):
                    raise
                time.sleep(min(90.0, 3.0 ** attempt))
        raise EmbeddingError(str(last_error) if last_error else "embedding_retry_exhausted")

    def _embed_batch_once(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps(build_embedding_request(self.settings.model, texts if len(texts) > 1 else texts[0])).encode("utf-8")
        url = self.settings.base_url.rstrip("/") + "/embeddings"
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
                self.settings.request_auth_header: f"Bearer {self.settings.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as response:
                raw_bytes = response.read()
        except urllib.error.HTTPError as exc:
            raise EmbeddingError(f"embedding_http_{exc.code}") from None
        except (TimeoutError, socket.timeout):
            raise EmbeddingError("embedding_timeout") from None
        except ssl.SSLError:
            raise EmbeddingError("embedding_ssl") from None
        except urllib.error.URLError:
            raise EmbeddingError("embedding_unreachable") from None
        except OSError:
            raise EmbeddingError("embedding_network") from None
        except ProviderError as exc:
            raise EmbeddingError(str(exc)) from None
        try:
            raw = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise EmbeddingError("embedding_invalid_json") from None
        return parse_embedding_response(raw, expected=len(texts))


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._rows: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk_id = str(row.get("chunk_id") or "")
                    if chunk_id:
                        self._rows[chunk_id] = row

    def get(self, chunk_id: str, digest: str, model: str, dimension: int) -> list[float] | None:
        row = self._rows.get(chunk_id)
        if not row:
            return None
        if row.get("text_sha256") != digest or row.get("model") != model or int(row.get("dimension") or 0) != dimension:
            return None
        vector = row.get("vector")
        if not isinstance(vector, list) or len(vector) != dimension:
            return None
        return [float(value) for value in vector]

    def put(self, chunk_id: str, digest: str, model: str, dimension: int, vector: list[float]) -> None:
        row = {
            "chunk_id": chunk_id,
            "text_sha256": digest,
            "model": model,
            "dimension": dimension,
            "vector": vector,
        }
        self._rows[chunk_id] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def put_many(self, rows: list[tuple[str, str, str, int, list[float]]]) -> None:
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for chunk_id, digest, model, dimension, vector in rows:
                row = {
                    "chunk_id": chunk_id,
                    "text_sha256": digest,
                    "model": model,
                    "dimension": dimension,
                    "vector": vector,
                }
                self._rows[chunk_id] = row
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def write(self) -> None:
        return None
