"""Construct the shared generator from explicit config or an injected provider."""

from __future__ import annotations

from answering.generator import DeterministicGroundedGenerator, Generator
from answering.openai_compatible import OpenAICompatibleProvider
from answering.provider import CompletionProvider
from answering.semantic import SemanticGenerator
from config.llm import LLMSettings, load_llm_settings

_SEMANTIC_CONFIG_ERROR = (
    "semantic generator requires a configured provider: set LLM_PROVIDER=openai_compatible "
    "with LLM_BASE_URL, LLM_MODEL, and LLM_API_KEY, or pass provider="
)


def build_generator(
    kind: str | None = None,
    *,
    provider: CompletionProvider | None = None,
    settings: LLMSettings | None = None,
) -> Generator:
    resolved = (kind or "deterministic").strip().casefold()
    if resolved in {"", "none", "deterministic"}:
        return DeterministicGroundedGenerator()
    if resolved not in {"semantic", "llm"}:
        raise ValueError(f"unknown generator kind: {kind}")
    if provider is not None:
        return SemanticGenerator(provider)
    llm = settings if settings is not None else load_llm_settings()
    if not llm.configured:
        raise RuntimeError(_SEMANTIC_CONFIG_ERROR)
    return SemanticGenerator(OpenAICompatibleProvider(llm))
