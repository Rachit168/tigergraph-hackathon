"""Provider-neutral completion boundary. No vendor types in domain models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

TRANSIENT_PROVIDER_CODES = frozenset(
    {
        "timeout",
        "http_408",
        "http_429",
        "http_500",
        "http_502",
        "http_503",
        "http_504",
        "unreachable",
        "network",
        "ssl",
    }
)


def is_transient_provider_error(code: str) -> bool:
    """Retry only clearly transient transport/gateway failures, never truncation or 4xx auth."""

    return str(code).strip().casefold() in TRANSIENT_PROVIDER_CODES


class ProviderError(RuntimeError):
    """Controlled completion failure. `code` is safe to log; do not put secrets in it."""

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        self.code = str(code)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        super().__init__(message or self.code)


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class CompletionProvider(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> CompletionResult: ...


class FakeCompletionProvider:
    """Scripted provider for tests. Never opens a network connection."""

    def __init__(
        self,
        text: str | Sequence[str] | Callable[..., str] | None = None,
        *,
        finish_reason: str = "stop",
        error: Exception | None = None,
        script: Sequence[str | Exception | CompletionResult] | None = None,
        prompt_tokens: int = 1,
        completion_tokens: int = 1,
        total_tokens: int = 2,
    ) -> None:
        self._text = text
        self._finish_reason = finish_reason
        self._error = error
        self._script = None if script is None else tuple(script)
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._total_tokens = total_tokens
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, *, system: str | None = None) -> CompletionResult:
        self.calls.append((prompt, system))
        if self._script is not None:
            if not self._script:
                raise ProviderError("empty_script", "fake_provider_empty_script")
            index = min(len(self.calls) - 1, len(self._script) - 1)
            item = self._script[index]
            if isinstance(item, Exception):
                raise item
            if isinstance(item, CompletionResult):
                return item
            return CompletionResult(
                text=str(item),
                model="fake",
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                total_tokens=self._total_tokens,
                finish_reason=self._finish_reason,
            )
        if self._error is not None:
            raise self._error
        if callable(self._text):
            payload = self._text(prompt, system)
        elif isinstance(self._text, (list, tuple)):
            index = min(len(self.calls) - 1, len(self._text) - 1)
            payload = self._text[index]
        else:
            payload = self._text or ""
        return CompletionResult(
            text=str(payload),
            model="fake",
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._total_tokens,
            finish_reason=self._finish_reason,
        )
