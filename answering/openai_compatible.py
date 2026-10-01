"""OpenAI-compatible HTTP completion provider. Configuration stays at the boundary."""

from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.request
from typing import Any

from answering.provider import CompletionResult, ProviderError
from config.llm import LLMSettings

USER_AGENT = "tigergraph-graphrag/0.1"


class OpenAICompatibleProvider:
    def __init__(self, settings: LLMSettings) -> None:
        if not settings.configured:
            errors = settings.validate() or [
                "semantic HTTP provider requires LLM_PROVIDER=openai_compatible "
                "with LLM_BASE_URL, LLM_MODEL, and LLM_API_KEY"
            ]
            raise RuntimeError("; ".join(errors))
        self.settings = settings

    def complete(self, prompt: str, *, system: str | None = None) -> CompletionResult:
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.settings.model,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "messages": messages,
        }
        payload = json.dumps(body).encode("utf-8")
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
            raise ProviderError(f"http_{exc.code}", f"llm_provider_http_{exc.code}") from None
        except (TimeoutError, socket.timeout):
            raise ProviderError("timeout", "llm_provider_timeout") from None
        except ssl.SSLError:
            raise ProviderError("ssl", "llm_provider_ssl") from None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise ProviderError("timeout", "llm_provider_timeout") from None
            raise ProviderError("unreachable", "llm_provider_unreachable") from None
        except OSError:
            raise ProviderError("network", "llm_provider_network") from None
        try:
            raw = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderError("invalid_json", "llm_provider_invalid_json") from None
        return _from_chat_payload(raw, fallback_model=self.settings.model)


def _from_chat_payload(raw: Any, *, fallback_model: str) -> CompletionResult:
    if not isinstance(raw, dict):
        raise ProviderError("invalid_payload", "llm_provider_invalid_payload")
    choices = raw.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ProviderError("empty_choices", "llm_provider_empty_choices")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise ProviderError("invalid_content", "llm_provider_invalid_content")
    content = message.get("content")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        raise ProviderError("invalid_content", "llm_provider_invalid_content")
    finish_reason = choices[0].get("finish_reason")
    finish = str(finish_reason) if finish_reason else None
    usage = _parse_usage(raw)
    prompt_tokens, completion_tokens, total_tokens = usage if usage is not None else (0, 0, 0)
    if finish in {"length", "max_tokens"} and not text.strip():
        extra = {}
        if usage is not None:
            extra = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        raise ProviderError("truncated", "llm_provider_truncated", **extra)
    return CompletionResult(
        text=text,
        model=str(raw.get("model") or fallback_model),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        finish_reason=finish,
        raw={"id": raw.get("id"), "object": raw.get("object")},
    )


def _parse_usage(raw: dict[str, Any]) -> tuple[int, int, int] | None:
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    if not any(key in usage for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
        return None
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return prompt_tokens, completion_tokens, total_tokens