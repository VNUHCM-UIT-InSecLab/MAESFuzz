from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ollama import Client

from .base import ProviderBase, ProviderError, ProviderResult


class OllamaProvider(ProviderBase):
    """Ollama provider wrapper."""

    def __init__(
        self,
        *,
        model: str,
        endpoint: Optional[str] = None,
        thinking: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.endpoint = endpoint or "http://127.0.0.1:11434"
        self.client = Client(host=self.endpoint)
        self.thinking = thinking
        self.options = options or {}

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history_messages: Optional[Iterable[Dict[str, Any]]],
    ) -> Any:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history_messages: Optional[Iterable[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ProviderResult:
        messages = self._build_messages(prompt, system_prompt, history_messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if self.options or kwargs:
            merged_options = dict(self.options)
            merged_options.update(kwargs.get("options", {}))
            payload["options"] = merged_options

        think_value = kwargs.get("think", self.thinking)
        if think_value is not None:
            payload["think"] = think_value

        try:
            response = self.client.chat(**payload)
        except Exception as exc:  # pragma: no cover - surface provider errors
            raise ProviderError(f"Ollama generation failed: {exc}") from exc

        message = response.get("message", {})
        text = message.get("content", "")
        thinking = message.get("thinking")

        if not text and not thinking:
            raise ProviderError("Ollama returned an empty response")

        return ProviderResult(text=text, thinking=thinking, raw=response)

