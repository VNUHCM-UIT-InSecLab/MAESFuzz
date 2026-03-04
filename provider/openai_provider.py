from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, Optional

from openai import OpenAI

from .base import ProviderBase, ProviderError, ProviderResult


class OpenAIProvider(ProviderBase):
    """OpenAI provider wrapper."""

    _configure_lock = threading.Lock()
    _configured_clients: Dict[str, OpenAI] = {}

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str],
        **kwargs: Any,
    ) -> None:
        if not api_key:
            raise ProviderError("OpenAI provider requires an OPENAI_API_KEY")

        super().__init__(model=model, **kwargs)
        self.api_key = api_key

        with self._configure_lock:
            if api_key not in self._configured_clients:
                self._configured_clients[api_key] = OpenAI(api_key=api_key)
            self._client = self._configured_clients[api_key]

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history_messages: Optional[Iterable[Dict[str, Any]]],
    ) -> list[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in {"user", "assistant", "system"}:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history_messages: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResult:
        messages = self._build_messages(prompt, system_prompt, history_messages)

        # Một số model mini của OpenAI (vd: gpt-4o-mini, gpt-5-mini) chỉ chấp nhận temperature=1
        forced_temperature = temperature
        lowered_model = self.model.lower()
        if lowered_model.startswith(("gpt-4o", "gpt-5")):
            forced_temperature = 1

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=forced_temperature,
                **kwargs
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI generation failed: {exc}") from exc

        if not response.choices:
            raise ProviderError("OpenAI returned an empty response")

        text = response.choices[0].message.content
        if not text:
            raise ProviderError("OpenAI returned an empty response")

        return ProviderResult(text=text, raw=response)
