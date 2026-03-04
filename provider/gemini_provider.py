from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, Optional

import google.generativeai as genai

from .base import ProviderBase, ProviderError, ProviderResult


class GeminiProvider(ProviderBase):
    """Google Gemini provider wrapper."""

    _configure_lock = threading.Lock()
    _configured_keys: Dict[str, bool] = {}

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str],
        generation_config: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not api_key:
            raise ProviderError("Gemini provider requires a GOOGLE_API_KEY")

        super().__init__(model=model, **kwargs)
        self.api_key = api_key
        self._generation_config = generation_config or {"temperature": 0.3}
        self._safety_settings = safety_settings

        with self._configure_lock:
            if not self._configured_keys.get(api_key):
                genai.configure(api_key=api_key)
                self._configured_keys[api_key] = True

        self._model = genai.GenerativeModel(
            model,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
        )

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history_messages: Optional[Iterable[Dict[str, Any]]],
    ) -> Any:
        messages = []
        if system_prompt:
            messages.append({"role": "user", "parts": [f"[System Instruction] {system_prompt}"]})
        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                if role not in {"user", "model"}:
                    role = "user"
                messages.append({"role": role, "parts": [msg.get("content", "")]})
        messages.append({"role": "user", "parts": [prompt]})
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

        try:
            response = self._model.generate_content(messages, **kwargs)
        except Exception as exc:  # pragma: no cover - surface provider errors
            raise ProviderError(f"Gemini generation failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text and hasattr(response, "candidates"):
            for candidate in response.candidates:
                for part in getattr(candidate, "content", {}).get("parts", []):
                    text = getattr(part, "text", None)
                    if text:
                        break
                if text:
                    break

        if not text:
            raise ProviderError("Gemini returned an empty response")

        return ProviderResult(text=text, raw=response)

