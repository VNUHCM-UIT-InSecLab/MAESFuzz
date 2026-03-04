from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


class ProviderError(RuntimeError):
    """Raised when an LLM provider fails to generate a response."""


@dataclass
class ProviderResult:
    """Normalized LLM output."""

    text: str
    thinking: Optional[str] = None
    raw: Any = None


class ProviderBase:
    """Common interface for all LLM providers."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.metadata: Dict[str, Any] = dict(kwargs)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history_messages: Optional[Iterable[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ProviderResult:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history_messages: Optional[Iterable[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        result = self.generate(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
        try:
            return json.loads(result.text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Failed to decode provider response as JSON: {exc}") from exc

