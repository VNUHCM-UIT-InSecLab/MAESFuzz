from __future__ import annotations

from typing import Optional

from .base import ProviderBase

_provider: Optional[ProviderBase] = None


def set_provider(provider: ProviderBase) -> None:
    global _provider
    _provider = provider


def get_provider(optional: bool = False) -> Optional[ProviderBase]:
    if _provider is None and not optional:
        raise RuntimeError("LLM provider has not been initialized")
    return _provider


def clear_provider() -> None:
    global _provider
    _provider = None


def has_provider() -> bool:
    return _provider is not None

