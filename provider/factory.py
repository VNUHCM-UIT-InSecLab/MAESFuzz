from __future__ import annotations

from typing import Any, Optional

from .base import ProviderBase, ProviderError
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .logging_provider import LoggingProvider


def _normalize_provider_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if name.lower() == "auto":
        return None
    mapping = {
        "google": "gemini",
        "gemini": "gemini",
        "ollama": "ollama",
        "openai": "openai",
        "gpt": "openai",
    }
    return mapping.get(name.lower(), name.lower())


def infer_provider_name(model: Optional[str], default: Optional[str] = None) -> str:
    if default:
        normalized_default = _normalize_provider_name(default)
        if normalized_default:
            return normalized_default
    if model:
        lowered = model.lower()
        if lowered.startswith("gemini"):
            return "gemini"
        elif lowered.startswith("gpt") or lowered.startswith("o1"):
            return "openai"
    return "ollama"


def create_provider(
    *,
    model: Optional[str],
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    default_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    **kwargs: Any,
) -> ProviderBase:
    if not model:
        raise ProviderError("Model name is required to initialize provider")

    normalized = _normalize_provider_name(provider_name) or infer_provider_name(model, default_provider)

    provider: ProviderBase

    if normalized == "gemini":
        if not model.lower().startswith("gemini"):
            raise ProviderError(
                "Gemini provider requires a model name starting with 'gemini'. "
                f"Received '{model}'."
            )
        generation_config = kwargs.get("generation_config")
        safety_settings = kwargs.get("safety_settings")
        provider = GeminiProvider(
            model=model,
            api_key=api_key,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
    elif normalized == "ollama":
        if model.lower().startswith("gemini"):
            raise ProviderError(
                "Ollama provider cannot be used with Gemini hosted models. "
                "Please specify an Ollama model identifier (e.g., 'llama3.1:8b')."
            )
        endpoint = kwargs.get("ollama_endpoint") or kwargs.get("endpoint")
        thinking = kwargs.get("ollama_thinking") or kwargs.get("thinking")
        options = kwargs.get("options")
        provider = OllamaProvider(
            model=model,
            endpoint=endpoint,
            thinking=thinking,
            options=options,
        )
    elif normalized == "openai":
        if not openai_api_key:
            raise ProviderError(
                "OpenAI provider requires an OPENAI_API_KEY. "
                "Please provide --openai-api-key or set OPENAI_API_KEY environment variable."
            )
        provider = OpenAIProvider(
            model=model,
            api_key=openai_api_key,
        )
    else:
        raise ProviderError(f"Unsupported provider '{normalized}'")

    # Wrap with logging so every prompt/response is stored in /log/LLM.log
    try:
        provider = LoggingProvider(provider, log_file="log/LLM.log")
    except Exception:
        # fallback: still return the provider even if logging wrapper fails
        pass

    return provider

