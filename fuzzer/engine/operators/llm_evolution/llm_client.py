#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Client Wrapper for Provider System
Provides unified interface for LLM operations in evolution operators
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from provider.base import ProviderBase, ProviderError

logger = logging.getLogger("LLMClient")


class LLMClient:
    """Wrapper around ProviderBase for LLM evolution operators"""

    _log_initialized = False

    def __init__(self, provider: ProviderBase, temperature: float = 0.7):
        """
        Initialize LLM client

        Args:
            provider: ProviderBase instance (GeminiProvider, OllamaProvider, or OpenAIProvider)
            temperature: Default sampling temperature
        """
        self.provider = provider
        self.temperature = temperature
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0

        os.makedirs("log", exist_ok=True)
        self.log_file = "log/LLM.log"

        if not LLMClient._log_initialized:
            LLMClient._log_initialized = True
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== LLM Interactions Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

        logger.info(f"LLMClient initialized: {provider.model} ({type(provider).__name__})")

    def _log_to_file(self, message: str):
        """Write message to log file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message)
        except Exception:
            pass

    def generate_json(self, prompt: str, **kwargs) -> Optional[Dict]:
        """
        Generate JSON response from LLM

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Dict parsed from JSON or None on failure
        """
        self.total_calls += 1

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        call_id = f"LLM_CALL_{self.total_calls}"

        self._log_to_file(f"\n{'='*80}\n")
        self._log_to_file(f"[{timestamp}] {call_id} - LLM Evolution (Mutation/Crossover)\n")
        self._log_to_file(f"{'='*80}\n")
        self._log_to_file(f"Model: {self.provider.model}\n")
        self._log_to_file(f"Provider: {type(self.provider).__name__}\n")
        self._log_to_file(f"Temperature: {kwargs.get('temperature', self.temperature)}\n")
        self._log_to_file(f"\n--- REQUEST ---\n")
        self._log_to_file(f"{prompt}\n")
        self._log_to_file(f"\n--- END REQUEST ---\n\n")

        try:
            merged_kwargs = {"temperature": kwargs.get("temperature", self.temperature)}
            # Some OpenAI models (e.g., gpt-5-mini) only accept temperature=1
            if type(self.provider).__name__ == "OpenAIProvider":
                merged_kwargs["temperature"] = 1
            merged_kwargs.update({k: v for k, v in kwargs.items() if k != "temperature"})

            result = self.provider.generate_json(prompt, **merged_kwargs)
            self.successful_calls += 1

            self._log_to_file(f"--- RESPONSE ---\n")
            self._log_to_file(f"{json.dumps(result, indent=2, ensure_ascii=False)}\n")
            self._log_to_file(f"\n--- END RESPONSE ---\n")
            self._log_to_file(f"Status: SUCCESS\n")
            self._log_to_file(f"{'='*80}\n\n")

            return result

        except ProviderError as e:
            error_msg = str(e)
            logger.debug(f"Provider error: {e}")
            self.failed_calls += 1

            self._log_to_file(f"--- ERROR ---\n")
            self._log_to_file(f"{error_msg}\n")
            self._log_to_file(f"\n--- END ERROR ---\n")
            self._log_to_file(f"Status: FAILED\n")
            self._log_to_file(f"{'='*80}\n\n")

            return None
        except Exception as e:
            error_msg = str(e)
            logger.debug(f"Unexpected error: {e}")
            self.failed_calls += 1

            self._log_to_file(f"--- ERROR ---\n")
            self._log_to_file(f"{error_msg}\n")
            self._log_to_file(f"\n--- END ERROR ---\n")
            self._log_to_file(f"Status: FAILED\n")
            self._log_to_file(f"{'='*80}\n\n")

            return None

    def generate_text(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate text response from LLM

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Text response or None on failure
        """
        self.total_calls += 1

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        call_id = f"LLM_CALL_{self.total_calls}"

        self._log_to_file(f"\n{'='*80}\n")
        self._log_to_file(f"[{timestamp}] {call_id} - LLM Text Generation\n")
        self._log_to_file(f"{'='*80}\n")
        self._log_to_file(f"Model: {self.provider.model}\n")
        self._log_to_file(f"Provider: {type(self.provider).__name__}\n")
        self._log_to_file(f"\n--- REQUEST ---\n")
        self._log_to_file(f"{prompt}\n")
        self._log_to_file(f"\n--- END REQUEST ---\n\n")

        try:
            merged_kwargs = {"temperature": kwargs.get("temperature", self.temperature)}
            if type(self.provider).__name__ == "OpenAIProvider":
                merged_kwargs["temperature"] = 1
            merged_kwargs.update({k: v for k, v in kwargs.items() if k != "temperature"})

            result = self.provider.generate(prompt, **merged_kwargs)
            response_text = result.text
            self.successful_calls += 1

            self._log_to_file(f"--- RESPONSE ---\n")
            self._log_to_file(f"{response_text}\n")
            self._log_to_file(f"\n--- END RESPONSE ---\n")
            self._log_to_file(f"Status: SUCCESS\n")
            self._log_to_file(f"{'='*80}\n\n")

            return response_text

        except ProviderError as e:
            error_msg = str(e)
            logger.debug(f"Provider error: {e}")
            self.failed_calls += 1

            self._log_to_file(f"--- ERROR ---\n")
            self._log_to_file(f"{error_msg}\n")
            self._log_to_file(f"\n--- END ERROR ---\n")
            self._log_to_file(f"Status: FAILED\n")
            self._log_to_file(f"{'='*80}\n\n")

            return None
        except Exception as e:
            error_msg = str(e)
            logger.debug(f"Unexpected error: {e}")
            self.failed_calls += 1

            self._log_to_file(f"--- ERROR ---\n")
            self._log_to_file(f"{error_msg}\n")
            self._log_to_file(f"\n--- END ERROR ---\n")
            self._log_to_file(f"Status: FAILED\n")
            self._log_to_file(f"{'='*80}\n\n")

            return None

    @classmethod
    def from_config(cls,
                    model: str,
                    provider_name: Optional[str] = None,
                    api_key: Optional[str] = None,
                    ollama_endpoint: Optional[str] = None,
                    openai_api_key: Optional[str] = None,
                    temperature: float = 0.7) -> 'LLMClient':
        """
        Factory method to create LLMClient from configuration

        Args:
            model: Model name (e.g., "gemini-2.0-flash", "llama3.1:8b", "gpt-4o-mini")
            provider_name: "gemini", "ollama", or "openai" (auto-detect if None)
            api_key: API key for Gemini
            ollama_endpoint: Endpoint for Ollama
            openai_api_key: API key for OpenAI
            temperature: Sampling temperature

        Returns:
            LLMClient instance
        """
        from provider.factory import create_provider

        provider = create_provider(
            model=model,
            provider_name=provider_name,
            api_key=api_key,
            ollama_endpoint=ollama_endpoint,
            openai_api_key=openai_api_key
        )

        return cls(provider, temperature=temperature)

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0.0
        }
