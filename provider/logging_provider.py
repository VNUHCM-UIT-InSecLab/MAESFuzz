from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from .base import ProviderBase, ProviderResult, ProviderError


class LoggingProvider(ProviderBase):
    """
    Wrapper to log all provider requests/responses to /log/LLM.log.
    """

    def __init__(self, inner: ProviderBase, log_file: str = "log/LLM.log") -> None:
        super().__init__(model=inner.model, **getattr(inner, "metadata", {}))
        self.inner = inner
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.logger = logging.getLogger("LLM")

    def _log(self, title: str, prompt: str, response: Optional[Any] = None, error: Optional[str] = None) -> None:
        try:
            # Print to stdout for quick visibility during fuzz runs
            preview_prompt = (prompt[:200] + "..." ) if len(prompt) > 200 else prompt
            print(f"[LLM][{title}] {preview_prompt}", flush=True)
            if response is not None:
                preview_resp = response
                try:
                    preview_resp = json.dumps(response, ensure_ascii=False)
                except Exception:
                    preview_resp = str(response)
                if len(preview_resp) > 200:
                    preview_resp = preview_resp[:200] + "..."
                print(f"[LLM][resp] {preview_resp}", flush=True)
            if error is not None:
                print(f"[LLM][err] {error}", flush=True)

            # Also log via Python logger so it shows in terminal handlers
            if self.logger:
                msg = f"[{title}] prompt_preview={preview_prompt}"
                if response is not None:
                    msg += f" | resp_preview={preview_resp}"
                if error is not None:
                    msg += f" | error={error}"
                self.logger.info(msg)

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {title}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Model: {self.inner.model}\n")
                f.write(f"Provider: {type(self.inner).__name__}\n")
                f.write("--- REQUEST ---\n")
                f.write(f"{prompt}\n")
                f.write("--- END REQUEST ---\n")
                if response is not None:
                    try:
                        pretty = json.dumps(response, ensure_ascii=False, indent=2)
                    except Exception:
                        pretty = str(response)
                    f.write("--- RESPONSE ---\n")
                    f.write(f"{pretty}\n")
                    f.write("--- END RESPONSE ---\n")
                if error is not None:
                    f.write("--- ERROR ---\n")
                    f.write(f"{error}\n")
                    f.write("--- END ERROR ---\n")
                f.write(f"Status: {'FAILED' if error else 'SUCCESS'}\n")
                f.write(f"{'='*80}\n\n")
        except Exception:
            # Best-effort logging; never break provider flow
            pass

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history_messages: Optional[Iterable[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ProviderResult:
        self._log("LLM Request", prompt)
        try:
            result = self.inner.generate(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
            self._log("LLM Response", prompt, response=result.text)
            return result
        except Exception as exc:
            self._log("LLM Error", prompt, error=str(exc))
            raise

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history_messages: Optional[Iterable[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        self._log("LLM Request (JSON)", prompt)
        try:
            result = self.inner.generate_json(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
            self._log("LLM Response (JSON)", prompt, response=result)
            return result
        except Exception as exc:
            self._log("LLM Error (JSON)", prompt, error=str(exc))
            raise

