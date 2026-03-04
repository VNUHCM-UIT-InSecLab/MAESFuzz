from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from agents.executor_agent import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class Report:
    body: str
    output_path: str
    contract_name: str
    session_id: str
    llm_used: bool = False
    confirmed_count: int = 0
    inconclusive_count: int = 0
    rejected_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ReporterAgent:
    """Generates a Markdown audit report via LLM-as-a-Judge.

    Delegates to ``agents/reporter.py`` for prompt templates and LLM calls.
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        ollama_endpoint: Optional[str] = None,
        ollama_thinking: Optional[str] = None,
        use_llm_summary: bool = True,
        threshold_high: float = 70.0,
        threshold_low: float = 30.0,
        output_dir: Optional[str] = None,
        rag_storage_path: Optional[str] = None,
        analysis_path: Optional[str] = "dataflow_analysis_result.json",
        log_path: Optional[str] = "log/INFO.log",
    ) -> None:
        self._llm_model       = llm_model
        self._llm_provider    = llm_provider
        self._api_key         = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._openai_api_key  = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._ollama_endpoint = ollama_endpoint
        self._ollama_thinking = ollama_thinking
        self._use_llm_summary = use_llm_summary
        self._threshold_high  = threshold_high
        self._threshold_low   = threshold_low
        self._output_dir      = output_dir
        self._rag_storage_path = rag_storage_path
        self._analysis_path   = analysis_path
        self._log_path        = log_path

    def report(self, execution_result: ExecutionResult) -> Report:
        logger.info("[Reporter] %s (llm=%s)", execution_result.contract_name, self._use_llm_summary)

        session_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        argv = self._build_argv(execution_result, session_id)

        try:
            from agents.reporter import main as reporter_main
            code = reporter_main(argv)
            if code != 0:
                logger.warning("[Reporter] reporter.main() exited with code %d", code)
        except Exception as exc:
            logger.error("[Reporter] reporter.main() failed: %s", exc)

        output_path = self._resolve_output_path(execution_result.contract_name, session_id)
        body = self._read_body(output_path)

        return Report(
            body=body,
            output_path=str(output_path),
            contract_name=execution_result.contract_name,
            session_id=session_id,
            llm_used=self._use_llm_summary,
            confirmed_count=body.count("[CONFIRMED]"),
            inconclusive_count=body.count("[INCONCLUSIVE]"),
            rejected_count=body.count("[REJECTED]"),
        )

    def _build_argv(self, result: ExecutionResult, session_id: str) -> List[str]:
        argv = [
            "--result", result.result_json_path,
            "--contract", f"{result.contract_path}::{result.contract_name}",
            "--session-id", session_id,
            "--threshold-high", str(self._threshold_high),
            "--threshold-low",  str(self._threshold_low),
        ]

        if self._use_llm_summary: argv.append("--use-llm-summary")
        if self._llm_model:       argv += ["--model",    self._llm_model]
        if self._llm_provider:    argv += ["--provider", self._llm_provider]

        # Pick API key matching the provider
        prov = (self._llm_provider or "").lower()
        model = (self._llm_model or "").lower()
        if prov == "openai" or model.startswith("gpt") or model.startswith("o1"):
            key = self._openai_api_key or self._api_key
        else:
            key = self._api_key
        if key: argv += ["--api-key", key]

        if self._ollama_endpoint:  argv += ["--ollama-endpoint", self._ollama_endpoint]
        if self._ollama_thinking:  argv += ["--ollama-thinking",  self._ollama_thinking]
        if self._rag_storage_path: argv += ["--rag-storage",      self._rag_storage_path]
        if self._analysis_path and os.path.exists(self._analysis_path):
            argv += ["--analysis", self._analysis_path]
        if self._log_path and os.path.exists(self._log_path):
            argv += ["--log", self._log_path]

        return argv

    def _resolve_output_path(self, contract_name: str, session_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", contract_name) or "contract"
        if self._output_dir:
            return Path(self._output_dir) / f"fuzz_report_{session_id}.md"
        root = Path(__file__).resolve().parents[1]
        return root / "reports" / slug / f"fuzz_report_{session_id}.md"

    @staticmethod
    def _read_body(path: Path) -> str:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("[Reporter] Could not read report at %s: %s", path, exc)
        return "(Report not found — see console output above.)"
