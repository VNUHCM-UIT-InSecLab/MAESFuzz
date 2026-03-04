from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.analyzer_agent import AnalysisResult
from agents.generator_agent import SeedSet

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    result_json_path: str
    result_saved_path: str
    code_coverage: float = 0.0
    branch_coverage: float = 0.0
    total_transactions: int = 0
    unique_transactions: int = 0
    execution_time: float = 0.0
    errors: Dict[str, Any] = field(default_factory=dict)
    analysis_result: Optional[AnalysisResult] = None
    contract_path: str = ""
    contract_name: str = ""


class ExecutorAgent:
    """Runs the fuzzing subprocess and collects the result."""

    def __init__(
        self,
        python_executable: str,
        fuzzer_script: str,
        fuzz_time: int = 60,
        generations: Optional[int] = None,
        max_individual_length: int = 10,
        duplication: str = "0",
        result_json_path: str = "result/res.json",
        result_saved_path: str = "result/results.json",
        use_rag: bool = True,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        ollama_endpoint: Optional[str] = None,
        ollama_thinking: Optional[str] = None,
        use_llm_judge: bool = False,
        judge_threshold: float = 60.0,
        extra_argv: Optional[List[str]] = None,
    ) -> None:
        self._python              = python_executable
        self._fuzzer_script       = fuzzer_script
        self._fuzz_time           = fuzz_time
        self._generations         = generations
        self._max_individual_length = max_individual_length
        self._duplication         = duplication
        self._result_json_path    = result_json_path
        self._result_saved_path   = result_saved_path
        self._use_rag             = use_rag
        self._api_key             = api_key
        self._openai_api_key      = openai_api_key
        self._llm_model           = llm_model
        self._llm_provider        = llm_provider
        self._ollama_endpoint     = ollama_endpoint
        self._ollama_thinking     = ollama_thinking
        self._use_llm_judge       = use_llm_judge
        self._judge_threshold     = judge_threshold
        self._extra_argv          = extra_argv or []

    def execute(self, seed_set: SeedSet) -> ExecutionResult:
        analysis = seed_set.analysis_result
        if analysis is None:
            raise ValueError("SeedSet must contain an AnalysisResult")
        cmd = self._build_command(analysis, seed_set)
        self._run_subprocess(cmd)
        return self._collect_result(analysis)

    def _build_command(self, analysis: AnalysisResult, seed_set: SeedSet) -> str:
        depend_str      = " ".join(analysis.depend_contracts) if analysis.depend_contracts else ""
        constructor_str = self._format_constructor_args(analysis.constructor_args)

        llm_model     = seed_set.llm_model     or self._llm_model
        llm_provider  = seed_set.llm_provider  or self._llm_provider
        openai_key    = seed_set.openai_api_key or self._openai_api_key
        ollama_ep     = seed_set.ollama_endpoint or self._ollama_endpoint

        parts = [
            f"{self._python} {self._fuzzer_script}",
            f"-s {analysis.contract_path}",
            f"-c {analysis.contract_name}",
            f"--solc v{analysis.solc_version}",
            f"--evm {analysis.evm_version}",
            f"--generations {self._generations}" if self._generations else f"-t {self._fuzz_time}",
            f"--result {self._result_json_path}",
            "--cross-contract 1",
            "--open-trans-comp 1",
            f"--depend-contracts {depend_str}",
            f"--constructor-args {constructor_str}",
            "--constraint-solving 1",
            f"--max-individual-length {self._max_individual_length}",
            f"--solc-path-cross {analysis.solc_path}",
            "--p-open-cross 80",
            "--cross-init-mode 1",
            "--trans-mode 1",
            f"--duplication {self._duplication}",
        ]

        if self._use_rag:             parts.append("--use-rag")
        if self._api_key:             parts.append(f"--api-key {self._api_key}")
        if openai_key:                parts.append(f"--openai-api-key {openai_key}")
        if llm_model:                 parts.append(f"--model {llm_model}")
        if llm_provider:              parts.append(f"--provider {llm_provider}")
        if ollama_ep:                 parts.append(f"--ollama-endpoint {ollama_ep}")
        if self._ollama_thinking:     parts.append(f"--ollama-thinking {self._ollama_thinking}")
        if self._use_llm_judge:
            parts += [f"--use-llm-judge", f"--judge-threshold {self._judge_threshold}"]

        for flag in ("--no-plateau", "--plateau-generations", "--plateau-threshold"):
            if flag in sys.argv:
                idx = sys.argv.index(flag)
                parts.append(flag)
                if idx + 1 < len(sys.argv) and flag != "--no-plateau":
                    parts.append(sys.argv[idx + 1])

        parts += self._extra_argv
        return " ".join(parts)

    @staticmethod
    def _format_constructor_args(constructor_args: List[Any]) -> str:
        out = []
        for arg in constructor_args:
            if isinstance(arg, (tuple, list)) and len(arg) >= 3:
                out.append(f"{arg[0]} {arg[1]} {arg[2]}")
            elif isinstance(arg, dict):
                out.append(f"{arg.get('name','')} {arg.get('type','')} {arg.get('value','')}")
            else:
                out.append(str(arg))
        return " ".join(out)

    def _run_subprocess(self, cmd: str) -> None:
        redacted = cmd
        for secret in filter(None, [self._api_key, self._openai_api_key]):
            redacted = redacted.replace(secret, "***")
        logger.info("[Executor] %s", redacted)
        print(redacted, flush=True)
        os.popen(cmd).readlines()

    def _collect_result(self, analysis: AnalysisResult) -> ExecutionResult:
        import json, shutil

        result = ExecutionResult(
            result_json_path=self._result_json_path,
            result_saved_path=self._result_saved_path,
            analysis_result=analysis,
            contract_path=analysis.contract_path,
            contract_name=analysis.contract_name,
        )

        if not os.path.exists(self._result_json_path):
            logger.warning("[Executor] Result file not found: %s", self._result_json_path)
            return result

        try:
            with open(self._result_json_path, encoding="utf-8") as fh:
                raw = json.load(fh)

            section = raw.get(analysis.contract_name) or (
                next(iter(raw.values()), {}) if len(raw) == 1 else {}
            )
            result.code_coverage       = section.get("code_coverage",   {}).get("percentage", 0.0) or 0.0
            result.branch_coverage     = section.get("branch_coverage",  {}).get("percentage", 0.0) or 0.0
            result.total_transactions  = section.get("transactions",     {}).get("total",  0) or 0
            result.unique_transactions = section.get("transactions",     {}).get("unique", 0) or 0
            result.execution_time      = section.get("execution_time",   0.0) or 0.0
            result.errors              = section.get("errors", {})
        except Exception as exc:
            logger.warning("[Executor] Could not parse result JSON: %s", exc)

        try:
            os.makedirs(os.path.dirname(self._result_saved_path), exist_ok=True)
            shutil.copyfile(self._result_json_path, self._result_saved_path)
            logger.info("[Executor] Results saved to %s", self._result_saved_path)
            print(f"Results saved: {self._result_saved_path}", flush=True)
        except Exception as exc:
            logger.warning("[Executor] Could not save results: %s", exc)

        return result
