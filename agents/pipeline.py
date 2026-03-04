from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # Contract
    contract_path: str
    contract_name: str

    # Compiler / EVM
    solc_version: str
    evm_version: str
    solc_path: str

    # Fuzzing
    fuzz_time: int = 60
    generations: Optional[int] = None
    max_individual_length: int = 10
    duplication: str = "0"

    # Output
    result_json_path: str = "result/res.json"
    result_saved_path: str = "result/results.json"

    # Constructor params ("auto" = infer via Slither)
    constructor_params_path: str = "auto"

    # LLM / RAG
    use_rag: bool = True
    api_key: Optional[str] = None          # Google / Gemini
    openai_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    ollama_endpoint: Optional[str] = None
    ollama_thinking: Optional[str] = None
    disable_rag_llm: bool = False

    # Reporter
    use_llm_judge: bool = False
    judge_threshold: float = 60.0
    use_llm_report: bool = True
    report_threshold_high: float = 70.0
    report_threshold_low: float = 30.0

    # Internal
    _python_executable: str = field(default_factory=lambda: sys.executable, repr=False)
    _fuzzer_script: str = field(default="fuzzer/main.py", repr=False)


class MAESFuzzPipeline:
    """Orchestrates the four agents: Analyzer → Generator → Executor → Reporter."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self):
        from agents.analyzer_agent import AnalyzerAgent
        from agents.generator_agent import GeneratorAgent
        from agents.executor_agent import ExecutorAgent
        from agents.reporter_agent import ReporterAgent

        cfg = self._config

        print(
            f"\n{'='*60}\n"
            f"  MAESFuzz — 4-Agent Pipeline\n"
            f"  Contract : {cfg.contract_name}\n"
            f"  Source   : {cfg.contract_path}\n"
            f"{'='*60}\n",
            flush=True,
        )

        # 1. Analyzer
        print("[1/4] Analyzer …", flush=True)
        analyzer = AnalyzerAgent(
            solc_version=cfg.solc_version,
            solc_path=cfg.solc_path,
            evm_version=cfg.evm_version,
            api_key=cfg.api_key,
            openai_api_key=cfg.openai_api_key,
            llm_model=cfg.llm_model,
            llm_provider=cfg.llm_provider,
            ollama_endpoint=cfg.ollama_endpoint,
            constructor_params_path=cfg.constructor_params_path,
        )
        analysis_result = analyzer.analyze(cfg.contract_path, cfg.contract_name)

        if analysis_result.constructor_args is None:
            logger.error("[Pipeline] Constructor analysis failed — aborting")
            sys.exit(-2)

        # 2. Generator
        print("[2/4] Generator …", flush=True)
        generator = GeneratorAgent(
            api_key=cfg.api_key,
            openai_api_key=cfg.openai_api_key,
            llm_model=cfg.llm_model,
            llm_provider=cfg.llm_provider,
            ollama_endpoint=cfg.ollama_endpoint,
            use_rag=cfg.use_rag,
            max_individual_length=cfg.max_individual_length,
            disable_rag_llm=cfg.disable_rag_llm,
        )
        seed_set = generator.generate(analysis_result, fuzzer_context={
            "interface": None, "bytecode": None, "accounts": [],
            "contract_address": None, "interface_mapper": None,
            "dependent_generators": [],
        })

        # 3. Executor
        print("[3/4] Executor …", flush=True)
        executor = ExecutorAgent(
            python_executable=cfg._python_executable,
            fuzzer_script=cfg._fuzzer_script,
            fuzz_time=cfg.fuzz_time,
            generations=cfg.generations,
            max_individual_length=cfg.max_individual_length,
            duplication=cfg.duplication,
            result_json_path=cfg.result_json_path,
            result_saved_path=cfg.result_saved_path,
            use_rag=cfg.use_rag,
            api_key=cfg.api_key,
            openai_api_key=cfg.openai_api_key,
            llm_model=seed_set.llm_model or cfg.llm_model,
            llm_provider=seed_set.llm_provider or cfg.llm_provider,
            ollama_endpoint=seed_set.ollama_endpoint or cfg.ollama_endpoint,
            ollama_thinking=cfg.ollama_thinking,
            use_llm_judge=cfg.use_llm_judge,
            judge_threshold=cfg.judge_threshold,
        )
        execution_result = executor.execute(seed_set)

        # 4. Reporter
        print("[4/4] Reporter …", flush=True)
        reporter = ReporterAgent(
            llm_model=cfg.llm_model,
            llm_provider=cfg.llm_provider,
            api_key=cfg.api_key,
            openai_api_key=cfg.openai_api_key,
            ollama_endpoint=cfg.ollama_endpoint,
            ollama_thinking=cfg.ollama_thinking,
            use_llm_summary=cfg.use_llm_report,
            threshold_high=cfg.report_threshold_high,
            threshold_low=cfg.report_threshold_low,
        )
        report = reporter.report(execution_result)

        logger.info(
            "[Pipeline] Done. CONFIRMED=%d INCONCLUSIVE=%d REJECTED=%d",
            report.confirmed_count, report.inconclusive_count, report.rejected_count,
        )
        print(
            f"\n{'='*60}\n"
            f"  Report saved: {report.output_path}\n"
            f"{'='*60}\n",
            flush=True,
        )
        return report
