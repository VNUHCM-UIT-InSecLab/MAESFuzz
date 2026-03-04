from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.analyzer_agent import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class SeedSet:
    """Generator output consumed by ExecutorAgent.

    ``generator`` is None when produced at pipeline-orchestration time
    (before the fuzzer subprocess is spawned).  The Executor Agent forwards
    the LLM / RAG settings to ``fuzzer/main.py`` via CLI flags.
    """

    generator: Any
    dependent_generators: List[Any] = field(default_factory=list)
    analysis_result: Optional[AnalysisResult] = None
    rag_enabled: bool = False
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_endpoint: Optional[str] = None
    disable_rag_llm: bool = False


class GeneratorAgent:
    """RAG-enhanced seed population synthesis.

    Operates in two modes:
    - Orchestration-time: EVM objects unavailable → returns a settings-only SeedSet.
    - Runtime (inside fuzzer/main.py): constructs the actual RAGEnhancedGenerator.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        ollama_endpoint: Optional[str] = None,
        use_rag: bool = True,
        max_individual_length: int = 10,
        disable_rag_llm: bool = False,
    ) -> None:
        self._api_key = api_key
        self._openai_api_key = openai_api_key
        self._llm_model = llm_model
        self._llm_provider = llm_provider
        self._ollama_endpoint = ollama_endpoint
        self._use_rag = use_rag
        self._max_individual_length = max_individual_length
        self._disable_rag_llm = disable_rag_llm

    def generate(
        self,
        analysis_result: AnalysisResult,
        fuzzer_context: Dict[str, Any],
    ) -> SeedSet:
        logger.info("[Generator] use_rag=%s", self._use_rag)

        interface        = fuzzer_context.get("interface")
        contract_address = fuzzer_context.get("contract_address")
        dependent_generators = fuzzer_context.get("dependent_generators", [])

        # Orchestration-time: no EVM context yet
        if interface is None or contract_address is None:
            return SeedSet(
                generator=None,
                dependent_generators=dependent_generators,
                analysis_result=analysis_result,
                rag_enabled=self._use_rag,
                llm_model=self._llm_model,
                llm_provider=self._llm_provider,
                openai_api_key=self._openai_api_key,
                ollama_endpoint=self._ollama_endpoint,
                disable_rag_llm=self._disable_rag_llm,
            )

        # Runtime: full EVM context available
        bytecode       = fuzzer_context["bytecode"]
        accounts       = fuzzer_context["accounts"]
        interface_mapper = fuzzer_context["interface_mapper"]
        raw_analysis   = self._to_raw_analysis(analysis_result)

        generator, rag_enabled = (None, False)
        if self._use_rag:
            generator, rag_enabled = self._try_rag_generator(
                interface, bytecode, accounts, contract_address,
                interface_mapper, dependent_generators,
                analysis_result.contract_name, analysis_result.contract_path,
                raw_analysis,
            )

        if generator is None:
            generator = self._create_standard_generator(
                interface, bytecode, accounts, contract_address,
                interface_mapper, dependent_generators,
                analysis_result.contract_name, analysis_result.contract_path,
            )

        return SeedSet(
            generator=generator,
            dependent_generators=dependent_generators,
            analysis_result=analysis_result,
            rag_enabled=rag_enabled,
            llm_model=self._llm_model,
            llm_provider=self._llm_provider,
            openai_api_key=self._openai_api_key,
            ollama_endpoint=self._ollama_endpoint,
            disable_rag_llm=self._disable_rag_llm,
        )

    def _to_raw_analysis(self, ar: AnalysisResult) -> Optional[Dict[str, Any]]:
        if not ar.critical_paths and not ar.test_sequences:
            return None
        return {
            "critical_paths": ar.critical_paths,
            "test_sequences":  ar.test_sequences,
            "vulnerabilities": ar.vulnerabilities,
        }

    def _try_rag_generator(
        self, interface, bytecode, accounts, contract_address,
        interface_mapper, dependent_generators, contract_name, sol_path, raw_analysis,
    ):
        try:
            from fuzzer.engine.components.rag_enhanced_generator import create_rag_enhanced_generator

            gen = create_rag_enhanced_generator(
                interface=interface, bytecode=bytecode, accounts=accounts,
                contract=contract_address, api_key=self._api_key,
                analysis_result=raw_analysis, contract_name=contract_name,
                sol_path=sol_path, other_generators=dependent_generators,
                interface_mapper=interface_mapper,
                max_individual_length=self._max_individual_length,
                llm_model=self._llm_model, llm_provider=self._llm_provider,
                openai_api_key=self._openai_api_key,
                disable_rag_llm=self._disable_rag_llm,
                adaptive_llm_controller=None,
            )
            logger.info("[Generator] RAG-enhanced generator created")
            return gen, True
        except ImportError:
            logger.warning("[Generator] RAG components not available")
        except Exception as exc:
            logger.warning("[Generator] RAG init failed: %s", exc)
        return None, False

    def _create_standard_generator(
        self, interface, bytecode, accounts, contract_address,
        interface_mapper, dependent_generators, contract_name, sol_path,
    ) -> Any:
        from fuzzer.engine.components.generator import Generator
        return Generator(
            interface=interface, bytecode=bytecode, accounts=accounts,
            contract=contract_address, other_generators=dependent_generators,
            interface_mapper=interface_mapper,
            contract_name=contract_name, sol_path=sol_path,
        )
