from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    contract_path: str
    contract_name: str
    solc_version: str
    evm_version: str
    solc_path: str

    depend_contracts: List[str] = field(default_factory=list)
    constructor_args: List[Any] = field(default_factory=list)

    dataflow_graph: Optional[Dict[str, Any]] = None
    critical_paths: List[Any] = field(default_factory=list)
    test_sequences: List[Any] = field(default_factory=list)
    vulnerabilities: List[Any] = field(default_factory=list)

    _slither_instance: Any = field(default=None, repr=False)


class AnalyzerAgent:
    """Static contract analysis + LLM-guided dataflow extraction.

    Produces an AnalysisResult consumed by GeneratorAgent.
    """

    def __init__(
        self,
        solc_version: str,
        solc_path: str,
        evm_version: str,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        ollama_endpoint: Optional[str] = None,
        constructor_params_path: str = "auto",
    ) -> None:
        self._solc_version = solc_version
        self._solc_path = solc_path
        self._evm_version = evm_version
        self._api_key = api_key
        self._openai_api_key = openai_api_key
        self._llm_model = llm_model
        self._llm_provider = llm_provider
        self._ollama_endpoint = ollama_endpoint
        self._constructor_params_path = constructor_params_path

    def analyze(self, contract_path: str, contract_name: str) -> AnalysisResult:
        logger.info("[Analyzer] %s::%s", contract_path, contract_name)

        result = AnalysisResult(
            contract_path=contract_path,
            contract_name=contract_name,
            solc_version=self._solc_version,
            evm_version=self._evm_version,
            solc_path=self._solc_path,
        )

        self._extract_structure(result)
        self._run_llm_analysis(result)

        logger.info(
            "[Analyzer] done: deps=%s args=%d paths=%d seqs=%d",
            result.depend_contracts,
            len(result.constructor_args),
            len(result.critical_paths),
            len(result.test_sequences),
        )
        return result

    def _extract_structure(self, result: AnalysisResult) -> None:
        from fuzzer.utils.comp import analysis_depend_contract
        from fuzzer.engine.analyzers.Analysis import analysis_main_contract_constructor

        try:
            depend_contracts, slither_instance = analysis_depend_contract(
                file_path=result.contract_path,
                _contract_name=result.contract_name,
                _solc_version=result.solc_version,
                _solc_path=result.solc_path,
            )
            result.depend_contracts = depend_contracts or []
            result._slither_instance = slither_instance
        except Exception as exc:
            logger.warning("[Analyzer] Slither failed: %s", exc)
            result.depend_contracts = []
            result._slither_instance = None

        if self._constructor_params_path != "auto":
            result.constructor_args = self._load_constructor_params_from_file()
        else:
            try:
                args = analysis_main_contract_constructor(
                    file_path=result.contract_path,
                    contract_name=result.contract_name,
                    sl=result._slither_instance,
                )
                result.constructor_args = args or []
            except Exception as exc:
                logger.warning("[Analyzer] Constructor extraction failed: %s", exc)
                result.constructor_args = []

    def _load_constructor_params_from_file(self) -> List[Any]:
        import json
        try:
            with open(self._constructor_params_path, "r", encoding="utf-8") as fh:
                params = json.load(fh)
            return [
                f"{name} {detail['type']} {detail['value']}"
                for name, detail in params.items()
            ]
        except Exception as exc:
            logger.warning("[Analyzer] Could not load constructor params: %s", exc)
            return []

    def _run_llm_analysis(self, result: AnalysisResult) -> None:
        if not result.contract_path or os.environ.get("SKIP_ANALYZER"):
            return

        self._ensure_provider_context()

        try:
            from fuzzer.engine.analyzers.dataflow_analyzer import SmartContractAnalyzer

            data = SmartContractAnalyzer(
                sol_path=result.contract_path,
                api_key=self._api_key,
                solc_path=result.solc_path,
            ).analyze()

            ar = data.get("analysis_result") or {}
            result.dataflow_graph = data.get("dataflow_graph")
            result.critical_paths = ar.get("critical_paths", [])
            result.test_sequences  = ar.get("test_sequences", [])
            result.vulnerabilities = ar.get("vulnerabilities", [])

            logger.info(
                "[Analyzer] LLM: %d paths, %d seqs, %d vulns",
                len(result.critical_paths),
                len(result.test_sequences),
                len(result.vulnerabilities),
            )
        except Exception as exc:
            logger.warning("[Analyzer] LLM analysis skipped: %s", exc)

    def _ensure_provider_context(self) -> None:
        """Initialise the global provider context if not already set."""
        try:
            from provider import context as provider_context
            from provider import factory as provider_factory
            import config

            if provider_context.get_provider(optional=True) is not None:
                return

            model     = self._llm_model    or config.get_default_llm_model()
            provider  = self._llm_provider or config.get_default_llm_provider()
            endpoint  = self._ollama_endpoint or config.get_default_ollama_endpoint()

            kwargs = {
                "ollama_endpoint": endpoint,
                "ollama_thinking": config.get_default_ollama_thinking(),
            }

            prov_lower = (provider or "").lower()
            if prov_lower == "openai":
                key = self._openai_api_key or os.environ.get("OPENAI_API_KEY")
                if not key:
                    logger.warning("[Analyzer] No OpenAI key — LLM analysis skipped")
                    return
                instance = provider_factory.create_provider(
                    model=model, provider_name=provider,
                    openai_api_key=key, default_provider=provider, **kwargs,
                )
            elif prov_lower in ("gemini", "google"):
                key = self._api_key or config.get_google_api_key()
                if not key:
                    logger.warning("[Analyzer] No Google key — LLM analysis skipped")
                    return
                instance = provider_factory.create_provider(
                    model=model, provider_name=provider,
                    api_key=key, default_provider=provider, **kwargs,
                )
            elif prov_lower == "ollama":
                instance = provider_factory.create_provider(
                    model=model, provider_name=provider,
                    default_provider=provider, **kwargs,
                )
            else:
                # Auto-detect from model name prefix
                if model and (model.lower().startswith("gpt") or model.lower().startswith("o1")):
                    key = self._openai_api_key or os.environ.get("OPENAI_API_KEY")
                    if not key:
                        return
                    instance = provider_factory.create_provider(
                        model=model, provider_name="openai",
                        openai_api_key=key, default_provider="openai", **kwargs,
                    )
                else:
                    key = self._api_key or config.get_google_api_key()
                    if not key:
                        return
                    instance = provider_factory.create_provider(
                        model=model, provider_name=provider,
                        api_key=key, default_provider=provider, **kwargs,
                    )

            provider_context.set_provider(instance)
            logger.info("[Analyzer] Provider ready: %s / %s", provider, model)
        except Exception as exc:
            logger.warning("[Analyzer] Provider init failed: %s", exc)
