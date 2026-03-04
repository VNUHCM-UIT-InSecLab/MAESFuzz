#!/usr/bin/env python3
"""MAESFuzz — Multi-Agent Evolutionary Smart Contract Fuzzer.

Usage:
  python MAESFuzz.py <contract.sol> <ContractName> [options]

Options:
  --solc-version <ver>         Solidity compiler version  (default: 0.8.26)
  --fuzz-time <sec>            Fuzzing duration in seconds (default: 60)
  -g, --generations <n>        Number of generations instead of time budget
  --max-trans-length <n>       Max transaction sequence length (default: 10)
  --result-path <path>         Output JSON path (default: result/results.json)
  --constructor-params <path>  Path to constructor params JSON, or "auto"
  --duplication <0|1>          Allow duplicate transactions (default: 0)
  --no-rag                     Disable RAG-enhanced seed generation
  --api-key <key>              Google / Gemini API key
  --openai-api-key <key>       OpenAI API key
  --model <name>               LLM model  (default: gpt-4o)
  --provider <name>            LLM provider: openai | gemini | ollama | auto
  --ollama-endpoint <url>      Ollama endpoint (default: http://127.0.0.1:11434)
  --ollama-thinking <val>      Enable thinking trace for Ollama
  --use-llm-judge              Enable LLM-as-a-Judge evaluation
  --judge-threshold <0-100>    Confidence threshold for LLM judge (default: 60)

Examples:
  python MAESFuzz.py examples/reentrance.sol Reentrance
  python MAESFuzz.py examples/reentrance.sol Reentrance --fuzz-time 120 --openai-api-key sk-...
  python MAESFuzz.py examples/reentrance.sol Reentrance --provider ollama --model deepseek-r1:7b
"""

from __future__ import annotations

import os
import sys

import config
from config import (
    get_solc_path, get_default_solc_version, get_default_fuzz_time,
    get_default_max_trans_length, get_default_duplication, get_default_result_path,
    get_default_constructor_params, get_evm_version_for_solc, get_default_use_rag,
    get_google_api_key, get_default_llm_model, get_default_llm_provider,
    get_default_ollama_endpoint, get_default_ollama_thinking,
    get_default_use_llm_judge, get_default_judge_threshold,
)
from agents.pipeline import MAESFuzzPipeline, PipelineConfig


def _parse_argv() -> PipelineConfig:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    contract_path = sys.argv[1]
    contract_name = sys.argv[2]

    solc_version          = get_default_solc_version()
    max_trans_length      = get_default_max_trans_length()
    fuzz_time             = get_default_fuzz_time()
    generations           = None
    result_saved_path     = get_default_result_path()
    solc_path             = get_solc_path()
    constructor_params    = get_default_constructor_params()
    duplication           = get_default_duplication()
    use_rag               = get_default_use_rag()
    api_key               = get_google_api_key()
    openai_api_key        = os.environ.get("OPENAI_API_KEY")
    llm_model             = get_default_llm_model()
    llm_provider          = get_default_llm_provider()
    ollama_endpoint       = get_default_ollama_endpoint()
    ollama_thinking       = get_default_ollama_thinking()
    use_llm_judge         = get_default_use_llm_judge()
    judge_threshold       = get_default_judge_threshold()

    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        def _next():
            nonlocal i
            i += 1
            return sys.argv[i]
        if   arg == "--solc-version":        solc_version       = _next()
        elif arg == "--max-trans-length":    max_trans_length   = int(_next())
        elif arg == "--fuzz-time":           fuzz_time          = int(_next())
        elif arg in ("-g", "--generations"): generations        = int(_next())
        elif arg == "--result-path":         result_saved_path  = _next()
        elif arg == "--constructor-params":  constructor_params = _next()
        elif arg == "--duplication":         duplication        = _next()
        elif arg == "--no-rag":              use_rag            = False
        elif arg == "--api-key":             api_key            = _next()
        elif arg == "--openai-api-key":      openai_api_key     = _next()
        elif arg == "--model":               llm_model          = _next()
        elif arg == "--provider":            llm_provider       = _next()
        elif arg == "--ollama-endpoint":     ollama_endpoint    = _next()
        elif arg == "--ollama-thinking":     ollama_thinking    = _next()
        elif arg == "--use-llm-judge":       use_llm_judge      = True
        elif arg == "--judge-threshold":     judge_threshold    = float(_next())
        i += 1

    evm_version = get_evm_version_for_solc(solc_version)
    os.makedirs(os.path.dirname(result_saved_path) or ".", exist_ok=True)

    return PipelineConfig(
        contract_path=contract_path,
        contract_name=contract_name,
        solc_version=solc_version,
        evm_version=evm_version,
        solc_path=solc_path,
        fuzz_time=fuzz_time,
        generations=generations,
        max_individual_length=max_trans_length,
        duplication=duplication,
        result_json_path="result/res.json",
        result_saved_path=result_saved_path,
        constructor_params_path=constructor_params,
        use_rag=use_rag,
        api_key=api_key,
        openai_api_key=openai_api_key,
        llm_model=llm_model,
        llm_provider=llm_provider,
        ollama_endpoint=ollama_endpoint,
        ollama_thinking=ollama_thinking,
        use_llm_judge=use_llm_judge,
        judge_threshold=judge_threshold,
        use_llm_report=True,
        _python_executable=sys.executable,
        _fuzzer_script="fuzzer/main.py",
    )


if __name__ == "__main__":
    MAESFuzzPipeline(_parse_argv()).run()
