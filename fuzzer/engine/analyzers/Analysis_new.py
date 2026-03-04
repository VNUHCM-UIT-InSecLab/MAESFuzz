#!/usr/bin/env python3
import re
import json
import logging
import os
import argparse
import subprocess
from typing import Optional

from slither.slither import Slither
from slither.core.expressions import Identifier, TypeConversion, AssignmentOperation

from provider import context as provider_context
from provider.factory import create_provider
from provider.base import ProviderError, ProviderResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_solc_version(contract_path):
    with open(contract_path, "r") as f:
        content = f.read()
    match = re.search(r'pragma\s+solidity\s+([^;]+);', content)
    if match:
        raw = match.group(1).strip()
        vm = re.search(r'(\d+\.\d+\.\d+)', raw)
        if vm:
            return vm.group(1)
    return "unknown"


def remove_redundant_fields(data):
    keys_to_remove = {
        "lines", "filename_used", "filename_relative",
        "filename_absolute", "filename_short",
        "first_markdown_element", "id", "start",
        "length", "is_dependency",
        "starting_column", "ending_column"
    }
    if isinstance(data, dict):
        return {k: remove_redundant_fields(v) for k, v in data.items() if k not in keys_to_remove}
    elif isinstance(data, list):
        return [remove_redundant_fields(item) for item in data]
    return data


def extract_slither_data(contract_path, solc_path):
    if os.path.exists("./slither_output.json"):
        os.remove("./slither_output.json")

    result = subprocess.run(
        ["slither", contract_path, "--json", "./slither_output.json", "--solc", solc_path],
        capture_output=True, text=True
    )
    logging.info("Slither stdout:\n%s", result.stdout)
    logging.info("Slither stderr:\n%s", result.stderr)

    if not os.path.exists("./slither_output.json"):
        logging.warning("slither_output.json not found")
        return None

    with open("./slither_output.json", "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            logging.warning("JSON parse error: %s", e)
            return None

    data = remove_redundant_fields(data)

    with open("./slither_output.json", "w") as file:
        json.dump(data, file, indent=4)

    solc_version = extract_solc_version(contract_path)
    contracts = data.get("results", {}).get("contracts", [])
    main_contract = contracts[-1].get("name") if contracts else None

    constructor_args = []
    if main_contract:
        try:
            constructor_args = analysis_main_contract_constructor(contract_path, main_contract, solc_path=solc_path)
        except Exception as e:
            logging.warning("Constructor analysis failed: %s", e)

    updated_data = {
        "solc_version": solc_version,
        **data,
        "constructor_args": constructor_args
    }
    return updated_data


def extract_param_contract_map(conversion):
    if hasattr(conversion, 'expression') and isinstance(conversion.expression, Identifier):
        return conversion.expression.value.name, str(conversion.type)
    return None, None


def analysis_main_contract_constructor(file_path, contract_name, sl=None, solc_path="/usr/bin/solc"):
    try:
        if sl is None:
            sl = Slither(file_path, solc=solc_path)
        contracts = sl.get_contract_from_name(contract_name)
        if not contracts or len(contracts) != 1:
            logging.error(f"No contract or multiple contracts found for name: {contract_name}")
            return []
        contract = contracts[0]

        constructor = contract.constructor
        if not constructor:
            return []

        params = []
        for p in constructor.parameters:
            ptype = p.type.name if hasattr(p.type, "name") else str(p.type)
            params.append({"name": p.name or "unnamed", "type": ptype, "value": None})

        for expr in constructor.expressions:
            if isinstance(expr, AssignmentOperation):
                left, right = expr.expression_left, expr.expression_right
                if isinstance(right, Identifier) and isinstance(left, Identifier):
                    for param in params:
                        if param["name"] == right.value.name:
                            param["value"] = left.value.name
                elif isinstance(right, TypeConversion) and isinstance(left, Identifier):
                    pname, contract_map = extract_param_contract_map(right)
                    if pname:
                        for param in params:
                            if param["name"] == pname:
                                param["value"] = contract_map
            elif isinstance(expr, TypeConversion):
                pname, contract_map = extract_param_contract_map(expr)
                if pname:
                    for param in params:
                        if param["name"] == pname:
                            param["value"] = contract_map

        for param in params:
            if param["value"] is None:
                if "address" in param["type"]:
                    param["value"] = "0x0000000000000000000000000000000000000000"
                else:
                    param["value"] = "0"

        return params
    except Exception as e:
        logging.warning(f"Constructor analysis failed: {e}")
        return []


def init_provider(
    model: str,
    provider_name: Optional[str],
    api_key: Optional[str],
    openai_api_key: Optional[str],
    ollama_endpoint: Optional[str],
):
    provider_context.clear_provider()
    provider = create_provider(
        model=model,
        provider_name=provider_name,
        api_key=api_key,
        openai_api_key=openai_api_key,
        ollama_endpoint=ollama_endpoint,
    )
    provider_context.set_provider(provider)
    return provider_context.get_provider()


def call_provider(prompt: str) -> Optional[str]:
    provider = provider_context.get_provider(optional=True)
    if provider is None:
        logging.error("No provider initialized")
        return None
    try:
        result: ProviderResult = provider.generate(prompt, temperature=1)
        return result.text
    except ProviderError as e:
        logging.error(f"Provider error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Smart Contract Analysis with provider system")
    parser.add_argument("--contract-path", required=True, help="Path to the Solidity smart contract")
    parser.add_argument("--solc-path", default="/usr/bin/solc", help="Path to solc binary")
    parser.add_argument("--output", default="analysis_output.txt", help="File to write analysis result")
    parser.add_argument("--model", required=True, help="LLM model name, e.g., gpt-4o-mini")
    parser.add_argument("--provider", default=None, help="Provider name: openai|gemini|ollama")
    parser.add_argument("--api-key", default=None, help="Gemini API key (if using Gemini)")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key")
    parser.add_argument("--ollama-endpoint", default=None, help="Ollama endpoint")
    args = parser.parse_args()

    try:
        init_provider(
            model=args.model,
            provider_name=args.provider,
            api_key=args.api_key,
            openai_api_key=args.openai_api_key or os.environ.get("OPENAI_API_KEY"),
            ollama_endpoint=args.ollama_endpoint,
        )
    except ProviderError as e:
        logging.error(f"Provider initialization failed: {e}")
        return

    if not os.path.exists(args.contract_path):
        logging.error(f"Contract not found: {args.contract_path}")
        return

    slither_data = extract_slither_data(args.contract_path, args.solc_path)
    if not slither_data:
        logging.error("Slither analysis failed")
        return

    with open("./slither_output.json", "r") as f:
        lines = [line for _, line in zip(range(200), f)]
        slither_json_str = ''.join(lines)

    max_chars = 20000
    if len(slither_json_str) > max_chars:
        logging.warning("Slither JSON too long, truncating for prompt")
        slither_json_str = slither_json_str[:max_chars]

    prompt = f"""
You are an expert blockchain security analyst.
You analyze a Solidity contract at {args.contract_path}.
Below is truncated Slither JSON output:

{slither_json_str}

Return ONLY JSON with keys:
- contract_name
- file_path
- solc_version
- depend_contracts
- constructor_args (list of objects: name, type, value)
- functions (list of objects: name, visibility, parameters)
- vulnerabilities (list of objects: name, severity, location, description, swc_id if applicable)
- functions_to_fuzz (list of function names that are public/external and risky)
- sequences (step-by-step interactions to trigger vulnerabilities)
- seeds (concrete input values for constructor/addresses/uints/calldata)
- testcases (id, sequence_id, seed, expected_outcome, reproducer)
- fuzzing_recommendations (per-function guidance)
"""

    llm_resp = call_provider(prompt)
    if llm_resp is None:
        logging.warning("No response from provider")
        return

    try:
        with open(args.output, "w") as f:
            f.write(llm_resp)
        logging.info(f"Analysis saved to {args.output}")
        print("LLM Analysis Result:")
        print(llm_resp)
    except Exception as e:
        logging.error(f"Failed to write analysis result: {e}")


if __name__ == "__main__":
    main()
