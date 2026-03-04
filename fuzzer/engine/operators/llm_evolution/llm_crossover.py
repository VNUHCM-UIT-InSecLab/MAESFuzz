#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-Based Crossover Operator

Replaces traditional crossover with LLM-driven intelligent combination.
Uses LLM + RAG to semantically combine two parent testcases into
offspring that explore new vulnerability patterns.
"""

import logging
import json
import random
from fuzzer.utils import settings
import ast
import re
from typing import Dict, List, Any, Optional, Tuple

from ...plugin_interfaces.operators.crossover import Crossover
from ...components.individual import Individual
from .llm_client import LLMClient
from .rag_context_provider import RAGContextProvider
from .execution_feedback import ExecutionFeedbackAnalyzer
from .testcase_formatter import TestcaseFormatter
from .coverage_context import CoverageContextProvider

try:
    from ..crossover.crossover import Crossover as TraditionalCrossover
    TRADITIONAL_CROSSOVER_AVAILABLE = True
except ImportError:
    TRADITIONAL_CROSSOVER_AVAILABLE = False

logger = logging.getLogger("LLMCrossover")


class LLMCrossover(Crossover):
    """LLM-driven crossover operator using Provider System"""

    def __init__(self,
                 llm_client: LLMClient,
                 crossover_probability: float = 0.3,
                 adaptive_llm_controller=None,
                 rag_context: Optional[RAGContextProvider] = None,
                 feedback_analyzer: Optional[ExecutionFeedbackAnalyzer] = None,
                 coverage_context: Optional[CoverageContextProvider] = None):
        """
        Initialize LLM crossover operator

        Args:
            llm_client: LLMClient instance (configured with Provider)
            crossover_probability: Probability to use LLM crossover (0.0-1.0)
            rag_context: RAG context provider
            feedback_analyzer: Execution feedback analyzer
            coverage_context: Coverage context provider
        """
        self.llm_client = llm_client
        self.crossover_probability = crossover_probability
        self.adaptive_llm_controller = adaptive_llm_controller
        self.rag_context = rag_context or RAGContextProvider()
        self.feedback_analyzer = feedback_analyzer or ExecutionFeedbackAnalyzer()
        self.coverage_context = coverage_context or CoverageContextProvider()
        self.formatter = None

        if TRADITIONAL_CROSSOVER_AVAILABLE:
            self.traditional_crossover_op = TraditionalCrossover(pc=0.8)
        else:
            self.traditional_crossover_op = None

        self.crossovers_performed = 0
        self.llm_crossovers = 0
        self.traditional_crossovers = 0
        self.successful_crossovers = 0
        self.failed_crossovers = 0

        logger.info(f"LLMCrossover initialized (prob={crossover_probability})")

    def cross(self, father, mother):
        """Cross two individuals using LLM or traditional crossover"""
        if mother is None:
            return father.clone(), father.clone()

        self.crossovers_performed += 1

        alpha = self._get_alpha()

        if random.random() > alpha:
            self.traditional_crossovers += 1
            return self._traditional_crossover(father, mother)

        self.llm_crossovers += 1

        if self.formatter is None and hasattr(father.generator, 'interface_mapper'):
            self.formatter = TestcaseFormatter(father.generator.interface_mapper)
        elif self.formatter is None:
            self.formatter = TestcaseFormatter()

        try:
            alpha = self._get_alpha()
            rand_val = random.random()
            logger.info("[LLM-CROSS] gen=%s alpha=%.3f rand=%.3f father_len=%d mother_len=%d",
                        getattr(settings, "CURRENT_ENGINE", None) and getattr(settings.CURRENT_ENGINE, "current_generation", -1),
                        alpha, rand_val, len(father.tx_sequence), len(mother.tx_sequence))
            if rand_val > alpha:
                self.traditional_crossovers += 1
                logger.info("[LLM-CROSS] using traditional crossover (rand > alpha)")
                return self._traditional_crossover(father, mother)

            logger.info("[LLM-CROSS] calling LLM for crossover")
            prompt = self._build_crossover_prompt(father, mother)
            llm_response = self.llm_client.generate_json(prompt, temperature=0.7)

            if llm_response and isinstance(llm_response, dict):
                child1, child2 = self._create_offspring(father, mother, llm_response)
                if child1 and child2:
                    self.successful_crossovers += 1
                    logger.info(
                        "LLMCrossover: success | child1_len=%d child2_len=%d | strategy=%s",
                        len(child1.tx_sequence),
                        len(child2.tx_sequence),
                        llm_response.get("strategy"),
                    )
                    if self.adaptive_llm_controller:
                        self.adaptive_llm_controller.record_llm_success()
                    return child1, child2

        except Exception as e:
            logger.debug(f"LLM crossover error: {e}")

        if self.adaptive_llm_controller:
            self.adaptive_llm_controller.record_llm_error()

        self.failed_crossovers += 1
        return self._traditional_crossover(father, mother)

    def _get_alpha(self) -> float:
        if self.adaptive_llm_controller is None:
            return self.crossover_probability

        engine = getattr(settings, "CURRENT_ENGINE", None)
        if engine is None:
            return self.crossover_probability

        try:
            current_cov = self._get_coverage_from_engine(engine)
            return self.adaptive_llm_controller.get_alpha(
                getattr(engine, "current_generation", 0),
                current_cov,
            )
        except Exception:
            return self.crossover_probability

    def _build_crossover_prompt(self, father, mother) -> str:
        """Build crossover prompt with contract context"""
        prompt_parts = []

        prompt_parts.append("# Smart Contract Fuzzing - Crossover Request\n")

        contract_name = getattr(father.generator, 'contract_name', 'Unknown')
        prompt_parts.append(f"## Contract: {contract_name}\n")

        source_code = getattr(father.generator, 'source_code', None)
        if not source_code:
            sol_path = getattr(father.generator, 'sol_path', None)
            if sol_path:
                try:
                    with open(sol_path, 'r') as f:
                        source_code = f.read()
                except Exception:
                    pass

        if source_code:
            source_excerpt = source_code[:1500] + "\n... (truncated)" if len(source_code) > 1500 else source_code
            prompt_parts.append("## Contract Source Code:")
            prompt_parts.append("```solidity")
            prompt_parts.append(source_excerpt)
            prompt_parts.append("```\n")

        try:
            from fuzzer.utils import settings
            env = getattr(settings, 'GLOBAL_ENV', None)
            if env:
                covered_pcs = len(getattr(env, 'code_coverage', set()))
                total_pcs = len(getattr(env, 'overall_pcs', set()))
                if total_pcs > 0:
                    code_cov = (covered_pcs / total_pcs) * 100
                    prompt_parts.append(f"## Coverage: {code_cov:.1f}% ({covered_pcs}/{total_pcs} PCs)")

                    all_pcs = set(env.overall_pcs)
                    uncovered_pcs = all_pcs - env.code_coverage
                    if uncovered_pcs:
                        sample_uncovered = list(uncovered_pcs)[:3]
                        prompt_parts.append(f"## UNCOVERED PCs: {[hex(pc) for pc in sample_uncovered]}")
        except Exception:
            pass

        p1_len = len(father.chromosome)
        p2_len = len(mother.chromosome)

        prompt_parts.append(f"\n## Parents:")
        prompt_parts.append(f"Parent1: {p1_len} transactions")
        prompt_parts.append(f"Parent2: {p2_len} transactions")

        prompt_parts.append(f"""
## Task: Combine 2 parents into 2 children to maximize coverage

Respond with ONLY this JSON:
{{"strategy": "sequential", "reasoning": "combine both", "expected_outcome": "more coverage", "child1": {{"description": "first half of each", "transactions": [{{"source": "parent1", "transaction_index": 0}}, {{"source": "parent2", "transaction_index": 0}}]}}, "child2": {{"description": "second half", "transactions": [{{"source": "parent2", "transaction_index": 0}}, {{"source": "parent1", "transaction_index": 0}}]}}}}

Rules:
- source: "parent1" or "parent2"
- transaction_index: 0 to {min(p1_len, p2_len) - 1}
- Return ONLY JSON, nothing else""")

        return "\n".join(prompt_parts)

    def _create_offspring(self, father, mother, llm_response: Dict) -> Tuple[Optional[object], Optional[object]]:
        """Create offspring based on LLM response"""
        try:
            child1_spec = llm_response.get("child1", {})
            child2_spec = llm_response.get("child2", {})

            child1 = self._build_child_from_spec(child1_spec, father, mother)
            child2 = self._build_child_from_spec(child2_spec, father, mother)

            if child1 and child2:
                return child1, child2
            else:
                return None, None

        except Exception as e:
            logger.error(f"Failed to create offspring: {e}")
            return None, None

    def _build_child_from_spec(self, spec: Dict, father, mother) -> Optional[object]:
        """Build child individual from LLM specification"""
        try:
            transactions = spec.get("transactions", [])
            if not transactions:
                return None

            child = Individual(generator=father.generator,
                             other_generators=father.other_generators)

            chromosome = []

            for tx_spec in transactions:
                source = tx_spec.get("source", "parent1")
                tx_index = tx_spec.get("transaction_index", 0)

                if source == "parent1" and tx_index < len(father.chromosome):
                    base_tx = father.chromosome[tx_index].copy()
                elif source == "parent2" and tx_index < len(mother.chromosome):
                    base_tx = mother.chromosome[tx_index].copy()
                elif source == "new":
                    base_tx = self._create_new_transaction(tx_spec, father.generator)
                else:
                    continue

                if not base_tx:
                    continue

                modifications = tx_spec.get("modifications", {})
                if modifications:
                    self._apply_modifications(base_tx, modifications, father.generator)

                chromosome.append(base_tx)

            if chromosome:
                for tx in chromosome:
                    func_hash = tx.get("arguments", [None])[0]
                    if func_hash and hasattr(father.generator, 'interface'):
                        param_types = father.generator.interface.get(func_hash, [])
                        current_args = tx.get("arguments", [])
                        actual_arg_count = len(current_args) - 1 if current_args else 0

                        if actual_arg_count < len(param_types):
                            for arg_idx in range(actual_arg_count, len(param_types)):
                                arg_type = param_types[arg_idx]
                                random_val = father.generator.get_random_argument(arg_type, func_hash, arg_idx)
                                tx["arguments"].append(random_val)

                child.chromosome = chromosome
                try:
                    child.solution = child.decode()
                    return child
                except Exception as decode_error:
                    logger.error(f"Failed to decode child chromosome: {decode_error}")
                    return None
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to build child: {e}")
            return None

    def _create_new_transaction(self, tx_spec: Dict, generator) -> Optional[Dict]:
        """Create new transaction from specification"""
        try:
            function_sig = tx_spec.get("function", "")
            function_hash = self._get_function_hash_from_name(function_sig, generator)

            if not function_hash or function_hash not in generator.interface:
                return None

            tx = {
                "account": generator.get_random_account(function_hash),
                "contract": generator.contract,
                "amount": 0,
                "gaslimit": generator.get_random_gaslimit(function_hash),
                "arguments": [function_hash],
                "timestamp": generator.get_random_timestamp(function_hash),
                "blocknumber": generator.get_random_blocknumber(function_hash),
                "balance": generator.get_random_balance(function_hash),
                "call_return": {},
                "extcodesize": {},
                "returndatasize": {}
            }

            if "arguments" in tx_spec and isinstance(tx_spec["arguments"], list):
                for arg_idx, arg_type in enumerate(generator.interface[function_hash]):
                    if arg_idx < len(tx_spec["arguments"]):
                        llm_value = tx_spec["arguments"][arg_idx]
                        arg_value = self._parse_llm_argument(llm_value, arg_type)
                    else:
                        arg_value = generator.get_random_argument(arg_type, function_hash, arg_idx)
                    tx["arguments"].append(arg_value)
            else:
                for arg_idx, arg_type in enumerate(generator.interface[function_hash]):
                    arg_value = generator.get_random_argument(arg_type, function_hash, arg_idx)
                    tx["arguments"].append(arg_value)

            return tx

        except Exception as e:
            logger.error(f"Failed to create new transaction: {e}")
            return None

    def _apply_modifications(self, tx: Dict, modifications: Dict, generator):
        """Apply modifications to transaction"""
        for field, value in modifications.items():
            if field == "account" or field == "from":
                tx["account"] = value
            elif field == "amount" or field == "value":
                tx["amount"] = self._parse_transaction_value(value)
            elif field == "gaslimit" or field == "gas":
                tx["gaslimit"] = self._parse_value(value)
            elif field == "arguments":
                if isinstance(value, list):
                    func_hash = tx["arguments"][0] if tx["arguments"] else None
                    param_types = []
                    if func_hash and hasattr(generator, 'interface') and func_hash in generator.interface:
                        param_types = generator.interface[func_hash]

                    if len(value) == 0 and len(param_types) > 0:
                        continue

                    if len(param_types) == 0:
                        tx["arguments"] = [func_hash]
                    elif len(param_types) == 1 and '[]' in param_types[0] and len(value) > 1:
                        if not isinstance(value[0], list):
                            base_type = param_types[0].replace('[]', '')
                            parsed_array = [self._parse_llm_argument(v, base_type) for v in value]
                            tx["arguments"] = [func_hash, parsed_array]
                        else:
                            parsed_val = self._parse_llm_argument(value[0], param_types[0])
                            tx["arguments"] = [func_hash, parsed_val]
                    else:
                        tx["arguments"] = [func_hash]
                        for idx, arg_val in enumerate(value):
                            arg_type = ""
                            if idx < len(param_types):
                                arg_type = param_types[idx]
                            parsed_val = self._parse_llm_argument(arg_val, arg_type)
                            tx["arguments"].append(parsed_val)
            elif field.startswith("arg_") or field.startswith("argument_"):
                arg_index = int(field.split("_")[1])
                if arg_index < len(tx["arguments"]) - 1:
                    arg_type = ""
                    func_hash = tx["arguments"][0]
                    if func_hash and hasattr(generator, 'interface') and func_hash in generator.interface:
                        if arg_index < len(generator.interface[func_hash]):
                            arg_type = generator.interface[func_hash][arg_index]
                    parsed_val = self._parse_llm_argument(value, arg_type)
                    tx["arguments"][arg_index + 1] = parsed_val

    def _parse_value(self, value):
        """Parse value from LLM with bounds checking"""
        MAX_UINT256 = 2**256 - 1

        if isinstance(value, (int, float)):
            result = int(value)
            return max(0, min(result, MAX_UINT256))

        if isinstance(value, str):
            if value.lower() == "max":
                return MAX_UINT256
            elif value.lower() == "zero":
                return 0
            elif value.startswith("0x"):
                result = int(value, 16)
                return min(result, MAX_UINT256)
            else:
                try:
                    result = int(value)
                    return max(0, min(result, MAX_UINT256))
                except ValueError:
                    return value

        return value

    def _parse_transaction_value(self, value):
        """Parse transaction value with realistic bounds"""
        MAX_TX_VALUE = 10000 * 10**18

        if isinstance(value, (int, float)):
            result = int(value)
            return max(0, min(result, MAX_TX_VALUE))

        if isinstance(value, str):
            if value.lower() == "max":
                return MAX_TX_VALUE
            elif value.lower() == "zero":
                return 0
            elif value.startswith("0x"):
                result = int(value, 16)
                return min(result, MAX_TX_VALUE)
            else:
                try:
                    result = int(value)
                    return max(0, min(result, MAX_TX_VALUE))
                except ValueError:
                    return 0

        return 0

    def _parse_llm_argument(self, value, arg_type: str = ""):
        """Parse LLM-generated argument value"""
        if isinstance(value, list):
            return [self._parse_llm_argument(v, arg_type.replace('[]', '')) for v in value]

        if 'address' in arg_type.lower():
            if isinstance(value, int):
                hex_str = hex(value)[2:]
                if len(hex_str) > 40:
                    hex_str = hex_str[:40]
                else:
                    hex_str = hex_str.zfill(40)
                return f"0x{hex_str}"

            if isinstance(value, str) and value.startswith('0x'):
                hex_part = value[2:]
                if len(hex_part) > 40:
                    hex_part = hex_part[:40]
                elif len(hex_part) < 40:
                    hex_part = hex_part.ljust(40, '0')
                    return f"0x{hex_part}"

            if isinstance(value, str):
                return "0x0000000000000000000000000000000000000000"

        MAX_UINT256 = 2**256 - 1
        MAX_ADDRESS = 2**160 - 1

        if isinstance(value, int):
            if 'uint' in arg_type.lower():
                if value > MAX_UINT256:
                    return MAX_UINT256
                elif value < 0:
                    return 0
                return value

            if not arg_type:
                if value <= MAX_ADDRESS:
                    hex_str = hex(value)[2:].zfill(40)
                    return f"0x{hex_str}"
                elif value <= MAX_UINT256:
                    return value
                else:
                    address_value = value % (2**160)
                    hex_str = hex(address_value)[2:].zfill(40)
                    return f"0x{hex_str}"

            if value > MAX_UINT256:
                return MAX_UINT256

            return value

        if not isinstance(value, str):
            return value

        if 'generate_large_array' in value or 'generate_array' in value:
            return []

        if value.startswith('generate_') or value.endswith('()'):
            if '[]' in arg_type:
                return []
            else:
                return 0

        if value.startswith('[') or value.startswith('{'):
            try:
                parsed = json.loads(value)
                return parsed
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(value)
                    return parsed
                except:
                    if '[]' in arg_type:
                        return []

        result = self._parse_value(value)

        if isinstance(result, int) and result > MAX_UINT256:
                result = MAX_UINT256

        return result

    def _get_function_hash_from_name(self, function_name: str, generator) -> Optional[str]:
        """Get function hash from function name/signature"""
        try:
            clean_name = function_name.strip()

            if re.match(r'^(0x)?[0-9a-fA-F]{8}$', clean_name):
                if not clean_name.startswith('0x'):
                    clean_name = '0x' + clean_name

                if hasattr(generator, 'interface') and clean_name in generator.interface:
                    return clean_name

                clean_name_lower = clean_name.lower()
                if hasattr(generator, 'interface') and clean_name_lower in generator.interface:
                    return clean_name_lower

            if hasattr(generator, 'interface_mapper'):
                interface_mapper = generator.interface_mapper

                if isinstance(interface_mapper, dict):
                    if function_name in interface_mapper:
                        return interface_mapper[function_name]

                    for sig, hash_val in interface_mapper.items():
                        if sig.startswith(function_name + "(") or sig == function_name:
                            return hash_val

                    for sig, hash_val in interface_mapper.items():
                        if hash_val == function_name or hash_val == '0x' + function_name:
                                return hash_val

            if hasattr(generator, 'interface'):
                test_hash = function_name if function_name.startswith('0x') else '0x' + function_name
                if test_hash in generator.interface:
                    return test_hash

            return None

        except Exception as e:
            logger.error(f"Error getting function hash: {e}")
            return None

    def _traditional_crossover(self, father, mother):
        """Traditional genetic crossover as fallback"""
        if self.traditional_crossover_op is not None:
            return self.traditional_crossover_op.cross(father, mother)
        else:
            return father.clone(), mother.clone()

    def get_statistics(self) -> Dict[str, Any]:
        """Get crossover statistics"""
        return {
            "total_crossovers": self.crossovers_performed,
            "llm_crossovers": self.llm_crossovers,
            "traditional_crossovers": self.traditional_crossovers,
            "successful_crossovers": self.successful_crossovers,
            "failed_crossovers": self.failed_crossovers,
            "llm_client_stats": self.llm_client.get_statistics()
        }
