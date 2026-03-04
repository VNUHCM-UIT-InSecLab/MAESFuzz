#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-Based Mutation Operator

Replaces traditional random mutation with LLM-driven intelligent mutation.
Uses LLM + RAG to make informed decisions about how to mutate testcases
based on execution feedback, coverage, and vulnerability knowledge.
"""

import logging
import json
import random
from fuzzer.utils import settings
import ast
import re
from typing import Dict, List, Any, Optional

from ...plugin_interfaces.operators.mutation import Mutation
from .llm_client import LLMClient
from .rag_context_provider import RAGContextProvider
from .execution_feedback import ExecutionFeedbackAnalyzer
from .testcase_formatter import TestcaseFormatter
from .coverage_context import CoverageContextProvider

try:
    from ..mutation.mutation import Mutation as TraditionalMutation
    TRADITIONAL_MUTATION_AVAILABLE = True
except ImportError:
    TRADITIONAL_MUTATION_AVAILABLE = False

logger = logging.getLogger("LLMMutation")


class LLMMutation(Mutation):
    """LLM-driven mutation operator using Provider System"""

    def __init__(self,
                 llm_client: LLMClient,
                 mutation_probability: float = 0.5,
                 adaptive_llm_controller=None,
                 rag_context: Optional[RAGContextProvider] = None,
                 feedback_analyzer: Optional[ExecutionFeedbackAnalyzer] = None,
                 coverage_context: Optional[CoverageContextProvider] = None):
        """
        Initialize LLM mutation operator

        Args:
            llm_client: LLMClient instance (configured with Provider)
            mutation_probability: Probability to use LLM mutation (0.0-1.0)
            rag_context: RAG context provider
            feedback_analyzer: Execution feedback analyzer
            coverage_context: Coverage context provider
        """
        self.llm_client = llm_client
        self.mutation_probability = mutation_probability
        self.adaptive_llm_controller = adaptive_llm_controller
        self.rag_context = rag_context or RAGContextProvider()
        self.feedback_analyzer = feedback_analyzer or ExecutionFeedbackAnalyzer()
        self.coverage_context = coverage_context or CoverageContextProvider()
        self.formatter = None

        if TRADITIONAL_MUTATION_AVAILABLE:
            self.traditional_mutator = TraditionalMutation(pm=0.1)
        else:
            self.traditional_mutator = None

        self.mutations_performed = 0
        self.llm_mutations = 0
        self.traditional_mutations = 0
        self.failed_mutations = 0

        logger.info(f"LLMMutation initialized (prob={mutation_probability})")

    def mutate(self, individual, engine):
        """Mutate individual using LLM or traditional mutation"""
        self.mutations_performed += 1

        alpha = self._get_alpha(engine)
        rand_val = random.random()

        # In UniFuzz, the sequence is stored in `chromosome`, not `tx_sequence`
        tx_seq = getattr(individual, "chromosome", None)
        tx_count = len(tx_seq) if tx_seq is not None else 0

        logger.info(
            "[LLM-MUTATE] gen=%s alpha=%.3f rand=%.3f tx_count=%d",
            getattr(engine, "current_generation", -1),
            alpha,
            rand_val,
            tx_count,
            )

        if rand_val > alpha:
            self.traditional_mutations += 1
            logger.info("[LLM-MUTATE] using traditional mutation (rand > alpha)")
        return self._traditional_mutation(individual, engine)

        self.llm_mutations += 1

        if self.formatter is None and hasattr(individual.generator, 'interface_mapper'):
            self.formatter = TestcaseFormatter(individual.generator.interface_mapper)
        elif self.formatter is None:
            self.formatter = TestcaseFormatter()

        try:
            logger.info(
                "LLMMutation: start | tx_count=%d | sample=%s",
                tx_count,
                tx_seq[:3] if tx_seq is not None else [],
            )
            execution_result = getattr(individual, 'execution_result', None)
            feedback = None
            if execution_result:
                feedback = self.feedback_analyzer.analyze_execution(individual, execution_result)

            prompt = self._build_mutation_prompt(individual, feedback, engine)
            logger.info("[LLM-MUTATE] calling LLM for mutation")
            llm_response = self.llm_client.generate_json(prompt, temperature=0.8)

            if llm_response:
                mutated = self._apply_llm_mutation(individual, llm_response, engine)
                if mutated:
                    logger.info(
                        "LLMMutation: applied | mutation_type=%s | expected_outcome=%s",
                        llm_response.get("mutation_type"),
                        llm_response.get("expected_outcome"),
                    )
                    if self.adaptive_llm_controller:
                        self.adaptive_llm_controller.record_llm_success()
                    logger.info("[LLM-MUTATE] LLM mutation succeeded")
                    return mutated

        except Exception as e:
            logger.debug(f"LLM mutation error: {e}")

        if self.adaptive_llm_controller:
            self.adaptive_llm_controller.record_llm_error()

        self.failed_mutations += 1
        logger.info("[LLM-MUTATE] falling back to traditional mutation after error/none")
        return self._traditional_mutation(individual, engine)

    def _get_alpha(self, engine) -> float:
        if self.adaptive_llm_controller is None:
            return self.mutation_probability

        current_cov = self._get_coverage_from_engine(engine)
        try:
            return self.adaptive_llm_controller.get_alpha(
                getattr(engine, "current_generation", 0),
                current_cov,
            )
        except Exception:
            return self.mutation_probability

    def _get_coverage_from_engine(self, engine) -> float:
        env = getattr(settings, "GLOBAL_ENV", None)
        if env and hasattr(env, "overall_pcs") and env.overall_pcs:
            return len(env.code_coverage) / len(env.overall_pcs) * 100.0
        if hasattr(engine, "coverage_history") and engine.coverage_history:
            return engine.coverage_history[-1]
        return 0.0

    def _build_mutation_prompt(self, individual, feedback: Optional[Dict], engine) -> str:
        """Build mutation prompt with contract context and coverage info"""
        prompt_parts = []

        prompt_parts.append("# Smart Contract Fuzzing - Mutation Request\n")

        contract_name = getattr(individual.generator, 'contract_name', 'Unknown')
        prompt_parts.append(f"## Contract: {contract_name}\n")

        source_code = getattr(individual.generator, 'source_code', None)
        if not source_code:
            sol_path = getattr(individual.generator, 'sol_path', None)
            if sol_path:
                try:
                    with open(sol_path, 'r') as f:
                        source_code = f.read()
                except Exception as e:
                    logger.debug(f"Could not read source code: {e}")

        if source_code:
            source_excerpt = source_code[:2000] + "\n... (truncated)" if len(source_code) > 2000 else source_code
            prompt_parts.append("## Contract Source Code:")
            prompt_parts.append("```solidity")
            prompt_parts.append(source_excerpt)
            prompt_parts.append("```\n")

        prompt_parts.append("## Available Functions:")
        if hasattr(individual.generator, 'interface') and individual.generator.interface:
            for func_hash, arg_types in list(individual.generator.interface.items())[:10]:
                func_name = func_hash[:10] + "..."
                if hasattr(individual.generator, 'interface_mapper'):
                    func_name = individual.generator.interface_mapper.get(func_hash, func_hash[:10])
                prompt_parts.append(f"- {func_name}({', '.join(arg_types) if arg_types else ''})")

        prompt_parts.append("\n## Coverage Status:")
        code_cov = getattr(individual, 'code_coverage', 0)
        branch_cov = getattr(individual, 'branch_coverage', 0)

        try:
            from fuzzer.utils import settings
            env = getattr(settings, 'GLOBAL_ENV', None)
            if env:
                covered_pcs = len(getattr(env, 'code_coverage', set()))
                total_pcs = len(getattr(env, 'overall_pcs', set()))
                if total_pcs > 0:
                    code_cov = (covered_pcs / total_pcs) * 100

                visited_branches = getattr(env, 'visited_branches', {})
                overall_jumpis = getattr(env, 'overall_jumpis', {})
                total_branches = len(overall_jumpis) * 2
                covered_branch_count = sum(len(v) for v in visited_branches.values())
                if total_branches > 0:
                    branch_cov = (min(covered_branch_count, total_branches) / total_branches) * 100

                prompt_parts.append(f"- Code Coverage: {code_cov:.1f}% ({covered_pcs}/{total_pcs} PCs)")
                prompt_parts.append(f"- Branch Coverage: {branch_cov:.1f}% ({covered_branch_count}/{total_branches} branches)")

                if covered_pcs > 0:
                    sample_covered = list(env.code_coverage)[:5]
                    prompt_parts.append(f"- Sample Covered PCs: {[hex(pc) for pc in sample_covered]}")

                all_pcs = set(env.overall_pcs) if hasattr(env, 'overall_pcs') else set()
                uncovered_pcs = all_pcs - env.code_coverage
                if uncovered_pcs:
                    sample_uncovered = list(uncovered_pcs)[:5]
                    prompt_parts.append(f"- UNCOVERED PCs (TARGET THESE!): {[hex(pc) for pc in sample_uncovered]}")

                if visited_branches:
                    branch_info = []
                    for pc, branches in list(visited_branches.items())[:3]:
                        branch_info.append(f"{hex(pc)}:{list(branches)}")
                    prompt_parts.append(f"- Visited Branches: {', '.join(branch_info)}")
            else:
                prompt_parts.append(f"- Code Coverage: {code_cov:.1f}%")
                prompt_parts.append(f"- Branch Coverage: {branch_cov:.1f}%")
        except Exception as e:
            logger.debug(f"Could not get global coverage: {e}")
            prompt_parts.append(f"- Code Coverage: {code_cov:.1f}%")
            prompt_parts.append(f"- Branch Coverage: {branch_cov:.1f}%")

        if self.coverage_context and self.coverage_context.covered_pcs:
            covered_pcs = len(self.coverage_context.covered_pcs)
            total_pcs = self.coverage_context.total_pcs or "?"
            covered_branches = len(self.coverage_context.covered_branches)
            total_branches = self.coverage_context.total_branches or "?"
            prompt_parts.append(f"- Coverage Context: {covered_pcs}/{total_pcs} PCs, {covered_branches}/{total_branches} branches")

        prompt_parts.append("\n## Current Testcase:")
        for i, tx in enumerate(individual.chromosome[:5]):
            func_hash = tx.get("arguments", [None])[0]
            func_name = func_hash[:10] if func_hash else "unknown"
            if hasattr(individual.generator, 'interface_mapper'):
                func_name = individual.generator.interface_mapper.get(func_hash, func_name)
            amount = tx.get("amount", 0)
            account = tx.get("account", "?")[:10]
            prompt_parts.append(f"  Tx{i}: {func_name}() from {account}... value={amount}")

        prompt_parts.append("\n## Task: Mutate to increase coverage and find vulnerabilities")
        prompt_parts.append("""
Return ONLY this JSON:
{
  "mutation_type": "value_mutation|account_mutation|argument_mutation|sequence_mutation",
  "reasoning": "why this mutation will help",
  "expected_outcome": "what branch/code you expect to cover",
  "mutations": [{"transaction_index": 0, "changes": {"amount": 0}}]
}

Rules: integers for small nums, strings for large nums, addresses "0x"+40hex chars""")

        return "\n".join(prompt_parts)

    def _apply_llm_mutation(self, individual, llm_response: Dict, engine) -> Optional[object]:
        """Apply mutations suggested by LLM"""
        try:
            mutations = llm_response.get("mutations", [])
            sequence_changes = llm_response.get("sequence_changes")

            mutated = individual.clone()

            reasoning = llm_response.get("reasoning", "No reasoning provided")
            logger.info(f"LLM reasoning: {reasoning}")

            for mutation in mutations:
                tx_index = mutation.get("transaction_index", -1)
                changes = mutation.get("changes", {})

                if tx_index == -1:
                    new_gene = self._create_transaction_from_llm(changes, individual.generator)
                    if new_gene:
                        mutated.chromosome.append(new_gene)
                        logger.info(f"Added new transaction: {changes.get('function', 'unknown')}")

                elif tx_index == -2:
                    remove_index = changes.get("index", 0)
                    if 0 <= remove_index < len(mutated.chromosome):
                        mutated.chromosome.pop(remove_index)
                        logger.info(f"Removed transaction at index {remove_index}")

                elif tx_index >= 0 and tx_index < len(mutated.chromosome):
                    self._apply_transaction_changes(
                        mutated.chromosome[tx_index],
                        changes,
                        individual.generator
                    )

            if sequence_changes:
                self._apply_sequence_changes(
                    mutated,
                    sequence_changes,
                    individual.generator
                )

            generator = individual.generator
            for tx in mutated.chromosome:
                func_hash = tx.get("arguments", [None])[0]
                if func_hash and hasattr(generator, 'interface'):
                    param_types = generator.interface.get(func_hash, [])
                    current_args = tx.get("arguments", [])
                    actual_arg_count = len(current_args) - 1 if current_args else 0

                    if actual_arg_count < len(param_types):
                        logger.warning(f"Fixing incomplete arguments: {func_hash}")
                        for arg_idx in range(actual_arg_count, len(param_types)):
                            arg_type = param_types[arg_idx]
                            random_val = generator.get_random_argument(arg_type, func_hash, arg_idx)
                            tx["arguments"].append(random_val)

            try:
                mutated.solution = mutated.decode()
            except Exception as decode_error:
                logger.error(f"Failed to decode mutated individual: {decode_error}")
                return None

            return mutated

        except Exception as e:
            logger.error(f"Failed to apply LLM mutation: {e}")
            return None

    def _apply_transaction_changes(self, gene: Dict, changes: Dict, generator):
        """Apply changes to a single transaction gene"""
        for field, new_value in changes.items():
            if field == "function" or field == "function_name":
                old_function_hash = gene["arguments"][0]
                new_function_hash = self._get_function_hash_from_name(new_value, generator)

                if new_function_hash:
                    gene["arguments"][0] = new_function_hash
                    logger.info(f"Changed function: {old_function_hash} → {new_function_hash}")

                    if "arguments" not in changes:
                        param_types = []
                        if hasattr(generator, 'interface') and new_function_hash in generator.interface:
                            param_types = generator.interface[new_function_hash]
                        gene["arguments"] = [new_function_hash]
                        logger.info(f"Reset arguments for new function (expects {len(param_types)} params)")

            elif field == "account" or field == "from":
                if isinstance(new_value, str):
                    if new_value.startswith("0x"):
                        gene["account"] = new_value
                    elif new_value == "random":
                        gene["account"] = generator.get_random_account(gene["arguments"][0])
                else:
                    gene["account"] = new_value

            elif field == "to" or field == "contract":
                if isinstance(new_value, str) and new_value.startswith("0x"):
                    gene["contract"] = new_value

            elif field == "amount" or field == "value":
                gene["amount"] = self._parse_transaction_value(new_value)

            elif field.startswith("arg_") or field.startswith("argument_"):
                arg_index = int(field.split("_")[1])
                if arg_index < len(gene["arguments"]) - 1:
                    function_hash = gene["arguments"][0]
                    arg_type = ""
                    if hasattr(generator, 'interface') and function_hash in generator.interface:
                        if arg_index < len(generator.interface[function_hash]):
                            arg_type = generator.interface[function_hash][arg_index]

                    parsed_value = self._parse_llm_argument(new_value, arg_type)
                    gene["arguments"][arg_index + 1] = parsed_value

            elif field == "arguments" and isinstance(new_value, list):
                function_hash = gene["arguments"][0]
                param_types = []
                if hasattr(generator, 'interface') and function_hash in generator.interface:
                    param_types = generator.interface[function_hash]

                if len(new_value) == 0 and len(param_types) > 0:
                    logger.warning(f"LLM returned empty arguments but function expects {len(param_types)} params")
                    continue

                if len(param_types) == 0:
                    if len(new_value) > 0:
                        logger.warning(f"Function takes no parameters but LLM provided {len(new_value)} args")
                    gene["arguments"] = [function_hash]

                elif len(param_types) == 1 and '[]' in param_types[0] and len(new_value) > 1:
                    if not isinstance(new_value[0], list):
                        base_type = param_types[0].replace('[]', '')
                        parsed_array = [self._parse_llm_argument(v, base_type) for v in new_value]
                        gene["arguments"] = [function_hash, parsed_array]
                    else:
                        parsed_val = self._parse_llm_argument(new_value[0], param_types[0])
                        gene["arguments"] = [function_hash, parsed_val]
                else:
                    parsed_args = []
                    for arg_val in new_value:
                        arg_idx = len(parsed_args)
                        arg_type = ""
                        if arg_idx < len(param_types):
                            arg_type = param_types[arg_idx]

                        parsed_val = self._parse_llm_argument(arg_val, arg_type)
                        parsed_args.append(parsed_val)

                    if len(parsed_args) < len(param_types):
                        logger.warning(f"LLM provided {len(parsed_args)} args but function expects {len(param_types)}")
                        for arg_idx in range(len(parsed_args), len(param_types)):
                            arg_type = param_types[arg_idx]
                            random_val = generator.get_random_argument(arg_type, function_hash, arg_idx)
                            parsed_args.append(random_val)

                    gene["arguments"] = [function_hash] + parsed_args

    def _apply_sequence_changes(self, individual, sequence_changes: Dict, generator):
        """Apply sequence-level changes (reorder, insert, remove)"""
        operation = sequence_changes.get("operation", "")
        details = sequence_changes.get("details", {})

        if operation == "swap":
            idx1 = details.get("index1", 0)
            idx2 = details.get("index2", 1)
            if (0 <= idx1 < len(individual.chromosome) and
                0 <= idx2 < len(individual.chromosome)):
                individual.chromosome[idx1], individual.chromosome[idx2] = \
                    individual.chromosome[idx2], individual.chromosome[idx1]

        elif operation == "insert":
            position = details.get("position", len(individual.chromosome))
            function_sig = details.get("function", "")
            function_hash = None
            if hasattr(generator, 'interface_mapper'):
                function_hash = generator.interface_mapper.get(function_sig)

            if function_hash and function_hash in generator.interface:
                new_tx = self._create_transaction(function_hash, details, generator)
                if new_tx:
                    individual.chromosome.insert(position, new_tx)

        elif operation == "remove":
            index = details.get("index", -1)
            if 0 <= index < len(individual.chromosome) and len(individual.chromosome) > 1:
                individual.chromosome.pop(index)

    def _create_transaction_from_llm(self, changes: Dict, generator) -> Optional[Dict]:
        """Create a new transaction from LLM mutation response"""
        try:
            function_name = changes.get("function") or changes.get("function_name")
            if not function_name:
                return None

            function_hash = self._get_function_hash_from_name(function_name, generator)
            if not function_hash:
                return None

            return self._create_transaction(function_hash, changes, generator)

        except Exception as e:
            logger.error(f"Failed to create transaction from LLM: {e}")
            return None

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

    def _create_transaction(self, function_hash: str, details: Dict, generator) -> Optional[Dict]:
        """Create a new transaction gene"""
        try:
            tx = {
                "account": details.get("from", generator.get_random_account(function_hash)),
                "contract": details.get("to", generator.contract),
                "amount": self._parse_value(details.get("value", 0)),
                "gaslimit": self._parse_value(details.get("gas", generator.get_random_gaslimit(function_hash))),
                "arguments": [function_hash],
                "timestamp": generator.get_random_timestamp(function_hash),
                "blocknumber": generator.get_random_blocknumber(function_hash),
                "balance": generator.get_random_balance(function_hash),
                "call_return": {},
                "extcodesize": {},
                "returndatasize": {}
            }

            args = details.get("arguments", [])
            if function_hash in generator.interface:
                for arg_idx, arg_type in enumerate(generator.interface[function_hash]):
                    if args and arg_idx < len(args):
                        arg_value = self._parse_llm_argument(args[arg_idx], arg_type)
                    else:
                        arg_value = generator.get_random_argument(arg_type, function_hash, arg_idx)
                    tx["arguments"].append(arg_value)

            return tx

        except Exception as e:
            logger.error(f"Failed to create transaction: {e}")
            return None

    def _parse_value(self, value):
        """Parse value from LLM with bounds checking"""
        MAX_UINT256 = 2**256 - 1

        if isinstance(value, (int, float)):
            result = int(value)
            if result > MAX_UINT256:
                return MAX_UINT256
            elif result < 0:
                return 0
            return result

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

    def _traditional_mutation(self, individual, engine):
        """Traditional genetic mutation as fallback"""
        if self.traditional_mutator is not None:
            mutated = individual.clone()
            return self.traditional_mutator.mutate(mutated, engine)
        else:
            return individual.clone()

    def get_statistics(self) -> Dict[str, Any]:
        """Get mutation statistics"""
        return {
            "total_mutations": self.mutations_performed,
            "llm_mutations": self.llm_mutations,
            "traditional_mutations": self.traditional_mutations,
            "failed_mutations": self.failed_mutations,
            "llm_client_stats": self.llm_client.get_statistics()
        }
