#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testcase Formatter for LLM

Converts UniFuzz testcase format (chromosome/individual) to human-readable
format that LLM can understand and vice versa.
"""

import json
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger("TestcaseFormatter")


class TestcaseFormatter:
    """Formats testcases for LLM consumption and parsing"""

    def __init__(self, interface_mapper: Optional[Dict] = None):
        """
        Initialize formatter

        Args:
            interface_mapper: Mapping from function signatures to hashes
        """
        self.interface_mapper = interface_mapper or {}
        # Reverse mapping: hash -> signature
        self.hash_to_signature = {v: k for k, v in self.interface_mapper.items()}

    def chromosome_to_readable(self, chromosome: List[Dict],
                              include_details: bool = True) -> str:
        """
        Convert chromosome (transaction sequence) to readable format

        Args:
            chromosome: List of transaction genes
            include_details: Include detailed transaction info

        Returns:
            Human-readable string representation
        """
        if not chromosome:
            return "Empty sequence"

        lines = []
        lines.append(f"Transaction Sequence (length: {len(chromosome)}):")
        lines.append("=" * 60)

        for idx, gene in enumerate(chromosome):
            lines.append(f"\nTransaction #{idx + 1}:")
            lines.append("-" * 40)

            # Function info
            function_hash = gene.get("arguments", [None])[0]
            function_sig = self.hash_to_signature.get(function_hash, function_hash)
            lines.append(f"  Function: {function_sig}")

            if include_details:
                # Transaction details
                lines.append(f"  From: {gene.get('account', 'N/A')}")
                lines.append(f"  To: {gene.get('contract', 'N/A')}")
                lines.append(f"  Value: {gene.get('amount', 0)} wei")
                lines.append(f"  Gas Limit: {gene.get('gaslimit', 'N/A')}")

                # Arguments
                args = gene.get("arguments", [])[1:]  # Skip function hash
                if args:
                    lines.append(f"  Arguments: {args}")

                # Block info
                if "timestamp" in gene:
                    lines.append(f"  Timestamp: {gene['timestamp']}")
                if "blocknumber" in gene:
                    lines.append(f"  Block Number: {gene['blocknumber']}")

                # Global state
                if "balance" in gene and gene["balance"]:
                    lines.append(f"  Balance Override: {gene['balance']}")

        return "\n".join(lines)

    def chromosome_to_json(self, chromosome: List[Dict]) -> str:
        """
        Convert chromosome to compact JSON representation

        Args:
            chromosome: List of transaction genes

        Returns:
            JSON string
        """
        compact = []
        for gene in chromosome:
            function_hash = gene.get("arguments", [None])[0]
            function_sig = self.hash_to_signature.get(function_hash, function_hash)

            tx = {
                "function": function_sig,
                "from": gene.get("account"),
                "value": gene.get("amount", 0),
                "args": gene.get("arguments", [])[1:],
                "gas": gene.get("gaslimit")
            }

            # Add optional fields
            if "timestamp" in gene:
                tx["timestamp"] = gene["timestamp"]
            if "blocknumber" in gene:
                tx["blocknumber"] = gene["blocknumber"]

            compact.append(tx)

        return json.dumps(compact, indent=2)

    def readable_to_chromosome(self, llm_output: Dict,
                              generator) -> Optional[List[Dict]]:
        """
        Convert LLM output back to chromosome format

        Args:
            llm_output: LLM-generated testcase description
            generator: Generator instance to create proper gene structure

        Returns:
            Chromosome (list of genes) or None on failure
        """
        try:
            # LLM should output a list of transactions
            transactions = llm_output.get("transactions", [])
            if not transactions:
                logger.error("No transactions in LLM output")
                return None

            chromosome = []
            for tx in transactions:
                gene = self._transaction_to_gene(tx, generator)
                if gene:
                    chromosome.append(gene)
                else:
                    logger.warning(f"Failed to convert transaction: {tx}")

            return chromosome if chromosome else None

        except Exception as e:
            logger.error(f"Failed to convert LLM output to chromosome: {e}")
            return None

    def _transaction_to_gene(self, tx: Dict, generator) -> Optional[Dict]:
        """
        Convert single transaction description to gene

        Args:
            tx: Transaction dict from LLM
            generator: Generator instance

        Returns:
            Gene dict or None
        """
        try:
            # Get function hash
            function_sig = tx.get("function", "")
            function_hash = self.interface_mapper.get(function_sig)

            if not function_hash:
                # Try to find by partial match
                for sig, hash_val in self.interface_mapper.items():
                    if function_sig in sig or sig.startswith(function_sig):
                        function_hash = hash_val
                        break

            if not function_hash:
                logger.error(f"Unknown function: {function_sig}")
                return None

            # Create gene structure
            gene = {
                "account": tx.get("from", generator.get_random_account(function_hash)),
                "contract": generator.contract_address,
                "amount": tx.get("value", 0),
                "gaslimit": tx.get("gas", generator.get_random_gaslimit(function_hash)),
                "arguments": [function_hash],
                "timestamp": tx.get("timestamp", generator.get_random_timestamp(function_hash)),
                "blocknumber": tx.get("blocknumber", generator.get_random_blocknumber(function_hash)),
                "balance": tx.get("balance", generator.get_random_balance(function_hash)),
                "call_return": tx.get("call_return", {}),
                "extcodesize": tx.get("extcodesize", {}),
                "returndatasize": tx.get("returndatasize", {})
            }

            # Add arguments
            args = tx.get("args", [])
            if args:
                gene["arguments"].extend(args)
            else:
                # Generate default arguments
                if function_hash in generator.interface:
                    for arg_idx, arg_type in enumerate(generator.interface[function_hash]):
                        arg_value = generator.get_random_argument(
                            arg_type, function_hash, arg_idx
                        )
                        gene["arguments"].append(arg_value)

            return gene

        except Exception as e:
            logger.error(f"Failed to convert transaction to gene: {e}")
            return None

    def format_execution_result(self, individual, execution_result: Dict) -> str:
        """
        Format execution result for LLM

        Args:
            individual: Individual that was executed
            execution_result: Result from execution

        Returns:
            Formatted string
        """
        lines = []
        lines.append("Execution Result:")
        lines.append("=" * 60)

        # Coverage info
        if "coverage" in execution_result:
            cov = execution_result["coverage"]
            lines.append(f"\nCoverage:")
            lines.append(f"  Instructions: {cov.get('instructions', 0)}")
            lines.append(f"  Branches: {cov.get('branches', 0)}")

        # Violations
        if "violations" in execution_result:
            violations = execution_result["violations"]
            if violations:
                lines.append(f"\n⚠️  Violations Found: {len(violations)}")
                for v in violations:
                    lines.append(f"  - {v.get('type', 'Unknown')}: {v.get('description', '')}")
            else:
                lines.append("\n✓ No violations found")

        # Execution traces (summarized)
        if "traces" in execution_result:
            traces = execution_result["traces"]
            lines.append(f"\nExecution Traces: {len(traces)} transactions")
            for idx, trace in enumerate(traces):
                if "storage_changes" in trace:
                    changes = len(trace["storage_changes"])
                    if changes > 0:
                        lines.append(f"  Tx #{idx + 1}: {changes} storage changes")

        return "\n".join(lines)

    def format_vulnerability_context(self, vulnerabilities: List[Dict]) -> str:
        """
        Format vulnerability information for LLM context

        Args:
            vulnerabilities: List of known vulnerabilities

        Returns:
            Formatted string
        """
        if not vulnerabilities:
            return "No known vulnerabilities"

        lines = []
        lines.append("Known Vulnerabilities:")
        lines.append("=" * 60)

        for vuln in vulnerabilities:
            vuln_type = vuln.get("type", "Unknown")
            functions = vuln.get("functions", [])
            description = vuln.get("description", "")

            lines.append(f"\n⚠️  {vuln_type}")
            if functions:
                lines.append(f"  Affected functions: {', '.join(functions)}")
            if description:
                lines.append(f"  Description: {description}")

        return "\n".join(lines)

    def format_contract_interface(self, interface: Dict) -> str:
        """
        Format contract interface for LLM

        Args:
            interface: Contract interface mapping

        Returns:
            Formatted string
        """
        lines = []
        lines.append("Contract Interface:")
        lines.append("=" * 60)

        for func_hash, arg_types in interface.items():
            func_sig = self.hash_to_signature.get(func_hash, func_hash)
            lines.append(f"\n{func_sig}")
            if arg_types:
                lines.append(f"  Arguments: {', '.join(arg_types)}")

        return "\n".join(lines)
