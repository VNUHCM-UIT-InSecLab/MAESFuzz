#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG-Enhanced Context Provider for LLM

Provides vulnerability knowledge and smart contract analysis context
to the LLM for better mutation/crossover decisions.

Uses SimpleRAG (FAISS + HuggingFace + Gemini) for knowledge retrieval.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger("RAGContextProvider")

# Try to import SimpleRAG
try:
    from fuzzer.engine.components.rag_enhanced_generator import SimpleRAG
    SIMPLE_RAG_AVAILABLE = True
except ImportError:
    SIMPLE_RAG_AVAILABLE = False
    SimpleRAG = None


class RAGContextProvider:
    """Provides RAG-enhanced context to LLM"""

    def __init__(self, rag_endpoint: str = "http://localhost:5000/request",
                 analysis_result: Optional[Dict] = None,
                 use_simple_rag: bool = True,
                 api_key: Optional[str] = None,
                 vector_db_path: Optional[str] = None):
        """
        Initialize RAG context provider

        Args:
            rag_endpoint: RAG server endpoint (unused, kept for compatibility)
            analysis_result: Contract analysis results (dataflow, vulnerabilities, etc.)
            use_simple_rag: Try to use SimpleRAG (default: True)
            api_key: Google API key for Gemini
            vector_db_path: Path to FAISS vector database
        """
        self.rag_endpoint = rag_endpoint
        self.analysis_result = analysis_result or {}
        self.timeout = 30

        # Extract key information from analysis
        self.critical_paths = self.analysis_result.get("critical_paths", [])
        self.vulnerabilities = self.analysis_result.get("vulnerabilities", [])
        self.test_sequences = self.analysis_result.get("test_sequences", [])

        # Initialize SimpleRAG
        self.simple_rag = None
        self.use_simple_rag = use_simple_rag

        if use_simple_rag and SIMPLE_RAG_AVAILABLE:
            try:
                api_key = api_key or os.environ.get("GOOGLE_API_KEY")
                # Default vector DB path
                if vector_db_path is None:
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
                    vector_db_path = os.path.join(base_dir, "RAG", "faiss_storage")

                if os.path.exists(vector_db_path):
                    self.simple_rag = SimpleRAG(
                        vector_db_path=vector_db_path,
                        api_key=api_key
                    )
                    if self.simple_rag.initialize():
                        logger.info("SimpleRAG initialized successfully")
                    else:
                        logger.warning("Failed to initialize SimpleRAG")
                        self.simple_rag = None
                else:
                    logger.warning(f"Vector DB path not found: {vector_db_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize SimpleRAG: {e}")
                self.simple_rag = None
        else:
            logger.info("SimpleRAG not available - using static knowledge base")

        logger.info(f"RAGContextProvider initialized with {len(self.vulnerabilities)} vulnerabilities")

    def get_vulnerability_knowledge(self, vuln_type: Optional[str] = None) -> str:
        """
        Get vulnerability knowledge from RAG

        Args:
            vuln_type: Specific vulnerability type to query

        Returns:
            Formatted vulnerability knowledge
        """
        if vuln_type:
            query = f"How to detect and exploit {vuln_type} vulnerabilities in smart contracts?"
        else:
            query = "Common smart contract vulnerabilities and how to detect them"

        # Query RAG
        rag_response = self._query_rag(query)

        if rag_response:
            return rag_response

        # Fallback: Return static knowledge
        return self._get_static_vulnerability_knowledge(vuln_type)

    def get_mutation_strategy(self, current_testcase: str,
                            execution_result: Optional[Dict] = None) -> str:
        """
        Get RAG suggestions for mutation strategy

        Args:
            current_testcase: Current testcase description
            execution_result: Execution result of current testcase

        Returns:
            Mutation strategy suggestions
        """
        # Build query
        query_parts = [
            "Smart contract fuzzing mutation strategies.",
            f"Current testcase: {current_testcase}"
        ]

        if execution_result:
            if execution_result.get("violations"):
                violations = execution_result["violations"]
                vuln_types = [v.get("type") for v in violations]
                query_parts.append(f"Found vulnerabilities: {', '.join(vuln_types)}")
                query_parts.append("How to mutate this testcase to explore similar vulnerabilities?")
            else:
                query_parts.append("No vulnerabilities found. How to mutate to explore new paths?")

        query = " ".join(query_parts)
        rag_response = self._query_rag(query)

        if rag_response:
            return rag_response

        # Fallback
        return self._get_static_mutation_strategies()

    def get_crossover_strategy(self, parent1: str, parent2: str,
                              fitness_info: Optional[Dict] = None) -> str:
        """
        Get RAG suggestions for crossover strategy

        Args:
            parent1: First parent testcase description
            parent2: Second parent testcase description
            fitness_info: Fitness information for parents

        Returns:
            Crossover strategy suggestions
        """
        query = f"""
Smart contract fuzzing: How to combine these two testcases effectively?

Parent 1:
{parent1}

Parent 2:
{parent2}

Goal: Create offspring that explores new vulnerabilities.
"""

        rag_response = self._query_rag(query)

        if rag_response:
            return rag_response

        # Fallback
        return self._get_static_crossover_strategies()

    def get_vulnerability_specific_context(self, vuln_type: str) -> Dict[str, Any]:
        """
        Get detailed context for specific vulnerability type

        Args:
            vuln_type: Vulnerability type (e.g., "reentrancy", "integer_overflow")

        Returns:
            Dict containing vulnerability-specific context
        """
        # Find matching vulnerabilities in analysis
        matching_vulns = [
            v for v in self.vulnerabilities
            if v.get("type", "").lower() == vuln_type.lower()
        ]

        context = {
            "type": vuln_type,
            "found_in_analysis": len(matching_vulns) > 0,
            "affected_functions": [],
            "exploitation_hints": [],
            "test_patterns": []
        }

        if matching_vulns:
            # Aggregate information from matching vulnerabilities
            for vuln in matching_vulns:
                context["affected_functions"].extend(vuln.get("functions", []))
                if "description" in vuln:
                    context["exploitation_hints"].append(vuln["description"])

        # Query RAG for additional context
        query = f"How to test for {vuln_type} in smart contracts? Provide specific transaction patterns."
        rag_response = self._query_rag(query)

        if rag_response:
            context["rag_suggestions"] = rag_response

        return context

    def get_function_interaction_patterns(self, functions: List[str]) -> str:
        """
        Get interaction patterns for specific functions

        Args:
            functions: List of function names

        Returns:
            Interaction patterns and suggestions
        """
        query = f"""
Smart contract function interaction patterns:
Functions: {', '.join(functions)}

What are dangerous interaction patterns between these functions?
How should they be tested in combination?
"""

        rag_response = self._query_rag(query)

        if rag_response:
            return rag_response

        # Check critical paths
        relevant_paths = [
            path for path in self.critical_paths
            if any(func in path for func in functions)
        ]

        if relevant_paths:
            return f"Critical paths found: {relevant_paths}"

        return "No specific interaction patterns found."

    def _query_rag(self, query: str) -> Optional[str]:
        """
        Query SimpleRAG for knowledge

        Args:
            query: Query string

        Returns:
            RAG response or None
        """
        if self.simple_rag:
            try:
                result = self.simple_rag.query(query, use_llm=True)
                if result:
                    logger.debug("Got response from SimpleRAG")
                    return result
                else:
                    logger.debug("SimpleRAG returned no result")
            except Exception as e:
                logger.debug(f"SimpleRAG query failed: {e}")

        # Return None to use static knowledge
        return None

    def _get_static_vulnerability_knowledge(self, vuln_type: Optional[str] = None) -> str:
        """Get static vulnerability knowledge (fallback)"""

        knowledge_base = {
            "reentrancy": """
Reentrancy Vulnerability:
- Occurs when external calls allow re-entering the function before state updates
- Test pattern: call1 -> externalCall -> call1 (recursive)
- Key functions: Those with external calls followed by state changes
- Mutation strategy: Increase call depth, manipulate balances, test callback scenarios
""",
            "integer_overflow": """
Integer Overflow/Underflow:
- Arithmetic operations exceed type bounds
- Test pattern: Large values near MAX_UINT, subtraction near 0
- Mutation strategy: Extreme values, boundary conditions, repeated operations
""",
            "access_control": """
Access Control Issues:
- Unauthorized access to privileged functions
- Test pattern: Call privileged functions from non-owner accounts
- Mutation strategy: Vary caller accounts, test permission boundaries
""",
            "unchecked_return": """
Unchecked Return Values:
- Ignoring return values from external calls
- Test pattern: Make external calls fail, observe state
- Mutation strategy: Manipulate call_return values, force failures
"""
        }

        if vuln_type and vuln_type.lower() in knowledge_base:
            return knowledge_base[vuln_type.lower()]

        # Return all knowledge
        return "\n\n".join(knowledge_base.values())

    def _get_static_mutation_strategies(self) -> str:
        """Get static mutation strategies (fallback)"""
        return """
Mutation Strategies for Smart Contract Fuzzing:

1. Value Mutation:
   - Test extreme values (0, MAX_UINT, negative if signed)
   - Test boundary conditions
   - Test precision loss scenarios

2. Account Mutation:
   - Switch between different privilege levels
   - Test unauthorized access
   - Test multi-party scenarios

3. Sequence Mutation:
   - Reorder transactions (swap dependent calls)
   - Insert re-entrant calls
   - Add/remove transactions

4. State Mutation:
   - Manipulate balances
   - Override storage values
   - Test different block states

5. Argument Mutation:
   - Array bounds (empty, large, malformed)
   - Address validity (zero, contract, EOA)
   - String/bytes manipulation
"""

    def _get_static_crossover_strategies(self) -> str:
        """Get static crossover strategies (fallback)"""
        return """
Crossover Strategies for Smart Contract Fuzzing:

1. Sequential Combination:
   - Combine sequences end-to-end
   - Preserve causal dependencies
   - Test extended interaction patterns

2. Interleaving:
   - Interleave transactions from both parents
   - Create complex interaction patterns
   - Test concurrent state changes

3. Function Extraction:
   - Extract high-value functions from both parents
   - Combine based on criticality
   - Focus on vulnerability-prone patterns

4. Semantic Merging:
   - Merge transactions with similar purposes
   - Preserve semantic meaning
   - Avoid redundant operations

5. Critical Path Fusion:
   - Combine critical paths from both parents
   - Maximize coverage potential
   - Explore unexplored combinations
"""

    def get_analysis_summary(self) -> str:
        """Get summary of contract analysis"""
        lines = []
        lines.append("Contract Analysis Summary:")
        lines.append("=" * 60)

        lines.append(f"\nCritical Paths: {len(self.critical_paths)}")
        if self.critical_paths:
            for idx, path in enumerate(self.critical_paths[:3]):
                lines.append(f"  {idx + 1}. {' -> '.join(path)}")

        lines.append(f"\nPotential Vulnerabilities: {len(self.vulnerabilities)}")
        vuln_types = set(v.get("type") for v in self.vulnerabilities)
        for vtype in vuln_types:
            count = sum(1 for v in self.vulnerabilities if v.get("type") == vtype)
            lines.append(f"  - {vtype}: {count}")

        lines.append(f"\nRecommended Test Sequences: {len(self.test_sequences)}")

        return "\n".join(lines)
