#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-Based Evolution System for Smart Contract Fuzzing

This package provides LLM-driven evolution operators that replace
traditional genetic algorithm operators with LLM + RAG powered decisions.

Main Components:
- LLMMutation: Uses LLM to intelligently mutate testcases
- LLMCrossover: Uses LLM to combine testcases semantically
- LLMClient: Wrapper for Provider System (Gemini/Ollama)
- RAGContextProvider: Provides vulnerability knowledge to LLM
- ExecutionFeedbackAnalyzer: Converts execution traces to LLM-readable format
- CoverageContextProvider: Tracks coverage for context-aware mutations
- TestcaseFormatter: Formats testcases for LLM prompts
"""

from .llm_client import LLMClient
from .llm_mutation import LLMMutation
from .llm_crossover import LLMCrossover
from .llm_controller import LLMController
from .rag_context_provider import RAGContextProvider
from .execution_feedback import ExecutionFeedbackAnalyzer
from .coverage_context import CoverageContextProvider
from .testcase_formatter import TestcaseFormatter

__all__ = [
    'LLMClient',
    'LLMMutation',
    'LLMCrossover',
    'RAGContextProvider',
    'ExecutionFeedbackAnalyzer',
    'CoverageContextProvider',
    'TestcaseFormatter',
]
