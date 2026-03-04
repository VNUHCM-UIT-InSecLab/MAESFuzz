#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Anything Client for LLM Operators

Integrates RAG Anything (LightRAG + Gemini) to provide vulnerability knowledge
for LLM-based fuzzing operators.
"""

import os
import asyncio
import logging
from typing import Optional
import sys

logger = logging.getLogger("RAGAnythingClient")

# Add RAG folder to path
RAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../RAG")
if os.path.exists(RAG_DIR):
    sys.path.insert(0, RAG_DIR)

try:
    import google.generativeai as genai
    from raganything import RAGAnything
    from lightrag.lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    RAG_ANYTHING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG Anything not available: {e}")
    RAG_ANYTHING_AVAILABLE = False


class RAGAnythingClient:
    """Client for querying RAG Anything system"""

    def __init__(self, api_key: str, working_dir: str = None):
        """
        Initialize RAG Anything client

        Args:
            api_key: Gemini API key
            working_dir: RAG storage directory
        """
        if not RAG_ANYTHING_AVAILABLE:
            raise ImportError("RAG Anything dependencies not available")

        self.api_key = api_key
        self.working_dir = working_dir or os.path.join(RAG_DIR, "rag_storage")

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize LLM and embedding functions
        self.llm_func = self._create_llm_func()
        self.embedding_func = self._create_embedding_func()

        # Initialize RAG instance (will be done lazily)
        self.rag_instance = None
        self._initialized = False

        logger.info(f"RAGAnythingClient initialized with working_dir: {self.working_dir}")

    def _create_llm_func(self):
        """Create Gemini LLM function"""
        async def gemini_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            def _call_sync():
                model = genai.GenerativeModel("gemini-2.5-flash")
                messages = []

                if system_prompt:
                    messages.append({"role": "user", "parts": [f"[System] {system_prompt}"]})

                if history_messages:
                    for msg in history_messages:
                        role = msg.get("role", "user")
                        if role not in ["user", "model"]:
                            role = "user"
                        content = msg.get("content", "")
                        messages.append({"role": role, "parts": [content]})

                messages.append({"role": "user", "parts": [prompt]})

                response = model.generate_content(messages)
                return response.text if response and response.text else ""

            return await asyncio.to_thread(_call_sync)

        return gemini_llm_func

    def _create_embedding_func(self):
        """Create Gemini embedding function"""
        async def gemini_embed(texts):
            def _call_sync():
                model = "models/embedding-001"
                embeddings = []
                for t in texts:
                    r = genai.embed_content(model=model, content=t)
                    emb = r.get("embedding") if isinstance(r, dict) else getattr(r, "embedding", None)
                    if emb is None:
                        raise ValueError("Gemini embedding returned empty!")
                    embeddings.append(emb)
                return embeddings

            return await asyncio.to_thread(_call_sync)

        return EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=gemini_embed
        )

    async def _initialize(self):
        """Initialize RAG instance (async)"""
        if self._initialized:
            return

        if not os.path.exists(self.working_dir) or not os.listdir(self.working_dir):
            logger.warning(f"RAG storage not found at {self.working_dir}")
            logger.warning("Please run ingest_raganything.py first!")
            return

        try:
            # Load LightRAG
            lightrag_instance = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self.llm_func,
                embedding_func=self.embedding_func,
            )

            # Initialize storages
            await lightrag_instance.initialize_storages()

            # Create RAGAnything wrapper
            self.rag_instance = RAGAnything(lightrag=lightrag_instance)
            self._initialized = True

            logger.info("✓ RAG Anything initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG Anything: {e}")
            self._initialized = False

    async def _aquery(self, question: str, mode: str = "hybrid") -> Optional[str]:
        """
        Async query RAG Anything

        Args:
            question: Question to ask
            mode: Query mode (hybrid, local, global, naive)

        Returns:
            Answer string or None
        """
        if not self._initialized:
            await self._initialize()

        if not self._initialized or not self.rag_instance:
            return None

        try:
            result = await self.rag_instance.aquery(
                question,
                mode=mode,
                vlm_enhanced=False
            )
            return result

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return None

    def query(self, question: str, mode: str = "hybrid") -> Optional[str]:
        """
        Sync wrapper for RAG query

        Args:
            question: Question to ask
            mode: Query mode

        Returns:
            Answer string or None
        """
        try:
            # Run async query in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new loop if current one is running
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._aquery(question, mode)
                    )
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self._aquery(question, mode))

        except Exception as e:
            logger.error(f"Sync query failed: {e}")
            return None

    def get_vulnerability_knowledge(self, vuln_type: str) -> Optional[str]:
        """
        Get vulnerability knowledge from RAG

        Args:
            vuln_type: Vulnerability type (e.g., "reentrancy")

        Returns:
            Knowledge string or None
        """
        question = f"""
What is {vuln_type} vulnerability in smart contracts?
How to detect it? How to exploit it for fuzzing?
Provide specific transaction patterns and test strategies.
"""
        return self.query(question, mode="hybrid")

    def get_mutation_strategy(self, testcase_desc: str, violations: list) -> Optional[str]:
        """
        Get mutation strategy suggestion from RAG

        Args:
            testcase_desc: Description of current testcase
            violations: List of violations found

        Returns:
            Strategy suggestion or None
        """
        if violations:
            vuln_types = ", ".join([v.get("type", "unknown") for v in violations])
            question = f"""
Current testcase: {testcase_desc}
Found vulnerabilities: {vuln_types}

How should I mutate this testcase to explore similar vulnerabilities?
Suggest specific mutation strategies for smart contract fuzzing.
"""
        else:
            question = f"""
Current testcase: {testcase_desc}
No vulnerabilities found yet.

How should I mutate this testcase to discover new vulnerabilities?
Suggest mutation strategies for smart contract fuzzing.
"""

        return self.query(question, mode="hybrid")

    def get_crossover_strategy(self, parent1_desc: str, parent2_desc: str) -> Optional[str]:
        """
        Get crossover strategy from RAG

        Args:
            parent1_desc: Description of first parent
            parent2_desc: Description of second parent

        Returns:
            Strategy suggestion or None
        """
        question = f"""
Smart contract fuzzing crossover:

Parent 1: {parent1_desc}
Parent 2: {parent2_desc}

How should I combine these two testcases effectively?
Suggest crossover strategies that preserve meaningful patterns
and maximize vulnerability discovery potential.
"""

        return self.query(question, mode="hybrid")


# Convenience function
def create_rag_anything_client(api_key: str = None) -> Optional[RAGAnythingClient]:
    """
    Create RAG Anything client if available

    Args:
        api_key: Gemini API key (uses env var if not provided)

    Returns:
        RAGAnythingClient or None
    """
    if not RAG_ANYTHING_AVAILABLE:
        logger.warning("RAG Anything not available")
        return None

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set")
        return None

    try:
        client = RAGAnythingClient(api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to create RAG Anything client: {e}")
        return None
