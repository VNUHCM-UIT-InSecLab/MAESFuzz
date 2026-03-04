#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import json
import logging
import os
import time
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# SimpleRAG imports (FAISS + HuggingFace + Gemini)
LANGCHAIN_AVAILABLE = False
FAISS = None
HuggingFaceEmbeddings = None
ChatGoogleGenerativeAI = None
ChatOpenAI = None
_LANGCHAIN_IMPORT_ERROR = None

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    _LANGCHAIN_IMPORT_ERROR = str(e)
    # Try to import what we can for better error messages
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        pass
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        pass
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        pass
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        pass

from fuzzer.utils.utils import initialize_logger
from fuzzer.utils import settings
from .generator import Generator
from provider import context as provider_context
from provider.base import ProviderError


class SimpleRAG:
    """
    Simple RAG implementation using FAISS + HuggingFace + Gemini.
    Replaces RAGAnything/LightRAG with a simpler, more stable approach.

    Usage:
        1. Run RAG/ingest.py to create FAISS index first
        2. Then run UniFuzz.py - it will only query, not ingest
    """

    def __init__(self,
                 vector_db_path: str,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 api_key: Optional[str] = None,
                 llm_model: str = "gemini-2.0-flash",
                 llm_provider: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        """
        Initialize SimpleRAG (query-only mode).

        Args:
            vector_db_path: Path to FAISS vector database
            embedding_model: HuggingFace embedding model name
            api_key: Google API key for Gemini
            llm_model: LLM model name (Gemini or OpenAI)
            llm_provider: "gemini" or "openai" (auto-detect from model if None)
            openai_api_key: OpenAI API key (required if provider is OpenAI)
        """
        self.vector_db_path = vector_db_path
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.llm_model = llm_model
        
        # Auto-detect provider from model name
        if llm_provider is None:
            if llm_model.lower().startswith("gpt") or llm_model.lower().startswith("o1"):
                self.llm_provider = "openai"
            elif llm_model.lower().startswith("gemini"):
                self.llm_provider = "gemini"
            else:
                self.llm_provider = "gemini"  # default
        else:
            self.llm_provider = llm_provider.lower()
        
        self.logger = logging.getLogger("SimpleRAG")
        # Shared LLM log file (same as LLM evolution)
        os.makedirs("log", exist_ok=True)
        self.log_file = "log/LLM.log"

        # External provider (UniFuzz global provider: OpenAI / Gemini / Ollama / ...)
        # This allows SimpleRAG to reuse the same provider as the rest of the system
        # instead of creating its own ChatOpenAI / ChatGoogleGenerativeAI instance.
        self.external_provider = None

        # Check if langchain is available
        if not LANGCHAIN_AVAILABLE:
            error_msg = f"LangChain not available: {_LANGCHAIN_IMPORT_ERROR}" if _LANGCHAIN_IMPORT_ERROR else "LangChain not available"
            self.logger.warning(f"{error_msg}. RAG will be disabled.")
            self.logger.warning("Please install: pip install langchain-community langchain-huggingface langchain-google-genai faiss-cpu")
            self.embeddings = None
            self.vector_store = None
            self.llm = None
            self._initialized = False
            return

        # Initialize embeddings
        self.logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Vector store and LLM
        self.vector_store = None
        self.llm = None
        self._initialized = False

    def _log_llm(self, title: str, prompt: str, response: Optional[str] = None, error: Optional[str] = None):
        """Log RAG LLM interactions to file only (avoid stdout spam)."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {title}\n")
                f.write(f"{'='*80}\n")
                f.write("--- REQUEST (RAG LLM) ---\n")
                f.write(f"Model: {self.llm_model}\n")
                f.write(f"Provider: {self.llm_provider}\n")
                f.write(f"{prompt}\n")
                f.write("--- END REQUEST ---\n")
                if response is not None:
                    f.write("--- RESPONSE (RAG LLM) ---\n")
                    f.write(f"{response}\n")
                    f.write("--- END RESPONSE ---\n")
                if error is not None:
                    f.write("--- ERROR (RAG LLM) ---\n")
                    f.write(f"{error}\n")
                    f.write("--- END ERROR ---\n")
                f.write(f"Status: {'FAILED' if error else 'SUCCESS'}\n")
                f.write(f"{'='*80}\n\n")
        except Exception:
            pass

    def initialize(self) -> bool:
        """
        Initialize vector store and LLM (query-only, no ingestion).

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True

        if not LANGCHAIN_AVAILABLE:
            error_msg = f"LangChain not available: {_LANGCHAIN_IMPORT_ERROR}" if _LANGCHAIN_IMPORT_ERROR else "LangChain not available"
            self.logger.warning(error_msg)
            return False

        try:
            # Check if FAISS exists
            index_file = os.path.join(self.vector_db_path, "index.faiss")
            pkl_file = os.path.join(self.vector_db_path, "index.pkl")

            if not os.path.exists(self.vector_db_path) or not (os.path.exists(index_file) or os.path.exists(pkl_file)):
                self.logger.warning(f"FAISS index not found at: {self.vector_db_path}")
                self.logger.warning("Please run 'python RAG/ingest.py' first to create the index")
                return False

            # Load existing FAISS vector store
            self.logger.info(f"Loading FAISS from: {self.vector_db_path}")
            self.vector_store = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # Try to reuse the global UniFuzz provider first (supports OpenAI / Gemini / Ollama)
            try:
                global_provider = provider_context.get_provider(optional=True)
            except Exception:
                global_provider = None

            if global_provider is not None:
                # Use the shared provider as primary LLM backend for RAG
                self.external_provider = global_provider
                self.logger.info(
                    "Using global provider %s for SimpleRAG answers",
                    type(global_provider).__name__,
                )
                # Keep self.llm = None in this mode; all generation goes through external_provider
            else:
                # Fallback: initialize an internal LangChain chat model if possible
                if self.llm_provider == "openai":
                    if self.openai_api_key:
                        self.logger.info(f"Initializing OpenAI LLM ({self.llm_model}) for SimpleRAG")
                        self.llm = ChatOpenAI(
                            model=self.llm_model,
                            temperature=0.3,
                            max_tokens=2048,
                            api_key=self.openai_api_key,
                        )
                        self.logger.info("OpenAI LLM initialized successfully")
                    else:
                        self.logger.warning("No OpenAI API key, will use similarity search only")
                elif self.llm_provider == "gemini":
                    if self.api_key:
                        self.logger.info(f"Initializing Gemini LLM ({self.llm_model}) for SimpleRAG")
                        self.llm = ChatGoogleGenerativeAI(
                            model=self.llm_model,
                            temperature=0.3,
                            max_tokens=2048,
                            google_api_key=self.api_key,
                        )
                        self.logger.info("Gemini LLM initialized successfully")
                    else:
                        self.logger.warning("No Google API key, will use similarity search only")
                else:
                    # Unknown / unsupported provider name when no global provider is set.
                    # In this case we fall back to similarity search only.
                    self.logger.warning(
                        "Unknown provider %s and no global provider set, will use similarity search only",
                        self.llm_provider,
                    )

            self._initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SimpleRAG: {e}")
            return False

    def query(self, question: str, use_llm: bool = True) -> Optional[str]:
        """
        Query the RAG system.

        Args:
            question: Question to ask
            use_llm: Whether to use LLM for answer generation

        Returns:
            Answer string or None
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Get relevant documents via similarity search
            docs = self.vector_store.similarity_search(question, k=3)
            if not docs:
                return None

            # Combine context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])

            if use_llm and (self.external_provider or self.llm):
                # Use LLM (global provider preferred) to generate answer based on context
                prompt = f"""You are a smart-contract security assistant.
Based on the following context, answer the question concisely and technically.

Context:
{context}

Question: {question}

Answer:"""
                try:
                    if self.external_provider is not None:
                        # Use UniFuzz provider abstraction (supports OpenAI / Gemini / Ollama / etc.)
                        result = self.external_provider.generate(
                            prompt,
                            system_prompt=(
                                "You are a RAG assistant for smart-contract fuzzing and security auditing. "
                                "Use the provided context, avoid hallucinations, and be concise."
                            ),
                        )
                        answer = result.text
                    else:
                        # Fallback to internal LangChain chat model
                        response = self.llm.invoke(prompt)
                        answer = response.content if hasattr(response, "content") else str(response)

                    self._log_llm("RAG LLM Query", prompt, response=answer)
                except Exception as e:
                    self._log_llm("RAG LLM Query", prompt, error=str(e))
                    raise

                # Check for no-context responses
                no_context_indicators = [
                    "i don't have", "no relevant", "cannot find",
                    "not able to", "no information", "i apologize"
                ]
                if any(ind in answer.lower() for ind in no_context_indicators):
                    self.logger.debug("LLM returned no-context response")
                    return None

                return answer
            else:
                # Return raw context without LLM
                return context

        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            return None

    def similarity_search(self, query: str, k: int = 3) -> List[Any]:
        """
        Perform similarity search without LLM.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of documents
        """
        if not self._initialized:
            if not self.initialize():
                return []

        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []


class RAGEnhancedGenerator(Generator):
    """
    Generator that uses dataflow analysis results and RAG to generate optimal transaction sequences.
    """
    
    def __init__(self, interface: Dict, 
                 bytecode: str, 
                 accounts: List[str], 
                 contract: str, 
                 api_key: Optional[str] = None,
                 analysis_result: Optional[Dict] = None,
                 contract_name: Optional[str] = None, 
                 sol_path: Optional[str] = None,
                 other_generators=None, 
                 interface_mapper=None,
                 max_individual_length: int = 10,
                 llm_model: Optional[str] = None,
                 llm_provider: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 disable_rag_llm: bool = False,
                 adaptive_llm_controller=None):
        """
        Initialize RAGEnhancedGenerator with additional parameters.
        """
        super().__init__(interface, bytecode, accounts, contract, 
                        other_generators=other_generators, 
                        interface_mapper=interface_mapper,
                        contract_name=contract_name, 
                        sol_path=sol_path)
        
        self.api_key = api_key
        self.provider = None
        self.logger = initialize_logger("RAGEnhancedGenerator")
        self._enforce_rate_limit = False
        self._refresh_provider()
        
        self.analysis_result = analysis_result or {
            "critical_paths": [],
            "test_sequences": [],
            "vulnerabilities": []
        }
        
        self.optimal_sequences = self.analysis_result.get("test_sequences", [])
        self.critical_paths = self.analysis_result.get("critical_paths", [])
        self.potential_vulnerabilities = self.analysis_result.get("vulnerabilities", [])
        self.arg_cache = {}
        self.good_sequences = []
        
        self._rag_instance = None
        self._rag_initialized = False
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.openai_api_key = openai_api_key
        self.disable_rag_llm = disable_rag_llm
        self.adaptive_llm_controller = adaptive_llm_controller
        # Use separate FAISS storage folder (not LightRAG storage)
        self.rag_working_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "RAG", "faiss_storage"
        )
        
        self.rag_requests = 0
        self.rag_successes = 0
        self.rag_failures = 0
        self.rag_cache_hits = 0
        self.max_individual_length = max_individual_length
        self._quota_exceeded = False  # Track if quota has been exceeded
        self._no_context_count = 0  # Track consecutive no-context responses
        self._max_no_context_before_disable = 3  # Disable RAG after 3 consecutive no-context responses
        # Rate limiting: Free tier allows 15 requests/minute
        self._last_request_time = 0
        self._request_times = []  # Track request times for rate limiting
        self._min_delay_between_requests = 4.0  # 4 seconds between requests (15 requests/minute)

    def _refresh_provider(self):
        provider = provider_context.get_provider(optional=True)
        self.provider = provider
        self._enforce_rate_limit = (
            provider is not None and provider.__class__.__name__ == "GeminiProvider"
        )
        return provider
    
    def _get_function_hash_by_name(self, function_name: str) -> Optional[str]:
        """Find function hash from function name."""
        if not self.interface_mapper:
            return None
            
        for fname, fhash in self.interface_mapper.items():
            # Chuẩn hóa tên hàm (loại bỏ phần tham số)
            normalized_name = fname.split("(")[0]
            if normalized_name == function_name:
                return fhash
        return None
    
    def _get_functions_in_critical_paths(self) -> set:
        """Get set of all function names in critical paths and optimal sequences."""
        functions = set()
        for path in self.critical_paths:
            functions.update(path)
        for seq in self.optimal_sequences:
            functions.update(seq)
        return functions
    
    def _get_vuln_type_for_function(self, function_name: str) -> Optional[str]:
        if not function_name:
            return None
        for vuln in self.potential_vulnerabilities or []:
            if isinstance(vuln, dict):
                candidates = vuln.get("functions")
                normalized = self._normalize_sequence(candidates)
                if function_name in normalized:
                    return vuln.get("type") or vuln.get("category") or "unknown"
            elif isinstance(vuln, (list, tuple)):
                if function_name in self._normalize_sequence(vuln):
                    return "unknown"
            elif isinstance(vuln, str) and function_name == vuln:
                return "unknown"
        return None
    
    
    def _get_function_args_from_rag(
        self,
        function_name: str,
        function_hash: str,
        argument_types: List[str],
        vuln_type: Optional[str] = None,
    ) -> Optional[List[Any]]:
        alpha = self._get_alpha()
        if random.random() > alpha:
            return None

        cache_key = f"{function_name}_{vuln_type}" if vuln_type else function_name
        if cache_key in self.arg_cache:
            self.rag_cache_hits += 1
            return self.arg_cache[cache_key]
        
        self.rag_requests += 1
                
        related_vulnerabilities: List[Dict[str, Any]] = []
        for vuln in self.potential_vulnerabilities or []:
            if not isinstance(vuln, dict):
                continue
            functions = vuln.get("functions", [])
            if isinstance(functions, list) and function_name in functions:
                    related_vulnerabilities.append(vuln)
        if vuln_type:
            related_vulnerabilities = [
                v for v in related_vulnerabilities if v.get("type") == vuln_type
            ]

        dataflow_context: Dict[str, Any] = {}
        if hasattr(self, "dataflow_graph"):
            graph = getattr(self, "dataflow_graph", {}) or {}
            for _, contract_info in graph.items():
                functions = contract_info.get("functions", {})
                if function_name in functions:
                    dataflow_context = functions[function_name]
                    break
            
        context_payload = {
            "contract": self.contract_name,
            "function": function_name,
            "function_hash": function_hash,
            "argument_types": argument_types,
            "vulnerability_type": vuln_type,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "dataflow": dataflow_context,
            "related_vulnerabilities": related_vulnerabilities,
            "dataflow_analysis": getattr(self, "analysis_result", None),
        }

        # Build comprehensive prompt for RAG query
        # SimpleRAG will: retrieve relevant docs -> inject context -> generate via LLM
        
        vulnerability_context = ""
        if vuln_type:
            vulnerability_context = f"Focus on {vuln_type} vulnerabilities. "
        if related_vulnerabilities:
            vuln_types = [v.get("type", "unknown") for v in related_vulnerabilities]
            vulnerability_context += f"Related vulnerability types: {', '.join(set(vuln_types))}. "
        
        # Build argument type description for clarity
        arg_desc = ", ".join([f"arg{i+1}: {arg_type}" for i, arg_type in enumerate(argument_types)])
        
        prompt = (
            "You are a smart-contract fuzzing specialist. Generate argument values that expose vulnerabilities.\n\n"
            f"Target Function: {function_name}\n"
            f"Function requires EXACTLY {len(argument_types)} arguments in this order: {arg_desc}\n"
            f"ABI Argument Types (call order): {argument_types}\n"
            f"Vulnerability Context: {vulnerability_context or 'General vulnerabilities'}\n\n"
            f"Contract Analysis Context:\n{json.dumps(context_payload, indent=2)}\n\n"
            "Use vulnerability patterns, examples, and edge cases from the RAG database context (automatically retrieved) "
            "to generate argument values. If no relevant patterns are found, use general fuzzing best practices.\n\n"
            "CRITICAL OUTPUT RULES:\n"
            "1) Return EXACTLY one JSON array and NOTHING ELSE (no text, no markdown, no comments, no code fences, no explanations).\n"
            "2) JSON does NOT support comments (// or /* */). Do NOT add any comments inside the JSON.\n"
            f"3) The array must contain EXACTLY {len(argument_types)} elements - no more, no less.\n"
            f"4) Element order MUST match: {arg_desc}\n"
            f"5) Do NOT add extra elements. Do NOT add elements for types not in the list above.\n\n"
            "Type encoding rules (MUST match exactly):\n"
            "   - address / address payable: JSON string, lowercase \"0x\" + 40 hex chars. (No checksum required; use lowercase.)\n"
            "   - uint<M> / int<M>: prefer **decimal string** for safety (e.g. \"115792089...\") if your consumer might lose precision; otherwise numeric literal is allowed. For unsigned, include edge values: 0, 1, max (2^M-1), max-1.\n"
            "   - bool: true or false (unquoted).\n"
            "   - string: JSON string (quoted). Can include long payloads, \\u0000, unicode, and format-injection patterns.\n"
            "   - bytesN (fixed): lowercase hex string \"0x\" + 2*N hex chars.\n"
            "   - bytes (dynamic): lowercase hex string \"0x\" + even number of hex chars (empty allowed as \"0x\").\n"
            "   - enum: return numeric index (integer) corresponding to enum variant.\n"
            "   - tuple/struct: ALWAYS represent as a JSON ARRAY matching ABI field order (do not use objects).\n"
            "   - arrays T[k] or T[]: JSON arrays. For fixed-size arrays T[k] produce exactly k elements.\n"
            "   - function: represent as address string of target contract (0x...40 hex).\n"
            "4) Value selection rules (based on vulnerability patterns from RAG database):\n"
            "   - For reentrancy: use addresses that could be malicious contracts, large amounts that could drain funds.\n"
            "   - For integer overflow/underflow: use max uint values, 0, 1, or values near boundaries.\n"
            "   - For access control: use zero address, invalid addresses, or addresses without permissions.\n"
            "   - For unchecked external calls: use addresses that might revert or consume gas.\n"
            "   - Prefer edge/interesting values (max, 0, -1 for signed, zero address, self-address, repeating patterns, very long strings/arrays, boundary lengths).\n"
            "   - Use patterns and examples from the RAG database context (automatically retrieved) as guidance.\n"
            "5) Never return null for any argument. Never return an empty outer array unless function has zero parameters.\n"
            "6) Format/hex case: all hex must be lowercase and 0x-prefixed.\n"
            "7) Fixed arrays/tuples: obey exact lengths and orders.\n\n"
            "Example output format (for illustration only — DO NOT output this exact example):\n"
            "[\"0x0000000000000000000000000000000000000000\", \"0x1111111111111111111111111111111111111111\", \"115792089237316195423570985008687907853269984665640564039457584007913129639935\"]\n\n"
            "CRITICAL REMINDERS:\n"
            "- NO comments in JSON (// or /* */ are NOT allowed)\n"
            "- NO markdown fences (```)\n"
            "- NO explanations or text before/after the array\n"
            "- EXACTLY {len(argument_types)} elements matching the types: {', '.join(argument_types)}\n"
            "- Return ONLY the JSON array starting with '[' and ending with ']'"
        )

        self.logger.info("Querying SimpleRAG for %s arguments", function_name)
        rag_response = self._fetch_rag_suggestion(prompt)
        if not rag_response or not rag_response.strip():
            self.logger.debug("RAG returned empty response for %s", function_name)
            return None

        # Log raw response for debugging
        self.logger.debug("Raw RAG response for %s (first 500 chars): %s", function_name, rag_response[:500] if rag_response else "None")
        self.logger.debug("Raw RAG response for %s (last 200 chars): %s", function_name, rag_response[-200:] if rag_response and len(rag_response) > 200 else rag_response)

        try:
            parsed_args = self._parse_rag_arguments(rag_response)
        except ValueError as exc:
            error_str = str(exc)
            # Use debug level for expected no-context errors to reduce log noise
            if "no-context" in error_str.lower():
                self._no_context_count += 1
                self.logger.debug("RAG returned no-context for %s (count: %d/%d)",
                                 function_name, self._no_context_count, self._max_no_context_before_disable)
                # Disable RAG after too many no-context responses (database likely has no relevant data)
                if self._no_context_count >= self._max_no_context_before_disable:
                    self.logger.info("RAG disabled due to %d consecutive no-context responses (database has no relevant data)",
                                    self._no_context_count)
                    self._quota_exceeded = True  # Reuse this flag to disable RAG
            else:
                self.logger.debug("Failed to parse RAG response for %s: %s", function_name, exc)
                self.logger.debug("Full response that failed to parse (first 1000 chars): %s", rag_response[:1000] if rag_response else "None")
                self.rag_failures += 1
                return None

        if not parsed_args:
            return None

        produced_args: List[Any] = []
        for index, arg_type in enumerate(argument_types):
            value = parsed_args[index] if index < len(parsed_args) else None
            coerced = self._coerce_argument_value(arg_type, value)
            produced_args.append(coerced)

        if len(produced_args) != len(argument_types):
            self.logger.warning("RAG args length mismatch for %s", function_name)
            return None

        self.logger.info("RAG generated args for %s: %s", function_name, produced_args)
        # Log the full chain from analysis context -> prompt -> args for transparency
        self.logger.debug(
            "RAG ctx for %s | vuln=%s | related=%s | dataflow=%s",
            function_name,
            vuln_type,
            [v.get("type") for v in related_vulnerabilities],
            dataflow_context,
        )
        self.arg_cache[cache_key] = produced_args
        self.rag_successes += 1
        self._no_context_count = 0  # Reset counter on success
        return produced_args

    def _parse_rag_arguments(self, response_text: str) -> Optional[List[Any]]:
        """Parse LLM response assuming strict JSON array output."""
        if not response_text or not response_text.strip():
            self.logger.debug("Empty response text received")
            raise ValueError("Empty response")

        clean_response = response_text.strip()

        # Log full response for debugging (first 500 chars)
        self.logger.debug("Parsing RAG response (first 500 chars): %s", clean_response[:500])
        self.logger.debug("Parsing RAG response (last 200 chars): %s", clean_response[-200:] if len(clean_response) > 200 else clean_response)

        # Check for "no-context" or error responses from RAG
        # RAG returns these when no relevant data found in database
        no_context_indicators = [
            "[no-context]",
            "no-context",
            "Sorry, I'm not able to provide",
            "I cannot provide",
            "I don't have enough",
            "No relevant information",
            "Unable to generate",
            "I apologize",
            "cannot answer",
            "no relevant",
            "I'm sorry",
            "don't have information",
            "no data available",
        ]
        lower_response = clean_response.lower()
        for indicator in no_context_indicators:
            if indicator.lower() in lower_response:
                self.logger.debug("RAG returned no-context response, skipping parse")
                self._quota_exceeded = True  # Disable RAG temporarily to avoid repeated failures
                raise ValueError(f"RAG returned no-context response: {indicator}")

        # Remove optional markdown fences if they slipped through
        if clean_response.startswith("```"):
            # Find the closing ```
            end_idx = clean_response.find("```", 3)
            if end_idx != -1:
                clean_response = clean_response[3:end_idx].strip()
                # Remove "json" if present
                if clean_response.lower().startswith("json"):
                    clean_response = clean_response[4:].strip()
            else:
                # No closing fence, just remove opening
                clean_response = clean_response[3:].strip()
                if clean_response.lower().startswith("json"):
                    clean_response = clean_response[4:].strip()
        
        clean_response = clean_response.strip()

        # Try to find JSON array in the response (in case there's extra text)
        # Look for first '[' and last ']'
        first_bracket = clean_response.find('[')
        last_bracket = clean_response.rfind(']')
        
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            clean_response = clean_response[first_bracket:last_bracket+1]
            self.logger.debug("Extracted JSON array from response: %s", clean_response[:200])
        elif first_bracket == -1 or last_bracket == -1:
            self.logger.debug("No JSON array brackets found in response. Full response: %s", clean_response[:500])
            raise ValueError(f"Response does not contain a JSON array. First 200 chars: {clean_response[:200]}")

        # Remove JSON comments (// ... and /* ... */)
        # JSON doesn't support comments, but LLMs often add them
        import re
        # Remove single-line comments (// ...)
        clean_response = re.sub(r'//.*?$', '', clean_response, flags=re.MULTILINE)
        # Remove multi-line comments (/* ... */)
        clean_response = re.sub(r'/\*.*?\*/', '', clean_response, flags=re.DOTALL)
        # Clean up extra whitespace
        clean_response = re.sub(r'\n\s*\n', '\n', clean_response)  # Remove empty lines
        clean_response = clean_response.strip()

        self.logger.debug("After removing comments: %s", clean_response[:200])

        # Single attempt to parse JSON array
        try:
            data = json.loads(clean_response)
        except json.JSONDecodeError as exc:
            self.logger.debug("JSON parse error. Response (first 500 chars): %s", clean_response[:500])
            self.logger.debug("JSON parse error details: %s at position %d", exc.msg, exc.pos)
            # Try to show context around error
            if exc.pos and exc.pos < len(clean_response):
                start = max(0, exc.pos - 50)
                end = min(len(clean_response), exc.pos + 50)
                self.logger.debug("Context around error: ...%s<ERROR>%s...", clean_response[start:exc.pos], clean_response[exc.pos:end])
            raise ValueError(f"Response is not valid JSON: {exc}") from exc

        if not isinstance(data, list):
            self.logger.debug("Parsed JSON is not a list, it's a %s: %s", type(data), data)
            raise ValueError(f"Response JSON is not an array, got {type(data)}")

        return data

    def _coerce_argument_value(self, arg_type: str, value: Any) -> Any:
        if value is None:
            return self._get_interesting_value_for_type(arg_type)

        # Integer-like (uint*/int*)
        if arg_type.startswith("uint") or arg_type.startswith("int"):
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                try:
                    if value.startswith("0x"):
                        return int(value, 16)
                    if value.replace("-", "").isdigit():
                        return int(value)
                except (ValueError, OverflowError):
                    pass
            return self._get_interesting_value_for_type(arg_type)

        # Address
        if arg_type == "address":
            if isinstance(value, str) and value.startswith("0x") and len(value) == 42:
                return value
            return random.choice(self.accounts) if self.accounts else "0x0000000000000000000000000000000000000000"

        # Boolean
        if arg_type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
            return bool(value)

        # bytes / bytesN (including bytes32)
        if arg_type.startswith("bytes"):
            # Already bytes-like → normalize to bytes
            if isinstance(value, (bytes, bytearray)):
                raw_bytes = bytes(value)
            elif isinstance(value, str):
                # Accept hex string (with/without 0x prefix)
                candidate = value.strip()
                if candidate.startswith("0x") or candidate.startswith("0X"):
                    candidate = candidate[2:]
                try:
                    raw_bytes = bytes.fromhex(candidate)
                except ValueError:
                    # Not valid hex → fall back to interesting value
                    return self._get_interesting_value_for_type(arg_type)
            else:
                # Unsupported type → fall back
                return self._get_interesting_value_for_type(arg_type)

            # For fixed-size bytesN, ensure correct length by trim/pad
            if arg_type != "bytes":
                import re

                m = re.match(r"bytes(\d+)$", arg_type)
                if m:
                    target_len = int(m.group(1))
                    if len(raw_bytes) == target_len:
                        return raw_bytes
                    if len(raw_bytes) > target_len:
                        # Keep least-significant bytes (right-trim)
                        return raw_bytes[-target_len:]
                    # Left-pad with zeros to required length
                    return raw_bytes.rjust(target_len, b"\x00")

            # Dynamic bytes → trả nguyên
            return raw_bytes

        # Mặc định: trả lại value (chuỗi, list...) để Individual/ABI tự xử lý
        return value
    
    def _get_interesting_value_for_type(self, type_str: str) -> Any:
        """Return an interesting (non-default) value for a data type to improve fuzzing."""
        if type_str.startswith("uint"):
            interesting_values = [0, 1, 2**256 - 1, 2**128 - 1, 2**64 - 1, 1000000, 10]
            return random.choice(interesting_values)
        elif type_str.startswith("int"):
            interesting_values = [0, 1, -1, 2**127 - 1, -(2**127), 1000000, -1000000]
            return random.choice(interesting_values)
        elif type_str == "address":
            interesting_addresses = [
                "0x0000000000000000000000000000000000000000",
                "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                "0x1000000000000000000000000000000000000000",
                "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
            ]
            return random.choice(interesting_addresses)
        elif type_str == "bool":
            return random.choice([True, False])
        elif type_str.startswith("bytes"):
            interesting_bytes = ["0x00", "0xFFFFFFFF", "0x1234567890ABCDEF", "0x" + "00" * 32]
            return random.choice(interesting_bytes)
        elif type_str == "string":
            interesting_strings = ["", "Hello", "A" * 100, "Special@#$%^&*()Characters"]
            return random.choice(interesting_strings)
        else:
            return None
    
    def _get_default_value_for_type(self, type_str: str) -> Any:
        """Return default value for a data type."""
        if type_str.startswith("uint") or type_str.startswith("int"):
            return 0
        elif type_str == "address":
            return "0x0000000000000000000000000000000000000000"
        elif type_str == "bool":
            return False
        elif type_str.startswith("bytes"):
            return "0x00"
        elif type_str == "string":
            return ""
        else:
            return None
    
    def generate_individual(self, function: str, argument_types: List[str], 
                           vuln_type: Optional[str] = None, default_value: bool = False) -> List[Dict]:
        """Generate transaction using RAG to get optimal parameter values."""
        # Get function name from interface_mapper
        function_name = None
        if self.interface_mapper:
            for fname, fhash in self.interface_mapper.items():
                if fhash == function:
                    function_name = fname.split("(")[0]  # Remove parameters
                    break
        
        rag_args = None
        # Refresh provider to know if global LLM evolution is active (for logging only)
        provider = self._refresh_provider()
        # Check if we've exceeded quota - if so, reduce RAG usage significantly
        quota_exceeded = getattr(self, '_quota_exceeded', False)
        if quota_exceeded:
            # If quota exceeded, only use RAG 10% of the time
            use_rag_probability = 0.1
        else:
            # Increase probability for RAG when vuln_type is specified or when in critical paths
            use_rag_probability = 0.8 if vuln_type or function_name in self._get_functions_in_critical_paths() else 0.5
        
        alpha = self._get_alpha()
        use_rag_probability *= alpha

        # Allow RAG argument generation even if LLM evolution (provider) is disabled.
        # SimpleRAG manages its own LLM client and rate limiting.
        if function_name and not default_value and random.random() < use_rag_probability:
            try:
                rag_args = self._get_function_args_from_rag(function_name, function, argument_types, vuln_type)
                # Reset quota flag if query succeeded
                if rag_args is not None:
                    self._quota_exceeded = False
                    self.logger.info("Using RAG args for %s (vuln=%s): %s", function_name, vuln_type, rag_args)
            except Exception as e:
                error_str = str(e)
                if isinstance(e, ProviderError) or "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    self._quota_exceeded = True
                    self.logger.debug("RAG quota exceeded, reducing RAG usage for remaining queries")
                self.logger.debug(f"RAG query failed for {function_name}: {e}")
        else:
            self.logger.debug(
                "Skip RAG for %s | prob=%.2f | provider=%s | quota_exceeded=%s",
                function_name, use_rag_probability, bool(provider), quota_exceeded
            )
        
        individual = []
        arguments = [function]
        
        if rag_args and len(rag_args) == len(argument_types):
            arguments.extend(rag_args)
            self.logger.debug(f"Using RAG args for {function_name}")
        else:
            for index in range(len(argument_types)):
                arguments.append(self.get_random_argument(argument_types[index], function, index))
            if rag_args is None:
                self.logger.debug("Fallback to random args for %s (no RAG args)", function_name)
            elif len(rag_args) != len(argument_types):
                self.logger.debug("Fallback to random args for %s (len mismatch)", function_name)
        
        # Log the transaction build for visibility
        self.logger.info(
            "Tx built for %s | via=%s | args=%s",
            function_name or function,
            "RAG" if rag_args else "random",
            arguments[1:],
        )
        
        individual.append({
            "account": self.get_random_account(function),
            "contract": self.contract,
            "amount": self.get_random_amount(function),
            "arguments": arguments,
            "blocknumber": self.get_random_blocknumber(function),
            "timestamp": self.get_random_timestamp(function),
            "gaslimit": self.get_random_gaslimit(function),
            "call_return": dict(),
            "extcodesize": dict(),
            "returndatasize": dict()
        })

        address, call_return_value = self.get_random_callresult_and_address(function)
        individual[-1]["call_return"] = {address: call_return_value}

        address, extcodesize_value = self.get_random_extcodesize_and_address(function)
        individual[-1]["extcodesize"] = {address: extcodesize_value}

        address, value = self.get_random_returndatasize_and_address(function)
        individual[-1]["returndatasize"] = {address: value}

        return individual
    

    def generate_random_individual(self, func_hash=None, func_args_types=None, default_value=False):
        """Generate smart transaction sequence based on dataflow analysis and successful sequence history."""
        if func_hash is not None and func_args_types is not None:
            individual = []
            individual.extend(self.generate_constructor())
            # Use self.generate_individual to enable RAG
            function_name = None
            if self.interface_mapper:
                for fname, fhash_val in self.interface_mapper.items():
                    if fhash_val == func_hash:
                        function_name = fname.split("(")[0]
                        break
            vuln_type = self._get_vuln_type_for_function(function_name) if function_name else None
            individual.extend(self.generate_individual(func_hash, func_args_types, vuln_type=vuln_type, default_value=default_value))
            return individual
        
        individual = []
        individual.extend(self.generate_constructor())
        
        original_functions = 0
        rag_enhanced_functions = 0
        self.logger.info(
            "Sequence generation start | strategies=%s | crit_paths=%d | optimal_seq=%d | funcs=%d",
            ["optimal_sequence", "critical_path", "mutation", "random", "combined"],
            len(self.critical_paths),
            len(self.optimal_sequences),
            len(self.interface),
        )
        
        available_functions = list(self.interface.keys())
        random.shuffle(available_functions)
        
        strategy = random.choices(
            ["optimal_sequence", "critical_path", "mutation", "random", "combined"],
            weights=[0.25, 0.25, 0.2, 0.1, 0.2]
        )[0]
        
        if not hasattr(self, '_strategy_counts'):
            self._strategy_counts = {
                "optimal_sequence": 0, 
                "critical_path": 0, 
                "mutation": 0, 
                "random": 0, 
                "combined": 0
            }
        self._strategy_counts[strategy] += 1
        self.logger.info("Chosen strategy: %s (counts=%s)", strategy, self._strategy_counts)
        
        MAX_SEQUENCE_LENGTH = max(1, min(settings.MAX_INDIVIDUAL_LENGTH, len(available_functions) or 1))

        if (strategy == "optimal_sequence" or strategy == "critical_path") and hasattr(self, '_last_sequence'):
            if self._last_sequence == strategy:
                self._repeat_count = getattr(self, '_repeat_count', 0) + 1
                if self._repeat_count > 5:
                    strategy = random.choice(["mutation", "combined", "random"])
                    self.logger.info(f"Switching to {strategy} to increase diversity")
                    self._repeat_count = 0
            else:
                self._repeat_count = 1
        
        self._last_sequence = strategy
        random.seed(time.time() + random.random())

        if strategy == "optimal_sequence" and self.optimal_sequences:
            selected_seq = random.choice(self.optimal_sequences)
            sequence_template = self._normalize_sequence(selected_seq)
            if sequence_template:
                sequence_template = sequence_template[: MAX_SEQUENCE_LENGTH - 1]
            self.logger.info("Using optimal sequence template: %s (from: %s)", sequence_template, selected_seq)
            if not sequence_template:
                self.logger.warning("Failed to normalize optimal sequence, will use random functions")
            else:
                for func_name in sequence_template:
                    func_hash = self._get_function_hash_by_name(func_name)
                    if func_hash and func_hash in self.interface:
                        vuln_type = self._get_vuln_type_for_function(func_name)
                        tx = self.generate_individual(
                            func_hash,
                            self.interface[func_hash],
                            vuln_type=vuln_type,
                            default_value=default_value,
                        )
                        if tx:
                            individual.extend(tx)
                            rag_enhanced_functions += 1
            
            if available_functions and len(individual) < MAX_SEQUENCE_LENGTH + 1:
                used_hashes = {
                    tx.get("arguments", [None])[0]
                    for tx in individual[1:]
                    if tx.get("arguments")
                }
                unused_functions = [f for f in available_functions if f not in used_hashes]
                if unused_functions:
                    random_func = random.choice(unused_functions)
                    tx = self.generate_individual(
                        random_func,
                        self.interface[random_func],
                        default_value=default_value,
                    )
                    if tx:
                        individual.extend(tx)
                        original_functions += 1
        
        elif strategy == "critical_path" and self.critical_paths:
            selected_path = random.choice(self.critical_paths)
            path = self._normalize_sequence(selected_path)
            if path:
                path = path[: MAX_SEQUENCE_LENGTH - 1]
            self.logger.info("Using critical path: %s (from: %s)", path, selected_path)
            if not path:
                self.logger.warning("Failed to normalize critical path, will use random functions")
            else:
                for func_name in path:
                    func_hash = self._get_function_hash_by_name(func_name)
                    if func_hash and func_hash in self.interface:
                        vuln_type = self._get_vuln_type_for_function(func_name)
                        tx = self.generate_individual(
                            func_hash,
                            self.interface[func_hash],
                            vuln_type=vuln_type,
                            default_value=default_value,
                        )
                        if tx:
                            individual.extend(tx)
                            rag_enhanced_functions += 1
            
            if available_functions and len(individual) < MAX_SEQUENCE_LENGTH + 1:
                used_hashes = set()
                for tx in individual[1:]:
                    if "arguments" in tx and tx["arguments"]:
                        used_hashes.add(tx["arguments"][0])
                
                unused_functions = [f for f in available_functions if f not in used_hashes]
                if unused_functions:
                    random_func = random.choice(unused_functions)
                    tx = self.generate_individual(
                        random_func,
                        self.interface[random_func],
                        default_value=default_value,
                    )
                    if tx:
                        individual.extend(tx)
                        original_functions += 1 
        
        elif strategy == "mutation" and hasattr(self, 'population') and self.population:
            if len(self.population) > 0:
                max_idx = len(self.population) - 1
                if max_idx > 10 and random.random() < 0.3:
                    best_idx = random.randint(5, min(10, max_idx))
                else:
                    best_idx = random.randint(0, min(5, max_idx))
                
                if hasattr(self.population[best_idx], "individual"):
                    base_sequence = self.population[best_idx]["individual"]
                elif hasattr(self.population[best_idx], "chromosome"):
                    base_sequence = self.population[best_idx].chromosome
                else:
                    base_sequence = self.population[best_idx]
                
                if base_sequence and len(base_sequence) > 1:
                    mutated_indiv = self.mutate(base_sequence)
                    self.logger.info(f"Applied full mutation to sequence from population index {best_idx}")
                    return mutated_indiv
        
        elif strategy == "combined":
            # Số giao dịch còn có thể thêm vào mà vẫn tôn trọng MAX_INDIVIDUAL_LENGTH
            remaining_slots = max(0, settings.MAX_INDIVIDUAL_LENGTH - len(individual))
            eff_max = min(MAX_SEQUENCE_LENGTH, remaining_slots)
            if eff_max <= 0:
                num_transactions = 0
            elif eff_max < 3:
                # Nếu còn rất ít slot (<3) thì lấy hết, tránh randint empty range
                num_transactions = eff_max
            else:
                num_transactions = random.randint(3, eff_max)
            sources = []
            
            if self.optimal_sequences:
                sampled = self._sample_function_from_sequence(random.choice(self.optimal_sequences))
                if sampled:
                    func_hash = self._get_function_hash_by_name(sampled)
                    if func_hash and func_hash in self.interface:
                        sources.append((func_hash, self.interface[func_hash]))
            
            if self.critical_paths:
                sampled = self._sample_function_from_sequence(random.choice(self.critical_paths))
                if sampled:
                    func_hash = self._get_function_hash_by_name(sampled)
                    if func_hash and func_hash in self.interface:
                        sources.append((func_hash, self.interface[func_hash]))
            self.logger.info("Combined strategy sources: %s", [s[0] for s in sources])
            
            for _ in range(2):
                if available_functions:
                    random_func = random.choice(available_functions)
                    sources.append((random_func, self.interface[random_func]))
            
            random.shuffle(sources)
            selected_sources = sources[:num_transactions] if num_transactions > 0 else []
            
            for func_hash, func_args_types in selected_sources:
                function_name = None
                if self.interface_mapper:
                    for fname, fhash_val in self.interface_mapper.items():
                        if fhash_val == func_hash:
                            function_name = fname.split("(")[0]
                            break
                vuln_type = self._get_vuln_type_for_function(function_name) if function_name else None
                tx = self.generate_individual(func_hash, func_args_types, vuln_type=vuln_type, default_value=default_value)
                if tx:
                    individual.extend(tx)
                    in_optimal = any(
                        self._get_function_hash_by_name(name) == func_hash
                        for seq in self.optimal_sequences
                        for name in self._normalize_sequence(seq)
                    )
                    in_critical = any(
                        self._get_function_hash_by_name(name) == func_hash
                        for path in self.critical_paths
                        for name in self._normalize_sequence(path)
                    )
                    if in_optimal or in_critical:
                        rag_enhanced_functions += 1
                    else:
                        original_functions += 1
        
        if len(individual) <= (1 if self.generate_constructor() else 0) or strategy == "random":
            remaining_slots = max(0, settings.MAX_INDIVIDUAL_LENGTH - len(individual))
            eff_max = min(MAX_SEQUENCE_LENGTH, remaining_slots)
            if eff_max <= 0:
                num_transactions = 0
            elif eff_max < 2:
                num_transactions = eff_max
            else:
                num_transactions = random.randint(2, eff_max)

            functions_to_use = available_functions[:num_transactions] if len(available_functions) >= num_transactions else available_functions
            self.logger.info("Random strategy: sampling %d functions", len(functions_to_use))
            
            for function in functions_to_use:
                argument_types = self.interface[function]
                # Use self.generate_individual to enable RAG
                tx = self.generate_individual(function, argument_types, default_value=default_value)
                if tx:
                    individual.extend(tx)
                    original_functions += 1
        
        if len(individual) > settings.MAX_INDIVIDUAL_LENGTH:
            if self.generate_constructor():
                individual = individual[:1] + individual[1:settings.MAX_INDIVIDUAL_LENGTH]
            else:
                individual = individual[:settings.MAX_INDIVIDUAL_LENGTH]

        if hasattr(self, 'good_sequences') and len(individual) > 1:
            if not isinstance(self.good_sequences, list):
                self.good_sequences = []
            if len(self.good_sequences) < 30:
                self.good_sequences.append(individual)
                    
        return individual

    def _initialize_rag(self):
        """Initialize SimpleRAG instance (lazy initialization)."""
        if self._rag_initialized:
            return self._rag_instance

        # Check if vector DB exists
        if not os.path.exists(self.rag_working_dir):
            self.logger.warning(f"RAG storage not found at {self.rag_working_dir}, RAG will not be available")
            self._rag_initialized = True
            return None

        # Get API keys and model/provider from config
        api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        openai_api_key = self.openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Try to get model/provider from args or config
        llm_model = getattr(self, 'llm_model', None) or os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        llm_provider = getattr(self, 'llm_provider', None) or os.environ.get("LLM_PROVIDER")
        
        # Auto-detect provider from model if not specified
        if not llm_provider:
            if llm_model.lower().startswith("gpt") or llm_model.lower().startswith("o1"):
                llm_provider = "openai"
            elif llm_model.lower().startswith("gemini"):
                llm_provider = "gemini"
            else:
                llm_provider = "gemini"  # default
        
        # Option to disable LLM in RAG (similarity-only)
        if getattr(self, "disable_rag_llm", False):
            llm_provider = None
            llm_model = None

        try:
            # Create SimpleRAG instance
            self._rag_instance = SimpleRAG(
                vector_db_path=self.rag_working_dir,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                api_key=api_key,
                llm_model=llm_model,
                llm_provider=llm_provider,
                openai_api_key=openai_api_key
            )

            # Initialize (load vector store and create QA chain)
            if self._rag_instance.initialize():
                self._rag_initialized = True
                self.logger.info("SimpleRAG instance initialized successfully")
                return self._rag_instance
            else:
                self.logger.warning("Failed to initialize SimpleRAG (initialize returned False)")
                self._rag_initialized = True
                return None

        except Exception as exc:
            import traceback
            tb_str = traceback.format_exc()
            self.logger.error(f"Failed to initialize RAG: {exc}\n{tb_str}")
            try:
                import logging
                logging.getLogger("RAGInit").error("Failed to initialize RAG: %s\n%s", exc, tb_str)
            except Exception:
                pass
            self._rag_initialized = True
            return None
    
    def _wait_for_rate_limit(self):
        """Wait to respect rate limit (15 requests/minute for Gemini free tier)."""
        if not self._enforce_rate_limit:
            return
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if current_time - t < 60]
        
        # Gemini free tier: 15 requests/minute
        max_requests_per_minute = 14  # Leave some buffer
        
        if len(self._request_times) >= max_requests_per_minute:
            oldest_request = min(self._request_times)
            wait_time = 60 - (current_time - oldest_request) + 2  # Add 2 second buffer
            if wait_time > 0:
                self.logger.debug(f"Rate limit: waiting {wait_time:.1f}s ({max_requests_per_minute} requests/minute limit)")
                time.sleep(wait_time)
                # Update request times after waiting
                current_time = time.time()
                self._request_times = [t for t in self._request_times if current_time - t < 60]
        
        # Ensure minimum delay between requests (4 seconds = 15 requests/minute)
        min_delay = 4.0
        if self._last_request_time > 0:
            elapsed = current_time - self._last_request_time
            if elapsed < min_delay:
                wait_time = min_delay - elapsed
                self.logger.debug(f"Rate limit: waiting {wait_time:.1f}s (min delay between RAG queries)")
                time.sleep(wait_time)
                current_time = time.time()
        
        # Record this request
        self._request_times.append(current_time)
        self._last_request_time = current_time
    
    def _fetch_rag_suggestion(self, prompt: str) -> Optional[str]:
        """Query SimpleRAG to get suggestions with rate limiting and quota handling."""
        try:
            rag = self._initialize_rag()
            if rag is None:
                self.logger.debug("RAG instance is None, cannot query")
                return None
                
            # Apply rate limiting before making request
            self._wait_for_rate_limit()

            self.rag_requests += 1
            self.logger.debug("Querying SimpleRAG")

            # Use SimpleRAG's query method (synchronous, no modes needed)
            result = rag.query(prompt, use_llm=True)

            if result and result.strip():
                self.rag_successes += 1
                self.logger.debug(f"RAG query succeeded, response length: {len(result)}")
                return result
            else:
                self.logger.debug("RAG returned empty or None response")
                return None

        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__
            self.logger.debug(f"RAG query exception ({error_type}): {error_str[:200]}")

            # Check for quota/rate limit errors
            if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                self.logger.warning(f"RAG quota exceeded: {e}")
                self._quota_exceeded = True

            self.rag_failures += 1
            return None


    def mutate(self, individual):
        """Đột biến một cá thể để tạo biến thể mới với độ đa dạng cao hơn"""
        # Tạo một file log riêng cho quá trình đột biến
        log_dir = os.path.join(os.getcwd(), "fuzzing_logs")
        os.makedirs(log_dir, exist_ok=True)
        mutation_log_file = os.path.join(log_dir, "mutation_statistics.txt")
        
        # Khởi tạo hoặc cập nhật biến theo dõi số lượng đột biến tổng cộng
        if not hasattr(self, '_mutation_stats_total'):
            self._mutation_stats_total = {
                "replace": 0,
                "insert": 0,
                "remove": 0,
                "swap": 0,
                "param_mutate": 0,
                "total_mutations": 0,
                "successful_mutations": 0
            }
        
        # Khởi tạo hoặc cập nhật số lượng cá thể đã đột biến
        if not hasattr(self, '_mutated_individuals_count'):
            self._mutated_individuals_count = 0
        self._mutated_individuals_count += 1
        
        # Giữ nguyên constructor
        constructor = individual[:1] if len(individual) > 0 else []
        transactions = individual[1:] if len(individual) > 1 else []
        
        if not transactions:
            # Ghi log khi không có transaction để đột biến
            with open(mutation_log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] MUTATION STOPPED: No transactions to mutate. Creating random individual instead.\n")
            self.logger.warning("Mutation stopped: No transactions to mutate. Creating random individual instead.")
            return self.generate_random_individual()
        
        # Ghi log bắt đầu quá trình đột biến
        original_tx_count = len(transactions)
        with open(mutation_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] MUTATION STARTED: Original sequence has {original_tx_count} transactions\n")
        
        # Lưu fitness trước khi đột biến nếu có
        original_fitness = None
        original_coverage = None
        if hasattr(self, 'population'):
            for item in self.population:
                if isinstance(item, dict) and "individual" in item and item["individual"] == individual:
                    original_fitness = item.get("fitness", None)
                    original_coverage = item.get("coverage", None)
                    break
        
        # Chọn ngẫu nhiên một số lượng giao dịch để đột biến
        max_possible_mutations = max(2, len(transactions) // 2)
        num_mutations = random.randint(1, max_possible_mutations)
        
        # Cập nhật số lượng đột biến dự kiến trong thống kê tổng thể
        self._mutation_stats_total["total_mutations"] += num_mutations
        
        # Ghi log số lượng đột biến sẽ thực hiện
        with open(mutation_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Planning to apply {num_mutations} mutations out of maximum {max_possible_mutations}\n")
        
        self.logger.info(f"Mutation process: Planning to apply {num_mutations} mutations on sequence of {len(transactions)} transactions")
        
        # Khởi tạo thống kê đột biến
        mutation_stats = {
            "replace": 0,
            "insert": 0,
            "remove": 0, 
            "swap": 0,
            "param_mutate": 0
        }
        
        # Theo dõi các đột biến đã thực hiện để debug
        mutations_applied = []
        
        # Đếm số lần đột biến đã thực hiện
        mutations_performed = 0
        
        for i in range(num_mutations):
            # Tùy chỉnh phân phối của các loại đột biến 
            mutation_weights = [40, 20, 15, 25]  # Tỷ lệ % cho replace, insert, remove, swap
            mutation_type = random.choices(
                ["replace", "insert", "remove", "swap"],
                weights=mutation_weights
            )[0]
            
            mutation_success = False
            
            if mutation_type == "replace" and transactions:
                idx = random.randint(0, len(transactions) - 1)

                existing_functions = {
                    tx.get("arguments", [None])[0]
                    for tx in transactions
                    if tx.get("arguments")
                }

                available_functions = [f for f in self.interface.keys() if f not in existing_functions]

                if available_functions and random.random() < 0.8:
                    function = random.choice(available_functions)
                    argument_types = self.interface[function]
                else:
                    function, argument_types = self.get_random_function_with_argument_types()

                new_tx = self.generate_individual(function, argument_types)
                if new_tx:
                    old_func = transactions[idx].get("arguments", ["unknown"])[0]
                    transactions[idx] = new_tx[0]
                    mutations_applied.append(f"Replaced tx {idx+1}: {old_func[:8]} -> {function[:8]}")
                    mutation_stats["replace"] += 1
                    self._mutation_stats_total["replace"] += 1
                    mutation_success = True
            
            elif mutation_type == "insert" and len(transactions) < settings.MAX_INDIVIDUAL_LENGTH - 1:
                existing_functions = {
                    tx.get("arguments", [None])[0]
                    for tx in transactions
                    if tx.get("arguments")
                }

                available_functions = [f for f in self.interface.keys() if f not in existing_functions]

                if available_functions and random.random() < 0.8:
                    function = random.choice(available_functions)
                    argument_types = self.interface[function]
                else:
                    function, argument_types = self.get_random_function_with_argument_types()

                new_tx = self.generate_individual(function, argument_types)
                if new_tx:
                    idx = random.randint(0, len(transactions))
                    transactions.insert(idx, new_tx[0])
                    mutations_applied.append(f"Inserted tx at position {idx+1}: {function[:8]}")
                    mutation_stats["insert"] += 1
                    self._mutation_stats_total["insert"] += 1
                    mutation_success = True
            
            elif mutation_type == "remove" and len(transactions) > 1:
                # Xóa một giao dịch
                idx = random.randint(0, len(transactions) - 1)
                func_to_remove = transactions[idx].get("arguments", ["unknown"])[0]
                transactions.pop(idx)
                mutations_applied.append(f"Removed tx {idx+1}: {func_to_remove[:8]}")
                mutation_stats["remove"] += 1
                self._mutation_stats_total["remove"] += 1
                mutation_success = True
            
            elif mutation_type == "swap" and len(transactions) > 1:
                # Đổi vị trí hai giao dịch
                idx1 = random.randint(0, len(transactions) - 1)
                idx2 = random.randint(0, len(transactions) - 1)
                while idx1 == idx2:
                    idx2 = random.randint(0, len(transactions) - 1)
                    
                func1 = transactions[idx1].get("arguments", ["unknown"])[0]
                func2 = transactions[idx2].get("arguments", ["unknown"])[0]
                
                transactions[idx1], transactions[idx2] = transactions[idx2], transactions[idx1]
                mutations_applied.append(f"Swapped tx {idx1+1}:{func1[:8]} <-> tx {idx2+1}:{func2[:8]}")
                mutation_stats["swap"] += 1
                self._mutation_stats_total["swap"] += 1
                mutation_success = True
            
            # Thêm thuật toán đột biến tham số - đột biến giá trị của transaction
            elif mutation_type == "param_mutate" and transactions and random.random() < 0.3:
                idx = random.randint(0, len(transactions) - 1)
                if "arguments" in transactions[idx]:
                    func_hash = transactions[idx]["arguments"][0]
                    func_args_types = self.interface.get(func_hash, [])
                    
                    # Sinh lại transaction với tham số mới
                    new_tx = self.generate_individual(func_hash, func_args_types)
                    if new_tx:
                        transactions[idx] = new_tx[0]
                        mutations_applied.append(f"Mutated params of tx {idx+1}: {func_hash[:8]}")
                        mutation_stats["param_mutate"] += 1
                        self._mutation_stats_total["param_mutate"] += 1
                        mutation_success = True
            
            if mutation_success:
                mutations_performed += 1
                self._mutation_stats_total["successful_mutations"] += 1
            
            # Kiểm tra điều kiện dừng
            if len(transactions) == 0:
                with open(mutation_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] MUTATION STOPPED: All transactions were removed\n")
                self.logger.warning("Mutation stopped: All transactions were removed")
                function, argument_types = self.get_random_function_with_argument_types()
                new_tx = self.generate_individual(function, argument_types)
                if new_tx:
                    transactions.extend(new_tx)
                break
        
        # Tạo cá thể mới sau đột biến
        mutated_individual = constructor + transactions
        
        # So sánh số lượng transaction sau khi đột biến
        final_tx_count = len(transactions)
        tx_change = final_tx_count - original_tx_count
        
        # Ghi log tổng kết đột biến
        with open(mutation_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] MUTATION COMPLETED:\n")
            f.write(f"  - Planned mutations: {num_mutations}\n")
            f.write(f"  - Mutations performed: {mutations_performed}\n")
            f.write(f"  - Transaction count: {original_tx_count} -> {final_tx_count} ({tx_change:+d})\n")
            f.write(f"  - Mutation statistics: {json.dumps(mutation_stats)}\n")
            f.write(f"  - Applied mutations: {'; '.join(mutations_applied)}\n")
            if original_fitness is not None:
                f.write(f"  - Original fitness: {original_fitness}\n")
            if original_coverage is not None:
                f.write(f"  - Original coverage: {original_coverage}\n")
                f.write("\n")
            
        if mutations_applied:
            self.logger.info(f"Applied {len(mutations_applied)} mutations: {'; '.join(mutations_applied)}")
        else:
            self.logger.warning("No mutations were applied")
        
        return mutated_individual

    def _get_alpha(self) -> float:
        if self.adaptive_llm_controller is None:
            return 1.0
        engine = getattr(settings, "CURRENT_ENGINE", None)
        if engine is None:
            return 1.0
        try:
            current_cov = self._current_coverage()
            return self.adaptive_llm_controller.get_alpha(
                getattr(engine, "current_generation", 0),
                current_cov,
            )
        except Exception:
            return 1.0

    def _current_coverage(self) -> float:
        env = getattr(settings, "GLOBAL_ENV", None)
        if env and hasattr(env, "overall_pcs") and env.overall_pcs:
            return len(env.code_coverage) / len(env.overall_pcs) * 100.0
        return 0.0

    def crossover(self, parent1_index, parent2_index):
        """Lai ghép hai cá thể để tạo cá thể mới"""
        if not hasattr(self, 'population'):
            return self.generate_random_individual()
        
        if parent1_index >= len(self.population) or parent2_index >= len(self.population):
            return self.generate_random_individual()
        
        parent1 = self.population[parent1_index]["individual"]
        parent2 = self.population[parent2_index]["individual"]
        
        # Bỏ qua constructor vì nó nên giữ nguyên
        constructor = parent1[:1] if len(parent1) > 0 else []
        
        # Lấy phần còn lại của các cá thể cha mẹ
        parent1_txs = parent1[1:] if len(parent1) > 1 else []
        parent2_txs = parent2[1:] if len(parent2) > 1 else []
        
        if not parent1_txs or not parent2_txs:
            return self.generate_random_individual()
        
        # Chọn điểm cắt
        cut_point = random.randint(1, min(len(parent1_txs), len(parent2_txs)))
        
        # Tạo cá thể con bằng cách kết hợp các phần từ cha mẹ
        child_txs = parent1_txs[:cut_point] + parent2_txs[cut_point:]
        
        # Giới hạn số lượng giao dịch
        max_txs = settings.MAX_INDIVIDUAL_LENGTH - len(constructor)
        if len(child_txs) > max_txs:
            child_txs = child_txs[:max_txs]
        
        # Tạo cá thể hoàn chỉnh
        child = constructor + child_txs
        
        self.logger.info(f"Created new individual via crossover with {len(child)} transactions")
        return child

    def mutate(self, individual):
        """Đột biến một cá thể để tạo biến thể mới với độ đa dạng cao hơn"""
        # Tạo một file log riêng cho quá trình đột biến
        log_dir = os.path.join(os.getcwd(), "fuzzing_logs")
        os.makedirs(log_dir, exist_ok=True)
        mutation_log_file = os.path.join(log_dir, "mutation_statistics.txt")
        
        # Khởi tạo hoặc cập nhật biến theo dõi số lượng đột biến tổng cộng
        if not hasattr(self, '_mutation_stats_total'):
            self._mutation_stats_total = {
                "replace": 0,
                "insert": 0,
                "remove": 0,
                "swap": 0,
                "param_mutate": 0,
                "total_mutations": 0,
                "successful_mutations": 0
            }
        
        # Khởi tạo hoặc cập nhật số lượng cá thể đã đột biến
        if not hasattr(self, '_mutated_individuals_count'):
            self._mutated_individuals_count = 0
        self._mutated_individuals_count += 1
        
        # Giữ nguyên constructor
        constructor = individual[:1] if len(individual) > 0 else []
        transactions = individual[1:] if len(individual) > 1 else []
        
        if not transactions:
            # Ghi log khi không có transaction để đột biến
            with open(mutation_log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] MUTATION STOPPED: No transactions to mutate. Creating random individual instead.\n")
            self.logger.warning("Mutation stopped: No transactions to mutate. Creating random individual instead.")
            return self.generate_random_individual()
        
        # Ghi log bắt đầu quá trình đột biến
        original_tx_count = len(transactions)
        with open(mutation_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] MUTATION STARTED: Original sequence has {original_tx_count} transactions\n")
        
        # Lưu fitness trước khi đột biến nếu có
        original_fitness = None
        original_coverage = None
        if hasattr(self, 'population'):
            for item in self.population:
                if isinstance(item, dict) and "individual" in item and item["individual"] == individual:
                    original_fitness = item.get("fitness", None)
                    original_coverage = item.get("coverage", None)
                    break
        
        # Chọn ngẫu nhiên một số lượng giao dịch để đột biến
        # Tăng cường đột biến nhiều transaction hơn cho độ đa dạng cao
        max_possible_mutations = max(2, len(transactions) // 2)
        num_mutations = random.randint(1, max_possible_mutations)
        
        # Cập nhật số lượng đột biến dự kiến trong thống kê tổng thể
        self._mutation_stats_total["total_mutations"] += num_mutations
        
        # Ghi log số lượng đột biến sẽ thực hiện
        with open(mutation_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Planning to apply {num_mutations} mutations out of maximum {max_possible_mutations}\n")
        
        self.logger.info(f"Mutation process: Planning to apply {num_mutations} mutations on sequence of {len(transactions)} transactions")
        
        # Khởi tạo thống kê đột biến
        mutation_stats = {
            "replace": 0,
            "insert": 0,
            "remove": 0, 
            "swap": 0,
            "param_mutate": 0
        }
        
        # Theo dõi các đột biến đã thực hiện để debug
        mutations_applied = []
        
        # Đếm số lần đột biến đã thực hiện
        mutations_performed = 0
        
        for i in range(num_mutations):
            # Tùy chỉnh phân phối của các loại đột biến 
            # Ưu tiên "replace" và "swap" nhiều hơn để tăng độ đa dạng
            mutation_weights = [40, 20, 15, 25]  # Tỷ lệ % cho replace, insert, remove, swap
            mutation_type = random.choices(
                ["replace", "insert", "remove", "swap"],
                weights=mutation_weights
            )[0]
            
            mutation_success = False
            
            if mutation_type == "replace" and transactions:
                # Thay thế một giao dịch bằng giao dịch mới
                idx = random.randint(0, len(transactions) - 1)
                
                # Lấy danh sách hàm hiện có để tránh lặp lại
                existing_functions = set()
                for tx in transactions:
                    if "arguments" in tx and tx["arguments"]:
                        existing_functions.add(tx["arguments"][0])
                
                # Ưu tiên chọn một hàm mới chưa có trong sequence
                available_functions = [f for f in self.interface.keys() if f not in existing_functions]
                
                if available_functions and random.random() < 0.8:  # 80% thời gian chọn hàm mới
                    function = random.choice(available_functions)
                    argument_types = self.interface[function]
                else:
                    function, argument_types = self.get_random_function_with_argument_types()
                
                new_tx = self.generate_individual(function, argument_types)
                if new_tx:
                    old_func = transactions[idx].get("arguments", ["unknown"])[0]
                    transactions[idx] = new_tx[0]
                    mutations_applied.append(f"Replaced tx {idx+1}: {old_func[:8]} -> {function[:8]}")
                    mutation_stats["replace"] += 1
                    self._mutation_stats_total["replace"] += 1
                    mutation_success = True
            
            elif mutation_type == "insert" and len(transactions) < settings.MAX_INDIVIDUAL_LENGTH - 1:
                # Chèn một giao dịch mới
                # Ưu tiên chèn hàm mới chưa có trong sequence
                existing_functions = set()
                for tx in transactions:
                    if "arguments" in tx and tx["arguments"]:
                        existing_functions.add(tx["arguments"][0])
                
                available_functions = [f for f in self.interface.keys() if f not in existing_functions]
                
                if available_functions and random.random() < 0.8:  # 80% thời gian chọn hàm mới
                    function = random.choice(available_functions)
                    argument_types = self.interface[function]
                else:
                    function, argument_types = self.get_random_function_with_argument_types()
                
                new_tx = self.generate_individual(function, argument_types)
                if new_tx:
                    idx = random.randint(0, len(transactions))
                    transactions.insert(idx, new_tx[0])
                    mutations_applied.append(f"Inserted tx at position {idx+1}: {function[:8]}")
                    mutation_stats["insert"] += 1
                    self._mutation_stats_total["insert"] += 1
                    mutation_success = True
            
            elif mutation_type == "remove" and len(transactions) > 1:
                # Xóa một giao dịch
                idx = random.randint(0, len(transactions) - 1)
                func_to_remove = transactions[idx].get("arguments", ["unknown"])[0]
                transactions.pop(idx)
                mutations_applied.append(f"Removed tx {idx+1}: {func_to_remove[:8]}")
                mutation_stats["remove"] += 1
                self._mutation_stats_total["remove"] += 1
                mutation_success = True
            
            elif mutation_type == "swap" and len(transactions) > 1:
                # Đổi vị trí hai giao dịch
                idx1 = random.randint(0, len(transactions) - 1)
                idx2 = random.randint(0, len(transactions) - 1)
                while idx1 == idx2:  # Đảm bảo hai vị trí khác nhau
                    idx2 = random.randint(0, len(transactions) - 1)
                    
                func1 = transactions[idx1].get("arguments", ["unknown"])[0]
                func2 = transactions[idx2].get("arguments", ["unknown"])[0]
                
                transactions[idx1], transactions[idx2] = transactions[idx2], transactions[idx1]
                mutations_applied.append(f"Swapped tx {idx1+1}:{func1[:8]} <-> tx {idx2+1}:{func2[:8]}")
                mutation_stats["swap"] += 1
                self._mutation_stats_total["swap"] += 1
                mutation_success = True
            
            # Thêm thuật toán đột biến tham số - đột biến giá trị của transaction
            elif mutation_type == "param_mutate" and transactions and random.random() < 0.3:
                idx = random.randint(0, len(transactions) - 1)
                if "arguments" in transactions[idx]:
                    func_hash = transactions[idx]["arguments"][0]
                    func_args_types = self.interface.get(func_hash, [])
                    
                    # Sinh lại transaction với tham số mới
                    new_tx = self.generate_individual(func_hash, func_args_types)
                    if new_tx:
                        transactions[idx] = new_tx[0]
                        mutations_applied.append(f"Mutated params of tx {idx+1}: {func_hash[:8]}")
                        mutation_stats["param_mutate"] += 1
                        self._mutation_stats_total["param_mutate"] += 1
                        mutation_success = True
            
            if mutation_success:
                mutations_performed += 1
                self._mutation_stats_total["successful_mutations"] += 1
            else:
                # Ghi log khi đột biến thất bại
                with open(mutation_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] Mutation attempt {i+1} failed: Type={mutation_type}\n")
            
            # Kiểm tra điều kiện dừng: nếu không còn transaction sau khi xóa
            if len(transactions) == 0:
                with open(mutation_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] MUTATION STOPPED: All transactions were removed\n")
                self.logger.warning("Mutation stopped: All transactions were removed")
                # Thêm lại ít nhất một transaction mới
                function, argument_types = self.get_random_function_with_argument_types()
                new_tx = self.generate_individual(function, argument_types)
                if new_tx:
                    transactions.extend(new_tx)
                break
        
        # Tạo cá thể mới sau đột biến
        mutated_individual = constructor + transactions
        
        # So sánh số lượng transaction sau khi đột biến
        final_tx_count = len(transactions)
        tx_change = final_tx_count - original_tx_count
        
        # Ghi log tổng kết đột biến
        with open(mutation_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] MUTATION COMPLETED:\n")
            f.write(f"  - Planned mutations: {num_mutations}\n")
            f.write(f"  - Mutations performed: {mutations_performed}\n")
            f.write(f"  - Transaction count: {original_tx_count} -> {final_tx_count} ({tx_change:+d})\n")
            f.write(f"  - Mutation statistics: {json.dumps(mutation_stats)}\n")
            f.write(f"  - Applied mutations: {'; '.join(mutations_applied)}\n")
            if original_fitness is not None:
                f.write(f"  - Original fitness: {original_fitness}\n")
            if original_coverage is not None:
                f.write(f"  - Original coverage: {original_coverage}\n")
            f.write("\n")
        
        # Log thông tin chi tiết về quá trình đột biến
        if mutations_applied:
            self.logger.info(f"Applied {len(mutations_applied)} mutations: {'; '.join(mutations_applied)}")
            self.logger.info(f"Mutation statistics: Replace={mutation_stats['replace']}, Insert={mutation_stats['insert']}, "
                            f"Remove={mutation_stats['remove']}, Swap={mutation_stats['swap']}, "
                            f"ParamMutate={mutation_stats['param_mutate']}")
            self.logger.info(f"Transaction count changed from {original_tx_count} to {final_tx_count} ({tx_change:+d})")
        else:
            self.logger.warning("No mutations were applied")
        
        return mutated_individual


    def log_mutation_stats(self, generation_number):
        """
        Ghi log thống kê về các đột biến đã thực hiện trong mỗi thế hệ
        """
        # Tạo đường dẫn để lưu log
        log_dir = os.path.join(os.getcwd(), "fuzzing_logs")
        os.makedirs(log_dir, exist_ok=True)
        mutation_stats_file = os.path.join(log_dir, "mutation_stats_by_generation.txt")
        
        # Thu thập dữ liệu thống kê nếu có
        mutation_stats = getattr(self, '_mutation_stats_total', {
            "replace": 0,
            "insert": 0,
            "remove": 0,
            "swap": 0,
            "param_mutate": 0,
            "total_mutations": 0,
            "successful_mutations": 0
        })
        
        # Đếm số cá thể đã đột biến
        mutated_individuals = getattr(self, '_mutated_individuals_count', 0)
        
        # Ghi thống kê vào file
        with open(mutation_stats_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] GENERATION {generation_number} MUTATION STATISTICS:\n")
            f.write(f"  - Total individuals mutated: {mutated_individuals}\n")
            f.write(f"  - Total mutations attempted: {mutation_stats['total_mutations']}\n")
            f.write(f"  - Successful mutations: {mutation_stats['successful_mutations']} ({(mutation_stats['successful_mutations']/mutation_stats['total_mutations']*100) if mutation_stats['total_mutations'] > 0 else 0:.2f}%)\n")
            f.write(f"  - Replace operations: {mutation_stats['replace']}\n")
            f.write(f"  - Insert operations: {mutation_stats['insert']}\n")
            f.write(f"  - Remove operations: {mutation_stats['remove']}\n")
            f.write(f"  - Swap operations: {mutation_stats['swap']}\n")
            f.write(f"  - Parameter mutations: {mutation_stats['param_mutate']}\n")
            f.write("\n")
        
        self.logger.info(f"Generation {generation_number} mutation stats: {mutation_stats['successful_mutations']}/{mutation_stats['total_mutations']} successful mutations")
        return mutation_stats


    def generate_constructor(self):
        """
        Tạo constructor cho contract hiện tại trong sequence
        """
        if not self.interface or 'constructor' not in self.interface:
            return []

        # Sử dụng cách triển khai từ lớp gốc 
        return super().generate_constructor()

    def finalize_fuzzing(self, *args, **kwargs) -> None:
        """Cleanup RAG resources (accepts extra args to match caller signature)."""
        self.logger.debug("RAGEnhancedGenerator finalize_fuzzing called")
        # SimpleRAG doesn't require explicit cleanup
        self._rag_instance = None
        self._rag_initialized = False

    def _normalize_sequence(self, seq: Any) -> List[str]:
        """
        Normalize sequence from various formats to a list of function names.
        
        Handles formats:
        - test_sequences: {"description": "...", "steps": ["func1", "func2"]}
        - critical_paths: {"description": "...", "target": "func1 -> func2"}
        - Direct list: ["func1", "func2"]
        - String: "func1 -> func2" or single function name
        """
        if seq is None:
            return []
        
        if isinstance(seq, dict):
            # Priority 1: Check for "steps" key (test_sequences format)
            if "steps" in seq:
                steps = seq.get("steps")
                if isinstance(steps, (list, tuple)):
                    return [item for item in steps if isinstance(item, str)]
            
            # Priority 2: Check for "target" key (critical_paths format)
            if "target" in seq:
                target = seq.get("target")
                if isinstance(target, str):
                    # Parse "func1 -> func2" or "func1 -> func2 -> func3" format
                    if "->" in target:
                        # Split by "->" and clean up function names
                        funcs = [f.strip() for f in target.split("->")]
                        # Filter out empty strings and comments
                        funcs = [f for f in funcs if f and not f.startswith("(") and not f.startswith("/*")]
                        if funcs:
                            return funcs
                    else:
                        # Single function name
                        return [target.strip()] if target.strip() else []
            
            # Priority 3: Check for other common keys
            for key in ("sequence", "functions", "path"):
                value = seq.get(key)
                if isinstance(value, (list, tuple)):
                    return [item for item in value if isinstance(item, str)]
                elif isinstance(value, str):
                    # Try to parse string format
                    if "->" in value:
                        funcs = [f.strip() for f in value.split("->")]
                        funcs = [f for f in funcs if f and not f.startswith("(") and not f.startswith("/*")]
                        if funcs:
                            return funcs
                    return [value.strip()] if value.strip() else []
        
        if isinstance(seq, str):
            # Parse string format "func1 -> func2"
            if "->" in seq:
                funcs = [f.strip() for f in seq.split("->")]
                funcs = [f for f in funcs if f and not f.startswith("(") and not f.startswith("/*")]
                return funcs if funcs else []
            return [seq.strip()] if seq.strip() else []
        
        if isinstance(seq, (list, tuple)):
            return [item for item in seq if isinstance(item, str)]
        
        return []

    def _sample_function_from_sequence(self, seq: Any) -> Optional[str]:
        normalized = self._normalize_sequence(seq)
        return random.choice(normalized) if normalized else None

def create_rag_enhanced_generator(
        interface: Dict, 
        bytecode: str,
        accounts: List[str],
        contract: str,
        api_key: str,
        analysis_result: Optional[Dict] = None,
        contract_name: Optional[str] = None,
        sol_path: Optional[str] = None,
        other_generators=None,
        interface_mapper=None,
        max_individual_length:int = 10,
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        disable_rag_llm: bool = False,
        adaptive_llm_controller=None) -> RAGEnhancedGenerator:
    """
    Factory function to create RAGEnhancedGenerator instance.
    """
    return RAGEnhancedGenerator(
        interface=interface,
        bytecode=bytecode,
        accounts=accounts,
        contract=contract,
        api_key=api_key,
        analysis_result=analysis_result,
        contract_name=contract_name,
        sol_path=sol_path,
        other_generators=other_generators,
        interface_mapper=interface_mapper,
        max_individual_length=max_individual_length,
        llm_model=llm_model,
        llm_provider=llm_provider,
        openai_api_key=openai_api_key,
        disable_rag_llm=disable_rag_llm,
        adaptive_llm_controller=adaptive_llm_controller,
    )