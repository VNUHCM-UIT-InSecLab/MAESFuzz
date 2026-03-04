#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import re
import traceback
from typing import Dict, List, Any, Optional

from fuzzer.utils.utils import initialize_logger
from provider import context as provider_context
from provider.base import ProviderError

class SmartContractAnalyzer:
    
    """
    Phân tích smart contract để xác định dataflow và mối quan hệ phụ thuộc giữa các hàm
    """
    def __init__(self, sol_path: str, api_key: Optional[str], solc_path: Optional[str] = None):
        self.sol_path = sol_path
        self.api_key = api_key
        self.solc_path = solc_path
        self.analysis_result = None
        self.dataflow_graph = None
        self.logger = initialize_logger("DataflowAnalyzer")
        self.provider = provider_context.get_provider(optional=True)
        
    def analyze(self):
        """Phân tích smart contract bằng Slither và LLM"""
        try:
            self.logger.info(f"Analyzing smart contract: {self.sol_path}")
            
            # Phân tích với Slither
            try:
                from slither.slither import Slither
                
                # Sử dụng solc_path nếu được cung cấp
                slither_args = {"solc": self.solc_path} if self.solc_path else {}
                slither = Slither(self.sol_path, **slither_args)
                
                # Xây dựng đồ thị phụ thuộc dữ liệu
                self.dataflow_graph = {}
                
                # Phân tích từng hợp đồng
                for contract in slither.contracts:
                    contract_info = {
                        "name": contract.name,
                        "functions": {},
                        "state_variables": [],
                        "inheritance": [base.name for base in contract.inheritance]
                    }
                    
                    # Thu thập biến trạng thái
                    for var in contract.state_variables:
                        var_info = {
                            "name": var.name,
                            "type": str(var.type),
                            "visibility": var.visibility
                        }
                        contract_info["state_variables"].append(var_info)
                    
                    # Thu thập thông tin các hàm
                    for func in contract.functions:
                        if func.visibility in ["public", "external"]:
                            # Bỏ qua hàm constructor vì được xử lý riêng
                            if func.name != "constructor":
                                function_info = {
                                    "name": func.name,
                                    "visibility": func.visibility,
                                    "state_mutability": getattr(func, "state_mutability", "nonpayable"),
                                    "reads": [v.name for v in func.state_variables_read],
                                    "writes": [v.name for v in func.state_variables_written],
                                    "calls": [],  # Sẽ được cập nhật bên dưới
                                    "parameters": [
                                        {"name": p.name, "type": str(p.type)}
                                        for p in func.parameters
                                    ]
                                }
                                
                                # Thu thập các lời gọi hàm nội bộ
                                for call in func.internal_calls:
                                    if hasattr(call, 'name'):
                                        function_info["calls"].append(call.name)
                                        
                                contract_info["functions"][func.name] = function_info
                    
                    self.dataflow_graph[contract.name] = contract_info
                
                # Chuyển đổi thành JSON để gửi đến LLM
                
                self.logger.info(f"Dataflow graph built successfully for {len(self.dataflow_graph)} contracts")
            except ImportError:
                self.logger.error("Không thể import Slither. Có thể thư viện này chưa được cài đặt.")
                self.logger.info("Tạo dataflow graph sơ bộ dựa trên đọc file...")
                
                # Đọc file Solidity và phân tích sơ bộ
                import re
                with open(self.sol_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Phân tích contracts
                contract_pattern = r'contract\s+(\w+)(?:\s+is\s+([^{]+))?\s*{([^}]+)}'
                contracts = re.findall(contract_pattern, code)
                
                self.dataflow_graph = {}
                
                for contract_match in contracts:
                    contract_name = contract_match[0].strip()
                    inheritance = [base.strip() for base in contract_match[1].split(',')] if contract_match[1] else []
                    contract_body = contract_match[2]
                    
                    # Phân tích state variables
                    var_pattern = r'(\w+(?:\[\])?\s+(?:public|private|internal)?\s+(\w+))'
                    state_vars = re.findall(var_pattern, contract_body)
                    
                    # Phân tích functions
                    func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*(public|external|internal|private)?\s*(view|pure|payable)?\s*(?:returns\s*\(([^)]*)\))?\s*{([^}]*)}'
                    functions = re.findall(func_pattern, contract_body)
                    
                    contract_info = {
                        "name": contract_name,
                        "functions": {},
                        "state_variables": [],
                        "inheritance": inheritance
                    }
                    
                    # Thêm state variables
                    for var in state_vars:
                        var_info = {
                            "name": var[1],
                            "type": var[0],
                            "visibility": "unknown"
                        }
                        contract_info["state_variables"].append(var_info)
                    
                    # Thêm functions
                    for func in functions:
                        func_name = func[0]
                        params_str = func[1]
                        visibility = func[2] if func[2] else "public"
                        mutability = func[3] if func[3] else "nonpayable"
                        
                        if visibility in ["public", "external"]:
                            params = []
                            if params_str:
                                param_list = params_str.split(',')
                                for p in param_list:
                                    parts = p.strip().split()
                                    if len(parts) >= 2:
                                        params.append({"name": parts[1], "type": parts[0]})
                            
                            function_info = {
                                "name": func_name,
                                "visibility": visibility,
                                "state_mutability": mutability,
                                "reads": [],
                                "writes": [],
                                "calls": [],
                                "parameters": params
                            }
                            
                            contract_info["functions"][func_name] = function_info
                    
                    self.dataflow_graph[contract_name] = contract_info
                
                # Chuyển đổi thành JSON để gửi đến LLM
                
                self.logger.info(f"Basic dataflow graph built from source code for {len(self.dataflow_graph)} contracts")
            
            # Gửi đến LLM để phân tích sâu hơn
            simplified_dataflow, main_contract_name = self._build_simplified_dataflow()
            
            # Log simplified dataflow for debugging
            if simplified_dataflow:
                self.logger.debug("Simplified dataflow for %s: %s", 
                                main_contract_name, 
                                json.dumps(simplified_dataflow, indent=2))
            else:
                self.logger.warning("Simplified dataflow is empty for contract %s", main_contract_name)
            
            self.analysis_result = self._run_llm_analysis(simplified_dataflow, main_contract_name)
                
            # Log critical paths and test sequences for visibility
            if self.analysis_result:
                cp = self.analysis_result.get("critical_paths", [])
                ts = self.analysis_result.get("test_sequences", [])
                vulns = self.analysis_result.get("vulnerabilities", [])
                self.logger.info("Critical paths (%d): %s", len(cp), cp[:5])
                self.logger.info("Test sequences (%d): %s", len(ts), ts[:5])
                if vulns:
                    self.logger.info("Vulnerabilities (%d): %s", len(vulns), vulns[:5])

            try:
                with open("dataflow_analysis_result.json", "w") as f:
                    json.dump({
                        "dataflow_graph": simplified_dataflow,
                        "analysis_result": self.analysis_result
                    }, f, indent=2)
            except Exception as dump_error:
                self.logger.debug(f"Failed to persist dataflow analysis result: {dump_error}")
            
            return {
                "dataflow_graph": self.dataflow_graph,
                "analysis_result": self.analysis_result
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing smart contract: {str(e)}")
            # Trả về kết quả trống nếu có lỗi
            return {
                "dataflow_graph": {},
                "analysis_result": self._create_default_analysis()
            }
    
    def _create_default_analysis(self):
        """Tạo phân tích mặc định dựa trên tên hàm"""
        try:
            default_analysis = {
                "critical_paths": [],
                "test_sequences": [],
                "vulnerabilities": []
            }
            
            # Tìm các hàm từ dataflow graph
            functions = []
            if self.dataflow_graph:
                main_contract_name = os.path.basename(self.sol_path).split('.')[0]
                if main_contract_name in self.dataflow_graph:
                    functions = list(self.dataflow_graph[main_contract_name].get("functions", {}).keys())
            
            # Phân loại hàm dựa trên tên
            write_funcs = []
            read_funcs = []
            
            for func_name in functions:
                if func_name.lower().startswith(("set", "add", "create", "update", "delete", "remove", "transfer", "mint", "burn")):
                    write_funcs.append(func_name)
                elif func_name.lower().startswith(("get", "view", "is", "has", "balance", "total", "name", "symbol", "decimals")):
                    read_funcs.append(func_name)
            
            # Tạo test sequences đơn giản
            if write_funcs and read_funcs:
                # Tạo 2-3 sequences
                for i in range(min(3, len(write_funcs))):
                    write_func = write_funcs[i % len(write_funcs)]
                    read_func = read_funcs[i % len(read_funcs)]
                    default_analysis["test_sequences"].append([write_func, read_func])
                    default_analysis["critical_paths"].append([write_func, read_func])
            
            # Tạo các vulnerabilities đơn giản
            if "transfer" in functions or "transferFrom" in functions:
                default_analysis["vulnerabilities"].append({
                    "type": "reentrancy",
                    "functions": ["transfer"] if "transfer" in functions else ["transferFrom"]
                })
            
            if "approve" in functions:
                default_analysis["vulnerabilities"].append({
                    "type": "front-running",
                    "functions": ["approve"]
                })
            
            return default_analysis
        except Exception as e:
            self.logger.error(f"Error creating default analysis: {str(e)}")
            return {
                "critical_paths": [],
                "test_sequences": [],
                "vulnerabilities": []
            } 

    def _refresh_provider(self):
        self.provider = provider_context.get_provider(optional=True)
        return self.provider

    def _build_simplified_dataflow(self):
        simplified = {}
        
        # Try to get contract name from dataflow_graph first
        # If not found, try filename as fallback
        main_contract_name = None
        if self.dataflow_graph:
            # Get the first contract (usually the main one)
            contract_names = list(self.dataflow_graph.keys())
            if contract_names:
                main_contract_name = contract_names[0]
                self.logger.debug("Using contract name from dataflow_graph: %s", main_contract_name)
        
        # Fallback to filename if no contract found
        if not main_contract_name:
            main_contract_name = os.path.basename(self.sol_path).split('.')[0]
            self.logger.debug("Using contract name from filename: %s", main_contract_name)

        if self.dataflow_graph and main_contract_name in self.dataflow_graph:
            contract_info = self.dataflow_graph[main_contract_name]
            all_functions = contract_info.get("functions", {})
            
            self.logger.debug("Found %d functions in contract %s: %s", 
                            len(all_functions), main_contract_name, list(all_functions.keys()))
            
            # Include all functions, not just limited subset
            # Prioritize write functions, but include all for comprehensive analysis
            write_funcs = {
                name: info for name, info in all_functions.items() if info.get("writes")
            }
            read_funcs = {
                name: info for name, info in all_functions.items()
                if info.get("reads") and name not in write_funcs
            }
            other_funcs = {
                name: info for name, info in all_functions.items()
                if name not in write_funcs and name not in read_funcs
            }
            
            # Include all functions for comprehensive analysis
            functions = {}
            # Add write functions first (up to 5)
            for name, info in list(write_funcs.items())[:5]:
                functions[name] = info
            # Add read functions (up to 5)
            for name, info in list(read_funcs.items())[:5]:
                functions.setdefault(name, info)
            # Add other functions (up to 3)
            for name, info in list(other_funcs.items())[:3]:
                functions.setdefault(name, info)
            
            self.logger.debug("Included %d functions in simplified dataflow: %s", 
                            len(functions), list(functions.keys()))

            simplified[main_contract_name] = {
                "name": main_contract_name,
                "functions": functions,
                "state_variables": contract_info.get("state_variables", [])
            }
        else:
            self.logger.warning("Contract %s not found in dataflow_graph. Available contracts: %s", 
                              main_contract_name, list(self.dataflow_graph.keys()) if self.dataflow_graph else [])

        return simplified, main_contract_name

    def _parse_llm_json(self, text: str):
        text = text.strip()
        if not text:
            raise ValueError("LLM returned empty response")

        # First, try direct JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown fences if present
        if text.startswith("```"):
            fence_end = text.find("```", 3)
            if fence_end != -1:
                inner = text[3:fence_end].strip()
                if inner.lower().startswith("json"):
                    inner = inner[4:].strip()
                text = inner

        # Extract first JSON object by brace depth
        def _extract_first_json_object(s: str) -> str:
            start = s.find("{")
            if start == -1:
                return ""
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(s)):
                ch = s[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
            return ""

        candidate = _extract_first_json_object(text)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Last fallback: greedy regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError("LLM response does not contain a valid JSON object")

    def _run_llm_analysis(self, simplified_dataflow: Dict[str, Any], main_contract_name: str) -> Dict[str, Any]:
        provider = self._refresh_provider()
        if provider is None:
            self.logger.warning("No LLM provider available, using default analysis")
            return self._create_default_analysis()

        simplified_json = json.dumps(simplified_dataflow, indent=2)
        
        # Build function list for context
        all_function_names = []
        if simplified_dataflow and main_contract_name in simplified_dataflow:
            all_function_names = list(simplified_dataflow[main_contract_name].get("functions", {}).keys())
        
        minimal_prompt = (
            "You are a smart contract security analyst. Analyze the following Solidity contract's dataflow and suggest fuzzing targets.\n\n"
            "Contract Information:\n"
            f"- Contract name: {main_contract_name}\n"
            f"- Available functions: {', '.join(all_function_names) if all_function_names else 'None'}\n\n"
            "Dataflow Analysis (JSON):\n"
            f"{simplified_json}\n\n"
            "Your Task:\n"
            "Analyze the contract and identify:\n"
            "1. Critical execution paths (sequences of functions that should be tested together)\n"
            "2. Test sequences (step-by-step function call sequences for fuzzing)\n"
            "3. Potential vulnerabilities (security issues to look for)\n\n"
            "Output Format:\n"
            "Return ONLY a valid JSON object with these exact keys:\n"
            "{\n"
            "  \"critical_paths\": [\n"
            "    {\"description\": \"Brief description of the path\", \"target\": \"Function name or path identifier\"}\n"
            "  ],\n"
            "  \"test_sequences\": [\n"
            "    {\"description\": \"What this sequence tests\", \"steps\": [\"function1\", \"function2\", ...]}\n"
            "  ],\n"
            "  \"vulnerabilities\": [\n"
            "    {\"type\": \"vulnerability type (e.g., reentrancy, access-control, integer-overflow)\", \"description\": \"Brief description\"}\n"
            "  ]\n"
            "}\n\n"
            "Guidelines:\n"
            "- For CREATE2 contracts: focus on deployment sequences and address computation\n"
            "- For state-modifying functions: create sequences that test state transitions\n"
            "- For view functions: include them in sequences after state changes\n"
            "- Look for access control issues, reentrancy risks, and state inconsistencies\n"
            "- If a function writes to storage, create test sequences that read that storage\n"
            "- If unsure about a specific item, still include it with a reasonable description\n\n"
            "Example for a CREATE2 factory contract:\n"
            "{\n"
            "  \"critical_paths\": [\n"
            "    {\"description\": \"Deploy contract via CREATE2\", \"target\": \"safeCreate2\"}\n"
            "  ],\n"
            "  \"test_sequences\": [\n"
            "    {\"description\": \"Compute address then deploy\", \"steps\": [\"findCreate2Address\", \"safeCreate2\"]},\n"
            "    {\"description\": \"Check deployment status\", \"steps\": [\"safeCreate2\", \"hasBeenDeployed\"]}\n"
            "  ],\n"
            "  \"vulnerabilities\": [\n"
            "    {\"type\": \"access-control\", \"description\": \"Salt validation in containsCaller modifier\"}\n"
            "  ]\n"
            "}\n\n"
            "CRITICAL: Return ONLY the JSON object. No markdown, no backticks, no explanations before or after.\n"
            "The response MUST start with '{' and end with '}'."
        )

        # Print prompt to terminal for visibility
        print(f"\n{'='*60}")
        print("[DATAFLOW ANALYZER] Sending prompt to LLM for analysis")
        print(f"{'='*60}")
        preview_prompt = (minimal_prompt[:800] + "\n... (truncated)") if len(minimal_prompt) > 800 else minimal_prompt
        print(f"[DATAFLOW][PROMPT]\n{preview_prompt}")
        import sys
        sys.stdout.flush()

        try:
            response = provider.generate(minimal_prompt)
            
            # Print response preview
            preview_resp = (response.text[:800] + "\n... (truncated)") if len(response.text) > 800 else response.text
            print(f"[DATAFLOW][RESPONSE]\n{preview_resp}")
            print(f"{'='*60}\n")
            sys.stdout.flush()
            
            # Log full response for debugging
            self.logger.debug("Full LLM response: %s", response.text)
            
            parsed = self._parse_llm_json(response.text)
            if not isinstance(parsed, dict):
                raise ValueError("LLM response is not a JSON object")
            
            # Log parsed result for debugging
            self.logger.debug("Parsed LLM result: %s", json.dumps(parsed, indent=2))

            self.logger.info(
                "LLM analysis complete. Found %d critical paths, %d test sequences, and %d potential vulnerabilities",
                len(parsed.get("critical_paths", [])),
                len(parsed.get("test_sequences", [])),
                len(parsed.get("vulnerabilities", [])),
            )
            
            # Log detailed results
            if parsed.get("critical_paths"):
                self.logger.info("Critical paths: %s", parsed.get("critical_paths"))
            if parsed.get("test_sequences"):
                self.logger.info("Test sequences: %s", parsed.get("test_sequences"))
            if parsed.get("vulnerabilities"):
                self.logger.info("Vulnerabilities: %s", parsed.get("vulnerabilities"))
            
            return parsed
        except (ProviderError, json.JSONDecodeError, ValueError) as exc:
            self.logger.warning(f"LLM analysis failed: {exc}")
            self.logger.debug("LLM analysis error details: %s", traceback.format_exc())
            return self._create_default_analysis() 