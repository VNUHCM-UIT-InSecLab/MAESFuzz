#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script để fuzz với whole_compile_info từ file
Sử dụng cho cross-contract fuzzing với bytecode
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Setup path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from main import Fuzzer, launch_argument_parser
from evm import InstrumentedEVM
from z3 import Solver
from utils import settings
from eth_utils import encode_hex, to_canonical_address

def main():
    parser = argparse.ArgumentParser(
        description="Fuzz với whole_compile_info từ file (cho cross-contract với bytecode)"
    )
    parser.add_argument("--whole-compile-info", type=str, required=True,
                       help="File JSON chứa whole_compile_info")
    parser.add_argument("--abi", type=str, required=True, help="ABI file")
    parser.add_argument("--contract", type=str, required=True, help="Contract address")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--depend-contracts", type=str, nargs="*", default=[],
                       help="Dependent contract names")
    parser.add_argument("--cross-contract", type=int, default=1)
    parser.add_argument("--open-trans-comp", type=int, default=1)
    parser.add_argument("--p-open-cross", type=int, default=80)
    parser.add_argument("--cross-init-mode", type=int, default=1)
    parser.add_argument("--trans-mode", type=int, default=1)
    parser.add_argument("--max-individual-length", type=int, default=10)
    parser.add_argument("--use-rag", action="store_true")
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--provider", type=str)
    
    args = parser.parse_args()
    
    # Load whole_compile_info
    with open(args.whole_compile_info, 'r') as f:
        whole_compile_info = json.load(f)
    
    # Load ABI
    with open(args.abi, 'r') as f:
        abi = json.load(f)
    
    # Setup EVM
    evm = InstrumentedEVM(settings.RPC_HOST, settings.RPC_PORT)
    evm.set_vm_by_name(settings.EVM_VERSION)
    evm.create_fake_accounts()
    
    # Get runtime bytecode
    runtime_bytecode = evm.get_code(to_canonical_address(args.contract)).hex()
    
    # Setup solver
    solver = Solver()
    solver.set("timeout", 30000)
    
    # Create mock args object
    class MockArgs:
        def __init__(self):
            self.source = None  # Bytecode mode
            self.depend_contracts = args.depend_contracts
            self.cross_contract = args.cross_contract
            self.open_trans_comp = args.open_trans_comp
            self.p_open_cross = args.p_open_cross
            self.cross_init_mode = args.cross_init_mode
            self.trans_mode = args.trans_mode
            self.results = args.results
            self.use_rag = args.use_rag
            self.api_key = args.api_key
            self.llm_model = args.model
            self.llm_provider = args.provider
            self.max_individual_length = args.max_individual_length
            self.global_timeout = args.timeout
            self.population_size = None
            self.generations = None
            self.probability_crossover = None
            self.probability_mutation = None
            self.trans_json_path = None
            self.constructor_args = None
    
    mock_args = MockArgs()
    
    # Get contract name từ whole_compile_info hoặc từ ABI
    contract_name = "Contract"  # Default
    if whole_compile_info:
        contract_name = list(whole_compile_info.keys())[0] if whole_compile_info else "Contract"
    
    # Create and run fuzzer
    fuzzer = Fuzzer(
        contract_name=contract_name,
        abi=abi,
        deployment_bytecode='',
        runtime_bytecode=runtime_bytecode,
        test_instrumented_evm=evm,
        blockchain_state=[],
        solver=solver,
        args=mock_args,
        seed=0.5,
        whole_compile_info=whole_compile_info
    )
    
    fuzzer.run()

if __name__ == "__main__":
    main()

