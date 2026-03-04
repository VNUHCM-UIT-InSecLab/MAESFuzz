#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Test Script - Chạy từ thư mục fuzzer/

Usage:
    cd fuzzer
    python manual_test.py
"""

import sys
import os

# Setup path giống như fuzzer/main.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from evm import InstrumentedEVM
from utils import settings
from utils.utils import initialize_logger
from eth_utils import encode_hex, keccak
from eth_abi import encode
from solcx import compile_source, set_solc_version
import solcx

logger = initialize_logger("ManualTest")

def test_transaction():
    logger.info("="*60)
    logger.info("Manual Transaction Test")
    logger.info("="*60)
    
    # Set solc version tương thích với byzantium
    try:
        set_solc_version('0.8.0')
        logger.info("Using solc 0.8.0")
    except:
        logger.warning("Could not set solc version, using default")
    
    # 1. Setup EVM
    logger.info("Setting up EVM...")
    settings.EVM_VERSION = "byzantium"
    evm = InstrumentedEVM()
    evm.set_vm_by_name("byzantium")
    evm.create_fake_accounts()
    logger.info(f"Created {len(evm.accounts)} accounts")
    
    # 2. Compile Contract
    logger.info("\nCompiling contract...")
    contract_source = """
pragma solidity ^0.8.0;
contract ABC {
    mapping(address => uint256) public balanceOf;
    
    constructor() {
        balanceOf[msg.sender] = 1000;
    }
    
    function transfer(address to, uint256 amount) public {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
    }
}
"""
    
    try:
        compiled = compile_source(contract_source)
        contract_interface = compiled['<stdin>:ABC']
        bytecode = contract_interface['bin']
        abi = contract_interface['abi']
        logger.info("✅ Contract compiled successfully")
    except Exception as e:
        logger.error(f"❌ Compilation failed: {e}")
        return 1
    
    # 3. Deploy Contract
    logger.info("\nDeploying contract...")
    result = evm.deploy_contract(
        creator=evm.accounts[0],
        bin_code=bytecode,
        amount=0,
        gas=settings.GAS_LIMIT
    )
    
    if result.is_error:
        logger.error(f"❌ Deploy failed: {result._error}")
        return 1
    
    contract_address = encode_hex(result.msg.storage_address)
    logger.info(f"✅ Contract deployed at: {contract_address}")
    
    # 4. Create Transaction Data
    logger.info("\nCreating transaction data...")
    function_name = "transfer"
    function_signature = "transfer(address,uint256)"
    function_hash = keccak(function_signature.encode())[:4].hex()
    
    to_address = "0x2222222222222222222222222222222222222222"
    amount = 100
    
    # Encode arguments
    to_bytes = bytes.fromhex(to_address[2:].zfill(40))
    args = encode(["address", "uint256"], [to_bytes, amount])
    data = f"0x{function_hash}{args.hex()}"
    
    logger.info(f"Function: {function_name}")
    logger.info(f"Function hash: 0x{function_hash}")
    logger.info(f"To: {to_address}")
    logger.info(f"Amount: {amount}")
    logger.info(f"Transaction data: {data}")
    
    # 5. Execute Transaction
    logger.info("\nExecuting transaction...")
    tx_input = {
        "transaction": {
            "from": evm.accounts[0],
            "to": contract_address,
            "value": 0,
            "gaslimit": settings.GAS_LIMIT,
            "data": data
        },
        "block": {},
        "global_state": {}
    }
    
    result = evm.deploy_transaction(tx_input)
    
    if result.is_error:
        logger.error(f"❌ Transaction failed: {result._error}")
        return 1
    
    logger.info("✅ Transaction executed successfully")
    logger.info(f"Gas used: {result.gas_used}")
    logger.info(f"Trace length: {len(result.trace)} instructions")
    
    # Show first 10 instructions
    logger.info("\nFirst 10 instructions:")
    for i, instruction in enumerate(result.trace[:10]):
        logger.info(f"  [{i}] PC={hex(instruction.get('pc', 0))} "
                    f"OP={instruction.get('op', 'UNKNOWN')}")
    
    logger.info("\n" + "="*60)
    logger.info("Test Complete")
    logger.info("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(test_transaction())

