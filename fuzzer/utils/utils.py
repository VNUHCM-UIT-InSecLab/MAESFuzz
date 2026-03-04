#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shlex
import solcx
import logging
import eth_utils
import subprocess

from solcx.install import _convert_and_validate_version
from web3 import Web3
from .settings import LOGGING_LEVEL


def initialize_logger(name):
    logger = logging.getLogger(name)
    logger.title = lambda *a: logger.info(*[bold(x) for x in a])
    logger_error = logger.error
    logger.error = lambda *a: logger_error(*[red(bold(x)) for x in a])
    logger_warning = logger.warning
    logger.warning = lambda *a: logger_warning(*[red(bold(x)) for x in a])
    logger.setLevel(level=LOGGING_LEVEL)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logger


def bold(x):
    return "".join(['\033[1m', x, '\033[0m']) if isinstance(x, str) else x


def red(x):
    return "".join(['\033[91m', x, '\033[0m']) if isinstance(x, str) else x


def code_bool(value: bool):
    return str(int(value)).zfill(64)


def code_uint(value):
    return hex(value).replace("0x", "").zfill(64)


def code_int(value):
    return hex(value).replace("0x", "").zfill(64)


def code_address(value):
    return value.zfill(64)


def code_bytes(value):
    return value.ljust(64, "0")


def code_type(value, type):
    if type == "bool":
        return code_bool(value)
    elif type.startswith("uint"):
        return code_uint(value)
    elif type.startswith("int"):
        return code_int(value)
    elif type == "address":
        return code_address(value)
    elif type.startswith("bytes"):
        return code_bytes(value)
    else:
        raise Exception()


def run_command(cmd):
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=FNULL)
    return p.communicate()[0]


def compile(solc_version, evm_version, source_code_file):
    out = None
    source_code = ""
    with open(source_code_file, 'r') as file:
        source_code = file.read()
    try:
        if not str(solc_version).startswith("v"):
            solc_version = "v" + str(solc_version)
        solc_version = _convert_and_validate_version(solc_version)
        if not solc_version in solcx.get_installed_solc_versions():
            solcx.install_solc(solc_version)
        solcx.set_solc_version(solc_version, True)
        out = solcx.compile_standard({
            'language': 'Solidity',
            'sources': {source_code_file: {'content': source_code}},
            'settings': {
                # "optimizer": {
                #     "enabled": True,
                #     "runs": 200
                # },
                "evmVersion": evm_version,
                "outputSelection": {
                    source_code_file: {
                        "*":
                            [
                                "abi",
                                "evm.deployedBytecode",
                                "evm.bytecode.object",
                                "evm.legacyAssembly",
                            ],
                    }
                }
            }
        }, allow_paths='.')
    except Exception as e:
        print("Error: Solidity compilation failed!")
        print(str(e))
    return out


def get_interface_from_abi(abi):
    interface = {}
    interface_mapper = {}  # 记录函数名和函数签名的映射
    for field in abi:
        if field['type'] == 'function':
            function_name = field['name']
            function_inputs = []
            signature = function_name + '('
            for i in range(len(field['inputs'])):
                input_type = field['inputs'][i]['type']
                function_inputs.append(input_type)
                signature += input_type
                if i < len(field['inputs']) - 1:
                    signature += ','
            signature += ')'
            hash = Web3.keccak(text=signature)[0:4].hex()
            interface[hash] = function_inputs
            interface_mapper[signature] = hash
        elif field['type'] == 'constructor':
            function_inputs = []
            for i in range(len(field['inputs'])):
                input_type = field['inputs'][i]['type']
                function_inputs.append(input_type)
            interface['constructor'] = function_inputs
    if not "fallback" in interface:
        interface["fallback"] = []
    return interface, interface_mapper


def get_function_signature_mapping(abi):
    mapping = {}
    for field in abi:
        if field['type'] == 'function':
            function_name = field['name']
            function_inputs = []
            signature = function_name + '('
            for i in range(len(field['inputs'])):
                input_type = field['inputs'][i]['type']
                signature += input_type
                if i < len(field['inputs']) - 1:
                    signature += ','
            signature += ')'
            hash = Web3.keccak(text=signature)[0:4].hex()
            mapping[hash] = signature
    if not "fallback" in mapping:
        mapping["fallback"] = "fallback"
    return mapping


def remove_swarm_hash(bytecode):
    if isinstance(bytecode, str):
        if bytecode.endswith("0029"):
            bytecode = re.sub(r"a165627a7a72305820\S{64}0029$", "", bytecode)
        if bytecode.endswith("0033"):
            bytecode = re.sub(r"5056fe.*?0033$", "5056", bytecode)
    return bytecode


def get_pcs_and_jumpis(bytecode):
    bytecode = bytes.fromhex(remove_swarm_hash(bytecode).replace("0x", ""))
    i = 0
    pcs = []
    jumpis = []
    while i < len(bytecode):
        opcode = bytecode[i]
        pcs.append(i)
        if opcode == 87:  # JUMPI
            jumpis.append(hex(i))
        if opcode >= 96 and opcode <= 127:  # PUSH
            size = opcode - 96 + 1
            i += size
        i += 1
    if len(pcs) == 0:
        pcs = [0]
    return (pcs, jumpis)


def convert_stack_value_to_int(stack_value):
    # Handle case where stack_value is already an int
    if isinstance(stack_value, int):
        return stack_value
    
    # Handle case where stack_value is not a list/tuple
    if not hasattr(stack_value, '__len__'):
        return 0
    
    if len(stack_value) < 2:
        return 0  # Default value if stack_value is malformed
    
    if stack_value[0] == int:
        return stack_value[1]
    elif stack_value[0] == bytes:
        return int.from_bytes(stack_value[1], "big")
    else:
        # Generic handler for any numeric type (0, 64, 169, etc.)
        # Assume the second element is the actual value
        try:
            return int(stack_value[1]) if len(stack_value) > 1 else 0
        except (ValueError, TypeError):
            return 0


def convert_stack_value_to_hex(stack_value):
    # Handle case where stack_value is already an int
    if isinstance(stack_value, int):
        return hex(stack_value).replace("0x", "").zfill(64)
    
    # Handle case where stack_value is not a list/tuple
    if not hasattr(stack_value, '__len__'):
        return "0" * 64
    
    if len(stack_value) < 2:
        return "0" * 64  # Default value if stack_value is malformed
    
    if stack_value[0] == int:
        return hex(stack_value[1]).replace("0x", "").zfill(64)
    elif stack_value[0] == bytes:
        return stack_value[1].hex().zfill(64)
    else:
        # Generic handler for any numeric type
        try:
            value = int(stack_value[1]) if len(stack_value) > 1 else 0
            return hex(value).replace("0x", "").zfill(64)
        except (ValueError, TypeError):
            return "0" * 64


def is_fixed(value):
    return isinstance(value, int)


def split_len(seq, length):
    return [seq[i:i + length] for i in range(0, len(seq), length)]


def print_individual_solution_as_transaction(logger, individual_solution, color="", function_signature_mapping={}, transaction_index=None):
    """
    Hiển thị các giao dịch trong một sequence đang được fuzzing theo định dạng dễ đọc
    """
    logger.info(f"\n===== Transaction sequence with {len(individual_solution)} transactions =====")
    
    for index, input in enumerate(individual_solution):
        transaction = input["transaction"]
        if transaction["to"] is None:
            continue
            
        # Lấy function hash hoặc tên hàm nếu có
        if transaction["data"].startswith("0x"):
            hash = transaction["data"][0:10]
        else:
            hash = transaction["data"][0:8]
        
        function_name = function_signature_mapping.get(hash, hash)
        
        # Hiển thị thông tin giao dịch theo định dạng dễ đọc
        logger.info(color + f"Transaction {index+1} - {function_name}")
        
        # Hiển thị các tham số nếu có
        if len(transaction["data"]) > 10 and hash != "constructor":
            # Nếu có data, hiển thị các tham số
            params = transaction["data"][10:]
            params_chunks = [params[i:i+64] for i in range(0, len(params), 64)]
            for i, param in enumerate(params_chunks):
                if param:
                    logger.info(color + f"    Param {i+1}: 0x{param}")
        
        # Hiển thị thông tin chi tiết về giao dịch
        logger.info(color + f"    From: {transaction['from']}")
        logger.info(color + f"    To: {transaction['to']}")
        logger.info(color + f"    Value: {transaction['value']} Wei")
        logger.info("")
        
        if transaction_index is not None and index + 1 > transaction_index:
            break


def normalize_32_byte_hex_address(value):
    as_bytes = eth_utils.to_bytes(hexstr=value)
    return eth_utils.to_normalized_address(as_bytes[-20:])
