#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import json
import psutil

from fuzzer.engine.environment import FuzzingEnvironment
from fuzzer.engine.plugin_interfaces import OnTheFlyAnalysis

from fuzzer.utils.utils import (
    initialize_logger,
    convert_stack_value_to_int,
    convert_stack_value_to_hex,
    normalize_32_byte_hex_address,
    get_function_signature_mapping,
    print_individual_solution_as_transaction,
)
from eth._utils.address import force_bytes_to_address
from eth_utils import to_hex, to_int, int_to_big_endian, encode_hex, ValidationError, to_canonical_address, \
    to_normalized_address

from z3 import simplify, BitVec, BitVecVal, Not, Optimize, sat, unsat, unknown, is_expr
from z3.z3util import get_vars

from fuzzer.utils import settings
import json


class ExecutionTraceAnalyzer(OnTheFlyAnalysis):
    def __init__(self, fuzzing_environment: FuzzingEnvironment) -> None:
        self.logger = initialize_logger("Analysis")
        self.env = fuzzing_environment
        self.symbolic_execution_count = 0

    def setup(self, ng, engine):
        pass

    def execute(self, population, engine):
        self.env.memoized_fitness.clear()
        self.env.memoized_storage.clear()
        self.env.memoized_symbolic_execution.clear()
        self.env.individual_branches.clear()

        executed_individuals = dict()
        for i, individual in enumerate(population.individuals):
            if individual.hash in executed_individuals:
                population.individuals[i] = executed_individuals[individual.hash]
                continue
            self.execution_function(individual, self.env)
            executed_individuals[individual.hash] = individual
        executed_individuals.clear()

        # Update statistic variables.
        engine._update_statvars()

    def register_step(self, g, population, engine):
        self.execute(population, engine)

        # Hiển thị thông tin về thế hệ hiện tại theo định dạng dễ đọc
        self.logger.info(f"\n===== Generation {g+1} =====")
        
        # Tính toán độ bao phủ
        code_coverage_percentage = 0
        if len(self.env.overall_pcs) > 0:
            code_coverage_percentage = (len(self.env.code_coverage) / len(self.env.overall_pcs)) * 100

        # Count unique branches visited (each JUMPI has 2 possible outcomes: 0 and 1)
        # Total branches = number of JUMPI instructions * 2 (each has true/false path)
        total_branches = len(self.env.overall_jumpis) * 2
        branch_coverage = 0
        for pc in self.env.visited_branches:
            branch_coverage += len(self.env.visited_branches[pc])
        branch_coverage_percentage = 0
        if total_branches > 0:
            # Ensure branch coverage doesn't exceed 100%
            branch_coverage = min(branch_coverage, total_branches)
            branch_coverage_percentage = (branch_coverage / total_branches) * 100

        # Thông tin về độ bao phủ
        self.logger.info(f"Code Coverage: {code_coverage_percentage:.2f}%")
        self.logger.info(f"Branch Coverage: {branch_coverage_percentage:.2f}%")
        self.logger.info(f"Total Transactions: {self.env.nr_of_transactions}")
        self.logger.info(f"Unique Transactions: {len(self.env.unique_individuals)}")
        
        # Save to results
        if "generations" not in self.env.results:
            self.env.results["generations"] = []

        self.env.results["generations"].append({
            "generation": g + 1,
            "time": time.time() - self.env.execution_begin,
            "total_transactions": self.env.nr_of_transactions,
            "unique_transactions": len(self.env.unique_individuals),
            "code_coverage": code_coverage_percentage,
            "branch_coverage": branch_coverage_percentage,
            "cross_transactions": settings.CROSS_TRANS_EXEC_COUNT
        })

        if len(self.env.code_coverage) == self.env.previous_code_coverage_length:  # 如果这一次覆盖率没有增加, 启动符号执行
            self.symbolic_execution(population.indv_generator, population.other_generators)
            if self.symbolic_execution_count == settings.MAX_SYMBOLIC_EXECUTION:
                del population.individuals[:]
                population.init(no_cross=True)
                self.logger.debug("Resetting population...")
                self.execute(population, engine)
                self.symbolic_execution_count = 0
            self.symbolic_execution_count += 1
        else:
            self.symbolic_execution_count = 0

        self.env.previous_code_coverage_length = len(self.env.code_coverage)

    def execution_function(self, indv, env: FuzzingEnvironment):
        env.unique_individuals.add(indv.hash)

        # Initialize metric
        branches = {}
        indv.data_dependencies = []
        contract_address = None

        env.detector_executor.initialize_detectors()

        for transaction_index, test in enumerate(indv.solution):

            this_error_cross_check = True

            transaction = test["transaction"]

            _function_hash = transaction["data"][:10] if transaction["data"].startswith("0x") else transaction["data"][
                                                                                                   :8]
            _function_hash = "fallback" if _function_hash == '' else _function_hash
            _array_size_indexes = dict()

            if transaction["to"] is None and contract_address is not None:
                transaction["to"] = contract_address

            if transaction["to"] is None:
                continue

            try:
                result = env.instrumented_evm.deploy_transaction(test)  # 执行事务
            except ValidationError as e:
                self.logger.error("Validation error in %s : %s (ignoring for now)", indv.hash, e)
                continue

            if not result.is_error and transaction["to"] == b'':
                contract_address = encode_hex(result.msg.storage_address)
                self.logger.debug("(%s - %d) Contract deployed at %s", indv.hash, transaction_index, contract_address)

            for child_computation in result.children:
                if child_computation.msg.to not in env.other_contracts:
                    continue
                if child_computation.msg.to not in env.children_code_coverage:
                    env.children_code_coverage[child_computation.msg.to] = set()
                env.children_code_coverage[child_computation.msg.to].update([x["pc"] for x in child_computation.trace])

            env.nr_of_transactions += 1

            # ContraMaster: Check Balance Invariant after transaction execution
            if hasattr(env.detector_executor, 'bookkeeping_variable') and env.detector_executor.bookkeeping_variable and contract_address:
                try:
                    import config
                    if config.is_semantic_oracle_enabled():
                        self.logger.debug(f"[ContraMaster] Checking balance invariant after transaction {transaction_index}...")
                        is_violated, violation_info = env.detector_executor.balance_invariant_detector.detect_balance_invariant_violation(
                            instrumented_evm=env.instrumented_evm,
                            contract_address=contract_address,
                            bookkeeping_var_name=env.detector_executor.bookkeeping_variable,
                            accounts=env.instrumented_evm.accounts
                        )

                        if is_violated and violation_info:
                            # Report balance invariant violation
                            color = env.detector_executor.get_color_for_severity('High')
                            self.logger.title(color + "-----------------------------------------------------")
                            self.logger.title(color + "   !!! Balance Invariant Violation detected !!!     ")
                            self.logger.title(color + "-----------------------------------------------------")
                            self.logger.title(color + "SWC-ID:   SEMANTIC-BALANCE")
                            self.logger.title(color + "Severity: High")
                            self.logger.title(color + "-----------------------------------------------------")
                            self.logger.title(color + f"Expected delta (K): {violation_info['expected_delta']} Wei")
                            self.logger.title(color + f"Actual delta:       {violation_info['actual_delta']} Wei")
                            self.logger.title(color + f"Contract balance:   {violation_info['contract_balance']} Wei")
                            self.logger.title(color + f"Sum(bookkeeping):   {violation_info['sum_bookkeeping']} Wei")
                            self.logger.title(color + "-----------------------------------------------------")
                            self.logger.title(color + "Transaction sequence:")
                            self.logger.title(color + "-----------------------------------------------------")
                            from fuzzer.utils.utils import print_individual_solution_as_transaction
                            print_individual_solution_as_transaction(self.logger, indv.solution, color, env.detector_executor.function_signature_mapping, transaction_index)

                            # Add to errors
                            env.results["errors"][f"balance_invariant_{transaction_index}"] = {
                                "swc_id": "SEMANTIC-BALANCE",
                                "severity": "High",
                                "type": "Balance Invariant Violation",
                                "individual": indv.solution,
                                "violation_info": violation_info
                            }
                except Exception as e:
                    self.logger.debug(f"Semantic oracle check failed: {e}")

            previous_instruction = None
            previous_branch = []
            previous_branch_expression = None
            previous_branch_address = None
            previous_call_address = None
            sha3 = {}

            for i, instruction in enumerate(result.trace):
                if settings.MAIN_CONTRACT_NAME != "" and settings.MAIN_CONTRACT_NAME in settings.TRANS_INFO and settings.TRANS_INFO[settings.MAIN_CONTRACT_NAME] != \
                        test["transaction"]["to"]:
                    # 对于跨合约的情况, 暂时不统计其他合约
                    continue

                env.symbolic_taint_analyzer.propagate_taint(instruction, contract_address)

                env.detector_executor.run_detectors(previous_instruction, instruction, env.results["errors"],
                                                    env.symbolic_taint_analyzer.get_tainted_record(index=-2), indv, env,
                                                    previous_branch,
                                                    transaction_index)

                # If constructor, we don't have to take into account the constructor inputs because they will be part of the
                # state. We don't have to compute the code coverage, because the code is not the deployed one. We don't need
                # to compute the cfg because we are on a different code. We actually don't need analyzing its traces.
                if indv.chromosome[transaction_index]["arguments"][0] == "constructor":
                    continue

                # Code coverage, 判断是否trace属于主合约, 如果不属于, 那么不添加到code_coverage里

                env.code_coverage.add(hex(instruction["pc"]))

                # Dynamically build control flow graph
                if env.cfg:
                    env.cfg.execute(instruction["pc"], instruction["stack"], instruction["op"], env.visited_branches,
                                    env.results["errors"].keys())

                if previous_instruction and previous_instruction["op"] == "SHA3":
                    try:
                        if (len(instruction["stack"]) >= 1 and 
                            hasattr(instruction["stack"][-1], '__len__') and
                            len(instruction["stack"][-1]) >= 2):
                            sha3[instruction["stack"][-1][1]] = instruction["memory"]
                    except (IndexError, KeyError, TypeError):
                        pass

                elif previous_instruction and previous_instruction["op"] == "ADD":
                    try:
                        if (len(previous_instruction["stack"]) >= 1 and 
                            hasattr(previous_instruction["stack"][-1], '__len__') and
                            len(previous_instruction["stack"][-1]) >= 2 and
                            len(instruction["stack"]) >= 1 and
                            hasattr(instruction["stack"][-1], '__len__') and
                            len(instruction["stack"][-1]) >= 2):
                            if previous_instruction["stack"][-1][1] in sha3:
                                sha3[instruction["stack"][-1][1]] = sha3[previous_instruction["stack"][-1][1]]
                    except (IndexError, KeyError, TypeError):
                        pass
                    try:
                        if (len(previous_instruction["stack"]) >= 2 and 
                            hasattr(previous_instruction["stack"][-2], '__len__') and
                            len(previous_instruction["stack"][-2]) >= 2 and
                            len(instruction["stack"]) >= 1 and
                            hasattr(instruction["stack"][-1], '__len__') and
                            len(instruction["stack"][-1]) >= 2):
                            if previous_instruction["stack"][-2][1] in sha3:
                                sha3[instruction["stack"][-1][1]] = sha3[previous_instruction["stack"][-2][1]]
                    except (IndexError, KeyError, TypeError):
                        pass

                if instruction["op"] == "JUMPI":
                    jumpi_pc = hex(instruction["pc"])
                    if jumpi_pc not in env.visited_branches:
                        env.visited_branches[jumpi_pc] = {}
                    if jumpi_pc not in branches:
                        branches[jumpi_pc] = dict()

                    destination = convert_stack_value_to_int(instruction["stack"][-1])
                    jumpi_condition = convert_stack_value_to_int(instruction["stack"][-2])

                    if jumpi_condition == 0:
                        # don't jump, but increase pc
                        branches[jumpi_pc][hex(destination)] = False
                        branches[jumpi_pc][hex(instruction["pc"] + 1)] = True
                    else:
                        # jump to destination
                        branches[jumpi_pc][hex(destination)] = True
                        branches[jumpi_pc][hex(instruction["pc"] + 1)] = False

                    env.visited_branches[jumpi_pc][jumpi_condition] = {}
                    env.visited_branches[jumpi_pc][jumpi_condition]["indv_hash"] = indv.hash
                    env.visited_branches[jumpi_pc][jumpi_condition]["chromosome"] = indv.chromosome
                    env.visited_branches[jumpi_pc][jumpi_condition]["transaction_index"] = transaction_index

                    tainted_record = env.symbolic_taint_analyzer.check_taint(instruction=instruction)
                    if tainted_record and tainted_record.stack and tainted_record.stack[-2]:
                        if jumpi_condition != 0:
                            previous_branch.append(tainted_record.stack[-2][0] != 0)
                        else:
                            previous_branch.append(tainted_record.stack[-2][0] == 0)
                        previous_branch_expression = previous_branch[-1]
                        env.visited_branches[jumpi_pc][jumpi_condition]["expression"] = previous_branch.copy()
                    else:
                        env.visited_branches[jumpi_pc][jumpi_condition]["expression"] = None
                        previous_branch_expression = None

                    previous_branch_address = jumpi_pc

                # Extract data dependencies (read-after-write)
                elif instruction["op"] == "SLOAD":
                    storage_slot = 0  # Default value
                    try:
                        if (len(instruction["stack"]) >= 1 and 
                            hasattr(instruction["stack"][-1], '__len__') and
                            len(instruction["stack"][-1]) >= 2 and
                            instruction["stack"][-1][1] in sha3):
                            hash = instruction["stack"][-1][1]
                            while hash in sha3:
                                if len(sha3[hash]) == 64:
                                    hash = sha3[hash][32:64]
                                else:
                                    hash = sha3[hash]
                            storage_slot = int.from_bytes(hash, byteorder='big')
                        else:
                            storage_slot = convert_stack_value_to_int(instruction["stack"][-1])
                    except (IndexError, KeyError, TypeError, ValueError):
                        storage_slot = convert_stack_value_to_int(instruction["stack"][-1])

                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                    if _function_hash not in self.env.data_dependencies:
                        self.env.data_dependencies[_function_hash] = {"read": set(), "write": set()}
                    self.env.data_dependencies[_function_hash]["read"].add(storage_slot)
                    temp_dict = settings.GLOBAL_DATA_INFO.get(indv.hash, {})
                    temp_dict_2 = temp_dict.get(transaction_index, {})
                    temp_dict_3 = temp_dict_2.get("read", set())
                    temp_dict_3.add(storage_slot)
                    temp_dict_2["read"] = temp_dict_3
                    temp_dict[transaction_index] = temp_dict_2
                    settings.GLOBAL_DATA_INFO[indv.hash] = temp_dict


                elif instruction["op"] == "SSTORE":
                    storage_slot = 0  # Default value
                    try:
                        if (len(instruction["stack"]) >= 1 and 
                            hasattr(instruction["stack"][-1], '__len__') and
                            len(instruction["stack"][-1]) >= 2 and
                            instruction["stack"][-1][1] in sha3):
                            hash = instruction["stack"][-1][1]
                            while hash in sha3:
                                if len(sha3[hash]) == 64:
                                    hash = sha3[hash][32:64]
                                else:
                                    hash = sha3[hash]
                            storage_slot = int.from_bytes(hash, byteorder='big')
                        else:
                            storage_slot = convert_stack_value_to_int(instruction["stack"][-1])
                    except (IndexError, KeyError, TypeError, ValueError):
                        storage_slot = convert_stack_value_to_int(instruction["stack"][-1])

                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                    if _function_hash not in self.env.data_dependencies:
                        self.env.data_dependencies[_function_hash] = {"read": set(), "write": set()}
                    self.env.data_dependencies[_function_hash]["write"].add(storage_slot)
                    temp_dict = settings.GLOBAL_DATA_INFO.get(indv.hash, {})
                    temp_dict_2 = temp_dict.get(transaction_index, {})
                    temp_dict_3 = temp_dict_2.get("write", set())
                    temp_dict_3.add(storage_slot)
                    temp_dict_2["write"] = temp_dict_3
                    temp_dict[transaction_index] = temp_dict_2
                    settings.GLOBAL_DATA_INFO[indv.hash] = temp_dict


                # If something goes wrong, we need to clean some pools
                elif instruction["op"] in ["REVERT", "INVALID", "ASSERTFAIL"]:
                    if this_error_cross_check and settings.TRANS_COMP_OPEN and random.randint(0,
                                                                                              100) <= settings.P_OPEN_CROSS:
                        hash_4_chromosome = count_hash_4_chromosome(indv.chromosome)
                        if hash_4_chromosome not in settings.TRANS_CROSS_BAD_INDVS_HASH:
                            settings.TRANS_MODE = "cross"  # 启动交叉模式
                            indv_success = (
                                indv,
                                transaction_index)  # 从0到transaction_index-1的事务都执行成功了, transaction_index是当前事务,这个执行失败了
                            settings.TRANS_CROSS_BAD_INDVS.append(indv_success)
                            settings.TRANS_CROSS_BAD_INDVS_HASH.add(hash_4_chromosome)
                            this_error_cross_check = False
                    if previous_branch_expression is not None and is_expr(previous_branch_expression):
                        # Only remove from pool when you are sure which variable caused the exception
                        if len(get_vars(previous_branch_expression)) == 1:
                            for var in get_vars(previous_branch_expression):
                                _str_var = str(var)
                                if _str_var.startswith("calldataload_") or str(var).startswith("calldatacopy_"):
                                    _parameter_index = int(str(var).split("_")[-1])
                                    _transaction_index = int(str(var).split("_")[-2])
                                    _function_hash = indv.chromosome[_transaction_index]["arguments"][0]
                                    _argument = indv.chromosome[_transaction_index]["arguments"][_parameter_index + 1]
                                    indv.generator.remove_argument_from_pool(_function_hash, _parameter_index,
                                                                             _argument)

                                elif _str_var.startswith("callvalue_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _amount = transaction["value"]
                                    if _amount == 0 or _amount == 1:
                                        indv.generator.remove_amount_from_pool(_function_hash, _amount)

                                elif _str_var.startswith("caller_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _caller = transaction["from"]
                                    indv.generator.remove_account_from_pool(_function_hash, _caller)

                                elif _str_var.startswith("gas_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _gas_limit = indv.chromosome[transaction_index]["gaslimit"]
                                    indv.generator.remove_gaslimit_from_pool(_function_hash, _gas_limit)

                                elif _str_var.startswith("blocknumber_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _blocknumber = indv.chromosome[transaction_index]["blocknumber"]
                                    indv.generator.remove_blocknumber_from_pool(_function_hash, _blocknumber)

                                elif _str_var.startswith("timestamp_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _timestamp = indv.chromosome[transaction_index]["timestamp"]
                                    indv.generator.remove_timestamp_from_pool(_function_hash, _timestamp)

                                elif _str_var.startswith("call_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _var_split = str(var).split("_")
                                    _address = to_normalized_address(_var_split[2])
                                    _result = int(_var_split[3], 16)
                                    indv.generator.remove_callresult_from_pool(_function_hash, _address, _result)

                                elif _str_var.startswith("extcodesize"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _var_split = str(var).split("_")
                                    _address = to_normalized_address(_var_split[2])
                                    _size = int(_var_split[3], 16)
                                    indv.generator.remove_extcodesize_from_pool(_function_hash, _address, _size)
                                elif _str_var.startswith("returndatasize"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _var_split = str(var).split("_")
                                    _address = to_normalized_address(_var_split[2])
                                    _size = int(_var_split[3], 16)
                                    indv.generator.remove_returndatasize_from_pool(_function_hash, _address, _size)

                elif instruction["op"] == "BALANCE":
                    taint = BitVec("_".join(["balance", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] in ["CALL", "STATICCALL"]:
                    _address_as_hex = to_hex(force_bytes_to_address(
                        int_to_big_endian(convert_stack_value_to_int(result.trace[i]["stack"][-2]))))
                    if i + 1 < len(result.trace):
                        _result_as_hex = convert_stack_value_to_hex(result.trace[i + 1]["stack"][-1])
                    else:
                        _result_as_hex = ""
                    previous_call_address = _address_as_hex
                    call_type = "call"
                    if instruction["op"] == "STATICCALL":
                        call_type = "staticcall"
                    taint = BitVec("_".join(
                        [call_type, str(transaction_index), str(_address_as_hex), str(_result_as_hex),
                         str(instruction["pc"])]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "CALLER":
                    taint = BitVec("_".join(["caller", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "CALLDATALOAD":
                    input_index = convert_stack_value_to_int(instruction["stack"][-1])
                    if input_index > 0 and _function_hash in env.interface:
                        input_index = int((input_index - 4) / 32)
                        if input_index < len(env.interface[_function_hash]):
                            parameter_type = env.interface[_function_hash][input_index]
                            if '[' in parameter_type:
                                array_size_index = convert_stack_value_to_int(result.trace[i + 1]["stack"][-1]) / 32
                                _array_size_indexes[array_size_index] = input_index
                            elif "bytes" in parameter_type:
                                pass
                            else:
                                taint = BitVec("_".join(["calldataload",
                                                         str(transaction_index),
                                                         str(input_index)
                                                         ]), 256)
                                env.symbolic_taint_analyzer.introduce_taint(taint, instruction)
                        else:
                            if input_index in _array_size_indexes:
                                array_size = convert_stack_value_to_int(result.trace[i + 1]["stack"][-1])
                                taint = BitVec("_".join(["inputarraysize",
                                                         str(transaction_index),
                                                         str(_array_size_indexes[input_index])
                                                         ]), 256)
                                env.symbolic_taint_analyzer.introduce_taint(taint, instruction)
                            else:
                                pass

                elif instruction["op"] == "CALLDATACOPY":
                    destOffset = convert_stack_value_to_int(instruction["stack"][-1])
                    offset = convert_stack_value_to_int(instruction["stack"][-2])
                    array_start_index = (offset - 4) / 32
                    lenght = convert_stack_value_to_int(instruction["stack"][-3])

                    if array_start_index - 1 in _array_size_indexes:
                        taint = BitVec("_".join(["calldatacopy",
                                                 str(transaction_index),
                                                 str(_array_size_indexes[array_start_index - 1])
                                                 ]), 256)
                        env.symbolic_taint_analyzer.introduce_taint(taint, instruction)
                    else:
                        pass

                elif instruction["op"] == "CALLDATASIZE":
                    taint = BitVec("_".join(["calldatasize", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "CALLVALUE":
                    taint = BitVec("_".join(["callvalue", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "GAS":
                    taint = BitVec("_".join(["gas", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                # BLOCK Opcodes
                elif instruction["op"] == "BLOCKHASH":
                    taint = BitVec("_".join(["blockhash", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "COINBASE":
                    taint = BitVec("_".join(["coinbase", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "TIMESTAMP":
                    taint = BitVec("_".join(["timestamp", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "NUMBER":
                    taint = BitVec("_".join(["blocknumber", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "DIFFICULTY":
                    taint = BitVec("_".join(["difficulty", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "GASLIMIT":
                    taint = BitVec("_".join(["gaslimit", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "EXTCODESIZE":
                    _address_as_hex = to_hex(
                        force_bytes_to_address(
                            int_to_big_endian(convert_stack_value_to_int(result.trace[i]["stack"][-1]))))
                    if i + 1 < len(result.trace):
                        _result_as_hex = convert_stack_value_to_hex(result.trace[i + 1]["stack"][-1])
                    else:
                        _result_as_hex = ""
                    taint = BitVec(
                        "_".join(["extcodesize", str(transaction_index), str(_address_as_hex), str(_result_as_hex)]),
                        256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "RETURNDATASIZE":
                    if previous_call_address:
                        if i + 1 < len(result.trace):
                            _size = convert_stack_value_to_int(result.trace[i + 1]["stack"][-1])
                        else:
                            _size = 0
                        taint = BitVec(
                            "_".join(["returndatasize", str(transaction_index), previous_call_address, str(_size)]),
                            256)
                        env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                previous_instruction = instruction

            env.symbolic_taint_analyzer.clear_callstack()

            if not result.is_error and not transaction["to"]:
                contract_address = encode_hex(result.msg.storage_address)

        env.individual_branches[indv.hash] = branches

        # Debug: log full transaction sequence (addresses, values, calldata) for this individual.
        # This uses the same pretty-printer as detectors, but is always emitted for easier analysis.
        # DISABLED: Commented out to reduce log verbosity
        # try:
        #     func_sig_mapping = getattr(
        #         env.detector_executor, "function_signature_mapping", {}
        #     )
        #     self.logger.info(
        #         "[DEBUG-SEQ] Transaction sequence for contract %s (indv_hash=%s)",
        #         getattr(env, "contract_name", "<unknown>"),
        #         indv.hash,
        #     )
        #     print_individual_solution_as_transaction(
        #         self.logger,
        #         indv.solution,
        #         function_signature_mapping=func_sig_mapping,
        #     )
        # except Exception as debug_exc:
        #     # Do not break fuzzing because of debug printing
        #     self.logger.debug("Failed to print transaction sequence: %s", debug_exc)

        env.symbolic_taint_analyzer.clear_storage()
        env.instrumented_evm.restore_from_snapshot()

    def get_coverage_with_children(self, children_code_coverage, code_coverage):
        code_coverage = len(code_coverage)

        for child_cc in children_code_coverage:
            code_coverage += len(child_cc)
        return code_coverage

    def symbolic_execution(self, indv_generator, other_generators):
        # Debug: log whenever symbolic execution is invoked and whether the flag is enabled
        # DISABLED: Commented out to reduce log verbosity
        # constraint_flag = getattr(self.env.args, "constraint_solving", None)
        # self.logger.info(
        #     "[DEBUG-SE] symbolic_execution called | constraint_solving=%s",
        #     constraint_flag,
        # )

        if not self.env.args.constraint_solving:  # 是否启用符号执行模块
            return

        for index, pc in enumerate(self.env.visited_branches):
            self.logger.debug("b(%d) pc : %s - visited branches : %s", index, pc, self.env.visited_branches[pc].keys())

            if len(self.env.visited_branches[pc]) != 1:  # 如果这个分支, 已经有2条路径可以出发了, 那么跳过, 没必要为其生成了
                continue

            branch, _d = next(iter(self.env.visited_branches[pc].items()))

            if not _d["expression"]:
                self.logger.debug("No expression for b(%d) pc : %s", index, pc)
                continue

            negated_branch = simplify(Not(_d["expression"][-1]))  # 反转最后一个条件

            if negated_branch in self.env.memoized_symbolic_execution:
                continue

            self.env.solver.reset()
            for expression_index in range(len(_d["expression"]) - 1):  # 除了最后一个条件, 其他的条件加入到约束求解器里
                expression = simplify(_d["expression"][expression_index])
                self.env.solver.add(expression)
            self.env.solver.add(negated_branch)  # 将反转条件加入到约束求解器里

            check = self.env.solver.check()  # 判断是否满足, 如果满足了, 说明反转条件在约束求解器里可以满足, 存在这么一个值, 让另一个分支成立, 但是! 不一定满足真实情况

            if check == sat:
                model = self.env.solver.model()

                # Log symbolic model at INFO level so it is visible by default
                # DISABLED: Commented out to reduce log verbosity
                # try:
                #     model_str = "; ".join(
                #         [f"{str(x)} ({model[x]})" for x in model]
                #     )
                # except Exception:
                #     model_str = str(model)
                # self.logger.info(
                #     "[DEBUG-SE] (%s) Symbolic solution to branch %s: %s",
                #     _d["indv_hash"],
                #     pc,
                #     model_str,
                # )

                for variable in model:
                    if str(variable).startswith("underflow"):
                        continue

                    var_split = str(variable).split("_")
                    transaction_index = int(var_split[1])

                    if str(variable).startswith("balance"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        opt = Optimize()
                        for expression_index in range(len(_d["expression"]) - 1):
                            opt.add(_d["expression"][expression_index])
                        opt.add(negated_branch)
                        check = opt.check()
                        if check == sat:
                            opt_model = opt.model()
                            balance = int(opt_model[variable].as_long())
                            if _d["chromosome"][transaction_index]["contract"]:
                                indv_generator.add_balance_to_pool(_function_hash,
                                                                   self.env.instrumented_evm.get_balance(
                                                                       to_canonical_address(
                                                                           _d["chromosome"][transaction_index][
                                                                               "contract"])))
                            indv_generator.add_balance_to_pool(_function_hash, balance)

                    elif str(variable).startswith("blocknumber"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        blocknumber = int(model[variable].as_long())
                        indv_generator.add_blocknumber_to_pool(_function_hash,
                                                               self.env.instrumented_evm.vm.state.block_number)
                        indv_generator.add_blocknumber_to_pool(_function_hash, blocknumber)

                    elif str(variable).startswith("call_") or str(variable).startswith("staticcall_"):
                        address = to_normalized_address(var_split[2])
                        old_result = int(var_split[3], 16)
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        new_result = 1 - old_result
                        # indv_generator.add_callresult_to_pool(_function_hash, address, old_result)
                        # indv_generator.add_callresult_to_pool(_function_hash, address, new_result)

                    elif str(variable).startswith("caller_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        if model[variable].as_long() > 8 and model[variable].as_long() < 2 ** 160:
                            account_address = normalize_32_byte_hex_address(
                                "0x" + hex(model[variable].as_long()).replace("0x", "").zfill(40))
                            if not self.env.instrumented_evm.has_account(account_address):
                                self.env.instrumented_evm.restore_from_snapshot()
                                self.env.instrumented_evm.accounts.append(
                                    self.env.instrumented_evm.create_fake_account(account_address))
                                self.env.instrumented_evm.create_snapshot()
                            indv_generator.add_account_to_pool(_function_hash,
                                                               _d["chromosome"][transaction_index]["account"])
                            indv_generator.add_account_to_pool(_function_hash, account_address)

                    elif str(variable).startswith("calldatacopy_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        parameter_index = int(var_split[2])
                        if "[" in indv_generator.interface[_function_hash][parameter_index]:
                            if indv_generator.interface[_function_hash][parameter_index].startswith("int"):
                                argument = model[variable].as_signed_long()
                            elif indv_generator.interface[_function_hash][parameter_index].startswith("address"):
                                try:
                                    _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                                    argument = normalize_32_byte_hex_address(hex(model[variable].as_long()))
                                    if not self.env.instrumented_evm.has_account(argument):
                                        self.env.instrumented_evm.restore_from_snapshot()
                                        self.env.instrumented_evm.accounts.append(
                                            self.env.instrumented_evm.create_fake_account(argument))
                                        self.env.instrumented_evm.create_snapshot()
                                except Exception as e:
                                    self.logger.error("(%s) [symbolic execution : calldatacopy ] %s", _function_hash,
                                                      e)
                                    continue
                            else:
                                argument = model[variable].as_long()
                            indv_generator.add_argument_to_pool(_function_hash, parameter_index,
                                                                _d["chromosome"][transaction_index]["arguments"][
                                                                    parameter_index + 1])
                            indv_generator.add_argument_to_pool(_function_hash, parameter_index, argument)

                    elif str(variable).startswith("calldataload_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        parameter_index = int(var_split[2])
                        # TODO: THE SOLVER DOES NOT CONSIDER THE MAX SIZE OF THE VARIABLE
                        #   GENERATING LATER A eth_abi.exceptions.ValueOutOfBounds
                        if "[" in indv_generator.interface[_function_hash][parameter_index]:  # 如果是数组?
                            if indv_generator.interface[_function_hash][parameter_index].startswith("int"):
                                argument = model[variable].as_signed_long()
                            elif indv_generator.interface[_function_hash][parameter_index].startswith("address"):
                                try:
                                    _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                                    argument = normalize_32_byte_hex_address(hex(model[variable].as_long()))
                                    if not self.env.instrumented_evm.has_account(argument):
                                        self.env.instrumented_evm.restore_from_snapshot()
                                        self.env.instrumented_evm.accounts.append(
                                            self.env.instrumented_evm.create_fake_account(argument))
                                        self.env.instrumented_evm.create_snapshot()
                                except Exception as e:
                                    self.logger.error("(%s) [symbolic execution : calldataload ] %s", _function_hash, e)
                                    continue

                        elif indv_generator.interface[_function_hash][parameter_index].startswith("int"):
                            argument = model[variable].as_signed_long()

                        elif indv_generator.interface[_function_hash][parameter_index] == "address":
                            try:
                                _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                                argument = to_hex(
                                    force_bytes_to_address(int_to_big_endian(int(model[variable].as_long()))))
                                if not self.env.instrumented_evm.has_account(argument):
                                    self.env.instrumented_evm.restore_from_snapshot()
                                    self.env.instrumented_evm.accounts.append(
                                        self.env.instrumented_evm.create_fake_account(argument))
                                    self.env.instrumented_evm.create_snapshot()
                            except Exception as e:
                                self.logger.error("(%s) [symbolic execution : calldataload ] %s", _function_hash, e)
                                continue

                        elif indv_generator.interface[_function_hash][parameter_index] == "string":
                            argument = _d["chromosome"][transaction_index]["arguments"][parameter_index + 1]
                        elif indv_generator.interface[_function_hash][parameter_index].startswith("uint"):
                            argument = model[variable].as_long()
                            bits = 256
                            if indv_generator.interface[_function_hash][parameter_index] != "uint":
                                bits = int(
                                    indv_generator.interface[_function_hash][parameter_index].replace("uint", ""))
                            base = 1 << bits
                            argument %= base
                        else:
                            argument = model[variable].as_long()
                            self.env.solver.add(BitVec(str(variable), 256) != BitVecVal(0, 256))
                            for variable_2 in model:
                                if variable_2 != variable and str(variable_2).startswith("callvalue"):
                                    callvalue_index = int(str(variable_2).split("_")[1])
                                    self.env.solver.add(BitVec(str(variable_2), 256) == BitVecVal(
                                        int(_d["chromosome"][callvalue_index]["amount"]), 256))
                            check = self.env.solver.check()
                            if check == sat:
                                model = self.env.solver.model()
                                argument = model[variable].as_long()

                        indv_generator.add_argument_to_pool(_function_hash, parameter_index,
                                                            _d["chromosome"][transaction_index]["arguments"][
                                                                parameter_index + 1])
                        indv_generator.add_argument_to_pool(_function_hash, parameter_index, argument)

                    elif str(variable).startswith("callvalue_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        amount = model[variable].as_long()
                        if amount > settings.ACCOUNT_BALANCE:
                            amount = settings.ACCOUNT_BALANCE
                        indv_generator.remove_amount_from_pool(_function_hash, 0)
                        indv_generator.remove_amount_from_pool(_function_hash, 1)
                        indv_generator.add_amount_to_pool(_function_hash, _d["chromosome"][transaction_index]["amount"])
                        indv_generator.add_amount_to_pool(_function_hash, amount)

                    elif str(variable).startswith("gas_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        indv_generator.add_gaslimit_to_pool(_function_hash,
                                                            _d["chromosome"][transaction_index]["gaslimit"])
                        indv_generator.add_gaslimit_to_pool(_function_hash, model[variable].as_long())

                    elif str(variable).startswith("inputarraysize"):
                        opt = Optimize()
                        for expression_index in range(len(_d["expression"]) - 1):
                            opt.add(_d["expression"][expression_index])
                        opt.add(negated_branch)
                        check = opt.check()
                        if check == sat:
                            opt_model = opt.model()
                            array_size = opt_model[variable].as_long()
                            _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                            parameter_index = int(var_split[2])
                            indv_generator.add_parameter_array_size(_function_hash, parameter_index, len(
                                _d["chromosome"][transaction_index]["arguments"][parameter_index + 1]))
                            indv_generator.add_parameter_array_size(_function_hash, parameter_index, array_size)

                    elif str(variable).startswith("timestamp"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        timestamp = int(model[variable].as_long())
                        indv_generator.add_timestamp_to_pool(_function_hash,
                                                             self.env.instrumented_evm.vm.state.timestamp)
                        indv_generator.add_timestamp_to_pool(_function_hash, timestamp)

                    elif str(variable).startswith("calldatasize"):
                        pass

                    elif str(variable).startswith("extcodesize"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        _address = to_normalized_address(var_split[2])
                        # indv_generator.add_extcodesize_to_pool(_function_hash, _address, int(var_split[3], 16))
                        # indv_generator.add_extcodesize_to_pool(_function_hash, _address, int(model[variable].as_long()))

                    elif str(variable).startswith("returndatasize"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        _address = to_normalized_address(var_split[2])
                        _size = int(var_split[3], 16)
                        # indv_generator.add_returndatasize_to_pool(_function_hash, _address, int(var_split[3], 16))
                        # indv_generator.add_returndatasize_to_pool(_function_hash, _address, int(model[variable].as_long()))

                    else:
                        self.logger.warning("Unknown symbolic variable: %s ", str(variable))

            self.env.memoized_symbolic_execution[negated_branch] = True

    def finalize(self, population, engine):
        execution_end = time.time()
        execution_delta = execution_end - self.env.execution_begin

        self.logger.title("-----------------------------------------------------")
        msg = 'Number of generations: \t {}'.format(engine.current_generation + 1)
        self.logger.info(msg)
        msg = 'Number of transactions: \t {} ({} unique)'.format(self.env.nr_of_transactions,
                                                                 len(self.env.unique_individuals))
        self.logger.info(msg)
        msg = 'Transactions per second: \t {:.0f}'.format(self.env.nr_of_transactions / execution_delta)
        self.logger.info(msg)
        code_coverage_percentage = 0
        if len(self.env.overall_pcs) > 0:
            code_coverage_percentage = (len(self.env.code_coverage) / len(self.env.overall_pcs)) * 100
        msg = 'Total code coverage: \t {:.2f}% ({}/{})'.format(code_coverage_percentage,
                                                               len(self.env.code_coverage),
                                                               len(self.env.overall_pcs))
        self.logger.info(msg)
        # Count unique branches visited (each JUMPI has 2 possible outcomes: 0 and 1)
        # Total branches = number of JUMPI instructions * 2 (each has true/false path)
        total_branches = len(self.env.overall_jumpis) * 2
        branch_coverage = 0
        for pc in self.env.visited_branches:
            branch_coverage += len(self.env.visited_branches[pc])
        branch_coverage_percentage = 0
        if total_branches > 0:
            # Ensure branch coverage doesn't exceed 100%
            branch_coverage = min(branch_coverage, total_branches)
            branch_coverage_percentage = (branch_coverage / total_branches) * 100
        msg = 'Total branch coverage: \t {:.2f}% ({}/{})'.format(branch_coverage_percentage,
                                                                 branch_coverage, total_branches)
        self.logger.info(msg)
        msg = 'Total execution time: \t {:.2f} seconds'.format(execution_delta)
        self.logger.info(msg)
        msg = 'Total memory consumption: \t {:.2f} MB'.format(
            psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        self.logger.info(msg)

        # Save to results
        self.env.results["transactions"] = {"total": self.env.nr_of_transactions,
                                            "per_second": self.env.nr_of_transactions / execution_delta}
        self.env.results["code_coverage"] = {"percentage": code_coverage_percentage,
                                             "covered": len(self.env.code_coverage),
                                             "total": len(self.env.overall_pcs),
                                             "covered_with_children": self.get_coverage_with_children(
                                                 self.env.children_code_coverage,
                                                 self.env.code_coverage),
                                             "total_with_children": self.env.len_overall_pcs_with_children
                                             }
        self.env.results["branch_coverage"] = {"percentage": branch_coverage_percentage,
                                               "covered": branch_coverage,
                                               "total": total_branches if len(self.env.overall_jumpis) > 0 else 0}
        self.env.results["execution_time"] = execution_delta
        self.env.results["memory_consumption"] = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.env.results["address_under_test"] = self.env.population.indv_generator.contract
        self.env.results["seed"] = self.env.seed

        self.env.results["cross_trans_count"] = settings.CROSS_TRANS_EXEC_COUNT

        self.env.results["total_op"] = list(self.env.overall_pcs)
        self.env.results["coverage_op"] = list(self.env.code_coverage)

        # Write results to file
        if self.env.args.results:
            results = {}
            # Đảm bảo thư mục chứa file kết quả tồn tại trước khi ghi
            if self.env.args.results.lower().endswith(".json"):
                results_path = self.env.args.results
                results_dir = os.path.dirname(results_path)
                if results_dir and not os.path.exists(results_dir):
                    os.makedirs(results_dir, exist_ok=True)

                if os.path.exists(results_path):
                    with open(results_path, 'r') as file:
                        results = json.load(file)
                results[self.env.contract_name] = self.env.results
                with open(results_path, 'w') as file:
                    json.dump(results, file)
            else:
                results_dir = self.env.args.results
                if results_dir and not os.path.exists(results_dir):
                    os.makedirs(results_dir, exist_ok=True)

                contract_json = os.path.join(
                    results_dir,
                    os.path.splitext(os.path.basename(self.env.contract_name))[0] + '.json',
                )
                if os.path.exists(contract_json):
                    with open(contract_json, 'r') as file:
                        results = json.load(file)
                results[self.env.contract_name] = self.env.results
                with open(contract_json, 'w') as file:
                    json.dump(results, file)

        diff = list(set(self.env.code_coverage).symmetric_difference(set([hex(x) for x in self.env.overall_pcs])))
        self.logger.debug("Instructions not executed: %s", sorted(diff))


def count_hash_4_chromosome(_chromosome: list):
    value = 0
    for ch in _chromosome:
        value += hash(ch["account"])
        value += hash(ch["contract"])
        for arg in ch["arguments"]:
            try:
                value += hash(arg)
            except TypeError:
                pass
    return value
