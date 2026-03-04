#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import string
import sys
import pickle
import logging
from typing import List

from eth import Chain, constants
from eth.chains.mainnet import (
    MAINNET_GENESIS_HEADER,
    HOMESTEAD_MAINNET_BLOCK,
    TANGERINE_WHISTLE_MAINNET_BLOCK,
    SPURIOUS_DRAGON_MAINNET_BLOCK,
    BYZANTIUM_MAINNET_BLOCK,
    PETERSBURG_MAINNET_BLOCK
)
from eth.constants import ZERO_ADDRESS, CREATE_CONTRACT_ADDRESS
from eth.db.atomic import AtomicDB
from eth.db.backends.memory import MemoryDB
from eth.rlp.accounts import Account
from eth.rlp.headers import BlockHeader
# from eth.tools.logging import DEBUG2_LEVEL_NUM  # Removed in newer versions
DEBUG2_LEVEL_NUM = 5  # Define our own debug level
from eth.validation import validate_uint256
from eth.vm.spoof import SpoofTransaction
from eth_utils import to_canonical_address, decode_hex, encode_hex
from web3 import HTTPProvider
from web3 import Web3

from .storage_emulation import (
    FrontierVMForFuzzTesting,
    HomesteadVMForFuzzTesting,
    TangerineWhistleVMForFuzzTesting,
    SpuriousDragonVMForFuzzTesting,
    ByzantiumVMForFuzzTesting,
    PetersburgVMForFuzzTesting
)

# 获取根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../'
# 将根目录添加到path中
sys.path.append(BASE_DIR)
from fuzzer.utils import settings
from fuzzer.utils.utils import initialize_logger
from eth_abi import encode


class InstrumentedEVM:
    def __init__(self, eth_node_ip=None, eth_node_port=None) -> None:
        chain_class = Chain.configure(
            __name__='Blockchain',
            vm_configuration=(
                (constants.GENESIS_BLOCK_NUMBER, FrontierVMForFuzzTesting),
                (HOMESTEAD_MAINNET_BLOCK, HomesteadVMForFuzzTesting),
                (TANGERINE_WHISTLE_MAINNET_BLOCK, TangerineWhistleVMForFuzzTesting),
                (SPURIOUS_DRAGON_MAINNET_BLOCK, SpuriousDragonVMForFuzzTesting),
                (BYZANTIUM_MAINNET_BLOCK, ByzantiumVMForFuzzTesting),
                (PETERSBURG_MAINNET_BLOCK, PetersburgVMForFuzzTesting),
            ),
        )

        class MyMemoryDB(MemoryDB):
            def __init__(self) -> None:
                self.kv_store = {'storage': dict(), 'account': dict(), 'code': dict()}

            def rst(self) -> None:
                self.kv_store = {'storage': dict(), 'account': dict(), 'code': dict()}

        if eth_node_ip and eth_node_port and settings.REMOTE_FUZZING:
            self.w3 = Web3(HTTPProvider('http://%s:%s' % (eth_node_ip, eth_node_port)))
        else:
            self.w3 = None
        self.chain = chain_class.from_genesis_header(AtomicDB(MyMemoryDB()), MAINNET_GENESIS_HEADER)
        self.logger = initialize_logger("EVM")
        self.accounts = list()
        self.snapshot = None
        self.vm = None

    def get_block_by_blockid(self, block_identifier):
        validate_uint256(block_identifier)
        return self.w3.eth.getBlock(block_identifier)

    def get_cached_block_by_id(self, block_number):
        block = None
        with open(os.path.dirname(os.path.abspath(__file__)) + "/" + ".".join([str(block_number), "block"]), "rb") as f:
            block = pickle.load(f)
        return block

    @property
    def storage_emulator(self):
        return self.vm.state._account_db

    def set_vm(self, block_identifier='latest'):
        _block = None
        if self.w3:
            if block_identifier == 'latest':
                block_identifier = self.w3.eth.blockNumber
            validate_uint256(block_identifier)
            _block = self.w3.eth.getBlock(block_identifier)
        if not _block:
            if block_identifier in [HOMESTEAD_MAINNET_BLOCK, BYZANTIUM_MAINNET_BLOCK, PETERSBURG_MAINNET_BLOCK]:
                _block = self.get_cached_block_by_id(block_identifier)
            else:
                self.logger.error("Unknown block identifier.")
                sys.exit(-4)
        block_header = BlockHeader(difficulty=_block.difficulty,
                                   block_number=_block.number,
                                   gas_limit=_block.gasLimit,
                                   timestamp=_block.timestamp,
                                   coinbase=ZERO_ADDRESS,  # default value
                                   parent_hash=_block.parentHash,
                                   uncles_hash=_block.uncles,
                                   state_root=_block.stateRoot,
                                   transaction_root=_block.transactionsRoot,
                                   receipt_root=_block.receiptsRoot,
                                   bloom=0,  # default value
                                   gas_used=_block.gasUsed,
                                   extra_data=_block.extraData,
                                   mix_hash=_block.mixHash,
                                   nonce=_block.nonce)
        self.vm = self.chain.get_vm(block_header)

    def execute(self, tx, debug=True):  # debug默认是False
        if debug:
            logging.getLogger('eth.vm.computation.Computation')
            logging.basicConfig(level=DEBUG2_LEVEL_NUM)
        return self.vm.state.apply_transaction(tx)

    def reset(self):
        self.storage_emulator._raw_store_db.wrapped_db.rst()

    def create_fake_account(self, address, nonce=0, balance=settings.ACCOUNT_BALANCE, code='', storage=None):
        if storage is None:
            storage = {}
        address = to_canonical_address(address)
        account = Account(nonce=nonce, balance=balance)
        self.vm.state._account_db._set_account(address, account)
        if code and code != '':
            self.vm.state._account_db.set_code(address, code)
        if storage:
            for k, v in storage.items():
                self.vm.state._account_db.set_storage(address, int.from_bytes(decode_hex(k), byteorder="big"),
                                                      int.from_bytes(decode_hex(v), byteorder="big"))
        self.logger.debug("Created account %s with balance %s", encode_hex(address), account.balance)
        return encode_hex(address)

    def has_account(self, address):
        address = to_canonical_address(address)
        return self.vm.state._account_db._has_account(address)

    def deploy_contract(self, creator, bin_code, amount=0, gas=settings.GAS_LIMIT, gas_price=settings.GAS_PRICE,
                        debug=False, deploy_args: List[str] = None, deploy_mode=1):
        """
        部署合约 - Triển khai hợp đồng
        """
        if deploy_args is not None:
            assert len(deploy_args) % 3 == 0, "deploy_args必须是3的倍数, [name, type, name对应的contract或者YA_DO_NOT_KNOW]"
            encode_types = []
            encode_values = []
            
            self.logger.info(f"Processing constructor args: {deploy_args}")
            
            for i in range(0, len(deploy_args), 3):
                param_name, param_type, param_value = deploy_args[i:i + 3]
                self.logger.info(f"Processing parameter: name={param_name}, type={param_type}, value={param_value}")
                
                # Xử lý địa chỉ và hợp đồng
                if (param_type == "address" or param_type == "contract"):
                    encode_types.append("address")
                    if param_value == "YA_DO_NOT_KNOW":
                        # Giá trị mặc định
                        if deploy_mode == 1:
                            encode_values.append(creator)
                        elif deploy_mode == 2:
                            encode_values.append(random.choice(self.accounts))
                        else:
                            encode_values.append("0x0000000000000000000000000000000000000000")
                    else:
                        # Giá trị cụ thể
                        if deploy_mode == 1 and param_value in settings.TRANS_INFO:
                            encode_values.append(settings.TRANS_INFO[param_value])
                        else:
                            encode_values.append(param_value)
                
                # Xử lý address[] (array of addresses)
                elif param_type == "address[]":
                    encode_types.append("address[]")
                    if param_value == "YA_DO_NOT_KNOW":
                        # Generate default array with 2-5 addresses
                        array_size = random.randint(2, 5)
                        default_array = []
                        for _ in range(array_size):
                            if deploy_mode == 1:
                                default_array.append(creator)
                            elif deploy_mode == 2:
                                default_array.append(random.choice(self.accounts))
                            else:
                                default_array.append("0x0000000000000000000000000000000000000000")
                        encode_values.append(default_array)
                    else:
                        # Try to parse as array
                        if isinstance(param_value, list):
                            encode_values.append(param_value)
                        elif isinstance(param_value, str):
                            # Try to parse string representation of array
                            # For now, create a simple array with one address
                            if deploy_mode == 1:
                                encode_values.append([creator])
                            else:
                                encode_values.append([random.choice(self.accounts)])
                        else:
                            # Fallback to default
                            encode_values.append([creator] if deploy_mode == 1 else [random.choice(self.accounts)])
                
                # Xử lý uint[] và int[] (array of integers)
                elif param_type.endswith("[]") and (param_type.startswith("uint") or param_type.startswith("int")):
                    encode_types.append(param_type)
                    if param_value == "YA_DO_NOT_KNOW":
                        # Generate default array with 2-5 elements
                        array_size = random.randint(2, 5)
                        default_array = [0] * array_size
                        encode_values.append(default_array)
                    else:
                        # Try to parse as array
                        if isinstance(param_value, list):
                            encode_values.append(param_value)
                        else:
                            # Fallback to default array
                            encode_values.append([0, 1])
                
                # Xử lý số nguyên uint
                elif param_type.startswith("uint"):
                    encode_types.append(param_type)
                    if param_value == "YA_DO_NOT_KNOW":
                        encode_values.append(0)
                    else:
                        try:
                            encode_values.append(int(param_value))
                        except ValueError:
                            self.logger.warning(f"Could not convert {param_value} to int, using 0 instead")
                            encode_values.append(0)
                
                # Xử lý boolean
                elif param_type == "bool":
                    encode_types.append(param_type)
                    if param_value == "YA_DO_NOT_KNOW":
                        encode_values.append(False)
                    else:
                        encode_values.append(param_value.lower() in ['true', '1', 'yes'])
                
                # Xử lý string - quan trọng cho ABE.sol
                elif param_type == "string":
                    encode_types.append(param_type)
                    if param_value == "YA_DO_NOT_KNOW":
                        encode_values.append("")
                    else:
                        # Sử dụng giá trị string trực tiếp
                        encode_values.append(param_value)
                
                # Xử lý bytes và byte arrays
                elif param_type.startswith("bytes"):
                    encode_types.append(param_type)
                    if param_value == "YA_DO_NOT_KNOW":
                        if param_type == "bytes":
                            bytes_size = random.randint(1, 32)
                        else:
                            bytes_size = int(param_type[5:])
                        encode_values.append(bytearray(0 for _ in range(bytes_size)))
                    else:
                        if param_value.startswith('0x'):
                            encode_values.append(bytearray.fromhex(param_value[2:]))
                        else:
                            encode_values.append(param_value.encode('utf-8'))
                
                # Xử lý số nguyên có dấu int
                elif param_type.startswith("int"):
                    encode_types.append(param_type)
                    if param_value == "YA_DO_NOT_KNOW":
                        encode_values.append(0)
                    else:
                        try:
                            encode_values.append(int(param_value))
                        except ValueError:
                            self.logger.warning(f"Could not convert {param_value} to int, using 0 instead")
                            encode_values.append(0)
                            
                # Kiểu dữ liệu không được hỗ trợ            
                else:
                    self.logger.warning(f"Unsupported parameter type: {param_type}, defaulting to string")
                    encode_types.append("string")
                    encode_values.append(str(param_value))
                    
            self.logger.info(f"encode_types: {encode_types}")
            self.logger.info(f"encode_values: {encode_values}")
            
            # Mã hóa constructor params
            if encode_types and encode_values and len(encode_types) == len(encode_values):
                try:
                    encoded_data = encode(encode_types, encode_values).hex()
                    bin_code += encoded_data
                    self.logger.info(f"Encoded constructor arguments: {encoded_data[:100]}...")
                except Exception as e:
                    self.logger.error(f"Error encoding constructor arguments: {e}")
                    for i, (t, v) in enumerate(zip(encode_types, encode_values)):
                        self.logger.error(f"  Param {i}: Type={t}, Value={v}, Value type={type(v)}")
                    # Thử điều chỉnh dữ liệu và mã hóa lại
                    try:
                        fixed_values = []
                        for i, (t, v) in enumerate(zip(encode_types, encode_values)):
                            if t == "string" and isinstance(v, str):
                                fixed_values.append(v)
                            elif t.startswith("uint") and not isinstance(v, int):
                                try:
                                    fixed_values.append(int(v))
                                except:
                                    fixed_values.append(0)
                            elif t == "address" and isinstance(v, str):
                                fixed_values.append(v)
                            else:
                                fixed_values.append(v)
                        self.logger.info(f"Retrying with fixed values: {fixed_values}")
                        encoded_data = encode(encode_types, fixed_values).hex()
                        bin_code += encoded_data
                        self.logger.info(f"Successfully encoded constructor arguments after fixing: {encoded_data[:100]}...")
                    except Exception as e2:
                        self.logger.error(f"Still failed to encode constructor arguments after fixing: {e2}")
                        # Nếu vẫn thất bại, không thêm tham số vào bytecode
        
        # Triển khai hợp đồng với bytecode đã chuẩn bị
        nonce = self.vm.state.get_nonce(decode_hex(creator))
        tx = self.vm.create_unsigned_transaction(
            nonce=nonce,
            gas_price=gas_price,
            gas=gas,
            to=CREATE_CONTRACT_ADDRESS,
            value=amount,
            data=decode_hex(bin_code),
        )
        tx = SpoofTransaction(tx, from_=decode_hex(creator))
        result = self.execute(tx, debug=debug)
        
        # Xử lý kết quả deploy
        if result.is_error:
            self.logger.error(f"Error deploying contract: {result._error}")
        else:
            address = to_canonical_address(encode_hex(result.msg.storage_address))
            self.storage_emulator.set_balance(address, 1)
            self.logger.info(f"Contract deployed at: {encode_hex(result.msg.storage_address)}")
            
        return result

    def deploy_transaction(self, input, gas_price=settings.GAS_PRICE, debug=False):
        transaction = input["transaction"]
        from_account = decode_hex(transaction["from"])
        nonce = self.vm.state.get_nonce(from_account)
        try:
            to = decode_hex(transaction["to"])
        except:
            to = transaction["to"]
        tx = self.vm.create_unsigned_transaction(
            nonce=nonce,
            gas_price=gas_price,
            gas=transaction["gaslimit"],
            to=to,
            value=transaction["value"],
            data=decode_hex(transaction["data"]),
        )
        tx = SpoofTransaction(tx, from_=from_account)

        block = input["block"]
        if "timestamp" in block and block["timestamp"] is not None:
            self.vm.state.fuzzed_timestamp = block["timestamp"]
        else:
            self.vm.state.fuzzed_timestamp = None
        if "blocknumber" in block and block["blocknumber"] is not None:
            self.vm.state.fuzzed_blocknumber = block["blocknumber"]
        else:
            self.vm.state.fuzzed_blocknumber = None

        global_state = input["global_state"]
        if "balance" in global_state and global_state["balance"] is not None:
            self.vm.state.fuzzed_balance = global_state["balance"]
        else:
            self.vm.state.fuzzed_balance = None

        if "call_return" in global_state and global_state["call_return"] is not None \
                and len(global_state["call_return"]) > 0:
            self.vm.state.fuzzed_call_return = global_state["call_return"]
        if "extcodesize" in global_state and global_state["extcodesize"] is not None \
                and len(global_state["extcodesize"]) > 0:
            self.vm.state.fuzzed_extcodesize = global_state["extcodesize"]

        environment = input["environment"]
        if "returndatasize" in environment and environment["returndatasize"] is not None:
            self.vm.state.fuzzed_returndatasize = environment["returndatasize"]

        self.storage_emulator.set_balance(from_account, settings.ACCOUNT_BALANCE)
        return self.execute(tx, debug=debug)

    def get_balance(self, address):
        return self.storage_emulator.get_balance(address)
    
    def get_storage(self, address, slot):
        """Get storage value at a specific slot for an address."""
        return self.storage_emulator.get_storage(address, slot)

    def get_code(self, address):
        return self.storage_emulator.get_code(address)

    def set_code(self, address, code):
        return self.storage_emulator.set_code(address, code)

    def create_snapshot(self):
        self.snapshot = self.storage_emulator.record()
        self.storage_emulator.set_snapshot(self.snapshot)

    def restore_from_snapshot(self):
        self.storage_emulator.discard(self.snapshot)

    def get_accounts(self):
        return [encode_hex(x) for x in self.storage_emulator._raw_store_db.wrapped_db["account"].keys()]

    def set_vm_by_name(self, EVM_VERSION):
        if EVM_VERSION == "homestead":
            self.set_vm(HOMESTEAD_MAINNET_BLOCK)
        elif EVM_VERSION == "byzantium":
            self.set_vm(BYZANTIUM_MAINNET_BLOCK)
        elif EVM_VERSION == "petersburg":
            self.set_vm(PETERSBURG_MAINNET_BLOCK)
        else:
            raise Exception("Unknown EVM version, please choose either 'homestead', 'byzantium' or 'petersburg'.")

    def create_fake_accounts(self):
        """
        Tạo các tài khoản giả cho fuzzing với số dư đủ lớn
        """
        # Tạo tài khoản chính (cafebabe)
        main_account = "0xcafebabecafebabecafebabecafebabecafebabe"
        self.accounts.append(self.create_fake_account(main_account, balance=10**30))
        self.logger.info(f"Created main fake account {main_account}")
        
        # Tạo tài khoản từ danh sách ATTACKER_ACCOUNTS
        for address in settings.ATTACKER_ACCOUNTS:
            self.accounts.append(self.create_fake_account(address, balance=10**30))
            self.logger.info(f"Created attacker account {address}")
        
        # Tạo thêm các tài khoản ngẫu nhiên
        additional_accounts = [
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "0x3333333333333333333333333333333333333333",
            "0x4444444444444444444444444444444444444444",
            "0x5555555555555555555555555555555555555555",
            "0x6666666666666666666666666666666666666666"
        ]
        
        for address in additional_accounts:
            self.accounts.append(self.create_fake_account(address, balance=10**20))
            self.logger.info(f"Created additional account {address}")
            
        self.logger.info(f"Created {len(self.accounts)} fake accounts for fuzzing")
        return self.accounts
