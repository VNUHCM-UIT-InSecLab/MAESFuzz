#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Balance Invariant Detector - Based on ContraMaster (Wang et al. 2020)

Implements Definition 3 from the paper:
    For every contract ⟨c, bal, A, σ⟩, ∑(a∈A) m_σ(a) - bal = K (constant)

This semantic oracle detects violations where the difference between:
- Sum of all bookkeeping balances (internal state)
- Contract balance (blockchain state)
changes unexpectedly, indicating exploitable vulnerabilities like:
- Reentrancy
- Integer overflow/underflow
- Logic errors in balance management
"""

from logging import getLogger

logger = getLogger(__name__)


class BalanceInvariantDetector:
    """
    Semantic detector for balance invariant violations.

    Unlike pattern-based detectors, this checks the actual runtime state
    to ensure bookkeeping consistency, reducing false positives.
    """

    def __init__(self):
        self.swc_id = 'SEMANTIC-BALANCE'
        self.severity = 'High'
        self.description = 'Balance Invariant Violation (Semantic Oracle)'

        # Track the invariant constant K
        self.previous_delta = None
        self.bookkeeping_var = None
        self.initial_state_recorded = False

        # Store violations for reporting
        self.violations = []

    def init(self, symbolic_taint_analyzer):
        """Initialize with symbolic taint analyzer reference."""
        self.symbolic_taint_analyzer = symbolic_taint_analyzer

    def detect_balance_invariant_violation(self, instrumented_evm, contract_address, bookkeeping_var_name, accounts):
        """
        Check if balance invariant holds: ∑(balances) - contract.balance = K

        Args:
            instrumented_evm: UniFuzz's instrumented EVM instance
            contract_address: Address of the contract to check
            bookkeeping_var_name: Name of the bookkeeping variable (e.g., 'balances')
            accounts: List of account addresses that have interacted with the contract

        Returns:
            tuple: (is_violated: bool, violation_info: dict or None)
        """
        try:
            # Get contract balance (blockchain state)
            contract_balance = instrumented_evm.get_balance(contract_address)

            # Get bookkeeping balances (internal state)
            sum_bookkeeping = self._get_sum_bookkeeping_balances_unifuzz(
                instrumented_evm,
                contract_address,
                bookkeeping_var_name,
                accounts
            )

            # Calculate current delta
            current_delta = sum_bookkeeping - contract_balance

            # First time: record the invariant constant K
            if not self.initial_state_recorded:
                self.previous_delta = current_delta
                self.initial_state_recorded = True
                logger.debug(f"Balance invariant K initialized: {current_delta}")
                return False, None

            # Check if invariant is violated (delta changed)
            if current_delta != self.previous_delta:
                violation_info = {
                    'type': 'balance_invariant_violation',
                    'expected_delta': self.previous_delta,
                    'actual_delta': current_delta,
                    'difference': current_delta - self.previous_delta,
                    'contract_balance': contract_balance,
                    'sum_bookkeeping': sum_bookkeeping
                }

                self.violations.append(violation_info)
                logger.warning(f"⚠️ Balance Invariant Violated! "
                             f"Expected delta: {self.previous_delta}, "
                             f"Actual delta: {current_delta}")
                return True, violation_info

            return False, None

        except Exception as e:
            logger.debug(f"Balance invariant check error: {e}")
            return False, None

    def _get_sum_bookkeeping_balances_unifuzz(self, instrumented_evm, contract_address, var_name, accounts):
        """
        Calculate sum of all bookkeeping balances from storage (UniFuzz version).

        For mapping(address => uint) balances, we iterate over all known addresses
        and sum their balances from contract storage.

        Args:
            instrumented_evm: UniFuzz's instrumented EVM
            contract_address: Contract address
            var_name: Bookkeeping variable name (e.g., 'balances')
            accounts: List of account addresses

        Returns:
            int: Sum of all bookkeeping balances
        """
        from eth_utils import to_canonical_address, keccak
        import eth_utils

        total = 0

        # Mapping slot for 'balances' is typically 0 in simple contracts
        # For Solidity: keccak256(key || slot)
        mapping_slot = 0  # Assume balances is the first state variable

        for account in accounts:
            try:
                # Convert account to canonical address
                if isinstance(account, str):
                    if account.startswith('0x'):
                        account_bytes = to_canonical_address(account)
                    else:
                        account_bytes = bytes.fromhex(account)
                else:
                    account_bytes = account

                # Calculate storage slot: keccak256(address + slot)
                # Solidity pads address to 32 bytes, then appends slot
                key = account_bytes.rjust(32, b'\x00') + mapping_slot.to_bytes(32, 'big')
                storage_key_bytes = keccak(key)

                # Convert bytes to int for UniFuzz's get_storage
                storage_key = int.from_bytes(storage_key_bytes, byteorder='big')

                # Read from storage
                balance_value = instrumented_evm.get_storage(contract_address, storage_key)

                if balance_value is not None and balance_value != 0:
                    total += balance_value
                    logger.debug(f"Account {eth_utils.to_hex(account_bytes)}: balance = {balance_value}")

            except Exception as e:
                logger.debug(f"Error reading balance for account {account}: {e}")
                continue

        logger.debug(f"Total bookkeeping balances: {total}")
        return total

    def _get_sum_bookkeeping_balances(self, global_state, var_name):
        """
        Calculate sum of all bookkeeping balances from storage (Mythril version - DEPRECATED).

        For mapping(address => uint) balances, we need to iterate
        over all known addresses and sum their balances.
        """
        total = 0
        storage = global_state.environment.active_account.storage

        # Get all accounts that have interacted with the contract
        accounts = self._get_known_accounts(global_state)

        for account_addr in accounts:
            # Calculate storage slot for mapping: keccak256(account_addr || mapping_slot)
            # Simplified: assume mapping is at slot 0 for demonstration
            # In production, you'd need to parse the contract to find the actual slot
            balance = self._read_mapping_value(storage, var_name, account_addr)
            if balance is not None:
                total += balance

        return total

    def _get_known_accounts(self, global_state):
        """Get list of accounts that have interacted with contract."""
        # Access the generator's account pool
        # This is simplified - in production, track accounts during execution
        accounts = []

        # Try to get accounts from environment
        if hasattr(global_state.environment, 'sender'):
            accounts.append(global_state.environment.sender)

        # Add caller
        if hasattr(global_state.environment, 'caller'):
            accounts.append(global_state.environment.caller)

        return list(set(accounts))  # Remove duplicates

    def _read_mapping_value(self, storage, mapping_name, key):
        """
        Read value from mapping in storage.

        This is a simplified implementation. In production:
        1. Parse contract ABI to find mapping slot
        2. Calculate keccak256(key || slot) for actual storage location
        3. Read from that location
        """
        try:
            # Simplified: try to read from storage directly
            # In real implementation, calculate proper storage slot
            storage_key = self._calculate_mapping_slot(mapping_name, key)
            value = storage.get(storage_key, 0)
            return value if isinstance(value, int) else 0
        except:
            return 0

    def _calculate_mapping_slot(self, mapping_name, key):
        """Calculate storage slot for mapping[key]."""
        # Simplified implementation
        # Real implementation needs keccak256(key || mapping_slot)
        return f"{mapping_name}[{key}]"

    def reset(self):
        """Reset detector state for new fuzzing round."""
        self.previous_delta = None
        self.initial_state_recorded = False
        self.violations = []


class TransactionInvariantDetector:
    """
    Transaction Invariant Detector - Based on ContraMaster (Wang et al. 2020)

    Implements Definition 4 from the paper:
        For every outgoing transaction ⟨c, r, v⟩:
        Δ(m_σ(r)) + Δ(r.bal) = 0

    Where Δ(x) = post(x) - pre(x)

    Detects:
    - Exception disorder
    - Gasless send
    - Unchecked call return values
    """

    def __init__(self):
        self.swc_id = 'SEMANTIC-TRANSACTION'
        self.severity = 'High'
        self.description = 'Transaction Invariant Violation (Semantic Oracle)'

        # Track pre/post transaction states
        self.pre_transaction_state = {}
        self.post_transaction_state = {}

        self.violations = []

    def init(self, symbolic_taint_analyzer):
        """Initialize detector."""
        self.symbolic_taint_analyzer = symbolic_taint_analyzer

    def record_pre_transaction(self, global_state, recipient_addr, bookkeeping_var):
        """
        Record state before outgoing transaction.

        Args:
            global_state: Current EVM state
            recipient_addr: Address receiving the transaction
            bookkeeping_var: Bookkeeping variable name
        """
        self.pre_transaction_state = {
            'recipient_balance': self._get_account_balance(global_state, recipient_addr),
            'recipient_bookkeeping': self._get_bookkeeping_balance(
                global_state, bookkeeping_var, recipient_addr
            ),
            'pc': global_state.mstate.pc
        }

    def detect_transaction_invariant_violation(self, global_state, recipient_addr,
                                               bookkeeping_var):
        """
        Check if transaction invariant holds after outgoing transaction.

        Returns:
            bool: True if invariant is violated
        """
        try:
            # Get post-transaction state
            post_balance = self._get_account_balance(global_state, recipient_addr)
            post_bookkeeping = self._get_bookkeeping_balance(
                global_state, bookkeeping_var, recipient_addr
            )

            # Calculate deltas
            delta_balance = post_balance - self.pre_transaction_state['recipient_balance']
            delta_bookkeeping = (post_bookkeeping -
                               self.pre_transaction_state['recipient_bookkeeping'])

            # Check invariant: Δ(bookkeeping) + Δ(balance) = 0
            invariant_sum = delta_bookkeeping + delta_balance

            if invariant_sum != 0:
                violation_info = {
                    'type': 'transaction_invariant_violation',
                    'delta_bookkeeping': delta_bookkeeping,
                    'delta_balance': delta_balance,
                    'invariant_sum': invariant_sum,
                    'recipient': recipient_addr,
                    'pc': global_state.mstate.pc,
                    'pre_state': self.pre_transaction_state.copy()
                }

                self.violations.append(violation_info)
                logger.warning(f"⚠️ Transaction Invariant Violated! "
                             f"Δ(bookkeeping)={delta_bookkeeping}, "
                             f"Δ(balance)={delta_balance}, "
                             f"Sum={invariant_sum} (should be 0)")
                return True

            return False

        except Exception as e:
            logger.debug(f"Transaction invariant check error: {e}")
            return False

    def _get_account_balance(self, global_state, address):
        """Get blockchain balance of an account."""
        try:
            # Access account from world state
            account = global_state.world_state.accounts.get(address)
            if account:
                return account.balance
            return 0
        except:
            return 0

    def _get_bookkeeping_balance(self, global_state, var_name, address):
        """Get bookkeeping balance for an address."""
        try:
            storage = global_state.environment.active_account.storage
            storage_key = self._calculate_mapping_slot(var_name, address)
            return storage.get(storage_key, 0)
        except:
            return 0

    def _calculate_mapping_slot(self, mapping_name, key):
        """Calculate storage slot for mapping[key]."""
        return f"{mapping_name}[{key}]"

    def reset(self):
        """Reset detector state."""
        self.pre_transaction_state = {}
        self.post_transaction_state = {}
        self.violations = []