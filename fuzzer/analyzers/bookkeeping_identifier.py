#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bookkeeping Variable Identifier - Based on ContraMaster (Wang et al. 2020)

Automatically identifies bookkeeping variables in smart contracts.

Characteristics of bookkeeping variables (from paper Section 3.2):
1. Type: mapping(address => uint*)
2. Updated at least once in a payable function
3. In normal transactions: amount received = balance increase

This enables automatic application of semantic oracles without manual specification.
"""

import random
from logging import getLogger
from typing import List, Dict, Optional

logger = getLogger(__name__)


class BookkeepingVariableIdentifier:
    """
    Automatically identify bookkeeping variables in smart contracts.

    Success rate: ~83.6% (as reported in ContraMaster paper on 514 contracts)
    """

    def __init__(self):
        self.identified_variables = []
        self.test_transactions = 10  # Number of test transactions
        self.test_amounts = [1, 10, 100, 1000, 10000]  # Test amounts in Wei

    def identify(self, contract_interface, contract_address, accounts, web3=None):
        """
        Identify bookkeeping variables in a contract.

        Args:
            contract_interface: List of contract functions (ABI format)
            contract_address: Deployed contract address
            accounts: List of test accounts
            web3: Web3 instance for testing (optional)

        Returns:
            List[str]: Names of identified bookkeeping variables
        """
        logger.info("Identifying bookkeeping variables...")

        # Step 1: Find candidate mapping variables
        candidates = self._find_mapping_candidates(contract_interface)
        logger.info(f"Found {len(candidates)} mapping variable candidates")

        if not candidates:
            logger.warning("No mapping variables found in contract")
            return []

        # Step 2: Find payable functions
        payable_functions = self._find_payable_functions(contract_interface)
        logger.info(f"Found {len(payable_functions)} payable functions")

        # Step 3: Test each candidate
        bookkeeping_vars = []
        for candidate in candidates:
            if self._test_bookkeeping_property(
                candidate,
                payable_functions,
                accounts,
                contract_address,
                web3
            ):
                bookkeeping_vars.append(candidate)
                logger.info(f"✅ Identified bookkeeping variable: {candidate['name']}")

        self.identified_variables = bookkeeping_vars
        return [var['name'] for var in bookkeeping_vars]

    def _find_mapping_candidates(self, contract_interface):
        """
        Find all mapping(address => uint*) variables from ABI.

        Args:
            contract_interface: Contract ABI

        Returns:
            List of candidate variable info dictionaries
        """
        candidates = []

        # Common bookkeeping variable names
        common_names = [
            'balances', 'balance', 'balanceOf', '_balances',
            'userBalances', 'accountBalances', 'funds', 'deposits'
        ]

        # Look for functions that return uint and take address parameter
        # These are likely balance getter functions
        for item in contract_interface:
            if item.get('type') == 'function':
                name = item.get('name', '')
                inputs = item.get('inputs', [])
                outputs = item.get('outputs', [])

                # Check if it's a getter: function(address) returns (uint)
                if (len(inputs) == 1 and
                    len(outputs) == 1 and
                    inputs[0].get('type') == 'address' and
                    outputs[0].get('type', '').startswith('uint')):

                    # Check if name matches common patterns
                    if (name.lower() in common_names or
                        'balance' in name.lower() or
                        'fund' in name.lower() or
                        'deposit' in name.lower()):

                        candidates.append({
                            'name': name,
                            'function': item,
                            'input_type': inputs[0]['type'],
                            'output_type': outputs[0]['type']
                        })

        # Also check for public state variables (they auto-generate getters)
        # These would appear as functions in the ABI
        return candidates

    def _find_payable_functions(self, contract_interface):
        """
        Find all payable functions in the contract.

        Args:
            contract_interface: Contract ABI

        Returns:
            List of payable function info
        """
        payable_funcs = []

        for item in contract_interface:
            if item.get('type') == 'function':
                # Check if function is payable
                if item.get('stateMutability') == 'payable' or item.get('payable') == True:
                    payable_funcs.append({
                        'name': item.get('name'),
                        'inputs': item.get('inputs', []),
                        'function': item
                    })

            # Also include fallback/receive functions
            elif item.get('type') in ['fallback', 'receive']:
                if item.get('stateMutability') == 'payable' or item.get('payable') == True:
                    payable_funcs.append({
                        'name': item.get('type'),
                        'inputs': [],
                        'function': item
                    })

        return payable_funcs

    def _test_bookkeeping_property(self, candidate, payable_functions,
                                   accounts, contract_address, web3):
        """
        Test if a candidate variable has bookkeeping properties.

        Property: amount sent = increase in balance for sender address

        Args:
            candidate: Candidate variable info
            payable_functions: List of payable functions
            accounts: Test accounts
            contract_address: Contract address
            web3: Web3 instance (if None, perform static analysis only)

        Returns:
            bool: True if candidate passes bookkeeping tests
        """
        if not web3 or not payable_functions:
            # Fallback: check name heuristics
            return self._heuristic_check(candidate)

        try:
            # Test with multiple transactions
            success_count = 0
            test_count = min(self.test_transactions, len(self.test_amounts))

            for i in range(test_count):
                amount = self.test_amounts[i % len(self.test_amounts)]
                sender = accounts[i % len(accounts)]

                # Read balance before transaction
                pre_balance = self._read_balance(
                    web3, contract_address, candidate, sender
                )

                # Send payable transaction
                if self._send_test_transaction(
                    web3, contract_address, payable_functions[0], amount, sender
                ):
                    # Read balance after transaction
                    post_balance = self._read_balance(
                        web3, contract_address, candidate, sender
                    )

                    # Check if balance increased by exactly the amount sent
                    increase = post_balance - pre_balance
                    if increase == amount:
                        success_count += 1

            # Require at least 80% success rate
            success_rate = success_count / test_count
            return success_rate >= 0.8

        except Exception as e:
            logger.debug(f"Test error for {candidate['name']}: {e}")
            # Fallback to heuristic
            return self._heuristic_check(candidate)

    def _heuristic_check(self, candidate):
        """
        Heuristic check based on variable name.

        Used when runtime testing is not available.
        """
        name_lower = candidate['name'].lower()

        # Strong indicators
        strong_indicators = ['balances', 'balanceof', '_balances']
        if name_lower in strong_indicators:
            return True

        # Weak indicators
        weak_indicators = ['balance', 'fund', 'deposit', 'amount']
        return any(indicator in name_lower for indicator in weak_indicators)

    def _read_balance(self, web3, contract_address, candidate, account):
        """Read balance from contract for a specific account."""
        try:
            # Call the balance getter function
            contract = web3.eth.contract(address=contract_address)
            balance_func = contract.functions[candidate['name']]
            balance = balance_func(account).call()
            return balance
        except:
            return 0

    def _send_test_transaction(self, web3, contract_address, payable_func,
                               amount, sender):
        """Send a test payable transaction."""
        try:
            contract = web3.eth.contract(address=contract_address)
            func = contract.functions[payable_func['name']]

            # Build transaction
            tx = func().buildTransaction({
                'from': sender,
                'value': amount,
                'gas': 200000
            })

            # Send and wait for receipt
            tx_hash = web3.eth.sendTransaction(tx)
            receipt = web3.eth.waitForTransactionReceipt(tx_hash)

            return receipt['status'] == 1  # Success

        except Exception as e:
            logger.debug(f"Test transaction error: {e}")
            return False

    def get_primary_bookkeeping_variable(self):
        """
        Get the primary bookkeeping variable (most likely candidate).

        Returns:
            Optional[str]: Name of primary bookkeeping variable
        """
        if not self.identified_variables:
            return None

        # Prioritize by name
        priority_names = ['balances', 'balanceOf', '_balances', 'balance']

        for priority_name in priority_names:
            for var in self.identified_variables:
                if var['name'] == priority_name:
                    return var['name']

        # Return first identified variable
        return self.identified_variables[0]['name']

    def supports_token_standard(self, contract_interface):
        """
        Check if contract follows ERC-20 or ERC-721 token standard.

        These standards have standardized balance getters:
        - ERC-20: balanceOf(address) returns (uint256)
        - ERC-721: balanceOf(address) returns (uint256)

        Returns:
            Dict: Token standard info
        """
        has_balanceOf = False
        has_totalSupply = False
        has_transfer = False

        for item in contract_interface:
            if item.get('type') != 'function':
                continue

            name = item.get('name', '')

            if name == 'balanceOf':
                has_balanceOf = True
            elif name == 'totalSupply':
                has_totalSupply = True
            elif name == 'transfer':
                has_transfer = True

        is_erc20_like = has_balanceOf and has_totalSupply and has_transfer

        return {
            'is_token': is_erc20_like,
            'has_balanceOf': has_balanceOf,
            'has_totalSupply': has_totalSupply,
            'standard_getter': 'balanceOf' if has_balanceOf else None
        }
