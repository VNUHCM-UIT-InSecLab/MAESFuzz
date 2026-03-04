#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transaction Sequence Mutation - Based on ContraMaster (Wang et al. 2020)

Implements data-driven transaction sequence mutation (Section 4.2 from the paper).

Key insight: Use data dependencies to guide sequence mutations.
If two transactions operate on the same state variable, switching their order
may expose vulnerabilities (e.g., deposit → withdraw for reentrancy).

This corresponds to Algorithm 1, Lines 19-21 in the paper.
"""

import random
from logging import getLogger

logger = getLogger(__name__)


class DataDrivenSequenceMutation:
    """
    Transaction sequence mutation guided by data dependencies.

    Analyzes which transactions share state variables and strategically
    mutates their ordering to expose vulnerabilities.
    """

    def __init__(self):
        self.data_dependencies = []
        self.interesting_sequences = []

    def analyze_data_dependencies(self, trace_sequence):
        """
        Analyze data dependencies between transactions in a sequence.

        Args:
            trace_sequence: List of execution traces for each transaction

        Returns:
            List[dict]: Data dependency information
        """
        dependencies = []

        for i, trace1 in enumerate(trace_sequence):
            for j, trace2 in enumerate(trace_sequence[i+1:], start=i+1):
                # Find shared state variables between two transactions
                shared_vars = self._find_shared_state_variables(trace1, trace2)

                if shared_vars:
                    dep_info = {
                        'tx_index_1': i,
                        'tx_index_2': j,
                        'shared_variables': shared_vars,
                        'should_switch': True,  # Candidate for order switching
                        'dependency_type': self._classify_dependency(trace1, trace2, shared_vars)
                    }
                    dependencies.append(dep_info)

        self.data_dependencies = dependencies
        return dependencies

    def _find_shared_state_variables(self, trace1, trace2):
        """
        Find state variables accessed by both transactions.

        Args:
            trace1, trace2: Execution traces

        Returns:
            List[str]: Shared state variable names/slots
        """
        # Extract storage accesses from traces
        storage1 = self._extract_storage_accesses(trace1)
        storage2 = self._extract_storage_accesses(trace2)

        # Find intersection
        shared = set(storage1.keys()) & set(storage2.keys())

        shared_info = []
        for slot in shared:
            shared_info.append({
                'slot': slot,
                'ops1': storage1[slot],  # Operations in trace1 (read/write)
                'ops2': storage2[slot]   # Operations in trace2
            })

        return shared_info

    def _extract_storage_accesses(self, trace):
        """
        Extract storage read/write operations from execution trace.

        Args:
            trace: Execution trace data

        Returns:
            Dict[str, List[str]]: Mapping from storage slot to operations
        """
        storage_accesses = {}

        if not trace or 'storage_accesses' not in trace:
            return storage_accesses

        for access in trace.get('storage_accesses', []):
            slot = access.get('slot')
            op_type = access.get('type')  # 'SLOAD' or 'SSTORE'

            if slot not in storage_accesses:
                storage_accesses[slot] = []

            storage_accesses[slot].append(op_type)

        return storage_accesses

    def _classify_dependency(self, trace1, trace2, shared_vars):
        """
        Classify the type of data dependency.

        Types:
        - RAW (Read-After-Write): trace1 writes, trace2 reads
        - WAR (Write-After-Read): trace1 reads, trace2 writes
        - WAW (Write-After-Write): Both write to same variable

        Args:
            trace1, trace2: Execution traces
            shared_vars: Shared variables info

        Returns:
            str: Dependency type
        """
        for var_info in shared_vars:
            ops1 = var_info['ops1']
            ops2 = var_info['ops2']

            # Check for writes in both
            has_write1 = 'SSTORE' in ops1
            has_write2 = 'SSTORE' in ops2
            has_read1 = 'SLOAD' in ops1
            has_read2 = 'SLOAD' in ops2

            if has_write1 and has_read2:
                return 'RAW'  # Read-After-Write
            elif has_read1 and has_write2:
                return 'WAR'  # Write-After-Read
            elif has_write1 and has_write2:
                return 'WAW'  # Write-After-Write

        return 'Unknown'

    def mutate_sequence(self, individual, trace_sequence=None):
        """
        Mutate transaction sequence based on data dependencies.

        Strategies (from ContraMaster):
        1. Switch order of transactions with data dependencies
        2. Insert new transactions
        3. Remove transactions

        Args:
            individual: Individual with chromosome (transaction sequence)
            trace_sequence: Optional execution traces for analysis

        Returns:
            Individual: Mutated individual
        """
        chromosome = individual.chromosome[:]

        if len(chromosome) < 2:
            # Need at least 2 transactions to mutate sequence
            return individual

        # Analyze dependencies if traces provided
        if trace_sequence:
            self.analyze_data_dependencies(trace_sequence)

        # Choose mutation strategy
        strategy = random.choice([
            'switch_dependent',
            'switch_random',
            'insert',
            'remove'
        ])

        if strategy == 'switch_dependent' and self.data_dependencies:
            chromosome = self._switch_dependent_transactions(chromosome)

        elif strategy == 'switch_random':
            chromosome = self._switch_random_transactions(chromosome)

        elif strategy == 'insert':
            chromosome = self._insert_transaction(chromosome, individual)

        elif strategy == 'remove' and len(chromosome) > 1:
            chromosome = self._remove_transaction(chromosome)

        # Update individual's chromosome
        individual.chromosome = chromosome
        return individual

    def _switch_dependent_transactions(self, chromosome):
        """
        Switch order of transactions that share state variables.

        This is the key mutation from ContraMaster (Algorithm 1, Line 20).
        """
        if not self.data_dependencies:
            return chromosome

        # Select a dependency to exploit
        dep = random.choice(self.data_dependencies)

        idx1 = dep['tx_index_1']
        idx2 = dep['tx_index_2']

        # Ensure indices are valid
        if idx1 >= len(chromosome) or idx2 >= len(chromosome):
            return chromosome

        # Switch the transactions
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

        logger.debug(f"Switched transactions {idx1} ↔ {idx2} "
                    f"(shared variables: {len(dep['shared_variables'])})")

        return chromosome

    def _switch_random_transactions(self, chromosome):
        """Switch two random transactions in sequence."""
        if len(chromosome) < 2:
            return chromosome

        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

        return chromosome

    def _insert_transaction(self, chromosome, individual):
        """
        Insert a new transaction at a random position.

        Args:
            chromosome: Current transaction sequence
            individual: Individual instance (for generating new transaction)

        Returns:
            List: Modified chromosome
        """
        # Generate a new transaction
        new_tx = self._generate_transaction(individual)

        # Random insertion point
        insert_pos = random.randint(0, len(chromosome))

        chromosome.insert(insert_pos, new_tx)

        logger.debug(f"Inserted transaction at position {insert_pos}")

        return chromosome

    def _remove_transaction(self, chromosome):
        """Remove a random transaction from sequence."""
        if len(chromosome) <= 1:
            return chromosome

        remove_idx = random.randint(0, len(chromosome) - 1)
        chromosome.pop(remove_idx)

        logger.debug(f"Removed transaction at position {remove_idx}")

        return chromosome

    def _generate_transaction(self, individual):
        """
        Generate a new transaction gene.

        Args:
            individual: Individual instance with generator

        Returns:
            dict: New transaction gene
        """
        # Get random function
        function_hash = random.choice(list(individual.generator.interface.keys()))

        # Generate transaction using individual's generator
        tx = {
            'account': individual.generator.get_random_account(function_hash),
            'contract': individual.generator.contract_address,
            'amount': individual.generator.get_random_amount(function_hash),
            'gaslimit': individual.generator.get_random_gaslimit(function_hash),
            'arguments': [function_hash],
            'timestamp': individual.generator.get_random_timestamp(function_hash),
            'blocknumber': individual.generator.get_random_blocknumber(function_hash),
            'balance': individual.generator.get_random_balance(function_hash),
            'call_return': {},
            'extcodesize': {},
            'returndatasize': {}
        }

        # Generate arguments for the function
        if function_hash in individual.generator.interface:
            for arg_idx, arg_type in enumerate(individual.generator.interface[function_hash]):
                arg_value = individual.generator.get_random_argument(
                    arg_type, function_hash, arg_idx
                )
                tx['arguments'].append(arg_value)

        return tx

    def record_interesting_sequence(self, sequence, reason):
        """
        Record transaction sequences that led to interesting behaviors.

        Args:
            sequence: Transaction sequence (chromosome)
            reason: Why this is interesting (e.g., 'vulnerability_found')
        """
        sequence_copy = [tx.copy() for tx in sequence]

        self.interesting_sequences.append({
            'sequence': sequence_copy,
            'reason': reason
        })

        # Keep bounded
        if len(self.interesting_sequences) > 20:
            self.interesting_sequences.pop(0)

    def get_similar_sequence(self):
        """
        Get a sequence similar to previously interesting ones.

        Returns:
            List: Transaction sequence or None
        """
        if not self.interesting_sequences:
            return None

        base_seq = random.choice(self.interesting_sequences)['sequence']

        # Make slight modifications
        modified_seq = [tx.copy() for tx in base_seq]

        # Small random mutation
        if len(modified_seq) >= 2 and random.random() < 0.5:
            idx1, idx2 = random.sample(range(len(modified_seq)), 2)
            modified_seq[idx1], modified_seq[idx2] = modified_seq[idx2], modified_seq[idx1]

        return modified_seq


class SequenceLengthMutation:
    """
    Mutation for varying transaction sequence lengths.

    Short sequences: Fast execution, good for simple bugs
    Long sequences: Complex interactions, good for state-dependent bugs
    """

    def __init__(self, min_length=1, max_length=5):
        self.min_length = min_length
        self.max_length = max_length

    def adjust_length(self, chromosome, target_length=None):
        """
        Adjust chromosome length to target or random length.

        Args:
            chromosome: Current transaction sequence
            target_length: Desired length (or None for random)

        Returns:
            List: Adjusted chromosome
        """
        current_length = len(chromosome)

        if target_length is None:
            target_length = random.randint(self.min_length, self.max_length)

        if target_length > current_length:
            # Need to add transactions
            return self._extend_sequence(chromosome, target_length)
        elif target_length < current_length:
            # Need to remove transactions
            return self._shorten_sequence(chromosome, target_length)
        else:
            return chromosome

    def _extend_sequence(self, chromosome, target_length):
        """Extend sequence by duplicating and mutating existing transactions."""
        while len(chromosome) < target_length:
            # Duplicate a random transaction with slight mutation
            if chromosome:
                tx_to_duplicate = random.choice(chromosome).copy()
                # TODO: Add slight mutation here
                chromosome.append(tx_to_duplicate)
            else:
                break

        return chromosome

    def _shorten_sequence(self, chromosome, target_length):
        """Shorten sequence by removing random transactions."""
        while len(chromosome) > target_length and len(chromosome) > 1:
            remove_idx = random.randint(0, len(chromosome) - 1)
            chromosome.pop(remove_idx)

        return chromosome
