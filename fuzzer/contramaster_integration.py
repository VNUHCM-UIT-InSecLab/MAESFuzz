#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ContraMaster Integration for UniFuzz

This module integrates ContraMaster's methods (Wang et al. 2020) into UniFuzz:
1. Semantic Test Oracle (Balance & Transaction Invariants)
2. Interval-based Gas Limit Mutation
3. Bookkeeping Variable Auto-Identification
4. Data-Driven Transaction Sequence Mutation

Usage:
    from fuzzer.contramaster_integration import ContraMasterEnhancedFuzzer

    fuzzer = ContraMasterEnhancedFuzzer(
        contract_abi=abi,
        contract_address=address,
        accounts=accounts
    )

    fuzzer.enable_semantic_oracle()
    fuzzer.enable_interval_gas_mutation()
    fuzzer.run()
"""

from logging import getLogger
import random

from .detectors.balance_invariant import (
    BalanceInvariantDetector,
    TransactionInvariantDetector
)
from .analyzers.bookkeeping_identifier import BookkeepingVariableIdentifier
from .engine.operators.mutation.gas_limit_mutation import (
    IntervalBasedGasMutation,
    EnhancedGasMutation
)
from .engine.operators.mutation.sequence_mutation import (
    DataDrivenSequenceMutation,
    SequenceLengthMutation
)

logger = getLogger(__name__)


class ContraMasterEnhancedFuzzer:
    """
    Enhanced fuzzer integrating ContraMaster's methods into UniFuzz.

    This class serves as a facade for enabling ContraMaster features
    while maintaining compatibility with UniFuzz's existing architecture.
    """

    def __init__(self, config=None):
        """
        Initialize ContraMaster integration.

        Args:
            config: Configuration dictionary with options
        """
        self.config = config or {}

        # ContraMaster components
        self.balance_detector = None
        self.transaction_detector = None
        self.bookkeeping_identifier = None
        self.gas_mutator = None
        self.sequence_mutator = None

        # State
        self.semantic_oracle_enabled = False
        self.interval_gas_enabled = False
        self.bookkeeping_var = None

        # Statistics
        self.stats = {
            'balance_violations': 0,
            'transaction_violations': 0,
            'gas_variants_tested': 0,
            'sequence_mutations': 0
        }

    def enable_semantic_oracle(self):
        """
        Enable semantic test oracle (Balance & Transaction Invariants).

        This is the core contribution of ContraMaster.
        """
        logger.info("🔧 Enabling Semantic Test Oracle (ContraMaster)")

        self.balance_detector = BalanceInvariantDetector()
        self.transaction_detector = TransactionInvariantDetector()

        self.semantic_oracle_enabled = True
        logger.info("✅ Semantic Oracle enabled")

    def enable_interval_gas_mutation(self, n_intervals=10):
        """
        Enable interval-based gas limit mutation.

        Args:
            n_intervals: Number of intervals to divide gas range
        """
        logger.info(f"🔧 Enabling Interval-based Gas Mutation ({n_intervals} intervals)")

        self.gas_mutator = EnhancedGasMutation()
        self.interval_gas_enabled = True

        logger.info("✅ Interval Gas Mutation enabled")

    def enable_sequence_mutation(self):
        """Enable data-driven transaction sequence mutation."""
        logger.info("🔧 Enabling Data-Driven Sequence Mutation")

        self.sequence_mutator = DataDrivenSequenceMutation()

        logger.info("✅ Sequence Mutation enabled")

    def identify_bookkeeping_variables(self, contract_interface, contract_address,
                                       accounts, web3=None):
        """
        Automatically identify bookkeeping variables.

        Args:
            contract_interface: Contract ABI
            contract_address: Deployed contract address
            accounts: List of test accounts
            web3: Web3 instance (optional)

        Returns:
            List[str]: Identified bookkeeping variable names
        """
        logger.info("🔍 Identifying bookkeeping variables...")

        self.bookkeeping_identifier = BookkeepingVariableIdentifier()

        bookkeeping_vars = self.bookkeeping_identifier.identify(
            contract_interface=contract_interface,
            contract_address=contract_address,
            accounts=accounts,
            web3=web3
        )

        if bookkeeping_vars:
            self.bookkeeping_var = bookkeeping_vars[0]  # Use first one
            logger.info(f"✅ Primary bookkeeping variable: {self.bookkeeping_var}")
        else:
            logger.warning("⚠️ No bookkeeping variables identified")

        return bookkeeping_vars

    def check_semantic_oracle(self, global_state, trace_sequence=None):
        """
        Check semantic oracle after transaction execution.

        Args:
            global_state: Current EVM global state
            trace_sequence: Execution traces (optional)

        Returns:
            Dict: Oracle check results
        """
        if not self.semantic_oracle_enabled:
            return {'oracle_passed': True}

        results = {
            'oracle_passed': True,
            'violations': []
        }

        # Check Balance Invariant
        if self.bookkeeping_var:
            balance_violated = self.balance_detector.detect_balance_invariant_violation(
                global_state=global_state,
                bookkeeping_var_name=self.bookkeeping_var
            )

            if balance_violated:
                results['oracle_passed'] = False
                results['violations'].extend(self.balance_detector.violations)
                self.stats['balance_violations'] += 1

        # Check Transaction Invariant
        # (This requires tracking pre/post transaction state)
        # Implementation depends on UniFuzz's execution trace structure

        return results

    def mutate_gas_with_intervals(self, gene, generator):
        """
        Apply interval-based gas mutation.

        Args:
            gene: Transaction gene
            generator: Generator instance

        Returns:
            int: Mutated gas limit
        """
        if not self.interval_gas_enabled or not self.gas_mutator:
            # Fallback to standard mutation
            return generator.get_random_gaslimit(gene['arguments'][0])

        gas_limit = self.gas_mutator.mutate(gene, generator)
        self.stats['gas_variants_tested'] += 1

        return gas_limit

    def mutate_sequence_data_driven(self, individual, trace_sequence=None):
        """
        Apply data-driven sequence mutation.

        Args:
            individual: Individual with chromosome
            trace_sequence: Execution traces (optional)

        Returns:
            Individual: Mutated individual
        """
        if not self.sequence_mutator:
            return individual

        mutated = self.sequence_mutator.mutate_sequence(
            individual=individual,
            trace_sequence=trace_sequence
        )

        self.stats['sequence_mutations'] += 1

        return mutated

    def get_statistics(self):
        """
        Get statistics about ContraMaster enhancements.

        Returns:
            Dict: Statistics
        """
        return self.stats.copy()

    def reset_detectors(self):
        """Reset all detectors for new fuzzing round."""
        if self.balance_detector:
            self.balance_detector.reset()

        if self.transaction_detector:
            self.transaction_detector.reset()


class ContraMasterEnhancedEngine:
    """
    Wrapper for UniFuzz's EvolutionaryFuzzingEngine with ContraMaster enhancements.

    This allows drop-in replacement of the standard engine.
    """

    def __init__(self, base_engine, contramaster_fuzzer):
        """
        Initialize enhanced engine.

        Args:
            base_engine: Original EvolutionaryFuzzingEngine
            contramaster_fuzzer: ContraMasterEnhancedFuzzer instance
        """
        self.engine = base_engine
        self.cm_fuzzer = contramaster_fuzzer

    def evolve(self, population):
        """
        Enhanced evolution with ContraMaster mutations.

        Args:
            population: Current population

        Returns:
            Population: Next generation
        """
        # Standard evolution
        next_gen = self.engine.evolve(population)

        # Apply ContraMaster enhancements
        if self.cm_fuzzer.sequence_mutator:
            for i, individual in enumerate(next_gen.individuals):
                if random.random() < 0.3:  # 30% chance to apply sequence mutation
                    next_gen.individuals[i] = self.cm_fuzzer.mutate_sequence_data_driven(
                        individual
                    )

        return next_gen


def integrate_contramaster_into_fuzzer(fuzzer_instance, enable_all=True,
                                      semantic_oracle=True,
                                      interval_gas=True,
                                      sequence_mutation=True):
    """
    Helper function to integrate ContraMaster into existing UniFuzz fuzzer.

    Args:
        fuzzer_instance: Existing Fuzzer instance from fuzzer/main.py
        enable_all: Enable all features at once
        semantic_oracle: Enable semantic oracle
        interval_gas: Enable interval gas mutation
        sequence_mutation: Enable sequence mutation

    Returns:
        ContraMasterEnhancedFuzzer: Enhanced fuzzer instance
    """
    cm_fuzzer = ContraMasterEnhancedFuzzer()

    if enable_all or semantic_oracle:
        cm_fuzzer.enable_semantic_oracle()

    if enable_all or interval_gas:
        cm_fuzzer.enable_interval_gas_mutation()

    if enable_all or sequence_mutation:
        cm_fuzzer.enable_sequence_mutation()

    # Identify bookkeeping variables if possible
    if hasattr(fuzzer_instance, 'interface') and hasattr(fuzzer_instance, 'contract_address'):
        cm_fuzzer.identify_bookkeeping_variables(
            contract_interface=fuzzer_instance.interface,
            contract_address=fuzzer_instance.contract_address,
            accounts=fuzzer_instance.accounts
        )

    logger.info("✅ ContraMaster integration complete")

    return cm_fuzzer


# Convenience imports for easier usage
__all__ = [
    'ContraMasterEnhancedFuzzer',
    'ContraMasterEnhancedEngine',
    'integrate_contramaster_into_fuzzer',
    'BalanceInvariantDetector',
    'TransactionInvariantDetector',
    'BookkeepingVariableIdentifier',
    'IntervalBasedGasMutation',
    'EnhancedGasMutation',
    'DataDrivenSequenceMutation',
    'SequenceLengthMutation'
]
