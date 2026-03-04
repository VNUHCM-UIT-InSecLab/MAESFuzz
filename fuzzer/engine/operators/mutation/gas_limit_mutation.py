#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gas Limit Mutation - Based on ContraMaster (Wang et al. 2020)

Implements interval-based gas limit mutation (Section 4.2 from the paper).

Instead of random gas limits, this divides [G_intrinsic, G_max] into n intervals
and systematically tests gas limits in each interval to trigger out-of-gas exceptions
at different points in execution.

This is effective for discovering:
- Gasless send vulnerabilities
- Exception disorder
- Out-of-gas related reentrancy issues
"""

import random
from logging import getLogger

logger = getLogger(__name__)


class IntervalBasedGasMutation:
    """
    Interval-based gas limit mutation strategy.

    Divides gas range into intervals and tests systematically,
    improving coverage of out-of-gas exception scenarios.
    """

    def __init__(self, n_intervals=10):
        """
        Initialize gas mutation strategy.

        Args:
            n_intervals: Number of intervals to divide gas range into
        """
        self.n_intervals = n_intervals

        # Gas constants (for Ethereum)
        self.BASE_TX_GAS = 21000  # Base transaction cost
        self.GAS_PER_BYTE = 68    # Cost per non-zero byte in calldata
        self.BLOCK_GAS_LIMIT = 8_000_000  # Approximate block gas limit

        # Cache for calculated gas limits
        self._gas_limit_cache = {}

    def generate_gas_limits(self, transaction, function_hash=None):
        """
        Generate multiple gas limit values for systematic testing.

        Args:
            transaction: Transaction dictionary with arguments
            function_hash: Function signature hash (optional)

        Returns:
            List[int]: List of gas limits to test
        """
        # Calculate intrinsic gas (minimum required)
        g_intrinsic = self._calculate_intrinsic_gas(transaction)

        # Use block gas limit as maximum
        g_max = self.BLOCK_GAS_LIMIT

        # Generate gas limits across intervals
        gas_limits = self._generate_interval_gas_limits(g_intrinsic, g_max)

        # Add some boundary values for thorough testing
        gas_limits.extend([
            g_intrinsic,  # Minimum gas
            g_intrinsic + 1,  # Just above minimum
            g_intrinsic + 2300,  # Enough for basic fallback
            g_intrinsic + 10000,  # Small amount
            g_max // 2,  # Mid-range
            g_max,  # Maximum
        ])

        # Remove duplicates and sort
        gas_limits = sorted(list(set(gas_limits)))

        logger.debug(f"Generated {len(gas_limits)} gas limit test cases "
                    f"in range [{g_intrinsic}, {g_max}]")

        return gas_limits

    def _calculate_intrinsic_gas(self, transaction):
        """
        Calculate intrinsic gas cost for a transaction.

        Intrinsic gas consists of:
        1. Base transaction cost (21000)
        2. Data cost (68 gas per non-zero byte, 4 gas per zero byte)

        Args:
            transaction: Transaction dictionary

        Returns:
            int: Intrinsic gas cost
        """
        base_gas = self.BASE_TX_GAS

        # Calculate calldata cost
        calldata_gas = 0
        if 'arguments' in transaction:
            arguments = transaction['arguments']

            # Estimate calldata size
            # Each argument typically encoded as 32 bytes (256 bits)
            for arg in arguments:
                if arg is not None:
                    # Simplified: assume each argument is 32 bytes
                    # Real calculation needs ABI encoding
                    calldata_gas += 32 * self.GAS_PER_BYTE

        total_intrinsic = base_gas + calldata_gas

        return total_intrinsic

    def _generate_interval_gas_limits(self, g_min, g_max):
        """
        Generate gas limits distributed across n intervals.

        Args:
            g_min: Minimum gas (intrinsic)
            g_max: Maximum gas (block limit)

        Returns:
            List[int]: Gas limits sampled from each interval
        """
        if g_min >= g_max:
            return [g_min]

        gas_limits = []
        interval_size = (g_max - g_min) // self.n_intervals

        if interval_size == 0:
            return [g_min, g_max]

        for i in range(self.n_intervals):
            # Start and end of this interval
            interval_start = g_min + (i * interval_size)
            interval_end = g_min + ((i + 1) * interval_size)

            # Random value within interval
            gas_limit = random.randint(interval_start, min(interval_end, g_max))
            gas_limits.append(gas_limit)

            # Also add interval boundaries
            gas_limits.append(interval_start)
            if i == self.n_intervals - 1:
                gas_limits.append(g_max)

        return gas_limits

    def mutate_gas_limit(self, gene, generator):
        """
        Apply interval-based gas mutation to a transaction gene.

        This method can be used as a drop-in replacement or supplement
        to the existing gas mutation in mutation.py.

        Args:
            gene: Transaction gene dictionary
            generator: Individual's generator with pools

        Returns:
            List[dict]: List of mutated genes with different gas limits
        """
        # Get base transaction info
        transaction = {
            'arguments': gene.get('arguments', [])
        }

        # Generate gas limit variants
        gas_limits = self.generate_gas_limits(transaction)

        # Create mutated versions
        mutated_genes = []
        for gas_limit in gas_limits:
            mutated_gene = gene.copy()
            mutated_gene['gaslimit'] = gas_limit
            mutated_genes.append(mutated_gene)

        return mutated_genes

    def select_strategic_gas_limit(self, transaction, strategy='low'):
        """
        Select a strategic gas limit for targeted testing.

        Strategies:
        - 'low': Just above intrinsic (test early out-of-gas)
        - 'fallback': Enough for fallback function (2300 gas)
        - 'medium': Mid-range (test partial execution)
        - 'high': Near block limit (test complete execution)
        - 'random_interval': Random from a specific interval

        Args:
            transaction: Transaction dictionary
            strategy: Gas limit selection strategy

        Returns:
            int: Selected gas limit
        """
        g_intrinsic = self._calculate_intrinsic_gas(transaction)
        g_max = self.BLOCK_GAS_LIMIT

        strategies = {
            'low': g_intrinsic + random.randint(0, 5000),
            'fallback': g_intrinsic + 2300,
            'medium': (g_intrinsic + g_max) // 2,
            'high': g_max - random.randint(0, 100000),
            'random_interval': self._random_interval_gas(g_intrinsic, g_max)
        }

        return strategies.get(strategy, strategies['medium'])

    def _random_interval_gas(self, g_min, g_max):
        """Select random gas from a random interval."""
        interval_idx = random.randint(0, self.n_intervals - 1)
        interval_size = (g_max - g_min) // self.n_intervals

        interval_start = g_min + (interval_idx * interval_size)
        interval_end = min(g_min + ((interval_idx + 1) * interval_size), g_max)

        return random.randint(interval_start, interval_end)


class EnhancedGasMutation:
    """
    Enhanced gas mutation that combines interval-based and adaptive strategies.
    """

    def __init__(self):
        self.interval_mutator = IntervalBasedGasMutation(n_intervals=10)
        self.adaptive_enabled = True

        # Track which gas limits lead to interesting behaviors
        self.interesting_gas_limits = []
        self.out_of_gas_points = []

    def mutate(self, gene, generator, feedback=None):
        """
        Enhanced gas mutation with adaptive learning.

        Args:
            gene: Transaction gene
            generator: Generator instance
            feedback: Optional feedback from previous executions

        Returns:
            int: Selected gas limit
        """
        # If we have feedback about interesting gas limits, use them
        if self.adaptive_enabled and self.interesting_gas_limits:
            if random.random() < 0.3:  # 30% chance to use learned limits
                return random.choice(self.interesting_gas_limits)

        # Otherwise, use interval-based approach
        gas_limits = self.interval_mutator.generate_gas_limits({
            'arguments': gene.get('arguments', [])
        })

        # Select one strategically
        if feedback and 'coverage_plateau' in feedback:
            # If coverage plateaued, try extreme values
            return random.choice([min(gas_limits), max(gas_limits)])
        else:
            # Normal case: random from generated limits
            return random.choice(gas_limits)

    def record_interesting_gas(self, gas_limit, reason):
        """
        Record gas limits that led to interesting behaviors.

        Args:
            gas_limit: Gas limit value
            reason: Why this is interesting ('out_of_gas', 'exception', 'new_coverage')
        """
        if gas_limit not in self.interesting_gas_limits:
            self.interesting_gas_limits.append(gas_limit)
            logger.debug(f"Recorded interesting gas limit: {gas_limit} ({reason})")

            # Keep list bounded
            if len(self.interesting_gas_limits) > 50:
                self.interesting_gas_limits.pop(0)

    def record_out_of_gas_point(self, gas_limit, instruction_count):
        """
        Record where out-of-gas occurred for adaptive mutation.

        Args:
            gas_limit: Gas limit that caused out-of-gas
            instruction_count: Number of instructions executed before OOG
        """
        self.out_of_gas_points.append({
            'gas_limit': gas_limit,
            'instruction_count': instruction_count
        })
