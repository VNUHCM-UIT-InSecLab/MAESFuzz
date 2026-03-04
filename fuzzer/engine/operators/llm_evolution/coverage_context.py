#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coverage Context Provider for LLM Fuzzing

Provides coverage information to help LLM make informed decisions:
- Which branches have been covered
- Which branches remain uncovered
- Coverage trends across generations
- High-value targets for increasing coverage
"""

import logging
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict

logger = logging.getLogger("CoverageContext")


class CoverageContextProvider:
    """Tracks and provides coverage information for LLM operators"""

    def __init__(self):
        """Initialize coverage context tracker"""
        self.covered_pcs: Set[int] = set()
        self.covered_branches: Set[tuple] = set()  # (pc, branch_id)
        self.uncovered_pcs: Set[int] = set()
        self.uncovered_branches: Set[tuple] = set()

        # Generation history
        self.generation_history: List[Dict] = []
        self.current_generation = 0

        # Statistics
        self.total_pcs = 0
        self.total_branches = 0
        self.best_coverage_individual = None
        self.best_coverage_value = 0.0

        # Branch coverage map
        self.pc_to_branches: Dict[int, List[int]] = defaultdict(list)

        logger.info("CoverageContextProvider initialized")

    def update_from_individual(self, individual):
        """
        Update coverage from an individual's execution

        Args:
            individual: Individual with execution results
        """
        if not hasattr(individual, 'execution_result'):
            return

        result = individual.execution_result

        # Track covered PCs
        if 'covered_pcs' in result:
            new_pcs = set(result['covered_pcs'])
            newly_covered = new_pcs - self.covered_pcs

            if newly_covered:
                logger.debug(f"Found {len(newly_covered)} newly covered PCs")
                self.covered_pcs.update(newly_covered)

        # Track branches
        if 'covered_branches' in result:
            for pc, branch_id in result['covered_branches']:
                self.covered_branches.add((pc, branch_id))
                self.pc_to_branches[pc].append(branch_id)

        # Update best individual
        coverage = getattr(individual, 'code_coverage', 0.0)
        if coverage > self.best_coverage_value:
            self.best_coverage_value = coverage
            self.best_coverage_individual = individual

    def update_from_population(self, population):
        """
        Update coverage from entire population

        Args:
            population: Population object
        """
        for individual in population.individuals:
            self.update_from_individual(individual)

        # Calculate coverage statistics
        coverage_values = [
            getattr(ind, 'code_coverage', 0.0)
            for ind in population.individuals
        ]

        fitness_values = [
            getattr(ind, 'fitness', 0.0)
            for ind in population.individuals
        ]

        gen_stats = {
            'generation': self.current_generation,
            'avg_coverage': sum(coverage_values) / len(coverage_values) if coverage_values else 0,
            'max_coverage': max(coverage_values) if coverage_values else 0,
            'min_coverage': min(coverage_values) if coverage_values else 0,
            'avg_fitness': sum(fitness_values) / len(fitness_values) if fitness_values else 0,
            'max_fitness': max(fitness_values) if fitness_values else 0,
            'total_covered_pcs': len(self.covered_pcs),
            'total_covered_branches': len(self.covered_branches)
        }

        self.generation_history.append(gen_stats)
        logger.info(f"Generation {self.current_generation}: "
                   f"Coverage={gen_stats['max_coverage']:.2f}%, "
                   f"PCs={len(self.covered_pcs)}")

    def set_total_coverage_targets(self, total_pcs: int, total_branches: int,
                                   all_pcs: Set[int], all_branches: Set[tuple]):
        """
        Set total coverage targets (from static analysis)

        Args:
            total_pcs: Total number of program counters
            total_branches: Total number of branches
            all_pcs: Set of all PC values
            all_branches: Set of all branch tuples
        """
        self.total_pcs = total_pcs
        self.total_branches = total_branches

        # Calculate uncovered
        self.uncovered_pcs = all_pcs - self.covered_pcs
        self.uncovered_branches = all_branches - self.covered_branches

        logger.info(f"Coverage targets set: {total_pcs} PCs, {total_branches} branches")
        logger.info(f"Currently uncovered: {len(self.uncovered_pcs)} PCs, "
                   f"{len(self.uncovered_branches)} branches")

    def get_coverage_summary(self) -> str:
        """
        Get human-readable coverage summary for LLM

        Returns:
            Formatted coverage summary
        """
        if not self.generation_history:
            return "No coverage data available yet (Generation 0)"

        latest = self.generation_history[-1]

        summary = f"""### Coverage Status (Generation {self.current_generation})

**Overall Coverage:**
- Code Coverage: {latest['max_coverage']:.2f}%
- Covered PCs: {len(self.covered_pcs)}/{self.total_pcs if self.total_pcs else '?'}
- Covered Branches: {len(self.covered_branches)}/{self.total_branches if self.total_branches else '?'}

**Progress Trends:**
"""

        # Show last 3 generations
        for gen_stat in self.generation_history[-3:]:
            summary += f"- Gen {gen_stat['generation']}: {gen_stat['max_coverage']:.1f}% coverage, "
            summary += f"{gen_stat['max_fitness']:.1f} max fitness\n"

        # Coverage improvement
        if len(self.generation_history) > 1:
            prev = self.generation_history[-2]
            improvement = latest['max_coverage'] - prev['max_coverage']
            if improvement > 0:
                summary += f"\n✅ Coverage improved by {improvement:.2f}% in last generation"
            elif improvement == 0:
                summary += f"\n⚠️  Coverage plateaued - need new strategies!"
            else:
                summary += f"\n⚠️  Coverage decreased slightly"

        return summary

    def get_uncovered_targets(self, limit: int = 10) -> str:
        """
        Get list of uncovered targets for LLM to aim for

        Args:
            limit: Maximum number of targets to return

        Returns:
            Formatted list of uncovered targets
        """
        if not self.uncovered_pcs and not self.uncovered_branches:
            return "All known targets covered! Focus on deeper exploration."

        targets = "### High-Priority Uncovered Targets\n\n"

        # Show some uncovered PCs
        if self.uncovered_pcs:
            sample_pcs = list(self.uncovered_pcs)[:limit]
            targets += f"**Uncovered Program Counters** ({len(self.uncovered_pcs)} remaining):\n"
            for pc in sample_pcs:
                targets += f"- PC {pc:#06x}\n"

        # Show some uncovered branches
        if self.uncovered_branches:
            sample_branches = list(self.uncovered_branches)[:limit]
            targets += f"\n**Uncovered Branches** ({len(self.uncovered_branches)} remaining):\n"
            for pc, branch_id in sample_branches:
                targets += f"- PC {pc:#06x}, Branch {branch_id}\n"

        targets += "\n💡 **Suggestion:** Mutate testcases to trigger these uncovered paths"

        return targets

    def get_best_individuals_summary(self, population, top_k: int = 3) -> str:
        """
        Get summary of best performing individuals

        Args:
            population: Current population
            top_k: Number of top individuals to summarize

        Returns:
            Formatted summary of best individuals
        """
        # Sort by coverage
        sorted_individuals = sorted(
            population.individuals,
            key=lambda x: getattr(x, 'code_coverage', 0.0),
            reverse=True
        )

        summary = f"### Top {top_k} Individuals (by Coverage)\n\n"

        for i, ind in enumerate(sorted_individuals[:top_k], 1):
            coverage = getattr(ind, 'code_coverage', 0.0)
            fitness = getattr(ind, 'fitness', 0.0)
            seq_len = len(ind.chromosome)

            summary += f"**#{i}:** {coverage:.2f}% coverage, "
            summary += f"fitness={fitness:.1f}, {seq_len} transactions\n"

            # Show first few transactions
            if seq_len > 0:
                summary += "  Sequence: "
                funcs = []
                for tx in ind.chromosome[:3]:
                    func_name = tx.get('function', '?')
                    funcs.append(func_name)
                summary += " → ".join(funcs)
                if seq_len > 3:
                    summary += f" → ... ({seq_len - 3} more)"
                summary += "\n"

        summary += "\n💡 **Learn from these:** They achieved high coverage - try similar strategies"

        return summary

    def get_generation_context(self) -> str:
        """
        Get overall generation context for LLM

        Returns:
            Formatted generation context
        """
        if not self.generation_history:
            return "Generation 0: Initial population (no history yet)"

        latest = self.generation_history[-1]

        context = f"""### Generation Context

**Current State:**
- Generation: {self.current_generation}
- Population Coverage: {latest['avg_coverage']:.2f}% avg, {latest['max_coverage']:.2f}% max
- Population Fitness: {latest['avg_fitness']:.1f} avg, {latest['max_fitness']:.1f} max

**Evolution Progress:**
"""

        if len(self.generation_history) >= 2:
            # Calculate trend
            recent_gens = self.generation_history[-5:]
            coverages = [g['max_coverage'] for g in recent_gens]

            if len(coverages) >= 2:
                trend = coverages[-1] - coverages[0]
                if trend > 1.0:
                    context += "📈 Coverage is improving (keep current strategies)\n"
                elif trend > 0:
                    context += "➡️  Coverage improving slowly (try more aggressive mutations)\n"
                else:
                    context += "📉 Coverage stagnant (need NEW strategies!)\n"

        return context

    def next_generation(self):
        """Mark transition to next generation"""
        self.current_generation += 1
        logger.info(f"Advancing to generation {self.current_generation}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get coverage statistics"""
        return {
            'generation': self.current_generation,
            'covered_pcs': len(self.covered_pcs),
            'covered_branches': len(self.covered_branches),
            'uncovered_pcs': len(self.uncovered_pcs),
            'uncovered_branches': len(self.uncovered_branches),
            'best_coverage': self.best_coverage_value,
            'total_generations': len(self.generation_history)
        }
