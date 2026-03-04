#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execution Feedback Analyzer for LLM

Analyzes execution traces and formats them into actionable feedback
that the LLM can use to make better mutation/crossover decisions.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ExecutionFeedbackAnalyzer")


class ExecutionFeedbackAnalyzer:
    """Analyzes execution results and provides feedback for LLM"""

    def __init__(self):
        """Initialize execution feedback analyzer"""
        self.execution_history = []
        self.max_history = 50

    def analyze_execution(self, individual, execution_result: Dict) -> Dict[str, Any]:
        """
        Analyze execution result and extract actionable insights

        Args:
            individual: Individual that was executed
            execution_result: Result from execution

        Returns:
            Dict containing analysis insights
        """
        analysis = {
            "testcase_id": individual.hash if hasattr(individual, 'hash') else None,
            "sequence_length": len(individual.chromosome),
            "coverage": self._analyze_coverage(execution_result),
            "violations": self._analyze_violations(execution_result),
            "state_changes": self._analyze_state_changes(execution_result),
            "interesting_behaviors": self._identify_interesting_behaviors(execution_result),
            "suggestions": []
        }

        # Generate suggestions based on analysis
        analysis["suggestions"] = self._generate_suggestions(analysis)

        # Store in history
        self._add_to_history(analysis)

        return analysis

    def _analyze_coverage(self, execution_result: Dict) -> Dict[str, Any]:
        """Analyze coverage metrics"""
        coverage = execution_result.get("coverage", {})

        return {
            "instruction_coverage": coverage.get("instructions", 0),
            "branch_coverage": coverage.get("branches", 0),
            "new_coverage": coverage.get("new_coverage", False),
            "coverage_increase": coverage.get("coverage_increase", 0)
        }

    def _analyze_violations(self, execution_result: Dict) -> Dict[str, Any]:
        """Analyze detected violations"""
        violations = execution_result.get("violations", [])

        analysis = {
            "count": len(violations),
            "types": [],
            "severity": "none",
            "details": []
        }

        if violations:
            # Extract violation types
            violation_types = set()
            for v in violations:
                vtype = v.get("type", "unknown")
                violation_types.add(vtype)
                analysis["details"].append({
                    "type": vtype,
                    "description": v.get("description", ""),
                    "location": v.get("location", ""),
                    "transaction_index": v.get("tx_index", -1)
                })

            analysis["types"] = list(violation_types)

            # Determine severity
            critical_types = {"reentrancy", "integer_overflow", "selfdestruct"}
            if any(vtype in critical_types for vtype in violation_types):
                analysis["severity"] = "critical"
            else:
                analysis["severity"] = "medium"

        return analysis

    def _analyze_state_changes(self, execution_result: Dict) -> Dict[str, Any]:
        """Analyze state changes during execution"""
        traces = execution_result.get("traces", [])

        state_analysis = {
            "total_storage_changes": 0,
            "balance_changes": 0,
            "external_calls": 0,
            "reverts": 0,
            "self_destructs": 0
        }

        for trace in traces:
            # Storage changes
            if "storage_changes" in trace:
                state_analysis["total_storage_changes"] += len(trace["storage_changes"])

            # Balance changes
            if "balance_changes" in trace:
                state_analysis["balance_changes"] += len(trace["balance_changes"])

            # External calls
            if "external_calls" in trace:
                state_analysis["external_calls"] += len(trace["external_calls"])

            # Reverts
            if trace.get("reverted", False):
                state_analysis["reverts"] += 1

            # Self destructs
            if "selfdestruct" in trace and trace["selfdestruct"]:
                state_analysis["self_destructs"] += 1

        return state_analysis

    def _identify_interesting_behaviors(self, execution_result: Dict) -> List[str]:
        """Identify interesting behaviors worth exploring"""
        interesting = []

        violations = execution_result.get("violations", [])
        if violations:
            interesting.append(f"Found {len(violations)} violations - explore similar patterns")

        coverage = execution_result.get("coverage", {})
        if coverage.get("new_coverage", False):
            interesting.append("New coverage achieved - promising direction")

        traces = execution_result.get("traces", [])

        # Check for complex interactions
        total_external_calls = sum(
            len(trace.get("external_calls", []))
            for trace in traces
        )
        if total_external_calls > 3:
            interesting.append("Complex external call pattern - test reentrancy scenarios")

        # Check for reverts
        reverted_count = sum(1 for trace in traces if trace.get("reverted", False))
        if reverted_count > 0:
            interesting.append(f"{reverted_count} transactions reverted - test boundary conditions")

        # Check for large state changes
        total_storage = sum(
            len(trace.get("storage_changes", []))
            for trace in traces
        )
        if total_storage > 5:
            interesting.append("Significant state changes - test state manipulation attacks")

        return interesting

    def _generate_suggestions(self, analysis: Dict) -> List[str]:
        """Generate mutation/crossover suggestions based on analysis"""
        suggestions = []

        # Coverage-based suggestions
        coverage = analysis["coverage"]
        if not coverage["new_coverage"]:
            suggestions.append("Try different function sequences to increase coverage")
            suggestions.append("Mutate function arguments to explore new branches")

        # Violation-based suggestions
        violations = analysis["violations"]
        if violations["count"] > 0:
            for vtype in violations["types"]:
                if vtype == "reentrancy":
                    suggestions.append("Deepen call stack to exploit reentrancy further")
                elif vtype == "integer_overflow":
                    suggestions.append("Test more extreme values and boundary conditions")
                elif vtype == "access_control":
                    suggestions.append("Try more unauthorized account combinations")
        else:
            suggestions.append("No violations yet - try more diverse transaction patterns")

        # State change suggestions
        state = analysis["state_changes"]
        # ✅ FIX BUG #17: sequence_length is already an int, don't use len()!
        sequence_length = analysis.get("sequence_length", 0)
        if state["reverts"] > sequence_length / 2:
            suggestions.append("Many reverts - adjust input values to valid ranges")

        if state["external_calls"] > 0:
            suggestions.append("External calls detected - test reentrancy and callback scenarios")

        return suggestions

    def _add_to_history(self, analysis: Dict):
        """Add analysis to history"""
        self.execution_history.append(analysis)

        # Keep bounded
        if len(self.execution_history) > self.max_history:
            self.execution_history.pop(0)

    def get_historical_insights(self, limit: int = 10) -> str:
        """
        Get insights from execution history

        Args:
            limit: Number of recent executions to analyze

        Returns:
            Formatted insights string
        """
        if not self.execution_history:
            return "No execution history available"

        recent = self.execution_history[-limit:]

        lines = []
        lines.append("Historical Execution Insights:")
        lines.append("=" * 60)

        # Aggregate statistics
        total_violations = sum(e["violations"]["count"] for e in recent)
        violation_types = set()
        for e in recent:
            violation_types.update(e["violations"]["types"])

        new_coverage_count = sum(
            1 for e in recent
            if e["coverage"]["new_coverage"]
        )

        lines.append(f"\nLast {len(recent)} executions:")
        lines.append(f"  - Total violations: {total_violations}")
        lines.append(f"  - Violation types: {', '.join(violation_types) if violation_types else 'None'}")
        lines.append(f"  - New coverage: {new_coverage_count}/{len(recent)}")

        # Most common suggestions
        all_suggestions = []
        for e in recent:
            all_suggestions.extend(e["suggestions"])

        if all_suggestions:
            suggestion_counts = {}
            for s in all_suggestions:
                suggestion_counts[s] = suggestion_counts.get(s, 0) + 1

            top_suggestions = sorted(
                suggestion_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            lines.append("\nTop suggestions:")
            for suggestion, count in top_suggestions:
                lines.append(f"  - {suggestion} ({count}x)")

        return "\n".join(lines)

    def get_violation_patterns(self) -> Dict[str, List[Dict]]:
        """
        Extract patterns from violations in history

        Returns:
            Dict mapping violation types to patterns
        """
        patterns = {}

        for execution in self.execution_history:
            for violation_detail in execution["violations"]["details"]:
                vtype = violation_detail["type"]
                if vtype not in patterns:
                    patterns[vtype] = []

                patterns[vtype].append({
                    "description": violation_detail["description"],
                    "transaction_index": violation_detail["transaction_index"],
                    "sequence_length": execution["sequence_length"]
                })

        return patterns

    def format_feedback_for_llm(self, analysis: Dict) -> str:
        """
        Format analysis as actionable feedback for LLM

        Args:
            analysis: Analysis dict

        Returns:
            Formatted feedback string
        """
        lines = []
        lines.append("Execution Feedback:")
        lines.append("=" * 60)

        # Coverage
        coverage = analysis["coverage"]
        lines.append("\n📊 Coverage:")
        lines.append(f"  - Instructions: {coverage['instruction_coverage']}")
        lines.append(f"  - Branches: {coverage['branch_coverage']}")
        if coverage["new_coverage"]:
            lines.append("  ✓ New coverage achieved!")

        # Violations
        violations = analysis["violations"]
        if violations["count"] > 0:
            lines.append(f"\n⚠️  Violations: {violations['count']}")
            lines.append(f"  Severity: {violations['severity']}")
            lines.append(f"  Types: {', '.join(violations['types'])}")

            for detail in violations["details"]:
                lines.append(f"\n  - {detail['type']}:")
                lines.append(f"    {detail['description']}")
                if detail['transaction_index'] >= 0:
                    lines.append(f"    (at transaction #{detail['transaction_index'] + 1})")
        else:
            lines.append("\n✓ No violations detected")

        # State changes
        state = analysis["state_changes"]
        lines.append(f"\n🔄 State Changes:")
        lines.append(f"  - Storage modifications: {state['total_storage_changes']}")
        lines.append(f"  - Balance changes: {state['balance_changes']}")
        lines.append(f"  - External calls: {state['external_calls']}")
        if state['reverts'] > 0:
            lines.append(f"  - Reverts: {state['reverts']}")

        # Interesting behaviors
        if analysis["interesting_behaviors"]:
            lines.append("\n💡 Interesting Behaviors:")
            for behavior in analysis["interesting_behaviors"]:
                lines.append(f"  - {behavior}")

        # Suggestions
        if analysis["suggestions"]:
            lines.append("\n🎯 Suggestions for Next Mutation:")
            for idx, suggestion in enumerate(analysis["suggestions"][:5], 1):
                lines.append(f"  {idx}. {suggestion}")

        return "\n".join(lines)
