import math
from typing import List


class LLMController:
    """
    Adaptive controller for LLM usage probability (alpha).

    Alpha is computed per-generation using:
      - Phase (exploration / transition / exploitation)
      - Coverage improvement rate over a sliding window
      - LLM error rate (disables when too high)
    """

    def __init__(
        self,
        max_generations: int,
        exploration_ratio: float = 0.2,
        transition_ratio: float = 0.3,
        high_improvement_threshold: float = 0.5,   # % per generation
        low_improvement_threshold: float = 0.1,    # % per generation
        coverage_adjustment: float = 0.15,
        error_rate_threshold: float = 0.3,
        coverage_window: int = 5,
    ) -> None:
        self.max_generations = max_generations
        self.exploration_ratio = exploration_ratio
        self.transition_ratio = transition_ratio
        self.high_improvement_threshold = high_improvement_threshold
        self.low_improvement_threshold = low_improvement_threshold
        self.coverage_adjustment = coverage_adjustment
        self.error_rate_threshold = error_rate_threshold
        self.coverage_window = coverage_window

        self.coverage_history: List[float] = []
        self.llm_error_count = 0
        self.llm_success_count = 0

    def get_alpha(self, current_generation: int, current_coverage: float) -> float:
        """Return alpha in [0,1] for this generation."""
        self._update_coverage(current_coverage)
        alpha = self._phase_alpha(current_generation)

        if len(self.coverage_history) >= self.coverage_window:
            improvement = self._improvement_rate()
            alpha = self._adjust_by_coverage(alpha, improvement)

        if self._should_disable_llm():
            return 0.0

        return max(0.0, min(1.0, alpha))

    def record_llm_success(self) -> None:
        self.llm_success_count += 1

    def record_llm_error(self) -> None:
        self.llm_error_count += 1

    # ---------------- Internal helpers ----------------

    def _update_coverage(self, coverage: float) -> None:
        self.coverage_history.append(coverage)
        max_len = max(self.max_generations, self.coverage_window)
        if len(self.coverage_history) > max_len:
            self.coverage_history.pop(0)

    def _phase_alpha(self, g: int) -> float:
        exploration_end = int(self.exploration_ratio * self.max_generations)
        transition_end = int(self.transition_ratio * self.max_generations)

        if g <= exploration_end:
            return 1.0
        if g <= transition_end:
            length = max(1, transition_end - exploration_end)
            progress = (g - exploration_end) / length
            return max(0.0, 1.0 - progress)
        return 0.0

    def _improvement_rate(self) -> float:
        if len(self.coverage_history) < self.coverage_window:
            return 0.0
        recent = self.coverage_history[-self.coverage_window:]
        return (recent[-1] - recent[0]) / self.coverage_window

    def _adjust_by_coverage(self, alpha: float, improvement: float) -> float:
        if improvement > self.high_improvement_threshold:
            return min(1.0, alpha + self.coverage_adjustment)
        if improvement < self.low_improvement_threshold:
            return max(0.0, alpha - self.coverage_adjustment)
        return alpha

    def _should_disable_llm(self) -> bool:
        total = self.llm_error_count + self.llm_success_count
        if total == 0:
            return False
        return (self.llm_error_count / total) > self.error_rate_threshold

    def stats(self) -> dict:
        total = self.llm_error_count + self.llm_success_count
        error_rate = (self.llm_error_count / total) if total > 0 else 0.0
        return {
            "total": total,
            "success": self.llm_success_count,
            "error": self.llm_error_count,
            "error_rate": error_rate,
            "history_len": len(self.coverage_history),
            "last_cov": self.coverage_history[-1] if self.coverage_history else 0.0,
        }

