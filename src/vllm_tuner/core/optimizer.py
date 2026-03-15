from __future__ import annotations

from vllm_tuner.core.models import BenchmarkResult, ConstraintSpec, ObjectiveSpec, TrialResult
from vllm_tuner.helper.logging import get_logger


logger = get_logger()


class Optimizer:
    """Computes objective values and evaluates constraints for a trial.

    Currently a mock — returns metric values from BenchmarkResult directly.
    """

    def __init__(self, objectives: list[ObjectiveSpec], constraints: list[ConstraintSpec] | None = None):
        self._objectives = objectives
        self._constraints = constraints or []

    def compute_objective_values(self, result: TrialResult) -> list[float]:
        """Extract objective values from trial results."""
        if result.benchmark is None:
            logger.warning("Trial #{} has no benchmark results, returning zeros", result.trial_number)
            return [0.0] * len(self._objectives)

        values = []
        for obj in self._objectives:
            value = self._extract_metric(result.benchmark, obj.metric)
            values.append(value)

        logger.debug("Trial #{} objective values: {}", result.trial_number, values)
        return values

    def evaluate_constraints(self, parameters: dict) -> list[float]:
        """Evaluate constraint expressions. Returns list of constraint values (>=0 means satisfied).

        Supports simple comparison expressions like:
            "max_num_batched_tokens >= max_num_seqs"
            "gpu_memory_utilization <= 0.95"
        Returns positive value if satisfied, negative if violated.
        """
        constraint_values = []
        for constraint in self._constraints:
            value = self._eval_constraint(constraint.expression, parameters)
            constraint_values.append(value)
            if value < 0:
                logger.debug("Constraint violated: {} (value={:.2f})", constraint.expression, value)
        return constraint_values

    @staticmethod
    def _eval_constraint(expression: str, parameters: dict) -> float:
        """Evaluate a single constraint expression. Returns >=0 if satisfied."""
        import operator

        ops = {
            ">=": operator.ge,
            "<=": operator.le,
            ">": operator.gt,
            "<": operator.lt,
            "==": operator.eq,
            "!=": operator.ne,
        }

        for op_str in [">=", "<=", "!=", "==", ">", "<"]:
            if op_str in expression:
                parts = expression.split(op_str, 1)
                if len(parts) == 2:
                    lhs_str = parts[0].strip()
                    rhs_str = parts[1].strip()

                    def _resolve(token: str) -> float:
                        try:
                            return float(token)
                        except ValueError:
                            return float(parameters.get(token, 0))

                    lhs = _resolve(lhs_str)
                    rhs = _resolve(rhs_str)
                    op_fn = ops[op_str]
                    return 1.0 if op_fn(lhs, rhs) else -1.0

        logger.warning("Cannot parse constraint expression: '{}'", expression)
        return 1.0  # Unknown expression — assume satisfied

    @staticmethod
    def _extract_metric(benchmark: BenchmarkResult, metric: str) -> float:
        """Extract a named metric from benchmark results."""
        metric_map = {
            "throughput_req_per_sec": benchmark.throughput_req_per_sec,
            "output_tokens_per_sec": benchmark.output_tokens_per_sec,
            "p50_latency_ms": benchmark.p50_latency_ms,
            "p95_latency_ms": benchmark.p95_latency_ms,
            "p99_latency_ms": benchmark.p99_latency_ms,
            "ttft_ms": benchmark.ttft_ms,
            "itl_ms": benchmark.itl_ms,
        }
        value = metric_map.get(metric, 0.0)
        if metric not in metric_map:
            logger.warning("Unknown metric '{}', returning 0.0", metric)
        return value
