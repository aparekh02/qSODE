"""
Evolution history tracking.

Records every iteration's parameters, metrics, agent suggestions,
and orchestrator decisions. Supports convergence detection and JSON persistence.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

from .parameter_space import ParameterSet
from .metrics import SimulationMetrics


@dataclass
class IterationRecord:
    """Record of one evolution iteration."""
    iteration: int
    parameters: Dict[str, float]
    metrics: Dict[str, Any]
    agent_suggestions: List[Dict[str, Any]]
    orchestrator_decision: Dict[str, float]
    improvement_score: float


class EvolutionHistory:
    """Tracks the full history of the evolution process."""

    def __init__(self):
        self.records: List[IterationRecord] = []

    def add(self, record: IterationRecord):
        self.records.append(record)

    def get_parameter_trajectory(self, param_name: str) -> List[float]:
        """Get how a specific parameter changed over iterations."""
        return [r.parameters.get(param_name, 0.0) for r in self.records]

    def get_metric_trajectory(self, metric_name: str) -> List[float]:
        """Get how a specific metric changed over iterations."""
        return [r.metrics.get(metric_name, 0.0) for r in self.records]

    def has_converged(self, window: int = 3, threshold: float = 0.01) -> bool:
        """Check if improvement has plateaued over recent iterations."""
        if len(self.records) < window + 1:
            return False
        recent = [r.improvement_score for r in self.records[-window:]]
        return all(abs(imp) < threshold for imp in recent)

    def save(self, path: str):
        """Save full history to JSON."""
        data = []
        for r in self.records:
            data.append({
                "iteration": r.iteration,
                "parameters": r.parameters,
                "metrics": r.metrics,
                "agent_suggestions": r.agent_suggestions,
                "orchestrator_decision": r.orchestrator_decision,
                "improvement_score": r.improvement_score,
            })
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> 'EvolutionHistory':
        """Load history from JSON."""
        history = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        for entry in data:
            record = IterationRecord(
                iteration=entry["iteration"],
                parameters=entry["parameters"],
                metrics=entry["metrics"],
                agent_suggestions=entry.get("agent_suggestions", []),
                orchestrator_decision=entry.get("orchestrator_decision", {}),
                improvement_score=entry.get("improvement_score", 0.0),
            )
            history.records.append(record)
        return history
