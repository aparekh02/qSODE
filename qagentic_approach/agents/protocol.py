"""
Structured message protocol for multi-agent communication.

All inter-agent communication uses typed dataclasses serializable to JSON.
No free-form text — agents produce and consume structured data only.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum
import json
import time


class MessageType(Enum):
    OBSERVATION = "observation"
    SUGGESTION = "suggestion"
    CONSENSUS = "consensus"
    CRITIQUE = "critique"


@dataclass
class AgentObservation:
    """What an agent observes from simulation results."""
    agent_role: str
    iteration: int
    metrics_analyzed: Dict[str, float]
    anomalies: List[str]
    confidence: float
    raw_data_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'AgentObservation':
        return cls(
            agent_role=d.get("agent_role", "unknown"),
            iteration=d.get("iteration", 0),
            metrics_analyzed=d.get("metrics_analyzed", {}),
            anomalies=d.get("anomalies", []),
            confidence=d.get("confidence", 0.5),
            raw_data_summary=d.get("raw_data_summary", {}),
        )


@dataclass
class ParameterSuggestion:
    """A single parameter adjustment suggestion."""
    param_name: str
    current_value: float
    suggested_value: float
    reasoning: str
    confidence: float

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ParameterSuggestion':
        return cls(
            param_name=d.get("param_name", ""),
            current_value=d.get("current_value", 0.0),
            suggested_value=d.get("suggested_value", 0.0),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", 0.5),
        )


@dataclass
class AgentSuggestion:
    """Full suggestion from one agent."""
    agent_role: str
    iteration: int
    observation: AgentObservation
    parameter_changes: List[ParameterSuggestion]
    overall_reasoning: str
    priority: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'AgentSuggestion':
        obs_data = d.get("observation", {})
        observation = AgentObservation.from_dict(obs_data)

        changes_data = d.get("parameter_changes", [])
        changes = [ParameterSuggestion.from_dict(c) for c in changes_data]

        return cls(
            agent_role=d.get("agent_role", "unknown"),
            iteration=d.get("iteration", 0),
            observation=observation,
            parameter_changes=changes,
            overall_reasoning=d.get("overall_reasoning", ""),
            priority=d.get("priority", 0.5),
        )


@dataclass
class AgentMessage:
    """Wrapper for all inter-agent messages."""
    msg_type: MessageType
    sender: str
    payload: Any
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        payload_dict = self.payload.to_dict() if hasattr(self.payload, 'to_dict') else self.payload
        return {
            "msg_type": self.msg_type.value,
            "sender": self.sender,
            "payload": payload_dict,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
