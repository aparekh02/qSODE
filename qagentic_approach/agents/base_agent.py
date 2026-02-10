"""
Base agent class for Claude-powered multi-agent collaboration.

Each agent specializes in a domain of the simulation, receives
relevant metrics, and produces structured parameter suggestions
via the Anthropic API.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import anthropic

from .protocol import AgentObservation, ParameterSuggestion, AgentSuggestion
from ..evolution.parameter_space import ParameterSet


class BaseAgent(ABC):
    """
    Abstract base class for all Claude-powered agents.

    Subclasses define:
    - role: agent's identity string
    - get_system_prompt(): domain-specific instructions
    - extract_relevant_data(): which metrics this agent cares about
    - get_owned_parameters(): which parameters this agent can adjust
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.role: str = ""

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt defining this agent's expertise."""
        pass

    @abstractmethod
    def extract_relevant_data(self, sim_metrics: Dict[str, Any],
                               current_params: ParameterSet) -> Dict[str, Any]:
        """Extract the data slice this agent cares about."""
        pass

    @abstractmethod
    def get_owned_parameters(self) -> List[str]:
        """Return list of parameter names this agent can suggest changes to."""
        pass

    def analyze(self, sim_metrics: Dict[str, Any],
                current_params: ParameterSet,
                iteration: int) -> AgentSuggestion:
        """
        Analyze simulation results and suggest parameter changes.

        Calls the Claude API with structured data, parses the JSON response.
        """
        relevant_data = self.extract_relevant_data(sim_metrics, current_params)
        owned = self.get_owned_parameters()

        # Build the owned parameters subset with current values and bounds
        bounds = ParameterSet.get_bounds()
        param_dict = current_params.to_dict()
        owned_params = {
            k: {"current": param_dict[k], "bounds": list(bounds[k])}
            for k in owned if k in param_dict
        }

        user_message = json.dumps({
            "iteration": iteration,
            "your_parameters": owned_params,
            "simulation_data": relevant_data,
            "instruction": (
                "Analyze the simulation data. For each parameter you control, "
                "decide whether to adjust it. Return a JSON object with these fields: "
                "agent_role (string), iteration (int), "
                "observation (object with: agent_role, iteration, metrics_analyzed, anomalies, confidence), "
                "parameter_changes (array of objects with: param_name, current_value, suggested_value, reasoning, confidence), "
                "overall_reasoning (string), priority (float 0-1). "
                "Only suggest changes you are confident about. If a parameter seems fine, omit it."
            ),
        }, indent=2)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": user_message}],
        )

        return self._parse_response(response, iteration, current_params)

    def _parse_response(self, response, iteration: int,
                        current_params: ParameterSet) -> AgentSuggestion:
        """Parse Claude's response into a structured AgentSuggestion."""
        text = response.content[0].text

        # Try to extract JSON from response (may be wrapped in ```json blocks)
        json_str = self._extract_json(text)

        try:
            data = json.loads(json_str)
            suggestion = AgentSuggestion.from_dict(data)
            # Ensure agent_role is correct
            suggestion.agent_role = self.role
            suggestion.iteration = iteration
            # Filter to only owned parameters
            owned = set(self.get_owned_parameters())
            suggestion.parameter_changes = [
                pc for pc in suggestion.parameter_changes
                if pc.param_name in owned
            ]
            return suggestion
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: return empty suggestion
            return AgentSuggestion(
                agent_role=self.role,
                iteration=iteration,
                observation=AgentObservation(
                    agent_role=self.role,
                    iteration=iteration,
                    metrics_analyzed={},
                    anomalies=["Failed to parse Claude response"],
                    confidence=0.0,
                ),
                parameter_changes=[],
                overall_reasoning=f"Parse error. Raw response: {text[:200]}",
                priority=0.0,
            )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text that may contain markdown code fences."""
        # Try to find ```json ... ``` blocks
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try to find raw JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0).strip()
        return text.strip()
