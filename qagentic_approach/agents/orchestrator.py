"""
Orchestrator: aggregates agent suggestions into final parameter updates.

Resolves conflicts via confidence-weighted averaging or Claude arbitration.
Applies learning rate damping and enforces parameter bounds.
"""

import json
import re
from typing import List, Dict, Optional
from collections import defaultdict

import anthropic

from .protocol import AgentSuggestion, ParameterSuggestion
from ..evolution.parameter_space import ParameterSet


class Orchestrator:
    """
    Aggregates agent suggestions into a single parameter update.

    Strategy:
    1. Group suggestions by parameter name.
    2. If agents agree on direction, use confidence-weighted average.
    3. If agents conflict, use Claude arbitration or highest-confidence pick.
    4. Apply learning rate: new = current + lr * (suggested - current).
    5. Clamp all values to physical bounds.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 learning_rate: float = 0.5,
                 use_claude_arbitration: bool = True):
        self.api_key = api_key
        self.model = model
        self.learning_rate = learning_rate
        self.use_claude_arbitration = use_claude_arbitration
        if use_claude_arbitration:
            self.client = anthropic.Anthropic(api_key=api_key)

    def synthesize(self, suggestions: List[AgentSuggestion],
                   current_params: ParameterSet) -> ParameterSet:
        """Combine all agent suggestions into a single parameter update."""
        # Group all parameter suggestions by param name
        param_suggestions: Dict[str, List[ParameterSuggestion]] = defaultdict(list)

        for suggestion in suggestions:
            for pc in suggestion.parameter_changes:
                param_suggestions[pc.param_name].append(pc)

        # Start from current params
        new_dict = current_params.to_dict()
        bounds = ParameterSet.get_bounds()

        for param_name, pcs in param_suggestions.items():
            if param_name not in new_dict:
                continue

            current_val = new_dict[param_name]

            if len(pcs) == 1:
                # Single suggestion — apply with learning rate
                target = pcs[0].suggested_value
            else:
                # Multiple suggestions — check for agreement
                directions = [
                    1 if pc.suggested_value > pc.current_value else -1
                    for pc in pcs
                ]
                all_agree = len(set(directions)) == 1

                if all_agree:
                    target = self._weighted_average(pcs)
                else:
                    target = self._resolve_conflict(param_name, pcs, current_params)

            # Apply learning rate damping
            new_val = current_val + self.learning_rate * (target - current_val)

            # Clamp to bounds
            if param_name in bounds:
                lo, hi = bounds[param_name]
                new_val = max(lo, min(hi, new_val))

            # Preserve int type for quantum_shots
            if param_name == "quantum_shots":
                new_val = int(round(new_val))

            new_dict[param_name] = new_val

        return ParameterSet.from_dict(new_dict)

    @staticmethod
    def _weighted_average(suggestions: List[ParameterSuggestion]) -> float:
        """Confidence-weighted average of suggested values."""
        total_weight = sum(s.confidence for s in suggestions)
        if total_weight < 1e-8:
            return suggestions[0].current_value
        return sum(s.suggested_value * s.confidence for s in suggestions) / total_weight

    def _resolve_conflict(self, param_name: str,
                          suggestions: List[ParameterSuggestion],
                          current_params: ParameterSet) -> float:
        """Resolve conflicting suggestions."""
        if not self.use_claude_arbitration:
            # Conservative: pick highest confidence
            best = max(suggestions, key=lambda s: s.confidence)
            return best.suggested_value

        # Claude arbitration
        prompt = json.dumps({
            "task": "resolve_parameter_conflict",
            "parameter": param_name,
            "current_value": getattr(current_params, param_name),
            "bounds": list(ParameterSet.get_bounds().get(param_name, (0, 1))),
            "suggestions": [
                {
                    "agent": s.param_name,
                    "value": s.suggested_value,
                    "reasoning": s.reasoning,
                    "confidence": s.confidence,
                }
                for s in suggestions
            ],
        }, indent=2)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=(
                    "You are a parameter optimization arbitrator. Given conflicting "
                    "suggestions for a physics parameter, pick the best value. "
                    "Respond with ONLY a JSON object: {\"value\": <float>, \"reasoning\": \"...\"}"
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return float(data["value"])
        except Exception:
            pass

        # Fallback: highest confidence
        best = max(suggestions, key=lambda s: s.confidence)
        return best.suggested_value
