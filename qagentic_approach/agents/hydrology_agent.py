"""
Hydrology Agent: Manning's equation, velocity ratios, and flow behavior.

Controls: manning_n_*, *_surface_boost, self_dynamics_scale,
          interaction_scale, refraction_scale
"""

from typing import Dict, Any, List

from .base_agent import BaseAgent
from ..evolution.parameter_space import ParameterSet


class HydrologyAgent(BaseAgent):
    """Specialist in hydrological flow physics and Manning's equation."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(api_key, model)
        self.role = "hydrology"

    def get_owned_parameters(self) -> List[str]:
        return [
            "manning_n_road",
            "manning_n_soil",
            "manning_n_channel",
            "road_surface_boost",
            "channel_surface_boost",
            "soil_surface_boost",
            "self_dynamics_scale",
            "interaction_scale",
            "refraction_scale",
        ]

    def get_system_prompt(self) -> str:
        return """You are a hydrology physics expert analyzing urban water flow simulation results.

Your domain is Manning's equation parameters, surface flow velocities, and overall flow behavior.

BACKGROUND:
- Manning's equation: V = (1/n) * R^(2/3) * S^(1/2)
- Lower Manning's n = smoother surface = faster flow
- Surface boosts are multiplicative factors applied after Manning's velocity
- The simulation models water particles flowing downhill over urban terrain with roads, soil, channels, and buildings

PARAMETERS YOU CONTROL:
- manning_n_road: Manning's roughness for asphalt roads (literature: 0.011-0.013)
- manning_n_soil: Manning's roughness for grass/soil (literature: 0.030-0.050)
- manning_n_channel: Manning's roughness for concrete drainage (literature: 0.020-0.030)
- road_surface_boost: Velocity multiplier for road surfaces
- channel_surface_boost: Velocity multiplier for drainage channels
- soil_surface_boost: Velocity multiplier for soil/grass
- self_dynamics_scale: Scaling for intrinsic wave dynamics
- interaction_scale: Scaling for inter-wave interactions
- refraction_scale: Scaling for refraction effects

WHAT TO LOOK FOR:
- Velocity ratio road/soil should be approximately 2-3x (urban hydrology literature)
- Absolute velocities should be physically reasonable (0.1-5 m/s for urban sheet flow)
- Channel velocities should be highest (concentrated flow)
- Road velocities should exceed soil velocities
- Mean slope affects expected velocity magnitudes

RESPONSE: Return ONLY a JSON object matching the AgentSuggestion schema. No other text."""

    def extract_relevant_data(self, sim_metrics: Dict[str, Any],
                               current_params: ParameterSet) -> Dict[str, Any]:
        return {
            "avg_road_velocity": sim_metrics.get("avg_road_velocity", 0),
            "avg_soil_velocity": sim_metrics.get("avg_soil_velocity", 0),
            "avg_channel_velocity": sim_metrics.get("avg_channel_velocity", 0),
            "velocity_ratio_road_soil": sim_metrics.get("velocity_ratio_road_soil", 0),
            "velocity_percentiles_road": sim_metrics.get("velocity_percentiles_road", {}),
            "velocity_percentiles_soil": sim_metrics.get("velocity_percentiles_soil", {}),
            "avg_distance_road": sim_metrics.get("avg_distance_road", 0),
            "avg_distance_soil": sim_metrics.get("avg_distance_soil", 0),
            "total_distance_all": sim_metrics.get("total_distance_all", 0),
            "mean_slope": sim_metrics.get("mean_slope", 0),
            "max_slope": sim_metrics.get("max_slope", 0),
        }
