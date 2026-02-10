"""
Surface Dynamics Agent: velocity modifiers, dispersion, surface differentiation.

Controls: *_velocity_base, *_velocity_runoff_coeff, *_dispersion
"""

from typing import Dict, Any, List

from .base_agent import BaseAgent
from ..evolution.parameter_space import ParameterSet


class SurfaceAgent(BaseAgent):
    """Specialist in surface water dynamics and particle behavior."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(api_key, model)
        self.role = "surface"

    def get_owned_parameters(self) -> List[str]:
        return [
            "soil_velocity_base",
            "soil_velocity_runoff_coeff",
            "road_velocity_base",
            "road_velocity_runoff_coeff",
            "road_dispersion",
            "soil_dispersion",
        ]

    def get_system_prompt(self) -> str:
        return """You are a surface water dynamics expert analyzing how water behaves differently on roads vs soil vs channels.

BACKGROUND:
- When water interacts with quantum soil, the result includes a runoff_factor (0-1)
- Velocity modifier for soil = soil_velocity_base + soil_velocity_runoff_coeff * runoff_factor
- Velocity modifier for road = road_velocity_base + road_velocity_runoff_coeff * runoff_factor
- Dispersion controls how much particles spread apart (repulsion force magnitude)
- Road water should channel (low dispersion) while soil water spreads (higher dispersion)

PARAMETERS YOU CONTROL:
- soil_velocity_base: Base velocity multiplier for soil (affects how much soil slows water)
- soil_velocity_runoff_coeff: How much runoff_factor accelerates soil flow
- road_velocity_base: Base velocity multiplier for roads (should be > 1.0)
- road_velocity_runoff_coeff: How much runoff_factor accelerates road flow
- road_dispersion: Particle spread force on roads (low = channelized flow)
- soil_dispersion: Particle spread force on soil (higher = sheet flow)

WHAT TO LOOK FOR:
- Clear velocity differentiation between road and soil particles
- Road particles should stay more clustered (channelized drainage)
- Soil particles should spread more (diffuse overland flow)
- Particles losing mass (infiltration) should slow down on soil
- Velocity modifiers should produce physically distinct behavior per surface
- Final particle distribution: most water on roads (accumulation) vs absorbed on soil

RESPONSE: Return ONLY a JSON object matching the AgentSuggestion schema. No other text."""

    def extract_relevant_data(self, sim_metrics: Dict[str, Any],
                               current_params: ParameterSet) -> Dict[str, Any]:
        return {
            "avg_road_velocity": sim_metrics.get("avg_road_velocity", 0),
            "avg_soil_velocity": sim_metrics.get("avg_soil_velocity", 0),
            "velocity_ratio_road_soil": sim_metrics.get("velocity_ratio_road_soil", 0),
            "num_particles_on_road_final": sim_metrics.get("num_particles_on_road_final", 0),
            "num_particles_on_soil_final": sim_metrics.get("num_particles_on_soil_final", 0),
            "dispersion_road": sim_metrics.get("dispersion_road", 0),
            "dispersion_soil": sim_metrics.get("dispersion_soil", 0),
            "avg_final_mass": sim_metrics.get("avg_final_mass", 0),
            "mass_loss_fraction": sim_metrics.get("mass_loss_fraction", 0),
            "avg_distance_road": sim_metrics.get("avg_distance_road", 0),
            "avg_distance_soil": sim_metrics.get("avg_distance_soil", 0),
        }
