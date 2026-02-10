"""
Quantum Soil Agent: infiltration fidelity, quantum parameters, soil interaction.

Controls: infiltration_loss_mult, entanglement_radius, quantum_shots, interaction_dt
"""

from typing import Dict, Any, List

from .base_agent import BaseAgent
from ..evolution.parameter_space import ParameterSet


class QuantumSoilAgent(BaseAgent):
    """Specialist in quantum soil modeling and infiltration physics."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(api_key, model)
        self.role = "quantum_soil"

    def get_owned_parameters(self) -> List[str]:
        return [
            "infiltration_loss_mult",
            "entanglement_radius",
            "quantum_shots",
            "interaction_dt",
        ]

    def get_system_prompt(self) -> str:
        return """You are a soil science and quantum computing expert analyzing quantum-enhanced soil infiltration modeling.

BACKGROUND:
- The simulation uses a quantum circuit (Qiskit) to model soil water absorption
- Each soil cell has a quantum state (moisture, saturation, coherence) that evolves as water passes
- Water particles lose mass based on infiltration_rate from quantum measurement
- Mass loss per interaction = infiltration_rate * infiltration_loss_mult
- Neighboring soil cells are entangled within entanglement_radius (correlated saturation)
- Quantum measurements happen every interaction_dt seconds
- More quantum_shots = more accurate probability estimation but slower

PARAMETERS YOU CONTROL:
- infiltration_loss_mult: How much mass a particle loses per infiltration event (0.02-0.15)
- entanglement_radius: Spatial correlation radius between soil cells (1.0-6.0 grid cells)
- quantum_shots: Number of quantum circuit measurement shots (128-2048)
- interaction_dt: Time interval between quantum soil interactions (0.05-0.3 seconds)

WHAT TO LOOK FOR:
- Total mass loss should be 20-40% for a mixed urban terrain (soil + roads)
- Mass loss curve should follow Green-Ampt pattern (high initial infiltration, decreasing over time)
- Saturation should increase from initial (~0) toward 0.3-0.6 after simulation
- Moisture should increase but not fully saturate (would mean soil capacity exceeded)
- Coherence should decay from 1.0 toward 0.7-0.9 (some decoherence, not total)
- Measurement count should be sufficient for statistical reliability
- Wave comparison data (if available) should show later waves infiltrate less (saturation effect)
- Higher entanglement_radius = more spatially correlated infiltration patterns
- More quantum_shots = less noise in infiltration rates

RESPONSE: Return ONLY a JSON object matching the AgentSuggestion schema. No other text."""

    def extract_relevant_data(self, sim_metrics: Dict[str, Any],
                               current_params: ParameterSet) -> Dict[str, Any]:
        return {
            "avg_final_mass": sim_metrics.get("avg_final_mass", 0),
            "mass_loss_fraction": sim_metrics.get("mass_loss_fraction", 0),
            "mass_loss_curve": sim_metrics.get("mass_loss_curve", []),
            "total_infiltration": sim_metrics.get("total_infiltration", 0),
            "total_runoff": sim_metrics.get("total_runoff", 0),
            "avg_saturation": sim_metrics.get("avg_saturation", 0),
            "avg_moisture": sim_metrics.get("avg_moisture", 0),
            "avg_coherence": sim_metrics.get("avg_coherence", 0),
            "measurement_count": sim_metrics.get("measurement_count", 0),
            "wave_comparison": sim_metrics.get("wave_comparison", {}),
        }
