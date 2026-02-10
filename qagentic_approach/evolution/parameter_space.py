"""
Evolvable parameter space for multi-agent simulation tuning.

Defines all physics constants that agents can adjust, their defaults
(matching the original hardcoded values), and physical bounds.
"""

from dataclasses import dataclass, asdict, fields
from typing import Dict, Tuple, Optional


@dataclass
class ParameterSet:
    """
    All evolvable parameters for the urban water simulation.

    Defaults match the original hardcoded values in urban_wave_simulation.py
    and qode_framework modules.
    """

    # --- Manning's Equation Roughness Coefficients ---
    manning_n_road: float = 0.012
    manning_n_soil: float = 0.035
    manning_n_channel: float = 0.025

    # --- Surface Velocity Boosts (multiplicative on Manning's V) ---
    road_surface_boost: float = 1.8
    channel_surface_boost: float = 2.5
    soil_surface_boost: float = 0.7

    # --- Velocity Modifiers (from quantum soil interaction result) ---
    soil_velocity_base: float = 0.5
    soil_velocity_runoff_coeff: float = 0.3
    road_velocity_base: float = 1.2
    road_velocity_runoff_coeff: float = 0.3

    # --- Infiltration ---
    infiltration_loss_mult: float = 0.08

    # --- Particle Dispersion ---
    road_dispersion: float = 0.1
    soil_dispersion: float = 0.25

    # --- Quantum Soil Grid ---
    entanglement_radius: float = 3.0
    quantum_shots: int = 512
    interaction_dt: float = 0.1

    # --- qODE Dynamics Scales ---
    self_dynamics_scale: float = 0.5
    interaction_scale: float = 0.3
    refraction_scale: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ParameterSet':
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @staticmethod
    def get_bounds() -> Dict[str, Tuple[float, float]]:
        """Physical bounds for each parameter."""
        return {
            "manning_n_road":             (0.008, 0.020),
            "manning_n_soil":             (0.020, 0.060),
            "manning_n_channel":          (0.015, 0.035),
            "road_surface_boost":         (1.0,   3.0),
            "channel_surface_boost":      (1.5,   4.0),
            "soil_surface_boost":         (0.3,   1.0),
            "soil_velocity_base":         (0.2,   0.8),
            "soil_velocity_runoff_coeff": (0.1,   0.5),
            "road_velocity_base":         (0.8,   2.0),
            "road_velocity_runoff_coeff": (0.1,   0.5),
            "infiltration_loss_mult":     (0.02,  0.15),
            "road_dispersion":            (0.02,  0.3),
            "soil_dispersion":            (0.1,   0.5),
            "entanglement_radius":        (1.0,   6.0),
            "quantum_shots":              (128,   2048),
            "interaction_dt":             (0.05,  0.3),
            "self_dynamics_scale":        (0.1,   1.0),
            "interaction_scale":          (0.1,   0.8),
            "refraction_scale":           (0.1,   1.0),
        }

    def clamp_to_bounds(self) -> 'ParameterSet':
        """Return a new ParameterSet with all values clamped to bounds."""
        bounds = self.get_bounds()
        d = self.to_dict()
        for k, (lo, hi) in bounds.items():
            if k in d:
                d[k] = type(d[k])(max(lo, min(hi, d[k])))
        return ParameterSet.from_dict(d)

    def diff(self, other: 'ParameterSet') -> Dict[str, Tuple[float, float]]:
        """Return parameters that differ: {name: (self_val, other_val)}."""
        d1, d2 = self.to_dict(), other.to_dict()
        return {k: (d1[k], d2[k]) for k in d1 if d1[k] != d2[k]}
