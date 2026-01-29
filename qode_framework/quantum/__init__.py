"""
Quantum Module for qODE Framework
=================================

Provides Qiskit-based quantum computing integration for modeling
soil water absorptivity through quantum superposition.

Key Components:
- QuantumSoilState: Quantum state representation of soil moisture
- QuantumAbsorptivityModel: Qiskit circuit for absorptivity calculation
- QuantumSoilGrid: Spatially distributed quantum soil model

Physics Mapping:
- Qubit amplitudes → probability distribution of infiltration rates
- Measurement collapse → water particle determines infiltration amount
- Quantum entanglement → correlates infiltration across neighboring cells
"""

from .soil_absorptivity import (
    SoilType,
    QuantumSoilState,
    QuantumAbsorptivityModel,
    QuantumSoilGrid,
    QuantumWaterDynamics,
    SurfaceStateSnapshot,
    SurfaceStateTracker,
)

__all__ = [
    "SoilType",
    "QuantumSoilState",
    "QuantumAbsorptivityModel",
    "QuantumSoilGrid",
    "QuantumWaterDynamics",
    "SurfaceStateSnapshot",
    "SurfaceStateTracker",
]
