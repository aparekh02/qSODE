"""
qODE Framework - Quantum ODE Simulation Framework
=================================================

A framework for simulating quantum ODE systems with interacting
wave fronts in various environments, now enhanced with Qiskit
quantum computing for soil water absorptivity modeling.

Main Components:
- core: Core dynamics, environments, and wave representations
- simulation: Simulation engine and runners
- visualization: Plotting, animation, and video generation
- quantum: Qiskit-based quantum computing for soil absorptivity

Example usage:
    from qode_framework import UrbanEnvironment, WaveSimulator, Visualizer

    env = UrbanEnvironment(width=100, height=100)
    sim = WaveSimulator(environment=env, num_waves=5)
    results = sim.run(t_end=10.0)

    viz = Visualizer(sim)
    viz.create_video('simulation.mp4')

Quantum-enhanced watershed example:
    from qode_framework.quantum import QuantumSoilGrid, QuantumWaterDynamics

    soil = QuantumSoilGrid(width=100, height=100)
    dynamics = QuantumWaterDynamics(terrain, soil, num_particles=50)
"""

__version__ = "2.0.0"
__author__ = "qODE Framework"

# Core components
from .core.environment import (
    BaseEnvironment,
    UrbanEnvironment,
    MediumType,
)
from .core.dynamics import (
    qODEDynamics,
    InteractionNetwork,
    SelfDynamicsNetwork,
    QuantumPhaseMixer,
)
from .core.wave import WaveState, WaveFront

# Simulation
from .simulation.simulator import WaveSimulator, SimulationConfig

# Visualization
from .visualization.visualizer import Visualizer, VideoConfig

# Quantum module (Qiskit-based soil absorptivity)
from .quantum import (
    SoilType,
    QuantumSoilState,
    QuantumAbsorptivityModel,
    QuantumSoilGrid,
    QuantumWaterDynamics,
)

__all__ = [
    # Core
    "BaseEnvironment",
    "UrbanEnvironment",
    "MediumType",
    "qODEDynamics",
    "InteractionNetwork",
    "SelfDynamicsNetwork",
    "QuantumPhaseMixer",
    "WaveState",
    "WaveFront",
    # Simulation
    "WaveSimulator",
    "SimulationConfig",
    # Visualization
    "Visualizer",
    "VideoConfig",
    # Quantum
    "SoilType",
    "QuantumSoilState",
    "QuantumAbsorptivityModel",
    "QuantumSoilGrid",
    "QuantumWaterDynamics",
]
