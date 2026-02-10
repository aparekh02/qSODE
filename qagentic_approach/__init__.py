"""
qAgentic Approach - Multi-Agent Self-Evolving qSODE Framework
=============================================================

Built on the qODE Framework for simulating quantum ODE systems with
interacting wave fronts, enhanced with a multi-agent collaboration
layer powered by Claude (Anthropic API) for self-evolving parameter
optimization.

Main Components:
- core: Core dynamics, environments, and wave representations
- simulation: Simulation engine and runners
- visualization: Plotting, animation, and video generation
- quantum: Qiskit-based quantum computing for soil absorptivity
- agents: Claude-powered specialist agents for parameter tuning
- evolution: Self-evolving loop, parameter space, metrics, and history

Example usage:
    from qagentic_approach import EvolutionLoop, EvolutionConfig, ParameterSet

    config = EvolutionConfig(max_iterations=10, learning_rate=0.3)
    loop = EvolutionLoop(config=config, sim_factory=my_sim_factory)
    final_params, history = loop.evolve()

Quantum-enhanced watershed example:
    from qagentic_approach.quantum import QuantumSoilGrid, QuantumWaterDynamics

    soil = QuantumSoilGrid(width=100, height=100)
    dynamics = QuantumWaterDynamics(terrain, soil, num_particles=50)
"""

__version__ = "3.0.0-qagentic"
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

# Multi-Agent Evolution
from .evolution.parameter_space import ParameterSet
from .evolution.evolution_loop import EvolutionLoop, EvolutionConfig
from .evolution.metrics import SimulationMetrics, MetricsExtractor
from .evolution.history import EvolutionHistory
from .agents.orchestrator import Orchestrator
from .agents.protocol import AgentMessage, AgentSuggestion

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
    # Multi-Agent Evolution
    "ParameterSet",
    "EvolutionLoop",
    "EvolutionConfig",
    "SimulationMetrics",
    "MetricsExtractor",
    "EvolutionHistory",
    "Orchestrator",
    "AgentMessage",
    "AgentSuggestion",
]
