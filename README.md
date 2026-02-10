# qSODE-powered Agents: Multi-Agent Self-Evolving Urban Watershed Modeling

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0+-purple.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange.svg)](https://www.anthropic.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Quantum-Enhanced ODE Framework with LLM-Driven Multi-Agent Collaboration for Autonomous Physics Parameter Calibration in Urban Watersheds**

[Overview](#overview) | [Multi-Agent Architecture](#multi-agent-architecture) | [Quantum Foundation](#quantum-foundation) | [Results](#results) | [Installation](#installation) | [Usage](#usage)

</div>

---

## Overview

Urban watershed modeling requires both accurate physics representation and careful calibration of dozens of parameters. Traditional approaches hardcode these values from literature, limiting adaptability to specific sites.

**qSODE-powered Agents** combines two innovations:

1. **Quantum Soil Model (qSODE)**: A 3-qubit quantum circuit encodes soil moisture states as quantum amplitudes, enabling probabilistic infiltration modeling through measurement collapse, spatially correlated saturation via entanglement, and persistent state evolution across rainfall events.

2. **Multi-Agent Self-Evolution**: Three domain-specialist Claude agents analyze simulation outputs each iteration and collaboratively tune 19 physics parameters toward target behaviors — without manual calibration.

<div align="center">

![Multi-Wave Water Flow Simulation](results-multiple-waves/multi_wave_flow.gif)

*Sequential water waves traversing urban terrain. Blue particles (Wave 1) encounter dry soil; red particles (Wave 3) encounter saturated conditions, resulting in faster runoff and reduced infiltration.*

</div>

---

## Multi-Agent Architecture

The self-evolving loop runs simulation → metrics → agent analysis → orchestrated update → repeat:

```
Simulation (ParameterSet) ──► MetricsExtractor ──► SimulationMetrics
                                                        │
                              ┌──────────────┬──────────┼──────────┐
                              ▼              ▼          ▼          │
                        HydrologyAgent  SurfaceAgent  QSoilAgent  │
                         (Claude API)   (Claude API)  (Claude API) │
                              │              │          │          │
                              ▼              ▼          ▼          │
                        AgentSuggestion x 3                       │
                              │              │          │          │
                              └──────────────┴──────┬───┘          │
                                                    ▼              │
                                              Orchestrator         │
                                            (aggregate +           │
                                             arbitrate)            │
                                                    │              │
                                                    ▼              │
                                            New ParameterSet ──────┘
```

### Three Specialist Agents

Each agent is a structured Claude API call with a physics-informed system prompt, receiving only the metrics and parameters relevant to its domain.

| Agent | Domain | Parameters Controlled |
|-------|--------|----------------------|
| **Hydrology Agent** | Manning's equation, flow physics | `manning_n_road`, `manning_n_soil`, `manning_n_channel`, `road_surface_boost`, `channel_surface_boost`, `soil_surface_boost`, `self_dynamics_scale`, `interaction_scale`, `refraction_scale` |
| **Surface Agent** | Surface differentiation, dispersion | `soil_velocity_base`, `soil_velocity_runoff_coeff`, `road_velocity_base`, `road_velocity_runoff_coeff`, `road_dispersion`, `soil_dispersion` |
| **Quantum Soil Agent** | Infiltration, quantum circuit tuning | `infiltration_loss_mult`, `entanglement_radius`, `quantum_shots`, `interaction_dt` |

### Orchestrator

The Orchestrator resolves all agent suggestions into a single parameter update:

- **Single suggestion** → apply directly with learning rate damping
- **Multiple agents agree** → confidence-weighted average
- **Agents conflict** → Claude arbitration call evaluates competing reasoning

All updates are damped:

$$p^{(t+1)} = p^{(t)} + \eta \cdot (p_{\text{suggested}} - p^{(t)})$$

where $\eta = 0.5$, and results are clamped to physical bounds.

### Communication Protocol

Agents communicate via structured JSON — no free-form text. Each agent returns an `AgentSuggestion`:

```json
{
  "agent_role": "hydrology",
  "observation": {
    "metrics_analyzed": {"velocity_ratio_road_soil": 2.70},
    "anomalies": ["Road velocity ratio above target"],
    "confidence": 0.90
  },
  "parameter_changes": [
    {
      "param_name": "manning_n_road",
      "current_value": 0.012,
      "suggested_value": 0.014,
      "reasoning": "Increase road roughness to reduce velocity ratio toward 2.5x target",
      "confidence": 0.85
    }
  ],
  "priority": 0.90
}
```

---

## Quantum Foundation

### Quantum-Enhanced Water Dynamics ODE

For each water particle $i$ at position $\mathbf{h}^i(t)$:

$$\frac{d\mathbf{h}^i(t)}{dt} = \mathbf{V}_{\text{Manning}}(\mathbf{h}^i) \cdot \Phi_Q(\mathbf{h}^i) + \sum_{j \neq i} \mathbf{F}_{\text{dispersion}}^{ij} + \mathbf{F}_{\text{obstacle}}^i$$

Where $\Phi_Q$ is the quantum velocity modifier derived from quantum soil measurements, and all Manning coefficients, surface boosts, and dispersion factors are read from the evolvable `ParameterSet`.

### 3-Qubit Soil State Encoding

| Qubit | Physical Meaning | State |0⟩ | State |1⟩ |
|-------|------------------|---------|---------|
| $q_0$ | Moisture level | Dry (absorbs) | Saturated (rejects) |
| $q_1$ | Saturation history | Fresh soil | Waterlogged |
| $q_2$ | Surface condition | Permeable | Sealed |

### Quantum Circuit

```
|0⟩ ──[Ry(θ_m)]──────●──────[CRx(φ)]──[M]──> infiltration_rate
                     │          │
|0⟩ ──[Ry(θ_s)]──────X────[H]──●──[H]──[M]──> runoff_factor
                              │
|0⟩ ──[Ry(θ_surf)]────────────●────────[M]──> saturation_probability
```

### Spatial Entanglement

Neighboring cells within $r_{entangle}$ exhibit correlated saturation:

$$\alpha(r) = \left(1 - \frac{r}{r_{entangle}}\right) \cdot 0.3, \quad r < r_{entangle}$$

---

## Results

### Multi-Wave Quantum Dynamics

<div align="center">

![Multi-Wave Comparison](results-multiple-waves/multi_wave_comparison.png)

*Three sequential water waves: (a) trajectory evolution, (b) saturation field, (c) wave statistics, (d) velocity changes.*

</div>

| Metric | Wave 1 | Wave 2 | Wave 3 | Trend |
|--------|--------|--------|--------|-------|
| Soil Saturation | 1.1% | 1.4% | 1.7% | +55% |
| Water Absorbed | 3.3% | 3.0% | 2.1% | -36% |
| Soil Velocity | 2.6 m/s | 2.9 m/s | 2.4 m/s | Variable |
| Road Velocity | 20.8 m/s | 19.9 m/s | 20.3 m/s | Stable |

### Multi-Agent Evolution (5 Iterations)

Starting from hardcoded defaults, the agents autonomously drive the simulation toward physically realistic targets:

| Metric | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Target |
|--------|--------|--------|--------|--------|--------|--------|
| Velocity Ratio (road/soil) | 2.70 | 3.28 | 2.45 | 2.60 | **2.54** | 2.5 |
| Mass Loss (%) | 3.3 | 12.6 | 11.6 | 19.0 | **21.4** | 30.0 |
| Avg Road Distance (m) | 28.2 | 44.9 | 43.2 | 50.4 | 44.7 | — |
| Avg Soil Distance (m) | 2.6 | 6.8 | 8.8 | 9.9 | 10.5 | — |

**Key parameter shifts by agents:**

| Parameter | Default | Final | Agent | Direction |
|-----------|---------|-------|-------|-----------|
| `manning_n_road` | 0.012 | 0.017 | Hydrology | Rougher roads → slower flow |
| `road_surface_boost` | 1.80 | 1.06 | Hydrology | Reduced boost → lower velocity ratio |
| `infiltration_loss_mult` | 0.08 | 0.14 | Quantum Soil | More infiltration → realistic mass loss |
| `entanglement_radius` | 3.0 | 4.0 | Quantum Soil | Wider coupling → smoother saturation |
| `road_velocity_base` | 1.2 | 1.43 | Surface | Faster base → better differentiation |
| `soil_dispersion` | 0.25 | 0.35 | Surface | More spread → realistic particle behavior |

The iteration 2 overshoot (velocity ratio 3.28) demonstrates self-correction: the hydrology agent adjusted `road_surface_boost` downward in iteration 3, stabilizing near the target.

---

## Project Structure

```
qSODE-urban-wsmodel/
├── qode_framework/                    # Original framework (untouched)
│   ├── quantum/                       #   Qiskit quantum circuits
│   ├── core/                          #   ODE dynamics, environments, waves
│   ├── simulation/                    #   ODE solver (torchdiffeq)
│   └── visualization/                 #   Matplotlib animations
│
├── qagentic_approach/                 # Multi-agent enhanced framework
│   ├── agents/
│   │   ├── base_agent.py              #   BaseAgent ABC (Claude API client)
│   │   ├── hydrology_agent.py         #   Manning's equation specialist
│   │   ├── surface_agent.py           #   Surface dynamics specialist
│   │   ├── quantum_soil_agent.py      #   Quantum soil specialist
│   │   ├── orchestrator.py            #   Conflict resolution + synthesis
│   │   └── protocol.py               #   Typed JSON message schemas
│   ├── evolution/
│   │   ├── parameter_space.py         #   19-parameter ParameterSet
│   │   ├── metrics.py                 #   SimulationMetrics extraction
│   │   ├── history.py                 #   EvolutionHistory tracking
│   │   └── evolution_loop.py          #   Self-evolving main loop
│   ├── quantum/                       #   Qiskit circuits (from qode_framework)
│   ├── core/                          #   Dynamics, environments (from qode_framework)
│   └── simulation/                    #   Solver (from qode_framework)
│
├── urban_wave_simulation.py           # Original simulation (hardcoded params)
├── qagentic_urban_wave_simulation.py  # Multi-agent simulation (evolvable params)
├── .env                               # ANTHROPIC_API_KEY
├── evolution_history.json             # Full iteration records (generated)
└── paper.md                           # Research paper
```

---

## Installation

### Prerequisites

- Python 3.10+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- CUDA-capable GPU (optional)

### Setup

```bash
# Clone repository
git clone https://github.com/qugena-labs/qSODE-urban-wsmodel.git
cd qSODE-urban-wsmodel

# Install dependencies
pip install -r requirements.txt
pip install anthropic python-dotenv

# Set your API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### Dependencies

```
numpy>=1.24.0
torch>=2.0.0
torchdiffeq>=0.2.3
matplotlib>=3.7.0
scipy>=1.10.0
qiskit>=1.0.0
qiskit-aer>=0.13.0
anthropic
python-dotenv
```

---

## Usage

### Run Multi-Agent Self-Evolving Simulation

```bash
python qagentic_urban_wave_simulation.py
```

This runs 5 evolution iterations where Claude agents collaboratively tune 19 parameters. Outputs:
- Console: per-iteration metrics and parameter changes
- `evolution_history.json`: full record of all iterations, suggestions, and decisions
- `results/`: simulation visualizations

### Run Original Simulation (No Agents)

```bash
python urban_wave_simulation.py
```

Runs the base qSODE simulation with hardcoded parameters for comparison.

### Custom Configuration

```python
from qagentic_approach.evolution.evolution_loop import EvolutionLoop, EvolutionConfig
from qagentic_approach.evolution.parameter_space import ParameterSet

config = EvolutionConfig(
    max_iterations=10,       # More iterations for better convergence
    learning_rate=0.3,       # Lower LR for more cautious updates
    convergence_window=3,    # Check last 3 iterations for plateau
)

loop = EvolutionLoop(config=config, sim_factory=my_sim_factory)
final_params, history = loop.evolve()
```

---

## Citation

```bibtex
@article{qsode-agents2025,
  title={qSODE-powered Agents: A Multi-Agent Self-Evolving Framework for
         Quantum Stochastic ODE-Based Urban Watershed Modeling},
  author={Parekh, Aksh},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Anthropic for Claude API powering the multi-agent system
- IBM Qiskit team for the quantum computing framework
- PyTorch team for differentiable ODE solvers
- USDA for soil classification data

---

<div align="center">

**qSODE-powered Agents** — *Quantum Physics Meets Multi-Agent Collaboration for Urban Hydrology*

[Report Issues](https://github.com/qugena-labs/qSODE-urban-wsmodel/issues) | [Contribute](https://github.com/qugena-labs/qSODE-urban-wsmodel/pulls)

</div>
