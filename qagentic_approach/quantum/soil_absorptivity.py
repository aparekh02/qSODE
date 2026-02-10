"""
Quantum Soil Absorptivity Model using Qiskit
=============================================

This module implements a quantum computing approach to modeling soil water
absorptivity using Qiskit. The key insight is that soil saturation states
can exist in superposition until water interacts with the soil, at which
point the state "collapses" to determine the actual infiltration behavior.

Physical Model:
--------------
Traditional Green-Ampt infiltration:
    f = K * (1 + ψΔθ/F)

Quantum Enhancement:
    - Soil moisture levels exist in superposition of dry/wet states
    - Water interaction causes measurement/collapse
    - Neighboring cells can be entangled (correlated saturation)
    - Post-measurement state affects future water movement

Quantum Circuit Design:
----------------------
    |0⟩ ──[Ry(θ_moisture)]──[CNOT]──[Measure]──> infiltration_rate
    |0⟩ ──[Ry(θ_saturation)]─────|──[Measure]──> surface_runoff_factor

Where:
    - θ_moisture encodes current soil moisture (0 = dry, π = saturated)
    - θ_saturation encodes cumulative saturation history
    - CNOT entangles moisture with saturation state
    - Measurement probabilities determine infiltration vs runoff

"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum

# Qiskit imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not installed. Using classical fallback.")
    print("Install with: pip install qiskit qiskit-aer")


class SoilType(Enum):
    """Soil types with base hydraulic properties."""
    SAND = (0.12, 0.417, 4.95e-3)      # (porosity, field_capacity, K_sat cm/s)
    LOAM = (0.43, 0.270, 1.32e-4)
    CLAY = (0.38, 0.385, 5.56e-6)
    IMPERVIOUS = (0.0, 0.0, 0.0)        # Roads, concrete

    def __init__(self, porosity, field_capacity, k_sat):
        self.porosity = porosity
        self.field_capacity = field_capacity
        self.k_sat = k_sat  # Saturated hydraulic conductivity


@dataclass
class QuantumSoilState:
    """
    Quantum state representation of a soil cell's water absorptivity.

    The state encodes:
    - moisture_amplitude: Superposition amplitude for moisture level
    - saturation_phase: Quantum phase encoding saturation history
    - coherence: Measure of quantum coherence (1 = pure, 0 = fully decohered)
    - entanglement_strength: Correlation with neighboring cells
    """
    x: int
    y: int
    soil_type: SoilType = SoilType.LOAM

    # Quantum state parameters (angles in radians)
    moisture_angle: float = 0.0          # Ry rotation angle for moisture qubit
    saturation_angle: float = 0.0        # Ry rotation angle for saturation qubit
    entanglement_angle: float = 0.0      # Controlled rotation for entanglement

    # Classical tracking
    cumulative_water: float = 0.0        # Total water that has passed through
    last_infiltration: float = 0.0       # Last measured infiltration rate
    measurement_count: int = 0           # Number of quantum measurements made

    # Quantum properties
    coherence: float = 1.0               # Quantum coherence (decays with measurements)

    def get_moisture_level(self) -> float:
        """Convert quantum angle to classical moisture level [0, 1]."""
        return np.sin(self.moisture_angle / 2) ** 2

    def get_saturation_factor(self) -> float:
        """Get saturation factor affecting water movement."""
        return np.sin(self.saturation_angle / 2) ** 2

    def absorb_water(self, amount: float) -> float:
        """
        Update quantum state when water is absorbed.
        Returns the actual amount absorbed (infiltrated).
        """
        # Update cumulative water
        self.cumulative_water += amount

        # Increase moisture angle (toward saturation)
        max_capacity = self.soil_type.porosity * 100  # cm of water
        if max_capacity > 0:
            self.moisture_angle = min(np.pi, self.moisture_angle + amount / max_capacity * np.pi)
            # Saturation increases more slowly (historical effect)
            self.saturation_angle = min(np.pi, self.saturation_angle + amount / (max_capacity * 3) * np.pi)
        else:
            # Impervious surface - already at max saturation
            self.moisture_angle = np.pi
            self.saturation_angle = np.pi

        # Coherence decays slightly with each interaction
        self.coherence *= 0.98

        return amount * (1 - self.get_saturation_factor())


class QuantumAbsorptivityModel:
    """
    Qiskit-based quantum circuit for computing soil absorptivity.

    This model uses a 3-qubit system:
    - Qubit 0: Moisture state (|0⟩ = dry, |1⟩ = saturated)
    - Qubit 1: Saturation history (|0⟩ = fresh, |1⟩ = waterlogged)
    - Qubit 2: Surface condition (|0⟩ = permeable, |1⟩ = sealed)

    The circuit creates superposition and entanglement to model the
    quantum uncertainty in infiltration rates.
    """

    def __init__(self, shots: int = 1024, use_statevector: bool = False):
        """
        Initialize the quantum absorptivity model.

        Args:
            shots: Number of measurement shots for probability estimation
            use_statevector: Use exact statevector simulation (deterministic)
        """
        self.shots = shots
        self.use_statevector = use_statevector

        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
            self._build_circuit()
        else:
            self.simulator = None

    def _build_circuit(self):
        """Build the parameterized quantum circuit."""
        # Quantum registers
        self.qr = QuantumRegister(3, 'soil')
        self.cr = ClassicalRegister(3, 'measure')

        # Parameters for the circuit
        self.theta_moisture = Parameter('θ_m')      # Moisture state
        self.theta_saturation = Parameter('θ_s')    # Saturation history
        self.theta_surface = Parameter('θ_surf')    # Surface condition
        self.phi_entangle = Parameter('φ_ent')      # Entanglement strength

        # Build parameterized circuit
        self.circuit = QuantumCircuit(self.qr, self.cr)

        # Initialize qubits based on soil state
        self.circuit.ry(self.theta_moisture, 0)      # Moisture superposition
        self.circuit.ry(self.theta_saturation, 1)    # Saturation superposition
        self.circuit.ry(self.theta_surface, 2)       # Surface superposition

        # Entangle moisture with saturation (wet soil stays wet)
        self.circuit.cx(1, 0)  # CNOT: saturation controls moisture

        # Controlled rotation: surface affects moisture-saturation correlation
        self.circuit.crx(self.phi_entangle, 2, 0)

        # Add interference effects
        self.circuit.h(1)  # Hadamard on saturation for interference
        self.circuit.cz(0, 1)  # Controlled-Z creates phase correlation
        self.circuit.h(1)  # Complete interference pattern

        # Measure all qubits
        self.circuit.measure(self.qr, self.cr)

    def compute_absorptivity(self, soil_state: QuantumSoilState,
                             water_amount: float = 1.0) -> Dict[str, float]:
        """
        Compute absorptivity using quantum circuit.

        Args:
            soil_state: Current quantum state of the soil
            water_amount: Amount of water attempting to infiltrate

        Returns:
            Dictionary with:
                - infiltration_rate: Fraction of water that infiltrates
                - runoff_factor: Fraction that runs off (easier surface flow)
                - saturation_probability: Probability soil becomes saturated
        """
        if not QISKIT_AVAILABLE:
            return self._classical_fallback(soil_state, water_amount)

        # Bind parameters based on soil state (assign_parameters in Qiskit 1.0+)
        bound_circuit = self.circuit.assign_parameters({
            self.theta_moisture: soil_state.moisture_angle,
            self.theta_saturation: soil_state.saturation_angle,
            self.theta_surface: soil_state.entanglement_angle,
            self.phi_entangle: np.pi * soil_state.coherence * 0.5
        })

        if self.use_statevector:
            # Exact statevector simulation (remove measurements)
            sv_circuit = bound_circuit.remove_final_measurements(inplace=False)
            statevector = Statevector(sv_circuit)
            probs = statevector.probabilities()

            # Convert to measurement-like results
            results = {format(i, '03b'): p for i, p in enumerate(probs)}
        else:
            # Run circuit with shots
            job = self.simulator.run(bound_circuit, shots=self.shots)
            result = job.result()
            counts = result.get_counts()

            # Normalize to probabilities
            results = {k: v / self.shots for k, v in counts.items()}

        # Interpret quantum results
        return self._interpret_results(results, soil_state, water_amount)

    def _interpret_results(self, results: Dict[str, float],
                          soil_state: QuantumSoilState,
                          water_amount: float) -> Dict[str, float]:
        """
        Interpret quantum measurement results as physical quantities.

        Bit interpretation (reading right to left):
        - bit 0 (rightmost): moisture (0=dry accepts water, 1=wet rejects)
        - bit 1: saturation (0=unsaturated, 1=saturated)
        - bit 2: surface (0=permeable, 1=sealed)
        """
        # Calculate probabilities for each physical outcome
        p_dry = sum(p for bits, p in results.items() if bits[-1] == '0')  # bit 0 = 0
        p_unsaturated = sum(p for bits, p in results.items() if bits[-2] == '0')  # bit 1 = 0
        p_permeable = sum(p for bits, p in results.items() if bits[-3] == '0')  # bit 2 = 0

        # Infiltration rate: higher when dry, unsaturated, and permeable
        # This is where quantum superposition shines - we get probabilistic mixing
        base_infiltration = soil_state.soil_type.k_sat * 1000  # Scale to reasonable range

        # Quantum-weighted infiltration
        infiltration_rate = base_infiltration * (
            0.6 * p_dry +           # Moisture state contribution
            0.3 * p_unsaturated +   # Saturation state contribution
            0.1 * p_permeable       # Surface state contribution
        )

        # Apply coherence factor (quantum advantage degrades with decoherence)
        infiltration_rate *= (0.5 + 0.5 * soil_state.coherence)

        # Runoff factor: water moves more easily over saturated/sealed surfaces
        # Higher runoff = easier water movement on surface
        runoff_factor = (
            0.3 * (1 - p_dry) +          # Wet surface aids runoff
            0.5 * (1 - p_unsaturated) +  # Saturated soil can't absorb
            0.2 * (1 - p_permeable)      # Sealed surface
        )

        # Saturation probability (for state update)
        saturation_probability = 1 - p_unsaturated

        return {
            'infiltration_rate': min(1.0, infiltration_rate),
            'runoff_factor': min(1.0, max(0.0, runoff_factor)),
            'saturation_probability': saturation_probability,
            'quantum_state': results  # Raw quantum results for analysis
        }

    def _classical_fallback(self, soil_state: QuantumSoilState,
                           water_amount: float) -> Dict[str, float]:
        """Classical approximation when Qiskit is not available."""
        moisture = soil_state.get_moisture_level()
        saturation = soil_state.get_saturation_factor()

        # Classical Green-Ampt approximation
        base_infiltration = soil_state.soil_type.k_sat * 1000
        infiltration_rate = base_infiltration * (1 - moisture) * (1 - saturation * 0.7)

        runoff_factor = 0.3 * moisture + 0.5 * saturation

        return {
            'infiltration_rate': min(1.0, infiltration_rate),
            'runoff_factor': min(1.0, max(0.0, runoff_factor)),
            'saturation_probability': saturation,
            'quantum_state': {'classical': 1.0}
        }


@dataclass
class SurfaceStateSnapshot:
    """Snapshot of surface state at a point in time."""
    time: float
    wave_id: int
    moisture_field: np.ndarray
    saturation_field: np.ndarray
    coherence_field: np.ndarray
    total_infiltration: float
    total_runoff: float
    measurement_count: int


class SurfaceStateTracker:
    """
    Tracks surface state changes over time across multiple water waves.

    This records how soil/road characteristics change as water passes through,
    enabling analysis of how subsequent waves behave differently due to
    saturation from previous waves.
    """

    def __init__(self):
        self.snapshots: List[SurfaceStateSnapshot] = []
        self.wave_statistics: List[Dict] = []
        self.current_wave_id = 0

    def record_snapshot(self, soil_grid: 'QuantumSoilGrid', time: float, wave_id: int):
        """Record current surface state."""
        snapshot = SurfaceStateSnapshot(
            time=time,
            wave_id=wave_id,
            moisture_field=soil_grid.get_moisture_field().copy(),
            saturation_field=soil_grid.get_saturation_field().copy(),
            coherence_field=soil_grid.get_coherence_field().copy(),
            total_infiltration=soil_grid.total_infiltration,
            total_runoff=soil_grid.total_runoff,
            measurement_count=len(soil_grid.measurement_history)
        )
        self.snapshots.append(snapshot)

    def start_wave(self, wave_id: int):
        """Mark the start of a new wave."""
        self.current_wave_id = wave_id
        self.wave_statistics.append({
            'wave_id': wave_id,
            'start_infiltration': 0.0,
            'start_runoff': 0.0,
            'end_infiltration': 0.0,
            'end_runoff': 0.0,
            'infiltration_delta': 0.0,
            'runoff_delta': 0.0,
            'avg_saturation_before': 0.0,
            'avg_saturation_after': 0.0,
        })

    def end_wave(self, soil_grid: 'QuantumSoilGrid', wave_id: int):
        """Mark the end of a wave and calculate statistics."""
        if wave_id < len(self.wave_statistics):
            stats = self.wave_statistics[wave_id]
            stats['end_infiltration'] = soil_grid.total_infiltration
            stats['end_runoff'] = soil_grid.total_runoff
            stats['infiltration_delta'] = stats['end_infiltration'] - stats['start_infiltration']
            stats['runoff_delta'] = stats['end_runoff'] - stats['start_runoff']
            stats['avg_saturation_after'] = soil_grid.get_saturation_field().mean()

    def record_wave_start(self, soil_grid: 'QuantumSoilGrid', wave_id: int):
        """Record state at wave start."""
        if wave_id < len(self.wave_statistics):
            stats = self.wave_statistics[wave_id]
            stats['start_infiltration'] = soil_grid.total_infiltration
            stats['start_runoff'] = soil_grid.total_runoff
            stats['avg_saturation_before'] = soil_grid.get_saturation_field().mean()

    def get_wave_comparison(self) -> Dict:
        """Get comparison data between waves."""
        if len(self.wave_statistics) < 2:
            return {}

        comparison = {
            'waves': self.wave_statistics,
            'saturation_increase': [],
            'infiltration_decrease': [],
            'runoff_increase': [],
        }

        for i in range(1, len(self.wave_statistics)):
            prev = self.wave_statistics[i - 1]
            curr = self.wave_statistics[i]

            comparison['saturation_increase'].append(
                curr['avg_saturation_before'] - prev['avg_saturation_before']
            )
            comparison['infiltration_decrease'].append(
                prev['infiltration_delta'] - curr['infiltration_delta']
            )
            comparison['runoff_increase'].append(
                curr['runoff_delta'] - prev['runoff_delta']
            )

        return comparison


class QuantumSoilGrid:
    """
    Spatially distributed quantum soil model.

    Manages a grid of QuantumSoilState objects with spatial entanglement
    between neighboring cells. When water affects one cell, entangled
    neighbors have correlated state updates.

    Now includes SurfaceStateTracker for recording changes across waves.
    """

    def __init__(self, width: int, height: int,
                 cell_size: float = 1.0,
                 entanglement_radius: float = 3.0,
                 seed: int = 42):
        """
        Initialize the quantum soil grid.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            cell_size: Physical size of each cell (meters)
            entanglement_radius: Radius for quantum entanglement between cells
            seed: Random seed for initialization
        """
        np.random.seed(seed)
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.entanglement_radius = entanglement_radius

        # Create quantum model
        self.quantum_model = QuantumAbsorptivityModel(shots=512)

        # Initialize soil grid
        self.grid: List[List[QuantumSoilState]] = []
        self._initialize_grid()

        # Track global statistics
        self.total_infiltration = 0.0
        self.total_runoff = 0.0
        self.measurement_history = []

        # Surface state tracker for multi-wave analysis
        self.state_tracker = SurfaceStateTracker()

    def _initialize_grid(self):
        """Initialize the soil grid with quantum states."""
        self.grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                # Determine soil type based on position
                # (can be customized for terrain features)
                soil_type = self._get_soil_type(x, y)

                # Create quantum soil state with slight random variation
                state = QuantumSoilState(
                    x=x, y=y,
                    soil_type=soil_type,
                    moisture_angle=np.random.uniform(0, 0.3),  # Start mostly dry
                    saturation_angle=np.random.uniform(0, 0.1),
                    entanglement_angle=np.random.uniform(0, 0.2),
                    coherence=1.0
                )
                row.append(state)
            self.grid.append(row)

    def _get_soil_type(self, x: int, y: int) -> SoilType:
        """Determine soil type at position (can be overridden)."""
        return SoilType.LOAM

    def set_soil_type(self, x: int, y: int, soil_type: SoilType):
        """Set soil type at a specific cell."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].soil_type = soil_type

    def set_soil_region(self, x1: int, y1: int, x2: int, y2: int,
                        soil_type: SoilType):
        """Set soil type for a rectangular region."""
        for y in range(max(0, y1), min(self.height, y2)):
            for x in range(max(0, x1), min(self.width, x2)):
                self.grid[y][x].soil_type = soil_type

    def get_state(self, x: float, y: float) -> Optional[QuantumSoilState]:
        """Get quantum soil state at continuous position."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.grid[iy][ix]

    def interact_water(self, x: float, y: float,
                       water_amount: float) -> Dict[str, float]:
        """
        Water interacts with soil at position (x, y).

        This triggers quantum measurement which:
        1. Computes infiltration vs runoff
        2. Updates the soil's quantum state
        3. Propagates entanglement effects to neighbors

        Returns:
            Dictionary with infiltration_rate, runoff_factor, etc.
        """
        state = self.get_state(x, y)
        if state is None:
            return {'infiltration_rate': 0.0, 'runoff_factor': 1.0}

        # Quantum computation of absorptivity
        result = self.quantum_model.compute_absorptivity(state, water_amount)

        # Update the soil state based on interaction
        actual_infiltration = water_amount * result['infiltration_rate']
        state.absorb_water(actual_infiltration)
        state.last_infiltration = result['infiltration_rate']
        state.measurement_count += 1

        # Propagate entanglement effects to neighbors
        self._propagate_entanglement(int(x), int(y), result['saturation_probability'])

        # Track statistics
        self.total_infiltration += actual_infiltration
        self.total_runoff += water_amount * result['runoff_factor']
        self.measurement_history.append({
            'x': x, 'y': y,
            'infiltration': result['infiltration_rate'],
            'runoff': result['runoff_factor']
        })

        return result

    def _propagate_entanglement(self, cx: int, cy: int, saturation_prob: float):
        """
        Propagate quantum entanglement effects to neighboring cells.

        When a cell is measured, entangled neighbors have their states
        partially updated based on the measurement outcome.
        """
        for dy in range(-int(self.entanglement_radius), int(self.entanglement_radius) + 1):
            for dx in range(-int(self.entanglement_radius), int(self.entanglement_radius) + 1):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                # Distance-weighted entanglement
                dist = np.sqrt(dx**2 + dy**2)
                if dist > self.entanglement_radius:
                    continue

                # Entanglement strength decays with distance
                entangle_strength = (1 - dist / self.entanglement_radius) * 0.3

                neighbor = self.grid[ny][nx]

                # Neighbor's saturation is correlated with measured cell
                neighbor.saturation_angle += entangle_strength * saturation_prob * 0.2
                neighbor.saturation_angle = min(np.pi, neighbor.saturation_angle)

                # Coherence slightly decays due to environmental decoherence
                neighbor.coherence *= (1 - entangle_strength * 0.05)

    def get_runoff_factor_at(self, x: float, y: float) -> float:
        """
        Get the runoff factor at a position (how easily water moves over surface).

        Higher runoff factor means water moves faster over the surface.
        """
        state = self.get_state(x, y)
        if state is None:
            return 0.5

        # Compute based on current quantum state without full measurement
        moisture = state.get_moisture_level()
        saturation = state.get_saturation_factor()

        # Impervious surfaces always have high runoff
        if state.soil_type == SoilType.IMPERVIOUS:
            return 1.0

        # Saturated soil has higher runoff
        return 0.2 + 0.6 * saturation + 0.2 * moisture

    def get_velocity_modifier(self, x: float, y: float) -> float:
        """
        Get velocity modifier for water movement at this position.

        Returns multiplier for water velocity (>1 means faster).
        """
        runoff = self.get_runoff_factor_at(x, y)
        # Higher runoff = easier movement = higher velocity
        return 0.5 + 1.5 * runoff

    def get_moisture_field(self) -> np.ndarray:
        """Get 2D array of moisture levels for visualization."""
        moisture = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                moisture[y, x] = self.grid[y][x].get_moisture_level()
        return moisture

    def get_saturation_field(self) -> np.ndarray:
        """Get 2D array of saturation levels for visualization."""
        saturation = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                saturation[y, x] = self.grid[y][x].get_saturation_factor()
        return saturation

    def get_coherence_field(self) -> np.ndarray:
        """Get 2D array of quantum coherence for visualization."""
        coherence = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                coherence[y, x] = self.grid[y][x].coherence
        return coherence


class QuantumWaterDynamics(torch.nn.Module):
    """
    Water particle dynamics enhanced with quantum soil absorptivity.

    This extends the classical Manning's equation with quantum effects:
    - Velocity is boosted over saturated/quantum-measured soil
    - Infiltration reduces particle "mass" (eventually removing it)
    - Quantum entanglement creates correlated flow patterns
    """

    def __init__(self, terrain, quantum_soil: QuantumSoilGrid,
                 num_particles: int):
        """
        Initialize quantum-enhanced water dynamics.

        Args:
            terrain: Terrain object with elevation and slope data
            quantum_soil: Quantum soil grid for absorptivity
            num_particles: Number of water particles
        """
        super().__init__()
        self.terrain = terrain
        self.quantum_soil = quantum_soil
        self.num_particles = num_particles
        self.eps = 1e-6

        # Track particle masses (for infiltration loss)
        self.particle_mass = torch.ones(num_particles)

        # Water interaction frequency (how often to query quantum model)
        self.interaction_dt = 0.1
        self.last_interaction_time = 0.0

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute derivatives for water particle positions.

        State: [x, y, z] for each particle (flattened)
        Returns: [dx/dt, dy/dt, dz/dt] for each particle
        """
        state = state.view(self.num_particles, 3)
        x, y, z = state[:, 0], state[:, 1], state[:, 2]

        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dz = torch.zeros_like(z)

        # Periodic quantum interaction
        current_t = float(t) if isinstance(t, torch.Tensor) else t
        do_quantum_interaction = (current_t - self.last_interaction_time) >= self.interaction_dt

        if do_quantum_interaction:
            self.last_interaction_time = current_t

        for i in range(self.num_particles):
            xi, yi = x[i].item(), y[i].item()

            # Skip particles outside bounds
            if xi < 2 or xi > self.terrain.width - 3 or yi < 2 or yi > self.terrain.height - 3:
                continue

            # Skip particles with no mass (fully infiltrated)
            if self.particle_mass[i] < 0.01:
                continue

            # Get terrain properties
            slope = self.terrain.get_slope(xi, yi)
            n = self.terrain.get_manning_n(xi, yi)
            flow_x, flow_y = self.terrain.get_flow_direction(xi, yi)
            surface = self.terrain.get_surface_type(xi, yi)

            # === QUANTUM ENHANCEMENT ===
            # Get quantum velocity modifier based on soil state
            velocity_modifier = self.quantum_soil.get_velocity_modifier(xi, yi)

            # Periodic quantum interaction: water affects soil
            if do_quantum_interaction:
                result = self.quantum_soil.interact_water(xi, yi, 0.1 * self.particle_mass[i].item())

                # Reduce particle mass based on infiltration
                infiltration = result['infiltration_rate']
                self.particle_mass[i] *= (1 - infiltration * 0.05)

                # Boost velocity based on runoff factor
                velocity_modifier *= (1 + result['runoff_factor'] * 0.5)

            # === MANNING'S EQUATION with quantum modification ===
            h = 0.1  # Water depth
            if slope > 0.001 and n > 0:
                V = (1.0 / n) * (h ** (2/3)) * np.sqrt(slope)
            else:
                V = 0.1

            # Apply quantum velocity modifier
            V *= velocity_modifier

            # Surface type boost (roads, channels)
            if surface == 0:  # Road
                V *= 1.5
            elif surface == 2:  # Channel
                V *= 2.0

            # Flow direction with mass-weighted velocity
            flow_mag = np.sqrt(flow_x**2 + flow_y**2) + self.eps
            vx = V * flow_x / flow_mag * self.particle_mass[i].item()
            vy = V * flow_y / flow_mag * self.particle_mass[i].item()

            # Particle-particle repulsion (dispersion)
            for j in range(self.num_particles):
                if i != j and self.particle_mass[j] > 0.01:
                    xj, yj = x[j].item(), y[j].item()
                    diff_x, diff_y = xi - xj, yi - yj
                    dist = np.sqrt(diff_x**2 + diff_y**2) + self.eps

                    if dist < 6:
                        force = 0.2 / (dist + 0.5)
                        vx += force * diff_x / dist
                        vy += force * diff_y / dist

            dx[i] = vx
            dy[i] = vy

            # Z follows terrain
            target_z = self.terrain.get_elevation(xi + vx * 0.1, yi + vy * 0.1)
            dz[i] = (target_z - z[i].item()) * 3.0

        return torch.stack([dx, dy, dz], dim=1).view(-1)

    def reset(self):
        """Reset particle masses for new simulation."""
        self.particle_mass = torch.ones(self.num_particles)
        self.last_interaction_time = 0.0
