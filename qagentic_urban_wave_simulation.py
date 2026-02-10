#!/usr/bin/env python3
"""
qAgentic Urban Water Flow Simulation
=====================================

Self-evolving multi-agent approach to urban water flow modeling.
Claude agents collaboratively tune 19 physics parameters across
iterations, starting from the same simulation as urban_wave_simulation.py
but with all constants read from an evolvable ParameterSet.

The original qode_framework/ and urban_wave_simulation.py are untouched.

Usage:
    1. Set ANTHROPIC_API_KEY in .env
    2. python qagentic_urban_wave_simulation.py
"""

import sys
import os

import numpy as np
import torch
from torchdiffeq import odeint
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from qagentic_approach.quantum import (  # noqa: E402
    QuantumAbsorptivityModel,
    QuantumSoilGrid,
    SoilType,
)
from qagentic_approach.evolution.parameter_space import ParameterSet  # noqa: E402
from qagentic_approach.evolution.evolution_loop import EvolutionLoop, EvolutionConfig  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# URBAN TERRAIN WITH BUILDINGS (parameterized Manning's n)
# =============================================================================

class UrbanHilltopTerrain:
    """
    Urban terrain with buildings on hilltop, road network, and soil patches.

    Surface Types:
        0 = ROAD (impervious, fast flow)
        1 = SOIL (permeable, quantum absorption)
        2 = CHANNEL (drainage)
        3 = BUILDING (obstacle, no flow)
    """

    ROAD = 0
    SOIL = 1
    CHANNEL = 2
    BUILDING = 3

    def __init__(self, width: int = 120, height: int = 120, seed: int = 42,
                 params: Optional[ParameterSet] = None):
        np.random.seed(seed)
        self.width = width
        self.height = height
        self.params = params or ParameterSet()

        # Grids
        self.elevation = np.zeros((height, width))
        self.surface_type = np.ones((height, width), dtype=int)  # Default soil

        # Building locations for visualization
        self.buildings = []

        # Generate terrain
        self._generate_terrain()
        self._generate_roads()
        self._generate_buildings_on_hilltop()
        self._generate_channel()
        self._compute_gradients()

        # Initialize quantum soil grid with evolvable entanglement_radius
        self.quantum_soil = QuantumSoilGrid(
            width=width,
            height=height,
            entanglement_radius=self.params.entanglement_radius,
            seed=seed
        )
        # Override shots if different from default
        self.quantum_soil.quantum_model = QuantumAbsorptivityModel(
            shots=self.params.quantum_shots
        )
        self._configure_soil_types()

        # Find hilltop
        self.hilltop = self._find_hilltop()

        # Calculate surface statistics
        total_cells = width * height
        road_cells = np.sum(self.surface_type == self.ROAD)
        soil_cells = np.sum(self.surface_type == self.SOIL)
        building_cells = np.sum(self.surface_type == self.BUILDING)

        print(f"Urban Hilltop Terrain: {width}x{height}")
        print("  Surface breakdown:")
        print(f"    Roads: {road_cells} ({100*road_cells/total_cells:.1f}%)")
        print(f"    Soil:  {soil_cells} ({100*soil_cells/total_cells:.1f}%)")
        print(f"    Buildings: {building_cells} ({100*building_cells/total_cells:.1f}%)")
        print(f"  Hilltop: {self.hilltop}")

    def _generate_terrain(self):
        """Generate terrain with prominent hill."""
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        base = 5 + 0.08 * (self.width - X + self.height - Y)
        self.elevation = base

        hill_cx, hill_cy = 35, 85
        hill = 40 * np.exp(-((X - hill_cx)**2 + (Y - hill_cy)**2) / (2 * 20**2))
        self.elevation += hill

        for cx, cy, amp, spread in [(75, 65, 15, 12), (90, 35, 10, 10)]:
            h = amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * spread**2))
            self.elevation += h

        self.elevation = gaussian_filter(self.elevation, sigma=2)
        self.elevation = np.maximum(self.elevation, 0)

    def _generate_roads(self):
        """Generate road grid."""
        road_width = 4

        for y in range(25, self.height - 10, 25):
            y_start = max(0, y - road_width // 2)
            y_end = min(self.height, y + road_width // 2 + 1)
            self.surface_type[y_start:y_end, :] = self.ROAD
            self.elevation[y_start:y_end, :] -= 0.5

        for x in range(25, self.width - 10, 25):
            x_start = max(0, x - road_width // 2)
            x_end = min(self.width, x + road_width // 2 + 1)
            self.surface_type[:, x_start:x_end] = self.ROAD
            self.elevation[:, x_start:x_end] -= 0.5

        hill_cx, hill_cy = 35, 85
        for angle in np.linspace(0, 2*np.pi, 100):
            for r in range(28, 32):
                x = int(hill_cx + r * np.cos(angle))
                y = int(hill_cy + r * np.sin(angle))
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.surface_type[y, x] = self.ROAD

    def _generate_buildings_on_hilltop(self):
        """Place buildings on the hilltop."""
        hill_cx, hill_cy = 35, 85

        self._place_building(hill_cx - 6, hill_cy - 6, 12, 12)

        building_positions = [
            (hill_cx - 18, hill_cy - 5, 8, 10),
            (hill_cx + 10, hill_cy - 5, 8, 10),
            (hill_cx - 5, hill_cy + 12, 10, 8),
            (hill_cx - 5, hill_cy - 18, 10, 8),
            (hill_cx - 16, hill_cy + 10, 6, 6),
            (hill_cx + 12, hill_cy + 10, 6, 6),
            (hill_cx - 16, hill_cy - 16, 6, 6),
            (hill_cx + 12, hill_cy - 16, 6, 6),
        ]
        for bx, by, bw, bh in building_positions:
            self._place_building(bx, by, bw, bh)

        for bx, by, bw, bh in [
            (55, 55, 8, 8), (80, 55, 10, 8), (55, 30, 8, 10),
            (80, 30, 8, 8), (100, 50, 6, 6),
        ]:
            self._place_building(bx, by, bw, bh)

    def _place_building(self, x: int, y: int, w: int, h: int):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(self.width, x + w), min(self.height, y + h)
        if x2 > x1 and y2 > y1:
            self.surface_type[y1:y2, x1:x2] = self.BUILDING
            self.elevation[y1:y2, x1:x2] += 2
            self.buildings.append((x1, y1, x2-x1, y2-y1))

    def _generate_channel(self):
        for x in range(self.width):
            y_center = int(10 + x * 0.05 + 3 * np.sin(x * 0.1))
            y_center = np.clip(y_center, 3, self.height - 3)
            for dy in range(-2, 3):
                y = y_center + dy
                if 0 <= y < self.height:
                    if self.surface_type[y, x] != self.BUILDING:
                        self.surface_type[y, x] = self.CHANNEL
                        self.elevation[y, x] = max(0, self.elevation[y, x] - 4)

    def _compute_gradients(self):
        self.grad_y, self.grad_x = np.gradient(self.elevation)
        self.slope = np.sqrt(self.grad_x**2 + self.grad_y**2)

    def _configure_soil_types(self):
        for y in range(self.height):
            for x in range(self.width):
                surface = self.surface_type[y, x]
                if surface == self.ROAD:
                    self.quantum_soil.set_soil_type(x, y, SoilType.IMPERVIOUS)
                elif surface == self.BUILDING:
                    self.quantum_soil.set_soil_type(x, y, SoilType.IMPERVIOUS)
                elif surface == self.CHANNEL:
                    self.quantum_soil.set_soil_type(x, y, SoilType.SAND)
                else:
                    if np.random.random() < 0.6:
                        self.quantum_soil.set_soil_type(x, y, SoilType.LOAM)
                    else:
                        self.quantum_soil.set_soil_type(x, y, SoilType.CLAY)

    def _find_hilltop(self) -> Tuple[int, int]:
        masked_elev = self.elevation.copy()
        masked_elev[self.surface_type == self.BUILDING] = -999
        idx = np.unravel_index(np.argmax(masked_elev), masked_elev.shape)
        return (idx[1], idx[0])

    def get_elevation(self, x: float, y: float) -> float:
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.elevation[iy, ix]

    def get_slope(self, x: float, y: float) -> float:
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.slope[iy, ix]

    def get_flow_direction(self, x: float, y: float) -> Tuple[float, float]:
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return -self.grad_x[iy, ix], -self.grad_y[iy, ix]

    def get_surface_type(self, x: float, y: float) -> int:
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return int(self.surface_type[iy, ix])

    def get_manning_n(self, x: float, y: float) -> float:
        """Manning's roughness — reads from ParameterSet."""
        surface = self.get_surface_type(x, y)
        return {
            self.ROAD: self.params.manning_n_road,
            self.SOIL: self.params.manning_n_soil,
            self.CHANNEL: self.params.manning_n_channel,
            self.BUILDING: 999.0,
        }.get(surface, self.params.manning_n_soil)

    def is_building(self, x: float, y: float) -> bool:
        return self.get_surface_type(x, y) == self.BUILDING


# =============================================================================
# URBAN WATER DYNAMICS — PARAMETERIZED
# =============================================================================

class UrbanWaterDynamics(torch.nn.Module):
    """
    Water dynamics with all physics constants read from ParameterSet.
    """

    def __init__(self, terrain: UrbanHilltopTerrain, num_particles: int,
                 params: Optional[ParameterSet] = None):
        super().__init__()
        self.terrain = terrain
        self.num_particles = num_particles
        self.params = params or ParameterSet()
        self.eps = 1e-6

        self.particle_mass = torch.ones(num_particles)
        self.particle_on_road = torch.zeros(num_particles, dtype=torch.bool)
        self.road_distance = torch.zeros(num_particles)
        self.soil_distance = torch.zeros(num_particles)

        self.interaction_dt = self.params.interaction_dt
        self.last_interaction_time = 0.0

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        state = state.view(self.num_particles, 3)
        x, y, z = state[:, 0], state[:, 1], state[:, 2]

        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dz = torch.zeros_like(z)

        current_t = float(t) if isinstance(t, torch.Tensor) else t
        do_quantum = (current_t - self.last_interaction_time) >= self.interaction_dt
        if do_quantum:
            self.last_interaction_time = current_t

        for i in range(self.num_particles):
            xi, yi = x[i].item(), y[i].item()

            if xi < 2 or xi > self.terrain.width - 3 or yi < 2 or yi > self.terrain.height - 3:
                continue
            if self.particle_mass[i] < 0.01:
                continue

            if self.terrain.is_building(xi, yi):
                flow_x, flow_y = self.terrain.get_flow_direction(xi, yi)
                dx[i] = -flow_x * 0.5
                dy[i] = -flow_y * 0.5
                continue

            surface = self.terrain.get_surface_type(xi, yi)
            slope = self.terrain.get_slope(xi, yi)
            n = self.terrain.get_manning_n(xi, yi)
            flow_x, flow_y = self.terrain.get_flow_direction(xi, yi)

            is_road = (surface == self.terrain.ROAD)
            self.particle_on_road[i] = is_road

            # === QUANTUM SOIL INTERACTION (parameterized) ===
            velocity_modifier = 1.0

            if do_quantum:
                result = self.terrain.quantum_soil.interact_water(
                    xi, yi, 0.1 * self.particle_mass[i].item()
                )

                if surface == self.terrain.SOIL:
                    infiltration = result['infiltration_rate']
                    self.particle_mass[i] *= (1 - infiltration * self.params.infiltration_loss_mult)
                    velocity_modifier = (self.params.soil_velocity_base
                                         + self.params.soil_velocity_runoff_coeff * result['runoff_factor'])
                else:
                    velocity_modifier = (self.params.road_velocity_base
                                         + self.params.road_velocity_runoff_coeff * result['runoff_factor'])

            # === MANNING'S EQUATION ===
            h = 0.1 * self.particle_mass[i].item()
            if slope > 0.001 and n < 100:
                V = (1.0 / n) * (h ** (2/3)) * np.sqrt(slope)
            else:
                V = 0.05

            V *= velocity_modifier

            # Surface-specific boosts (parameterized)
            if surface == self.terrain.ROAD:
                V *= self.params.road_surface_boost
            elif surface == self.terrain.CHANNEL:
                V *= self.params.channel_surface_boost
            elif surface == self.terrain.SOIL:
                V *= self.params.soil_surface_boost

            flow_mag = np.sqrt(flow_x**2 + flow_y**2) + self.eps
            vx = V * flow_x / flow_mag
            vy = V * flow_y / flow_mag

            # Particle dispersion (parameterized)
            dispersion_factor = self.params.road_dispersion if is_road else self.params.soil_dispersion
            for j in range(self.num_particles):
                if i != j and self.particle_mass[j] > 0.01:
                    xj, yj = x[j].item(), y[j].item()
                    diff_x, diff_y = xi - xj, yi - yj
                    dist = np.sqrt(diff_x**2 + diff_y**2) + self.eps
                    if dist < 5:
                        force = dispersion_factor / (dist + 0.5)
                        vx += force * diff_x / dist
                        vy += force * diff_y / dist

            # Avoid buildings
            for bx, by, bw, bh in self.terrain.buildings:
                cx, cy = bx + bw/2, by + bh/2
                diff_x, diff_y = xi - cx, yi - cy
                dist = np.sqrt(diff_x**2 + diff_y**2) + self.eps
                if dist < max(bw, bh) + 3:
                    force = 2.0 / (dist + 1)
                    vx += force * diff_x / dist
                    vy += force * diff_y / dist

            dx[i] = vx
            dy[i] = vy

            step_dist = np.sqrt(vx**2 + vy**2) * 0.05
            if is_road:
                self.road_distance[i] += step_dist
            else:
                self.soil_distance[i] += step_dist

            target_z = self.terrain.get_elevation(xi + vx * 0.1, yi + vy * 0.1)
            dz[i] = (target_z - z[i].item()) * 3.0

        return torch.stack([dx, dy, dz], dim=1).view(-1)

    def reset(self):
        self.particle_mass = torch.ones(self.num_particles)
        self.particle_on_road = torch.zeros(self.num_particles, dtype=torch.bool)
        self.road_distance = torch.zeros(self.num_particles)
        self.soil_distance = torch.zeros(self.num_particles)
        self.last_interaction_time = 0.0


# =============================================================================
# SIMULATION — PARAMETERIZED
# =============================================================================

class UrbanWaterSimulation:
    """Simulation comparing water flow on soil vs roads, with evolvable params."""

    def __init__(self, num_particles: int = 40, seed: int = 42,
                 params: Optional[ParameterSet] = None):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.params = params or ParameterSet()
        self.terrain = UrbanHilltopTerrain(width=120, height=120, seed=seed,
                                           params=self.params)
        self.num_particles = num_particles
        self.source = (35, 75)

        self.trajectories = None
        self.times = None
        self.dynamics = None

    def _create_particles(self) -> torch.Tensor:
        states = []
        for i in range(self.num_particles):
            angle = 2 * np.pi * i / self.num_particles
            radius = np.random.uniform(1, 4)
            x = self.source[0] + radius * np.cos(angle)
            y = self.source[1] + radius * np.sin(angle)

            attempts = 0
            while self.terrain.is_building(x, y) and attempts < 10:
                angle += 0.3
                x = self.source[0] + radius * np.cos(angle)
                y = self.source[1] + radius * np.sin(angle)
                attempts += 1

            z = self.terrain.get_elevation(x, y)
            states.append([x, y, z])

        return torch.tensor(states, dtype=torch.float32, device=device).view(-1)

    def run(self, t_duration: float = 8.0, num_steps: int = 120):
        print("\nRunning qAgentic Urban Water Flow Simulation...")
        print(f"  Source: {self.source}")
        print(f"  Particles: {self.num_particles}")
        print(f"  Duration: {t_duration}s")

        self.dynamics = UrbanWaterDynamics(self.terrain, self.num_particles,
                                           params=self.params)
        initial_state = self._create_particles()
        t_span = torch.linspace(0, t_duration, num_steps, device=device)

        with torch.no_grad():
            trajectories = odeint(
                self.dynamics, initial_state, t_span,
                method='euler', options={'step_size': 0.05}
            )

        self.trajectories = trajectories.view(num_steps, self.num_particles, 3).cpu().numpy()
        self.times = t_span.cpu().numpy()

        print("Simulation complete!")

        road_dist = self.dynamics.road_distance.numpy()
        soil_dist = self.dynamics.soil_distance.numpy()
        final_mass = self.dynamics.particle_mass.numpy()

        print("\nStatistics:")
        print(f"  Avg distance on roads: {road_dist.mean():.1f}")
        print(f"  Avg distance on soil: {soil_dist.mean():.1f}")
        print(f"  Avg final mass: {final_mass.mean():.2f}")
        print(f"  Water lost to infiltration: {(1 - final_mass.mean()) * 100:.1f}%")


# =============================================================================
# SIMULATION FACTORY FOR EVOLUTION LOOP
# =============================================================================

def create_simulation(params: ParameterSet, config: EvolutionConfig):
    """Factory function: creates a parameterized simulation instance."""
    return UrbanWaterSimulation(
        num_particles=config.sim_particles,
        seed=config.seed,
        params=params,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the multi-agent self-evolving urban water simulation."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in .env file.")
        print("Example: ANTHROPIC_API_KEY=sk-ant-api03-...")
        sys.exit(1)

    print("=" * 70)
    print("qAGENTIC: SELF-EVOLVING URBAN WATER FLOW SIMULATION")
    print("=" * 70)
    print()

    # Configure the evolution
    config = EvolutionConfig(
        max_iterations=5,
        convergence_threshold=0.01,
        convergence_window=3,
        learning_rate=0.5,
        params_to_evolve=None,  # Evolve all parameters
        use_claude_arbitration=True,
        claude_model="claude-sonnet-4-20250514",
        sim_particles=40,
        sim_duration=8.0,
        sim_steps=120,
        seed=42,
        verbose=True,
        save_history=True,
        history_path="evolution_history.json",
    )

    # Create and run the evolution loop
    loop = EvolutionLoop(
        config=config,
        api_key=api_key,
        initial_params=ParameterSet(),  # Start from defaults
        sim_factory=create_simulation,
    )

    final_params, history = loop.evolve()

    # Print final results
    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"\nIterations: {len(history.records)}")
    print("\nFinal Parameters:")
    default = ParameterSet()
    diff = default.diff(final_params)
    if diff:
        for name, (orig, final) in diff.items():
            print(f"  {name}: {orig} -> {final:.4f}")
    else:
        print("  (no changes from defaults)")

    # Run one final simulation with optimized params and generate visuals
    print("\nRunning final simulation with optimized parameters...")
    final_sim = UrbanWaterSimulation(
        num_particles=50, seed=42, params=final_params,
    )
    final_sim.run(t_duration=10.0, num_steps=150)

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nResults saved to {results_dir}/")
    print("Evolution history saved to evolution_history.json")

    return final_sim, final_params, history


if __name__ == "__main__":
    sim, params, history = main()
