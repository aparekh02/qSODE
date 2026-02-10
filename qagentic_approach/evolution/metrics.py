"""
Simulation metrics extraction for agent analysis.

MetricsExtractor pulls structured data from a completed simulation
so that each agent receives the numbers it needs to reason about.
"""

import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any


@dataclass
class SimulationMetrics:
    """Structured metrics from one simulation run."""

    # Velocity stats
    avg_road_velocity: float = 0.0
    avg_soil_velocity: float = 0.0
    avg_channel_velocity: float = 0.0
    velocity_ratio_road_soil: float = 0.0
    velocity_percentiles_road: Dict[str, float] = field(default_factory=dict)
    velocity_percentiles_soil: Dict[str, float] = field(default_factory=dict)

    # Distance stats
    avg_distance_road: float = 0.0
    avg_distance_soil: float = 0.0
    total_distance_all: float = 0.0

    # Mass / Infiltration stats
    avg_final_mass: float = 0.0
    mass_loss_fraction: float = 0.0
    mass_loss_curve: List[float] = field(default_factory=list)

    # Quantum soil stats
    total_infiltration: float = 0.0
    total_runoff: float = 0.0
    avg_saturation: float = 0.0
    avg_moisture: float = 0.0
    avg_coherence: float = 0.0
    measurement_count: int = 0

    # Surface state tracker data
    wave_comparison: Dict[str, Any] = field(default_factory=dict)

    # Particle behavior
    num_particles_on_road_final: int = 0
    num_particles_on_soil_final: int = 0
    dispersion_road: float = 0.0
    dispersion_soil: float = 0.0

    # Terrain stats
    mean_slope: float = 0.0
    max_slope: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsExtractor:
    """Extracts SimulationMetrics from a completed UrbanWaterSimulation."""

    @staticmethod
    def extract(sim) -> SimulationMetrics:
        """
        Pull all relevant metrics from simulation results.

        Args:
            sim: UrbanWaterSimulation with completed run
                 (has .trajectories, .times, .dynamics, .terrain)
        """
        trajectories = sim.trajectories   # [steps, particles, 3]
        times = sim.times                 # [steps]
        dynamics = sim.dynamics
        terrain = sim.terrain

        num_steps, num_particles, _ = trajectories.shape

        # --- Velocity by surface type ---
        road_velocities = []
        soil_velocities = []
        channel_velocities = []

        for t in range(1, num_steps):
            dt = times[t] - times[t - 1]
            if dt < 1e-8:
                continue
            for p in range(num_particles):
                dx = trajectories[t, p, 0] - trajectories[t - 1, p, 0]
                dy = trajectories[t, p, 1] - trajectories[t - 1, p, 1]
                v = np.sqrt(dx**2 + dy**2) / dt

                x, y = trajectories[t, p, 0], trajectories[t, p, 1]
                surface = terrain.get_surface_type(x, y)

                if surface == terrain.ROAD:
                    road_velocities.append(v)
                elif surface == terrain.SOIL:
                    soil_velocities.append(v)
                elif surface == terrain.CHANNEL:
                    channel_velocities.append(v)

        road_velocities = np.array(road_velocities) if road_velocities else np.array([0.0])
        soil_velocities = np.array(soil_velocities) if soil_velocities else np.array([0.0])
        channel_velocities = np.array(channel_velocities) if channel_velocities else np.array([0.0])

        avg_road_v = float(np.mean(road_velocities))
        avg_soil_v = float(np.mean(soil_velocities))
        avg_chan_v = float(np.mean(channel_velocities))

        def percentiles(arr):
            if len(arr) == 0:
                return {}
            return {
                "p10": float(np.percentile(arr, 10)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
            }

        # --- Distance stats ---
        road_dist = dynamics.road_distance.numpy()
        soil_dist = dynamics.soil_distance.numpy()

        # --- Mass stats ---
        final_mass = dynamics.particle_mass.numpy()
        avg_mass = float(np.mean(final_mass))

        # Mass loss curve: approximate by tracking particle positions over time
        # and checking mass at end (mass is only available at final state)
        mass_curve = []
        for t_idx in range(num_steps):
            # Linear interpolation of mass loss over time
            frac = t_idx / max(num_steps - 1, 1)
            mass_curve.append(float(1.0 - frac * (1.0 - avg_mass)))

        # --- Quantum soil stats ---
        soil_grid = terrain.quantum_soil
        sat_field = soil_grid.get_saturation_field()
        moisture_field = soil_grid.get_moisture_field()
        coherence_field = soil_grid.get_coherence_field()

        wave_comp = {}
        if hasattr(soil_grid, 'state_tracker'):
            wave_comp = soil_grid.state_tracker.get_wave_comparison()

        # --- Particle surface distribution at final step ---
        on_road_final = 0
        on_soil_final = 0
        road_positions = []
        soil_positions = []

        for p in range(num_particles):
            x, y = trajectories[-1, p, 0], trajectories[-1, p, 1]
            surface = terrain.get_surface_type(x, y)
            if surface == terrain.ROAD:
                on_road_final += 1
                road_positions.append([x, y])
            elif surface == terrain.SOIL:
                on_soil_final += 1
                soil_positions.append([x, y])

        road_positions = np.array(road_positions) if road_positions else np.zeros((0, 2))
        soil_positions = np.array(soil_positions) if soil_positions else np.zeros((0, 2))

        disp_road = float(np.std(road_positions)) if len(road_positions) > 1 else 0.0
        disp_soil = float(np.std(soil_positions)) if len(soil_positions) > 1 else 0.0

        return SimulationMetrics(
            avg_road_velocity=avg_road_v,
            avg_soil_velocity=avg_soil_v,
            avg_channel_velocity=avg_chan_v,
            velocity_ratio_road_soil=avg_road_v / max(avg_soil_v, 1e-6),
            velocity_percentiles_road=percentiles(road_velocities),
            velocity_percentiles_soil=percentiles(soil_velocities),
            avg_distance_road=float(np.mean(road_dist)),
            avg_distance_soil=float(np.mean(soil_dist)),
            total_distance_all=float(np.sum(road_dist) + np.sum(soil_dist)),
            avg_final_mass=avg_mass,
            mass_loss_fraction=float(1.0 - avg_mass),
            mass_loss_curve=mass_curve,
            total_infiltration=float(soil_grid.total_infiltration),
            total_runoff=float(soil_grid.total_runoff),
            avg_saturation=float(np.mean(sat_field)),
            avg_moisture=float(np.mean(moisture_field)),
            avg_coherence=float(np.mean(coherence_field)),
            measurement_count=len(soil_grid.measurement_history),
            wave_comparison=wave_comp,
            num_particles_on_road_final=on_road_final,
            num_particles_on_soil_final=on_soil_final,
            dispersion_road=disp_road,
            dispersion_soil=disp_soil,
            mean_slope=float(np.mean(terrain.slope)),
            max_slope=float(np.max(terrain.slope)),
        )
