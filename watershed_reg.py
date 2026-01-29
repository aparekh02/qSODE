#!/usr/bin/env python3
"""
Quantum-Enhanced Watershed Simulation
=====================================

This is the main entry point for the qODE framework's quantum watershed model.
It uses Qiskit quantum computing to model soil water absorptivity through
quantum superposition, where:

1. Soil moisture states exist in quantum superposition until water interacts
2. Quantum measurement determines actual infiltration vs surface runoff
3. Saturated soil (post-measurement) allows water to move more easily over surface
4. Neighboring soil cells are entangled, creating correlated saturation patterns

Physics Model:
-------------
- Water particles follow terrain gradients (gravity-driven flow)
- Manning's equation governs base flow velocity
- Quantum soil model modifies velocity based on saturation state
- Infiltration reduces particle mass; fully infiltrated particles stop

Quantum Enhancement:
-------------------
- 3-qubit circuit per soil cell: moisture, saturation, surface condition
- Measurement collapse determines infiltration rate
- Entanglement propagates saturation to neighbors
- Coherence decays with repeated measurements

Usage:
    python main.py                    # Run full quantum simulation
    python main.py --classical        # Run classical comparison
    python main.py --no-video         # Skip video generation
"""

import sys
import os
import argparse
import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')

# Add framework to path
sys.path.insert(0, os.path.dirname(__file__))

# Import quantum module
from qode_framework.quantum import (
    QuantumSoilState,
    QuantumAbsorptivityModel,
    QuantumSoilGrid,
    QuantumWaterDynamics,
    SoilType,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# TERRAIN WITH QUANTUM SOIL
# =============================================================================

class QuantumWatershedTerrain:
    """
    Terrain with integrated quantum soil model for water absorptivity.

    The terrain has:
    - Elevation map with hills and valleys
    - Road network (impervious surfaces)
    - Drainage channels
    - Quantum soil grid for absorptivity
    """

    def __init__(self, width: int = 120, height: int = 120, seed: int = 42):
        np.random.seed(seed)
        self.width = width
        self.height = height

        # Terrain grids
        self.elevation = np.zeros((height, width))
        self.surface_type = np.ones((height, width), dtype=int)  # 1 = soil default

        # Generate terrain
        self._generate_terrain()
        self._generate_roads()
        self._generate_channel()
        self._compute_gradients()

        # Initialize quantum soil grid
        self.quantum_soil = QuantumSoilGrid(
            width=width,
            height=height,
            entanglement_radius=3.0,
            seed=seed
        )
        self._configure_soil_types()

        # Find key locations
        self.hilltop = self._find_hilltop()
        self.flat_ground = self._find_flat_ground()

        print(f"Terrain initialized: {width}x{height}")
        print(f"  Hilltop: {self.hilltop} (elevation: {self.get_elevation(*self.hilltop):.1f})")
        print(f"  Flat ground: {self.flat_ground} (elevation: {self.get_elevation(*self.flat_ground):.1f})")

    def _generate_terrain(self):
        """Generate terrain with hills and valleys."""
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # Base slope toward drainage
        base = 5 + 0.08 * (self.width - X + self.height - Y)
        self.elevation = base

        # Main hill
        hill_cx, hill_cy = 30, 90
        hill = 35 * np.exp(-((X - hill_cx)**2 + (Y - hill_cy)**2) / (2 * 18**2))
        self.elevation += hill

        # Secondary hills
        for cx, cy, amp, spread in [(70, 70, 20, 12), (90, 40, 15, 10), (50, 50, 12, 10)]:
            h = amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * spread**2))
            self.elevation += h

        # Valley
        valley = -8 * np.exp(-((X - 85)**2 + (Y - 25)**2) / (2 * 20**2))
        self.elevation += valley

        # Smooth and clamp
        self.elevation = gaussian_filter(self.elevation, sigma=2)
        self.elevation = np.maximum(self.elevation, 0)

    def _generate_roads(self):
        """Generate road grid (impervious surfaces)."""
        spacing, width = 30, 3

        for y in range(spacing, self.height - 10, spacing):
            y_start = max(0, y - width // 2)
            y_end = min(self.height, y + width // 2 + 1)
            self.surface_type[y_start:y_end, :] = 0  # Road
            self.elevation[y_start:y_end, :] -= 1

        for x in range(spacing, self.width - 10, spacing):
            x_start = max(0, x - width // 2)
            x_end = min(self.width, x + width // 2 + 1)
            self.surface_type[:, x_start:x_end] = 0  # Road
            self.elevation[:, x_start:x_end] -= 1

    def _generate_channel(self):
        """Generate drainage channel."""
        for x in range(self.width):
            y_center = int(self.height - 15 - x * 0.7 + 5 * np.sin(x * 0.08))
            y_center = np.clip(y_center, 5, self.height - 5)

            for dy in range(-2, 3):
                y = y_center + dy
                if 0 <= y < self.height:
                    self.surface_type[y, x] = 2  # Channel
                    self.elevation[y, x] = max(0, self.elevation[y, x] - 6)

    def _compute_gradients(self):
        """Compute terrain gradients for flow direction."""
        self.grad_y, self.grad_x = np.gradient(self.elevation)
        self.slope = np.sqrt(self.grad_x**2 + self.grad_y**2)

    def _configure_soil_types(self):
        """Configure quantum soil types based on surface."""
        for y in range(self.height):
            for x in range(self.width):
                surface = self.surface_type[y, x]
                if surface == 0:  # Road
                    self.quantum_soil.set_soil_type(x, y, SoilType.IMPERVIOUS)
                elif surface == 2:  # Channel
                    self.quantum_soil.set_soil_type(x, y, SoilType.SAND)  # High permeability
                else:  # Soil
                    # Vary soil type slightly for realism
                    if np.random.random() < 0.7:
                        self.quantum_soil.set_soil_type(x, y, SoilType.LOAM)
                    else:
                        self.quantum_soil.set_soil_type(x, y, SoilType.CLAY)

    def _find_hilltop(self) -> Tuple[int, int]:
        """Find highest point."""
        idx = np.unravel_index(np.argmax(self.elevation), self.elevation.shape)
        return (idx[1], idx[0])

    def _find_flat_ground(self) -> Tuple[int, int]:
        """Find flat, low area."""
        search_region = self.elevation[10:50, 60:110]
        slope_region = self.slope[10:50, 60:110]
        score = search_region + slope_region * 10
        idx = np.unravel_index(np.argmin(score), score.shape)
        return (idx[1] + 60, idx[0] + 10)

    # Accessors
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
        """Manning's roughness coefficient."""
        surface = self.get_surface_type(x, y)
        return {0: 0.013, 1: 0.035, 2: 0.025}.get(surface, 0.035)


# =============================================================================
# CLASSICAL WATER DYNAMICS (for comparison)
# =============================================================================

class ClassicalWaterDynamics(torch.nn.Module):
    """Classical water dynamics without quantum effects."""

    def __init__(self, terrain: QuantumWatershedTerrain, num_particles: int):
        super().__init__()
        self.terrain = terrain
        self.num_particles = num_particles
        self.eps = 1e-6

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        state = state.view(self.num_particles, 3)
        x, y, z = state[:, 0], state[:, 1], state[:, 2]

        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dz = torch.zeros_like(z)

        for i in range(self.num_particles):
            xi, yi = x[i].item(), y[i].item()

            if xi < 2 or xi > self.terrain.width - 3 or yi < 2 or yi > self.terrain.height - 3:
                continue

            slope = self.terrain.get_slope(xi, yi)
            n = self.terrain.get_manning_n(xi, yi)
            flow_x, flow_y = self.terrain.get_flow_direction(xi, yi)
            surface = self.terrain.get_surface_type(xi, yi)

            h = 0.1
            if slope > 0.001 and n > 0:
                V = (1.0 / n) * (h ** (2/3)) * np.sqrt(slope)
            else:
                V = 0.1

            if surface == 0:
                V *= 1.5
            elif surface == 2:
                V *= 2.0

            flow_mag = np.sqrt(flow_x**2 + flow_y**2) + self.eps
            vx = V * flow_x / flow_mag
            vy = V * flow_y / flow_mag

            for j in range(self.num_particles):
                if i != j:
                    xj, yj = x[j].item(), y[j].item()
                    diff_x, diff_y = xi - xj, yi - yj
                    dist = np.sqrt(diff_x**2 + diff_y**2) + self.eps
                    if dist < 6:
                        force = 0.2 / (dist + 0.5)
                        vx += force * diff_x / dist
                        vy += force * diff_y / dist

            dx[i] = vx
            dy[i] = vy

            target_z = self.terrain.get_elevation(xi + vx * 0.1, yi + vy * 0.1)
            dz[i] = (target_z - z[i].item()) * 3.0

        return torch.stack([dx, dy, dz], dim=1).view(-1)


# =============================================================================
# QUANTUM WATERSHED SIMULATION
# =============================================================================

class QuantumWatershedSimulation:
    """
    Quantum-enhanced watershed simulation comparing hilltop vs flat ground flow,
    with and without quantum soil effects.
    """

    def __init__(self, num_particles: int = 30, num_waves: int = 3,
                 use_quantum: bool = True, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.terrain = QuantumWatershedTerrain(width=120, height=120, seed=seed)
        self.num_particles = num_particles
        self.num_waves = num_waves
        self.use_quantum = use_quantum

        # Sources
        self.hilltop_source = self.terrain.hilltop
        self.flat_source = self.terrain.flat_ground

        # Results storage
        self.hilltop_trajectories = []
        self.flat_trajectories = []
        self.hilltop_times = []
        self.flat_times = []

        # Quantum state snapshots (for visualization)
        self.moisture_snapshots = []
        self.saturation_snapshots = []
        self.coherence_snapshots = []

        print(f"\nSimulation mode: {'QUANTUM' if use_quantum else 'CLASSICAL'}")

    def _create_particles(self, source: Tuple[float, float]) -> torch.Tensor:
        """Create water particles at source."""
        states = []
        for i in range(self.num_particles):
            angle = 2 * np.pi * i / self.num_particles
            radius = np.random.uniform(0.5, 2.0)

            x = source[0] + radius * np.cos(angle)
            y = source[1] + radius * np.sin(angle)
            z = self.terrain.get_elevation(x, y)

            states.append([x, y, z])

        return torch.tensor(states, dtype=torch.float32, device=device).view(-1)

    def run_simulation(self, source: Tuple[float, float], name: str,
                       t_duration: float = 6.0, num_steps: int = 80) -> Tuple[List, List]:
        """Run multi-wave simulation from source."""

        print(f"\n  Running {name} ({'quantum' if self.use_quantum else 'classical'})...")
        print(f"    Source: {source}, Elevation: {self.terrain.get_elevation(*source):.1f}")

        all_trajectories = []
        all_times = []

        # Create dynamics model
        if self.use_quantum:
            dynamics = QuantumWaterDynamics(
                self.terrain, self.terrain.quantum_soil, self.num_particles
            )
        else:
            dynamics = ClassicalWaterDynamics(self.terrain, self.num_particles)

        for wave in range(self.num_waves):
            t_start = wave * t_duration
            print(f"    Wave {wave + 1}: t={t_start:.1f} to {t_start + t_duration:.1f}")

            # Reset dynamics for new wave
            if self.use_quantum:
                dynamics.reset()

            initial_state = self._create_particles(source)
            t_span = torch.linspace(0, t_duration, num_steps, device=device)

            with torch.no_grad():
                trajectories = odeint(
                    dynamics, initial_state, t_span,
                    method='euler', options={'step_size': 0.05}
                )

            trajectories = trajectories.view(num_steps, self.num_particles, 3).cpu().numpy()
            all_trajectories.append(trajectories)
            all_times.append(t_span.cpu().numpy() + t_start)

            # Capture quantum state snapshot after each wave
            if self.use_quantum:
                self.moisture_snapshots.append(
                    self.terrain.quantum_soil.get_moisture_field().copy()
                )
                self.saturation_snapshots.append(
                    self.terrain.quantum_soil.get_saturation_field().copy()
                )
                self.coherence_snapshots.append(
                    self.terrain.quantum_soil.get_coherence_field().copy()
                )

        return all_trajectories, all_times

    def run(self):
        """Run both hilltop and flat ground simulations."""
        print("=" * 70)
        print(f"QUANTUM WATERSHED SIMULATION ({'QUANTUM' if self.use_quantum else 'CLASSICAL'})")
        print("=" * 70)

        self.hilltop_trajectories, self.hilltop_times = self.run_simulation(
            self.hilltop_source, "HILLTOP"
        )

        self.flat_trajectories, self.flat_times = self.run_simulation(
            self.flat_source, "FLAT GROUND"
        )

        print("\nSimulations complete!")

        if self.use_quantum:
            soil = self.terrain.quantum_soil
            print(f"\nQuantum Statistics:")
            print(f"  Total infiltration: {soil.total_infiltration:.2f}")
            print(f"  Total runoff: {soil.total_runoff:.2f}")
            print(f"  Measurements made: {len(soil.measurement_history)}")

    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate comparison statistics."""
        stats = {'hilltop': {}, 'flat': {}}

        for name, trajectories, source in [
            ('hilltop', self.hilltop_trajectories, self.hilltop_source),
            ('flat', self.flat_trajectories, self.flat_source)
        ]:
            all_velocities = []
            all_distances = []

            for wave_traj in trajectories:
                for t in range(1, len(wave_traj)):
                    dx = wave_traj[t, :, 0] - wave_traj[t-1, :, 0]
                    dy = wave_traj[t, :, 1] - wave_traj[t-1, :, 1]
                    v = np.sqrt(dx**2 + dy**2) / 0.075
                    all_velocities.extend(v)

                for p in range(self.num_particles):
                    dist = np.sum(np.sqrt(
                        np.diff(wave_traj[:, p, 0])**2 +
                        np.diff(wave_traj[:, p, 1])**2
                    ))
                    all_distances.append(dist)

            stats[name] = {
                'start_elevation': self.terrain.get_elevation(*source),
                'mean_velocity': np.mean(all_velocities),
                'max_velocity': np.max(all_velocities),
                'mean_distance': np.mean(all_distances),
            }

        return stats

    def visualize(self, save_path: Optional[str] = None):
        """Create comprehensive visualization."""
        fig = plt.figure(figsize=(20, 16))

        stats = self.calculate_statistics()

        # 1. 3D terrain
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self._plot_3d_comparison(ax1)

        # 2. Top-down view
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_topdown_comparison(ax2)

        # 3. Quantum state evolution OR velocity comparison
        ax3 = fig.add_subplot(2, 2, 3)
        if self.use_quantum and len(self.saturation_snapshots) > 0:
            self._plot_quantum_evolution(ax3)
        else:
            self._plot_velocity_comparison(ax3)

        # 4. Statistics
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_statistics(ax4, stats)

        mode = "Quantum" if self.use_quantum else "Classical"
        plt.suptitle(f'{mode} Watershed Simulation: Hilltop vs Flat Ground',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def _plot_3d_comparison(self, ax):
        """Plot 3D terrain with trajectories."""
        x = np.arange(0, self.terrain.width, 2)
        y = np.arange(0, self.terrain.height, 2)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.terrain.elevation[
                    min(int(Y[i, j]), self.terrain.height - 1),
                    min(int(X[i, j]), self.terrain.width - 1)
                ]

        ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, linewidth=0)

        # Hilltop trajectories
        colors_hill = plt.cm.Reds(np.linspace(0.4, 0.9, self.num_waves))
        for wave_idx, traj in enumerate(self.hilltop_trajectories):
            for p in range(0, self.num_particles, 3):
                ax.plot(traj[:, p, 0], traj[:, p, 1], traj[:, p, 2] + 0.5,
                       color=colors_hill[wave_idx], linewidth=1, alpha=0.7)

        # Flat trajectories
        colors_flat = plt.cm.Blues(np.linspace(0.4, 0.9, self.num_waves))
        for wave_idx, traj in enumerate(self.flat_trajectories):
            for p in range(0, self.num_particles, 3):
                ax.plot(traj[:, p, 0], traj[:, p, 1], traj[:, p, 2] + 0.5,
                       color=colors_flat[wave_idx], linewidth=1, alpha=0.7)

        # Mark sources
        hill_z = self.terrain.get_elevation(*self.hilltop_source)
        flat_z = self.terrain.get_elevation(*self.flat_source)

        ax.scatter(*self.hilltop_source, hill_z + 3, color='red', s=200,
                  marker='^', edgecolor='white', linewidth=2, label='Hilltop')
        ax.scatter(*self.flat_source, flat_z + 3, color='blue', s=200,
                  marker='o', edgecolor='white', linewidth=2, label='Flat Ground')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain: Hilltop (Red) vs Flat Ground (Blue)', fontweight='bold')
        ax.legend(loc='upper left')
        ax.view_init(elev=25, azim=45)

    def _plot_topdown_comparison(self, ax):
        """Plot top-down view with quantum saturation overlay."""
        # Background: saturation if quantum, else elevation
        if self.use_quantum and len(self.saturation_snapshots) > 0:
            final_saturation = self.saturation_snapshots[-1]
            im = ax.imshow(final_saturation, origin='lower', cmap='Blues',
                          extent=[0, self.terrain.width, 0, self.terrain.height],
                          alpha=0.7, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='Saturation', shrink=0.8)
        else:
            im = ax.imshow(self.terrain.elevation, origin='lower', cmap='terrain',
                          extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.7)
            plt.colorbar(im, ax=ax, label='Elevation', shrink=0.8)

        ax.contour(self.terrain.elevation, levels=15, colors='brown', alpha=0.4,
                  extent=[0, self.terrain.width, 0, self.terrain.height], linewidths=0.5)

        # Trajectories
        for wave_idx, traj in enumerate(self.hilltop_trajectories):
            alpha = 0.4 + 0.2 * wave_idx / self.num_waves
            for p in range(self.num_particles):
                ax.plot(traj[:, p, 0], traj[:, p, 1], color='red', linewidth=0.8, alpha=alpha)
            ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color='darkred', s=10, alpha=0.5)

        for wave_idx, traj in enumerate(self.flat_trajectories):
            alpha = 0.4 + 0.2 * wave_idx / self.num_waves
            for p in range(self.num_particles):
                ax.plot(traj[:, p, 0], traj[:, p, 1], color='blue', linewidth=0.8, alpha=alpha)
            ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color='darkblue', s=10, alpha=0.5)

        ax.scatter(*self.hilltop_source, color='red', s=300, marker='^',
                  edgecolor='white', linewidth=2, zorder=20, label='Hilltop Start')
        ax.scatter(*self.flat_source, color='blue', s=300, marker='o',
                  edgecolor='white', linewidth=2, zorder=20, label='Flat Ground Start')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        title = 'Top-Down: Final Quantum Saturation' if self.use_quantum else 'Top-Down: Flow Paths'
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right')

    def _plot_quantum_evolution(self, ax):
        """Plot quantum state evolution over waves."""
        if len(self.saturation_snapshots) == 0:
            ax.text(0.5, 0.5, 'No quantum data', ha='center', va='center')
            return

        # Plot mean saturation per wave
        waves = list(range(1, len(self.saturation_snapshots) + 1))
        mean_saturation = [s.mean() for s in self.saturation_snapshots]
        max_saturation = [s.max() for s in self.saturation_snapshots]
        mean_coherence = [c.mean() for c in self.coherence_snapshots]

        ax.plot(waves, mean_saturation, 'b-o', linewidth=2, label='Mean Saturation', markersize=8)
        ax.plot(waves, max_saturation, 'r--s', linewidth=2, label='Max Saturation', markersize=8)
        ax.plot(waves, mean_coherence, 'g-.^', linewidth=2, label='Mean Coherence', markersize=8)

        ax.set_xlabel('Wave Number')
        ax.set_ylabel('Value')
        ax.set_title('Quantum State Evolution Over Waves', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        # Annotation
        ax.text(0.98, 0.02,
               'Saturation increases with each wave\n'
               'Coherence decays from measurements\n'
               'Higher saturation → faster surface flow',
               transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _plot_velocity_comparison(self, ax):
        """Plot velocity over time."""
        def calc_velocities(trajectories, times):
            all_v, all_t = [], []
            for wave_idx, (traj, time) in enumerate(zip(trajectories, times)):
                for t_idx in range(1, len(traj)):
                    dt = time[t_idx] - time[t_idx - 1]
                    if dt > 0:
                        dx = traj[t_idx, :, 0] - traj[t_idx-1, :, 0]
                        dy = traj[t_idx, :, 1] - traj[t_idx-1, :, 1]
                        v = np.mean(np.sqrt(dx**2 + dy**2)) / dt
                        all_v.append(v)
                        all_t.append(time[t_idx])
            return np.array(all_t), np.array(all_v)

        hill_t, hill_v = calc_velocities(self.hilltop_trajectories, self.hilltop_times)
        flat_t, flat_v = calc_velocities(self.flat_trajectories, self.flat_times)

        ax.plot(hill_t, hill_v, 'r-', linewidth=2, label='Hilltop', alpha=0.8)
        ax.plot(flat_t, flat_v, 'b-', linewidth=2, label='Flat Ground', alpha=0.8)

        if len(hill_v) > 10:
            hill_v_smooth = gaussian_filter(hill_v, sigma=3)
            flat_v_smooth = gaussian_filter(flat_v, sigma=3)
            ax.plot(hill_t, hill_v_smooth, 'r--', linewidth=3, alpha=0.9)
            ax.plot(flat_t, flat_v_smooth, 'b--', linewidth=3, alpha=0.9)

        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Velocity')
        ax.set_title('Velocity Comparison Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_statistics(self, ax, stats):
        """Plot statistics summary."""
        ax.axis('off')

        mode = "QUANTUM" if self.use_quantum else "CLASSICAL"
        text = f"{mode} WATERSHED STATISTICS\n"
        text += "=" * 50 + "\n\n"

        text += f"{'Metric':<25} {'Hilltop':>12} {'Flat Ground':>12}\n"
        text += "-" * 50 + "\n"

        text += f"{'Start Elevation':<25} {stats['hilltop']['start_elevation']:>12.1f} {stats['flat']['start_elevation']:>12.1f}\n"
        text += f"{'Mean Velocity':<25} {stats['hilltop']['mean_velocity']:>12.2f} {stats['flat']['mean_velocity']:>12.2f}\n"
        text += f"{'Max Velocity':<25} {stats['hilltop']['max_velocity']:>12.2f} {stats['flat']['max_velocity']:>12.2f}\n"
        text += f"{'Mean Distance':<25} {stats['hilltop']['mean_distance']:>12.1f} {stats['flat']['mean_distance']:>12.1f}\n"

        v_ratio = stats['hilltop']['mean_velocity'] / max(stats['flat']['mean_velocity'], 0.01)
        text += f"\nVelocity Ratio (Hill/Flat): {v_ratio:.2f}x\n\n"

        if self.use_quantum:
            soil = self.terrain.quantum_soil
            text += "QUANTUM EFFECTS:\n"
            text += "-" * 50 + "\n"
            text += f"{'Total Infiltration':<25} {soil.total_infiltration:>12.2f}\n"
            text += f"{'Total Runoff':<25} {soil.total_runoff:>12.2f}\n"
            text += f"{'Quantum Measurements':<25} {len(soil.measurement_history):>12d}\n"
            text += "\nQuantum superposition determines infiltration\n"
            text += "vs runoff at each soil cell interaction.\n"
            text += "Saturated cells boost surface water velocity.\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=10, family='monospace', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title('Simulation Statistics', fontweight='bold', pad=20)

    def create_video(self, save_path: str, fps: int = 12):
        """Create comparison animation."""
        print(f"Creating video: {save_path}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        for ax, title, source, color in [
            (ax1, 'HILLTOP START', self.hilltop_source, 'red'),
            (ax2, 'FLAT GROUND START', self.flat_source, 'blue')
        ]:
            ax.imshow(self.terrain.elevation, origin='lower', cmap='terrain',
                     extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.6)
            ax.contour(self.terrain.elevation, levels=10, colors='brown', alpha=0.3,
                      extent=[0, self.terrain.width, 0, self.terrain.height], linewidths=0.5)
            ax.scatter(*source, color=color, s=200,
                      marker='^' if color == 'red' else 'o',
                      edgecolor='white', linewidth=2, zorder=20)
            ax.set_xlim(0, self.terrain.width)
            ax.set_ylim(0, self.terrain.height)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(title, fontweight='bold', fontsize=14)

        hill_scatter = ax1.scatter([], [], color='red', s=20, alpha=0.8)
        flat_scatter = ax2.scatter([], [], color='blue', s=20, alpha=0.8)

        hill_trails = [ax1.plot([], [], 'r-', linewidth=0.5, alpha=0.3)[0]
                       for _ in range(self.num_particles)]
        flat_trails = [ax2.plot([], [], 'b-', linewidth=0.5, alpha=0.3)[0]
                       for _ in range(self.num_particles)]

        mode = "Quantum" if self.use_quantum else "Classical"
        time_text = fig.suptitle('', fontsize=14, fontweight='bold')
        hill_vel_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                                 va='top', bbox=dict(facecolor='white', alpha=0.8))
        flat_vel_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=10,
                                 va='top', bbox=dict(facecolor='white', alpha=0.8))

        all_times = []
        for t in self.hilltop_times + self.flat_times:
            all_times.extend(t)
        t_min, t_max = min(all_times), max(all_times)

        num_frames = int((t_max - t_min) * fps)
        frame_times = np.linspace(t_min, t_max, num_frames)
        trail_length = 15

        def animate(frame):
            t_target = frame_times[frame]

            hill_positions = np.empty((0, 2))
            hill_vel = 0
            for wave_idx, (traj, times) in enumerate(zip(self.hilltop_trajectories, self.hilltop_times)):
                if times[0] <= t_target <= times[-1]:
                    t_idx = np.argmin(np.abs(times - t_target))
                    hill_positions = traj[t_idx, :, :2]

                    if t_idx > 0:
                        dt = times[t_idx] - times[t_idx - 1]
                        dx = traj[t_idx, :, 0] - traj[t_idx-1, :, 0]
                        dy = traj[t_idx, :, 1] - traj[t_idx-1, :, 1]
                        hill_vel = np.mean(np.sqrt(dx**2 + dy**2)) / dt

                    trail_start = max(0, t_idx - trail_length)
                    for p in range(min(self.num_particles, len(hill_trails))):
                        hill_trails[p].set_data(traj[trail_start:t_idx+1, p, 0],
                                               traj[trail_start:t_idx+1, p, 1])
                elif t_target > times[-1]:
                    hill_positions = traj[-1, :, :2]

            hill_scatter.set_offsets(hill_positions)
            hill_vel_text.set_text(f'Velocity: {hill_vel:.2f}')

            flat_positions = np.empty((0, 2))
            flat_vel = 0
            for wave_idx, (traj, times) in enumerate(zip(self.flat_trajectories, self.flat_times)):
                if times[0] <= t_target <= times[-1]:
                    t_idx = np.argmin(np.abs(times - t_target))
                    flat_positions = traj[t_idx, :, :2]

                    if t_idx > 0:
                        dt = times[t_idx] - times[t_idx - 1]
                        dx = traj[t_idx, :, 0] - traj[t_idx-1, :, 0]
                        dy = traj[t_idx, :, 1] - traj[t_idx-1, :, 1]
                        flat_vel = np.mean(np.sqrt(dx**2 + dy**2)) / dt

                    trail_start = max(0, t_idx - trail_length)
                    for p in range(min(self.num_particles, len(flat_trails))):
                        flat_trails[p].set_data(traj[trail_start:t_idx+1, p, 0],
                                               traj[trail_start:t_idx+1, p, 1])
                elif t_target > times[-1]:
                    flat_positions = traj[-1, :, :2]

            flat_scatter.set_offsets(flat_positions)
            flat_vel_text.set_text(f'Velocity: {flat_vel:.2f}')

            time_text.set_text(f'{mode} Watershed | Time: {t_target:.2f}')

            return [hill_scatter, flat_scatter, time_text, hill_vel_text, flat_vel_text] + hill_trails + flat_trails

        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, blit=False)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        plt.close(fig)

        print(f"Video saved: {save_path}")
        return save_path

    def visualize_quantum_grid(self, save_path: Optional[str] = None):
        """Visualize the quantum soil grid state."""
        if not self.use_quantum:
            print("No quantum data available (running in classical mode)")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Final moisture field
        ax1 = axes[0, 0]
        if len(self.moisture_snapshots) > 0:
            im1 = ax1.imshow(self.moisture_snapshots[-1], origin='lower', cmap='Blues',
                            extent=[0, self.terrain.width, 0, self.terrain.height])
            plt.colorbar(im1, ax=ax1, label='Moisture Level')
        ax1.set_title('Final Soil Moisture', fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # 2. Final saturation field
        ax2 = axes[0, 1]
        if len(self.saturation_snapshots) > 0:
            im2 = ax2.imshow(self.saturation_snapshots[-1], origin='lower', cmap='Oranges',
                            extent=[0, self.terrain.width, 0, self.terrain.height])
            plt.colorbar(im2, ax=ax2, label='Saturation')
        ax2.set_title('Final Soil Saturation', fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # 3. Coherence field
        ax3 = axes[1, 0]
        if len(self.coherence_snapshots) > 0:
            im3 = ax3.imshow(self.coherence_snapshots[-1], origin='lower', cmap='Greens',
                            extent=[0, self.terrain.width, 0, self.terrain.height],
                            vmin=0, vmax=1)
            plt.colorbar(im3, ax=ax3, label='Quantum Coherence')
        ax3.set_title('Quantum Coherence (Decoherence Pattern)', fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        # 4. Evolution over waves
        ax4 = axes[1, 1]
        if len(self.saturation_snapshots) > 0:
            waves = list(range(1, len(self.saturation_snapshots) + 1))
            mean_sat = [s.mean() for s in self.saturation_snapshots]
            max_sat = [s.max() for s in self.saturation_snapshots]
            mean_coh = [c.mean() for c in self.coherence_snapshots]

            ax4.bar([w - 0.2 for w in waves], mean_sat, 0.2, label='Mean Saturation', color='orange', alpha=0.7)
            ax4.bar([w for w in waves], max_sat, 0.2, label='Max Saturation', color='red', alpha=0.7)
            ax4.bar([w + 0.2 for w in waves], mean_coh, 0.2, label='Mean Coherence', color='green', alpha=0.7)

            ax4.set_xlabel('Wave Number')
            ax4.set_ylabel('Value')
            ax4.set_title('Quantum State Evolution', fontweight='bold')
            ax4.legend()
            ax4.set_ylim(0, 1.1)

        plt.suptitle('Quantum Soil Grid Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Quantum-Enhanced Watershed Simulation using Qiskit'
    )
    parser.add_argument('--classical', action='store_true',
                       help='Run classical simulation (no quantum effects)')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip video generation')
    parser.add_argument('--particles', type=int, default=30,
                       help='Number of water particles (default: 30)')
    parser.add_argument('--waves', type=int, default=3,
                       help='Number of water waves (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for results')

    args = parser.parse_args()

    print("=" * 70)
    print("QUANTUM WATERSHED SIMULATION")
    print("Using Qiskit for soil water absorptivity superposition")
    print("=" * 70)
    print()

    use_quantum = not args.classical

    # Create simulation
    sim = QuantumWatershedSimulation(
        num_particles=args.particles,
        num_waves=args.waves,
        use_quantum=use_quantum,
        seed=args.seed
    )

    # Run simulation
    sim.run()

    # Calculate statistics
    stats = sim.calculate_statistics()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Hilltop':>12} {'Flat Ground':>12}")
    print("-" * 50)
    print(f"{'Start Elevation':<25} {stats['hilltop']['start_elevation']:>12.1f} {stats['flat']['start_elevation']:>12.1f}")
    print(f"{'Mean Velocity':<25} {stats['hilltop']['mean_velocity']:>12.2f} {stats['flat']['mean_velocity']:>12.2f}")
    print(f"{'Max Velocity':<25} {stats['hilltop']['max_velocity']:>12.2f} {stats['flat']['max_velocity']:>12.2f}")
    print(f"{'Mean Distance':<25} {stats['hilltop']['mean_distance']:>12.1f} {stats['flat']['mean_distance']:>12.1f}")

    v_ratio = stats['hilltop']['mean_velocity'] / max(stats['flat']['mean_velocity'], 0.01)
    print(f"\nHilltop flows {v_ratio:.1f}x faster than flat ground!")

    if use_quantum:
        soil = sim.terrain.quantum_soil
        print(f"\nQuantum Effects Summary:")
        print(f"  Total water infiltrated: {soil.total_infiltration:.2f}")
        print(f"  Total surface runoff: {soil.total_runoff:.2f}")
        print(f"  Quantum measurements: {len(soil.measurement_history)}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Always save images in a local './results' directory, next to this script
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    mode = "quantum" if use_quantum else "classical"

    sim.visualize(save_path=os.path.join(results_dir, f'{mode}_watershed_comparison.png'))

    if use_quantum:
        sim.visualize_quantum_grid(save_path=os.path.join(results_dir, f'{mode}_soil_grid.png'))

    if not args.no_video:
        sim.create_video(save_path=os.path.join(results_dir, f'{mode}_watershed_animation.gif'), fps=10)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return sim


if __name__ == "__main__":
    sim = main()
