#!/usr/bin/env python3
"""
Urban Water Flow Simulation: Soil vs Road Absorption
=====================================================

Demonstrates the difference in water mobility between:
- SOIL (green patches): Absorbs water via quantum superposition, slows flow
- ROADS (gray): Impervious surface, water flows fast with no absorption
- BUILDINGS: Obstacles that redirect water flow

The terrain features:
- Hilltop with buildings (urban development)
- Road network connecting buildings
- Green soil patches between roads
- Drainage channels at lower elevations

Key Physics:
- Quantum soil model determines infiltration vs runoff probabilistically
- Roads have runoff_factor ≈ 1.0 (all water stays on surface, moves fast)
- Soil has variable runoff based on saturation state
- Saturated soil behaves more like roads (faster flow)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')

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
# URBAN TERRAIN WITH BUILDINGS
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

    def __init__(self, width: int = 120, height: int = 120, seed: int = 42):
        np.random.seed(seed)
        self.width = width
        self.height = height

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

        # Initialize quantum soil grid
        self.quantum_soil = QuantumSoilGrid(
            width=width,
            height=height,
            entanglement_radius=3.0,
            seed=seed
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
        print(f"  Surface breakdown:")
        print(f"    Roads: {road_cells} ({100*road_cells/total_cells:.1f}%)")
        print(f"    Soil:  {soil_cells} ({100*soil_cells/total_cells:.1f}%)")
        print(f"    Buildings: {building_cells} ({100*building_cells/total_cells:.1f}%)")
        print(f"  Hilltop: {self.hilltop}")

    def _generate_terrain(self):
        """Generate terrain with prominent hill."""
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # Base slope toward drainage (bottom-right)
        base = 5 + 0.08 * (self.width - X + self.height - Y)
        self.elevation = base

        # Main hill in upper-left (where buildings will go)
        hill_cx, hill_cy = 35, 85
        hill = 40 * np.exp(-((X - hill_cx)**2 + (Y - hill_cy)**2) / (2 * 20**2))
        self.elevation += hill

        # Secondary smaller hills
        for cx, cy, amp, spread in [(75, 65, 15, 12), (90, 35, 10, 10)]:
            h = amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * spread**2))
            self.elevation += h

        # Smooth and clamp
        self.elevation = gaussian_filter(self.elevation, sigma=2)
        self.elevation = np.maximum(self.elevation, 0)

    def _generate_roads(self):
        """Generate road grid."""
        road_width = 4

        # Main grid roads
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

        # Ring road around hilltop
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

        # Central large building
        self._place_building(hill_cx - 6, hill_cy - 6, 12, 12)

        # Surrounding smaller buildings
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

        # Additional buildings along roads in lower areas
        for bx, by, bw, bh in [
            (55, 55, 8, 8),
            (80, 55, 10, 8),
            (55, 30, 8, 10),
            (80, 30, 8, 8),
            (100, 50, 6, 6),
        ]:
            self._place_building(bx, by, bw, bh)

    def _place_building(self, x: int, y: int, w: int, h: int):
        """Place a building at position."""
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + w)
        y2 = min(self.height, y + h)

        if x2 > x1 and y2 > y1:
            self.surface_type[y1:y2, x1:x2] = self.BUILDING
            # Buildings are slightly elevated
            self.elevation[y1:y2, x1:x2] += 2
            self.buildings.append((x1, y1, x2-x1, y2-y1))

    def _generate_channel(self):
        """Generate drainage channel."""
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
        """Compute terrain gradients."""
        self.grad_y, self.grad_x = np.gradient(self.elevation)
        self.slope = np.sqrt(self.grad_x**2 + self.grad_y**2)

    def _configure_soil_types(self):
        """Configure quantum soil types based on surface."""
        for y in range(self.height):
            for x in range(self.width):
                surface = self.surface_type[y, x]
                if surface == self.ROAD:
                    self.quantum_soil.set_soil_type(x, y, SoilType.IMPERVIOUS)
                elif surface == self.BUILDING:
                    self.quantum_soil.set_soil_type(x, y, SoilType.IMPERVIOUS)
                elif surface == self.CHANNEL:
                    self.quantum_soil.set_soil_type(x, y, SoilType.SAND)
                else:  # SOIL
                    # Mix of soil types for realism
                    if np.random.random() < 0.6:
                        self.quantum_soil.set_soil_type(x, y, SoilType.LOAM)
                    else:
                        self.quantum_soil.set_soil_type(x, y, SoilType.CLAY)

    def _find_hilltop(self) -> Tuple[int, int]:
        """Find highest point that isn't a building."""
        # Mask out buildings
        masked_elev = self.elevation.copy()
        masked_elev[self.surface_type == self.BUILDING] = -999
        idx = np.unravel_index(np.argmax(masked_elev), masked_elev.shape)
        return (idx[1], idx[0])

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
        # Road: smooth, fast flow
        # Soil: rough, slow flow
        # Channel: moderate
        # Building: infinite (no flow)
        return {
            self.ROAD: 0.012,      # Smooth asphalt
            self.SOIL: 0.035,      # Grass/soil
            self.CHANNEL: 0.025,   # Concrete channel
            self.BUILDING: 999.0,  # No flow
        }.get(surface, 0.035)

    def is_building(self, x: float, y: float) -> bool:
        """Check if position is inside a building."""
        return self.get_surface_type(x, y) == self.BUILDING


# =============================================================================
# URBAN WATER DYNAMICS WITH SOIL VS ROAD DIFFERENCE
# =============================================================================

class UrbanWaterDynamics(torch.nn.Module):
    """
    Water dynamics showing clear difference between soil and road.

    Key behaviors:
    - Roads: Fast flow, no infiltration, water accumulates
    - Soil: Slower flow, quantum infiltration reduces water volume
    - Buildings: Water redirects around obstacles
    """

    def __init__(self, terrain: UrbanHilltopTerrain, num_particles: int):
        super().__init__()
        self.terrain = terrain
        self.num_particles = num_particles
        self.eps = 1e-6

        # Track particle properties
        self.particle_mass = torch.ones(num_particles)
        self.particle_on_road = torch.zeros(num_particles, dtype=torch.bool)

        # Statistics
        self.road_distance = torch.zeros(num_particles)
        self.soil_distance = torch.zeros(num_particles)

        self.interaction_dt = 0.1
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

            # Boundary check
            if xi < 2 or xi > self.terrain.width - 3 or yi < 2 or yi > self.terrain.height - 3:
                continue

            # Skip dead particles
            if self.particle_mass[i] < 0.01:
                continue

            # Skip if in building
            if self.terrain.is_building(xi, yi):
                # Redirect away from building
                flow_x, flow_y = self.terrain.get_flow_direction(xi, yi)
                dx[i] = -flow_x * 0.5
                dy[i] = -flow_y * 0.5
                continue

            # Get terrain properties
            surface = self.terrain.get_surface_type(xi, yi)
            slope = self.terrain.get_slope(xi, yi)
            n = self.terrain.get_manning_n(xi, yi)
            flow_x, flow_y = self.terrain.get_flow_direction(xi, yi)

            # Track surface type
            is_road = (surface == self.terrain.ROAD)
            self.particle_on_road[i] = is_road

            # === QUANTUM SOIL INTERACTION ===
            velocity_modifier = 1.0

            if do_quantum:
                result = self.terrain.quantum_soil.interact_water(
                    xi, yi, 0.1 * self.particle_mass[i].item()
                )

                if surface == self.terrain.SOIL:
                    # Soil absorbs water - reduce mass
                    infiltration = result['infiltration_rate']
                    self.particle_mass[i] *= (1 - infiltration * 0.08)

                    # Soil slows water (lower runoff = slower)
                    velocity_modifier = 0.5 + 0.3 * result['runoff_factor']
                else:
                    # Road/channel - no absorption, fast flow
                    velocity_modifier = 1.2 + 0.3 * result['runoff_factor']

            # === MANNING'S EQUATION ===
            h = 0.1 * self.particle_mass[i].item()  # Depth proportional to mass
            if slope > 0.001 and n < 100:
                V = (1.0 / n) * (h ** (2/3)) * np.sqrt(slope)
            else:
                V = 0.05

            # Apply velocity modifier based on surface
            V *= velocity_modifier

            # Surface-specific boosts
            if surface == self.terrain.ROAD:
                V *= 1.8  # Roads are fast
            elif surface == self.terrain.CHANNEL:
                V *= 2.5  # Channels concentrate and speed up flow
            elif surface == self.terrain.SOIL:
                V *= 0.7  # Soil friction slows flow

            # Flow direction
            flow_mag = np.sqrt(flow_x**2 + flow_y**2) + self.eps
            vx = V * flow_x / flow_mag
            vy = V * flow_y / flow_mag

            # Particle dispersion (weaker on roads for channelized flow)
            dispersion_factor = 0.1 if is_road else 0.25
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

            # Track distance by surface type
            step_dist = np.sqrt(vx**2 + vy**2) * 0.05
            if is_road:
                self.road_distance[i] += step_dist
            else:
                self.soil_distance[i] += step_dist

            # Z follows terrain
            target_z = self.terrain.get_elevation(xi + vx * 0.1, yi + vy * 0.1)
            dz[i] = (target_z - z[i].item()) * 3.0

        return torch.stack([dx, dy, dz], dim=1).view(-1)

    def reset(self):
        """Reset for new simulation."""
        self.particle_mass = torch.ones(self.num_particles)
        self.particle_on_road = torch.zeros(self.num_particles, dtype=torch.bool)
        self.road_distance = torch.zeros(self.num_particles)
        self.soil_distance = torch.zeros(self.num_particles)
        self.last_interaction_time = 0.0


# =============================================================================
# SIMULATION AND VISUALIZATION
# =============================================================================

class UrbanWaterSimulation:
    """
    Simulation comparing water flow on soil vs roads.
    """

    def __init__(self, num_particles: int = 40, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.terrain = UrbanHilltopTerrain(width=120, height=120, seed=seed)
        self.num_particles = num_particles

        # Release point (on hilltop, near buildings)
        self.source = (35, 75)  # Just below the hilltop buildings

        # Results
        self.trajectories = None
        self.times = None
        self.dynamics = None

    def _create_particles(self) -> torch.Tensor:
        """Create particles at source with some spread."""
        states = []
        for i in range(self.num_particles):
            angle = 2 * np.pi * i / self.num_particles
            radius = np.random.uniform(1, 4)

            x = self.source[0] + radius * np.cos(angle)
            y = self.source[1] + radius * np.sin(angle)

            # Make sure not in a building
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
        """Run the simulation."""
        print(f"\nRunning Urban Water Flow Simulation...")
        print(f"  Source: {self.source}")
        print(f"  Particles: {self.num_particles}")
        print(f"  Duration: {t_duration}s")

        self.dynamics = UrbanWaterDynamics(self.terrain, self.num_particles)
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

        # Print statistics
        road_dist = self.dynamics.road_distance.numpy()
        soil_dist = self.dynamics.soil_distance.numpy()
        final_mass = self.dynamics.particle_mass.numpy()

        print(f"\nStatistics:")
        print(f"  Avg distance on roads: {road_dist.mean():.1f}")
        print(f"  Avg distance on soil: {soil_dist.mean():.1f}")
        print(f"  Avg final mass: {final_mass.mean():.2f} (1.0 = no infiltration)")
        print(f"  Total water lost to infiltration: {(1 - final_mass.mean()) * 100:.1f}%")

    def visualize(self, save_path: Optional[str] = None):
        """Create comprehensive visualization."""
        fig = plt.figure(figsize=(20, 16))

        # 1. Surface type map with trajectories
        ax1 = fig.add_subplot(2, 2, 1)
        self._plot_surface_map(ax1)

        # 2. 3D view
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        self._plot_3d_view(ax2)

        # 3. Velocity by surface type
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_velocity_by_surface(ax3)

        # 4. Statistics
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_statistics(ax4)

        plt.suptitle('Urban Water Flow: Soil Absorption vs Road Runoff',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def _plot_surface_map(self, ax):
        """Plot surface type map with trajectories."""
        # Create colored surface map
        surface_colors = np.zeros((*self.terrain.surface_type.shape, 3))

        # Color by surface type
        # Road = gray, Soil = green, Channel = blue, Building = dark gray
        for y in range(self.terrain.height):
            for x in range(self.terrain.width):
                s = self.terrain.surface_type[y, x]
                if s == self.terrain.ROAD:
                    surface_colors[y, x] = [0.5, 0.5, 0.5]  # Gray
                elif s == self.terrain.SOIL:
                    surface_colors[y, x] = [0.3, 0.7, 0.3]  # Green
                elif s == self.terrain.CHANNEL:
                    surface_colors[y, x] = [0.3, 0.5, 0.8]  # Blue
                elif s == self.terrain.BUILDING:
                    surface_colors[y, x] = [0.3, 0.3, 0.3]  # Dark gray

        ax.imshow(surface_colors, origin='lower',
                 extent=[0, self.terrain.width, 0, self.terrain.height])

        # Elevation contours
        ax.contour(self.terrain.elevation, levels=15, colors='white', alpha=0.3,
                  extent=[0, self.terrain.width, 0, self.terrain.height], linewidths=0.5)

        # Plot trajectories colored by mass (infiltration)
        final_mass = self.dynamics.particle_mass.numpy()
        colors = plt.cm.RdYlBu(final_mass)  # Blue = high mass, Red = low mass (infiltrated)

        for p in range(self.num_particles):
            ax.plot(self.trajectories[:, p, 0], self.trajectories[:, p, 1],
                   color=colors[p], linewidth=1.5, alpha=0.7)

        # End points
        ax.scatter(self.trajectories[-1, :, 0], self.trajectories[-1, :, 1],
                  c=final_mass, cmap='RdYlBu', s=30, edgecolor='white', linewidth=0.5)

        # Source
        ax.scatter(*self.source, color='yellow', s=200, marker='*',
                  edgecolor='black', linewidth=2, zorder=20, label='Source')

        # Legend
        legend_patches = [
            mpatches.Patch(color=[0.5, 0.5, 0.5], label='Road (fast, no absorption)'),
            mpatches.Patch(color=[0.3, 0.7, 0.3], label='Soil (slow, absorbs water)'),
            mpatches.Patch(color=[0.3, 0.5, 0.8], label='Drainage channel'),
            mpatches.Patch(color=[0.3, 0.3, 0.3], label='Building'),
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Surface Types & Water Trajectories\n(Color = remaining mass)', fontweight='bold')

    def _plot_3d_view(self, ax):
        """Plot 3D terrain with trajectories."""
        # Terrain mesh (subsampled)
        step = 3
        x = np.arange(0, self.terrain.width, step)
        y = np.arange(0, self.terrain.height, step)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=float)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.terrain.elevation[
                    min(int(Y[i, j]), self.terrain.height - 1),
                    min(int(X[i, j]), self.terrain.width - 1)
                ]

        ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, linewidth=0)

        # Plot trajectories
        final_mass = self.dynamics.particle_mass.numpy()
        colors = plt.cm.RdYlBu(final_mass)

        for p in range(0, self.num_particles, 2):  # Every other particle
            ax.plot(self.trajectories[:, p, 0], self.trajectories[:, p, 1],
                   self.trajectories[:, p, 2] + 0.5,
                   color=colors[p], linewidth=1, alpha=0.8)

        # Source marker
        sz = self.terrain.get_elevation(*self.source)
        ax.scatter(*self.source, sz + 3, color='yellow', s=100, marker='*',
                  edgecolor='black', zorder=20)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain View', fontweight='bold')
        ax.view_init(elev=30, azim=45)

    def _plot_velocity_by_surface(self, ax):
        """Plot velocity comparison between surfaces."""
        # Calculate instantaneous velocities and track surface type
        road_velocities = []
        soil_velocities = []

        for t in range(1, len(self.trajectories)):
            dt = self.times[t] - self.times[t-1]
            for p in range(self.num_particles):
                dx = self.trajectories[t, p, 0] - self.trajectories[t-1, p, 0]
                dy = self.trajectories[t, p, 1] - self.trajectories[t-1, p, 1]
                v = np.sqrt(dx**2 + dy**2) / dt

                # Check surface at this position
                x, y = self.trajectories[t, p, 0], self.trajectories[t, p, 1]
                surface = self.terrain.get_surface_type(x, y)

                if surface == self.terrain.ROAD:
                    road_velocities.append(v)
                elif surface == self.terrain.SOIL:
                    soil_velocities.append(v)

        # Box plot comparison
        data = [road_velocities, soil_velocities]
        bp = ax.boxplot(data, labels=['Road', 'Soil'], patch_artist=True)

        bp['boxes'][0].set_facecolor('gray')
        bp['boxes'][1].set_facecolor('green')

        ax.set_ylabel('Velocity')
        ax.set_title('Velocity Distribution by Surface Type', fontweight='bold')

        # Add mean annotations
        ax.annotate(f'Mean: {np.mean(road_velocities):.2f}',
                   xy=(1, np.mean(road_velocities)), fontsize=10)
        ax.annotate(f'Mean: {np.mean(soil_velocities):.2f}',
                   xy=(2, np.mean(soil_velocities)), fontsize=10)

        # Key insight
        ratio = np.mean(road_velocities) / max(np.mean(soil_velocities), 0.01)
        ax.text(0.5, 0.95, f'Road flow is {ratio:.1f}x faster than soil',
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _plot_statistics(self, ax):
        """Plot statistics summary."""
        ax.axis('off')

        road_dist = self.dynamics.road_distance.numpy()
        soil_dist = self.dynamics.soil_distance.numpy()
        final_mass = self.dynamics.particle_mass.numpy()

        soil = self.terrain.quantum_soil

        text = "SOIL VS ROAD: KEY DIFFERENCES\n"
        text += "=" * 50 + "\n\n"

        text += "WATER BEHAVIOR:\n"
        text += "-" * 50 + "\n"
        text += f"{'Metric':<30} {'Road':>10} {'Soil':>10}\n"
        text += "-" * 50 + "\n"
        text += f"{'Avg distance traveled':<30} {road_dist.mean():>10.1f} {soil_dist.mean():>10.1f}\n"
        text += f"{'Manning roughness (n)':<30} {'0.012':>10} {'0.035':>10}\n"
        text += f"{'Infiltration':<30} {'None':>10} {'Quantum':>10}\n"

        text += "\n" + "=" * 50 + "\n\n"

        text += "QUANTUM SOIL EFFECTS:\n"
        text += "-" * 50 + "\n"
        text += f"{'Water lost to infiltration':<30} {(1 - final_mass.mean()) * 100:>10.1f}%\n"
        text += f"{'Total quantum measurements':<30} {len(soil.measurement_history):>10d}\n"
        text += f"{'Total infiltrated volume':<30} {soil.total_infiltration:>10.2f}\n"
        text += f"{'Total surface runoff':<30} {soil.total_runoff:>10.2f}\n"

        text += "\n" + "=" * 50 + "\n\n"

        text += "PHYSICAL INTERPRETATION:\n"
        text += "-" * 50 + "\n"
        text += "* ROADS: Impervious surfaces cause fast runoff\n"
        text += "  - No water absorption\n"
        text += "  - Low friction (smooth surface)\n"
        text += "  - Water accumulates and accelerates\n\n"
        text += "* SOIL: Permeable surfaces absorb water\n"
        text += "  - Quantum superposition determines infiltration\n"
        text += "  - Higher friction (vegetation, roughness)\n"
        text += "  - Saturated soil -> higher runoff (more road-like)\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, family='monospace', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title('Simulation Statistics', fontweight='bold', pad=20)

    def create_video(self, save_path: str, fps: int = 15):
        """Create animation showing water flow."""
        print(f"Creating video: {save_path}")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create surface color map
        surface_colors = np.zeros((*self.terrain.surface_type.shape, 3))
        for y in range(self.terrain.height):
            for x in range(self.terrain.width):
                s = self.terrain.surface_type[y, x]
                if s == self.terrain.ROAD:
                    surface_colors[y, x] = [0.5, 0.5, 0.5]
                elif s == self.terrain.SOIL:
                    surface_colors[y, x] = [0.3, 0.7, 0.3]
                elif s == self.terrain.CHANNEL:
                    surface_colors[y, x] = [0.3, 0.5, 0.8]
                elif s == self.terrain.BUILDING:
                    surface_colors[y, x] = [0.3, 0.3, 0.3]

        ax.imshow(surface_colors, origin='lower',
                 extent=[0, self.terrain.width, 0, self.terrain.height])
        ax.contour(self.terrain.elevation, levels=12, colors='white', alpha=0.3,
                  extent=[0, self.terrain.width, 0, self.terrain.height], linewidths=0.5)

        # Source marker
        ax.scatter(*self.source, color='yellow', s=150, marker='*',
                  edgecolor='black', linewidth=2, zorder=20)

        # Particle scatter and trails
        scatter = ax.scatter([], [], c=[], cmap='RdYlBu', s=40,
                            edgecolor='white', linewidth=0.5, vmin=0, vmax=1)
        trails = [ax.plot([], [], 'b-', linewidth=0.8, alpha=0.4)[0]
                 for _ in range(self.num_particles)]

        # Info text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                           va='top', bbox=dict(facecolor='white', alpha=0.8))

        # Legend
        legend_patches = [
            mpatches.Patch(color=[0.5, 0.5, 0.5], label='Road'),
            mpatches.Patch(color=[0.3, 0.7, 0.3], label='Soil'),
            mpatches.Patch(color=[0.3, 0.5, 0.8], label='Channel'),
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

        ax.set_xlim(0, self.terrain.width)
        ax.set_ylim(0, self.terrain.height)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Urban Water Flow: Soil vs Road', fontweight='bold', fontsize=14)

        trail_length = 20
        final_mass = self.dynamics.particle_mass.numpy()

        def animate(frame):
            positions = self.trajectories[frame, :, :2]
            scatter.set_offsets(positions)
            scatter.set_array(final_mass)

            # Update trails
            trail_start = max(0, frame - trail_length)
            for p in range(self.num_particles):
                trails[p].set_data(
                    self.trajectories[trail_start:frame+1, p, 0],
                    self.trajectories[trail_start:frame+1, p, 1]
                )
                # Color trail by mass
                trails[p].set_color(plt.cm.RdYlBu(final_mass[p]))

            time_text.set_text(f'Time: {self.times[frame]:.2f}s\n'
                              f'Frame: {frame}/{len(self.times)-1}')

            return [scatter, time_text] + trails

        anim = FuncAnimation(fig, animate, frames=len(self.times),
                           interval=1000/fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        plt.close(fig)

        print(f"Video saved: {save_path}")


def main():
    """Run the urban water simulation."""
    print("=" * 70)
    print("URBAN WATER FLOW SIMULATION")
    print("Comparing Soil Absorption vs Road Runoff")
    print("=" * 70)
    print()

    # Create and run simulation
    sim = UrbanWaterSimulation(num_particles=50, seed=42)
    sim.run(t_duration=10.0, num_steps=150)

    # Output directory

    # Create visualizations
    print("\nGenerating visualizations...")
    # Ensure the 'results' directory exists before saving outputs
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    sim.visualize(save_path=os.path.join(results_dir, 'urban_soil_vs_road.png'))
    sim.create_video(save_path=os.path.join(results_dir, 'urban_water_flow.gif'), fps=12)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return sim


if __name__ == "__main__":
    sim = main()
