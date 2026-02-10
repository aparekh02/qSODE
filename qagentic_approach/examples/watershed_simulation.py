#!/usr/bin/env python3
"""
Watershed qODE Simulation with Multi-Wave Water Dispersion
==========================================================

Implements hydrological equations for realistic water flow modeling:

1. Manning's Equation for velocity: V = (1/n) * R^(2/3) * S^(1/2)
2. Green-Ampt Infiltration: f = K * (1 + (ψ * Δθ) / F)
3. Kinematic Wave Approximation for surface runoff
4. Saturation-dependent flow (easier movement as water accumulates)

Features:
- Multiple water release waves from the same source point
- Surface types: Road (impervious), Soil (pervious), Channel
- Saturation effects: infiltration decreases, runoff increases with water
- 3D terrain with elevation-driven flow
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# HYDROLOGICAL CONSTANTS AND EQUATIONS
# =============================================================================

class SurfaceType(Enum):
    """Surface types with hydrological properties."""
    ROAD = 0        # Impervious - asphalt/concrete
    SOIL = 1        # Pervious - natural soil
    CHANNEL = 2     # Water channel/stream
    VEGETATION = 3  # Vegetated soil


@dataclass
class HydrologicalProperties:
    """
    Hydrological properties for different surface types.

    Based on standard values from:
    - USDA Natural Resources Conservation Service
    - Urban Hydrology for Small Watersheds (TR-55)
    - Manning's n values from Chow (1959)
    """

    # Manning's roughness coefficient n
    # Lower n = faster flow
    manning_n: Dict[SurfaceType, float] = field(default_factory=lambda: {
        SurfaceType.ROAD: 0.013,       # Smooth asphalt
        SurfaceType.SOIL: 0.035,       # Bare soil
        SurfaceType.CHANNEL: 0.025,    # Natural channel
        SurfaceType.VEGETATION: 0.050  # Grass/vegetation
    })

    # Hydraulic conductivity K (m/s) - infiltration rate
    # Higher K = more infiltration into soil
    hydraulic_conductivity: Dict[SurfaceType, float] = field(default_factory=lambda: {
        SurfaceType.ROAD: 0.0,         # Impervious
        SurfaceType.SOIL: 1.0e-5,      # Sandy loam
        SurfaceType.CHANNEL: 0.0,      # Water stays in channel
        SurfaceType.VEGETATION: 5.0e-6 # Clay loam with roots
    })

    # Wetting front suction head ψ (m) for Green-Ampt
    suction_head: Dict[SurfaceType, float] = field(default_factory=lambda: {
        SurfaceType.ROAD: 0.0,
        SurfaceType.SOIL: 0.11,        # Sandy loam
        SurfaceType.CHANNEL: 0.0,
        SurfaceType.VEGETATION: 0.21   # Clay loam
    })

    # Porosity (void fraction)
    porosity: Dict[SurfaceType, float] = field(default_factory=lambda: {
        SurfaceType.ROAD: 0.0,
        SurfaceType.SOIL: 0.45,
        SurfaceType.CHANNEL: 1.0,
        SurfaceType.VEGETATION: 0.40
    })

    # Initial moisture content θ_i
    initial_moisture: Dict[SurfaceType, float] = field(default_factory=lambda: {
        SurfaceType.ROAD: 0.0,
        SurfaceType.SOIL: 0.15,
        SurfaceType.CHANNEL: 1.0,
        SurfaceType.VEGETATION: 0.20
    })


def manning_velocity(slope: float, hydraulic_radius: float, n: float) -> float:
    """
    Manning's equation for open channel/overland flow velocity.

    V = (1/n) * R^(2/3) * S^(1/2)

    Args:
        slope: Energy slope (dimensionless, typically terrain slope)
        hydraulic_radius: R = A/P (cross-sectional area / wetted perimeter)
                         For sheet flow, R ≈ water depth
        n: Manning's roughness coefficient

    Returns:
        Flow velocity (m/s)
    """
    if slope <= 0 or hydraulic_radius <= 0 or n <= 0:
        return 0.0

    return (1.0 / n) * (hydraulic_radius ** (2/3)) * (np.sqrt(abs(slope)))


def green_ampt_infiltration(
    K: float,           # Hydraulic conductivity
    psi: float,         # Suction head
    theta_s: float,     # Saturated moisture content (porosity)
    theta_i: float,     # Initial moisture content
    F: float,           # Cumulative infiltration
    dt: float = 0.1     # Time step
) -> Tuple[float, float]:
    """
    Green-Ampt infiltration model.

    f = K * (1 + (ψ * Δθ) / F)

    As cumulative infiltration F increases, infiltration rate f decreases
    (soil becomes saturated).

    Args:
        K: Saturated hydraulic conductivity
        psi: Wetting front suction head
        theta_s: Saturated moisture content
        theta_i: Initial moisture content
        F: Cumulative infiltration depth
        dt: Time step

    Returns:
        (infiltration_rate, new_cumulative_infiltration)
    """
    if K <= 0:
        return 0.0, F

    delta_theta = theta_s - theta_i

    # Avoid division by zero
    if F < 0.001:
        F = 0.001

    # Green-Ampt equation
    f = K * (1 + (psi * delta_theta) / F)

    # Update cumulative infiltration
    F_new = F + f * dt

    return f, F_new


def saturation_factor(
    cumulative_water: float,
    max_saturation: float = 100.0
) -> float:
    """
    Calculate saturation factor that increases runoff as water accumulates.

    As more water is present:
    - Soil infiltration capacity decreases
    - Surface runoff increases (water moves easier)

    Uses exponential saturation curve:
    S = 1 - exp(-water / max_sat)

    Returns value in [0, 1] where 1 = fully saturated
    """
    return 1.0 - np.exp(-cumulative_water / max_saturation)


# =============================================================================
# TERRAIN WITH SURFACE TYPES
# =============================================================================

@dataclass
class WatershedConfig:
    """Configuration for watershed terrain."""
    width: int = 120
    height: int = 120
    max_elevation: float = 30.0
    num_hills: int = 4
    road_width: int = 4
    road_spacing: int = 25
    channel_width: int = 3
    seed: int = 42


class WatershedTerrain:
    """
    Terrain with elevation and surface type classification.

    Tracks:
    - Elevation map (topography)
    - Surface types (road, soil, vegetation, channel)
    - Cumulative water depth at each cell
    - Soil moisture / saturation state
    """

    def __init__(self, config: WatershedConfig = None):
        self.config = config or WatershedConfig()
        np.random.seed(self.config.seed)

        self.width = self.config.width
        self.height = self.config.height

        # Initialize grids
        self.elevation = np.zeros((self.height, self.width))
        self.surface_type = np.full((self.height, self.width), SurfaceType.SOIL.value)
        self.water_depth = np.zeros((self.height, self.width))
        self.cumulative_infiltration = np.zeros((self.height, self.width))
        self.soil_moisture = np.zeros((self.height, self.width))

        # Hydrological properties
        self.hydro = HydrologicalProperties()

        # Generate terrain
        self._generate_terrain()
        self._generate_roads()
        self._generate_channel()
        self._compute_gradients()

        # Initialize soil moisture
        self._initialize_soil_moisture()

    def _generate_terrain(self):
        """Generate terrain with hills, valleys, and general slope toward channel."""
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # General slope toward bottom-right (where channel will be)
        base_slope = 0.15 * (self.width - X + self.height - Y) / (self.width + self.height) * self.config.max_elevation
        self.elevation = base_slope

        # Add hills
        for _ in range(self.config.num_hills):
            cx = np.random.uniform(15, self.width - 15)
            cy = np.random.uniform(15, self.height - 15)
            amplitude = np.random.uniform(0.4, 0.8) * self.config.max_elevation
            spread = np.random.uniform(10, 20)

            hill = amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * spread**2))
            self.elevation += hill

        # Add some valleys
        for _ in range(2):
            cx = np.random.uniform(20, self.width - 20)
            cy = np.random.uniform(20, self.height - 20)
            amplitude = np.random.uniform(0.2, 0.4) * self.config.max_elevation
            spread = np.random.uniform(8, 15)

            valley = -amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * spread**2))
            self.elevation += valley

        # Smooth
        self.elevation = gaussian_filter(self.elevation, sigma=2)
        self.elevation = np.maximum(self.elevation, 0)

    def _generate_roads(self):
        """Generate road network (impervious surfaces)."""
        spacing = self.config.road_spacing
        width = self.config.road_width

        # Horizontal roads
        for y in range(spacing, self.height - spacing, spacing):
            y_start = max(0, y - width // 2)
            y_end = min(self.height, y + width // 2 + 1)
            self.surface_type[y_start:y_end, :] = SurfaceType.ROAD.value
            # Roads are slightly lower
            self.elevation[y_start:y_end, :] *= 0.85

        # Vertical roads
        for x in range(spacing, self.width - spacing, spacing):
            x_start = max(0, x - width // 2)
            x_end = min(self.width, x + width // 2 + 1)
            self.surface_type[:, x_start:x_end] = SurfaceType.ROAD.value
            self.elevation[:, x_start:x_end] *= 0.85

        # Add vegetation patches between roads
        for bx in range(spacing // 2, self.width - spacing // 2, spacing):
            for by in range(spacing // 2, self.height - spacing // 2, spacing):
                # Random vegetation patches
                if np.random.random() < 0.4:
                    veg_size = np.random.randint(4, 10)
                    x_start = max(0, bx - veg_size // 2)
                    x_end = min(self.width, bx + veg_size // 2)
                    y_start = max(0, by - veg_size // 2)
                    y_end = min(self.height, by + veg_size // 2)

                    # Only if it's soil (not road)
                    mask = self.surface_type[y_start:y_end, x_start:x_end] == SurfaceType.SOIL.value
                    self.surface_type[y_start:y_end, x_start:x_end][mask] = SurfaceType.VEGETATION.value

    def _generate_channel(self):
        """Generate drainage channel (low elevation path)."""
        # Main channel from upper-left to lower-right
        channel_y = np.linspace(self.height - 10, 10, self.width).astype(int)

        for x in range(self.width):
            y_center = channel_y[x] + int(3 * np.sin(x * 0.1))  # Meandering
            y_center = np.clip(y_center, 5, self.height - 5)

            for dy in range(-self.config.channel_width, self.config.channel_width + 1):
                y = y_center + dy
                if 0 <= y < self.height:
                    self.surface_type[y, x] = SurfaceType.CHANNEL.value
                    # Channel is lower than surroundings
                    self.elevation[y, x] = max(0, self.elevation[y, x] - 5)

    def _compute_gradients(self):
        """Compute terrain gradients for flow direction."""
        self.grad_y, self.grad_x = np.gradient(self.elevation)
        self.slope = np.sqrt(self.grad_x**2 + self.grad_y**2)

    def _initialize_soil_moisture(self):
        """Initialize soil moisture based on surface type."""
        for surf_type in SurfaceType:
            mask = self.surface_type == surf_type.value
            self.soil_moisture[mask] = self.hydro.initial_moisture[surf_type]

    def get_surface_type(self, x: float, y: float) -> SurfaceType:
        """Get surface type at location."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return SurfaceType(self.surface_type[iy, ix])

    def get_manning_n(self, x: float, y: float) -> float:
        """Get Manning's n at location."""
        surface = self.get_surface_type(x, y)
        return self.hydro.manning_n[surface]

    def get_slope(self, x: float, y: float) -> float:
        """Get terrain slope at location."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.slope[iy, ix]

    def get_flow_direction(self, x: float, y: float) -> Tuple[float, float]:
        """Get flow direction (negative gradient = downhill)."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return -self.grad_x[iy, ix], -self.grad_y[iy, ix]

    def get_elevation(self, x: float, y: float) -> float:
        """Get elevation at location."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.elevation[iy, ix]

    def add_water(self, x: float, y: float, amount: float):
        """Add water at location."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        self.water_depth[iy, ix] += amount

    def get_local_saturation(self, x: float, y: float) -> float:
        """Get saturation factor based on accumulated water."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))

        # Check water in neighborhood
        radius = 3
        x_min = max(0, ix - radius)
        x_max = min(self.width, ix + radius + 1)
        y_min = max(0, iy - radius)
        y_max = min(self.height, iy + radius + 1)

        local_water = np.sum(self.water_depth[y_min:y_max, x_min:x_max])

        return saturation_factor(local_water, max_saturation=50.0)

    def update_infiltration(self, x: float, y: float, dt: float) -> float:
        """
        Update infiltration at location and return infiltration rate.

        Returns amount of water that infiltrates (removed from surface flow).
        """
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))

        surface = self.get_surface_type(x, y)

        K = self.hydro.hydraulic_conductivity[surface]
        psi = self.hydro.suction_head[surface]
        theta_s = self.hydro.porosity[surface]
        theta_i = self.soil_moisture[iy, ix]
        F = self.cumulative_infiltration[iy, ix]

        if K <= 0:
            return 0.0

        f, F_new = green_ampt_infiltration(K, psi, theta_s, theta_i, F, dt)

        self.cumulative_infiltration[iy, ix] = F_new
        self.soil_moisture[iy, ix] = min(theta_s, theta_i + f * dt / 0.1)

        return f * dt


# =============================================================================
# WATER PARTICLE DYNAMICS
# =============================================================================

class WaterParticleDynamics(torch.nn.Module):
    """
    Water particle dynamics using hydrological equations.

    Velocity determined by:
    1. Manning's equation (surface roughness, slope)
    2. Saturation effects (easier flow with more water)
    3. Gradient-driven flow (downhill)
    4. Inter-particle dispersion

    Key equation:
    V = (1/n_eff) * h^(2/3) * S^(1/2)

    where n_eff decreases with saturation (water flows easier on wet surfaces)
    """

    def __init__(
        self,
        terrain: WatershedTerrain,
        num_particles: int,
        base_water_depth: float = 0.1,
        dispersion_strength: float = 0.3
    ):
        super().__init__()

        self.terrain = terrain
        self.num_particles = num_particles
        self.base_water_depth = base_water_depth
        self.dispersion_strength = dispersion_strength
        self.eps = 1e-6

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute water particle velocities.

        State: [num_particles, 4] -> [x, y, z, water_mass]
        """
        state = state.view(self.num_particles, 4)

        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        water_mass = state[:, 3]

        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dz = torch.zeros_like(z)
        d_mass = torch.zeros_like(water_mass)

        for i in range(self.num_particles):
            xi, yi = x[i].item(), y[i].item()
            mass_i = water_mass[i].item()

            # Skip if out of bounds
            if xi < 1 or xi > self.terrain.width - 2 or yi < 1 or yi > self.terrain.height - 2:
                continue

            # 1. Get terrain properties
            slope = self.terrain.get_slope(xi, yi)
            n = self.terrain.get_manning_n(xi, yi)
            flow_dir_x, flow_dir_y = self.terrain.get_flow_direction(xi, yi)
            saturation = self.terrain.get_local_saturation(xi, yi)
            surface = self.terrain.get_surface_type(xi, yi)

            # 2. Calculate effective Manning's n (decreases with saturation)
            # Wet surfaces have less friction - water flows easier
            # n_eff = n * (1 - 0.5 * saturation)
            n_eff = n * (1.0 - 0.6 * saturation)
            n_eff = max(n_eff, 0.005)  # Minimum roughness

            # 3. Calculate hydraulic radius (approximate as water depth)
            h = self.base_water_depth + 0.05 * mass_i

            # 4. Manning's velocity
            V = manning_velocity(slope + 0.001, h, n_eff)

            # Boost velocity on roads (impervious, smooth)
            if surface == SurfaceType.ROAD:
                V *= 1.5
            elif surface == SurfaceType.CHANNEL:
                V *= 2.0  # Channels concentrate flow

            # 5. Velocity components (follow flow direction)
            flow_mag = np.sqrt(flow_dir_x**2 + flow_dir_y**2) + self.eps
            vx = V * flow_dir_x / flow_mag
            vy = V * flow_dir_y / flow_mag

            # 6. Dispersion from other particles
            disp_x, disp_y = 0.0, 0.0
            for j in range(self.num_particles):
                if i != j:
                    xj, yj = x[j].item(), y[j].item()
                    diff_x = xi - xj
                    diff_y = yi - yj
                    dist = np.sqrt(diff_x**2 + diff_y**2) + self.eps

                    if dist < 8:
                        # Dispersion inversely proportional to distance
                        force = self.dispersion_strength / (dist + 1)
                        disp_x += force * diff_x / dist
                        disp_y += force * diff_y / dist

            # 7. Apply saturation boost (more water = easier flow)
            saturation_boost = 1.0 + 0.8 * saturation

            # 8. Final velocity
            dx[i] = saturation_boost * (vx + disp_x)
            dy[i] = saturation_boost * (vy + disp_y)

            # 9. Z follows terrain
            target_z = self.terrain.get_elevation(xi + dx[i].item() * 0.1,
                                                   yi + dy[i].item() * 0.1)
            dz[i] = (target_z - z[i].item()) * 2.0

            # 10. Mass loss due to infiltration
            infiltration = self.terrain.update_infiltration(xi, yi, 0.1)
            d_mass[i] = -infiltration * 10  # Scale factor

            # 11. Update terrain water depth
            self.terrain.add_water(xi, yi, 0.01 * mass_i)

        dstate = torch.stack([dx, dy, dz, d_mass], dim=1)
        return dstate.view(-1)


# =============================================================================
# MULTI-WAVE WATERSHED SIMULATION
# =============================================================================

class WatershedSimulation:
    """
    Multi-wave watershed simulation.

    Features:
    - Multiple water release waves from same source
    - Each wave starts after previous has dispersed
    - Saturation effects accumulate across waves
    - Surface-dependent flow dynamics
    """

    def __init__(
        self,
        terrain_config: WatershedConfig = None,
        source_point: Tuple[float, float] = (20, 100),
        particles_per_wave: int = 25,
        num_waves: int = 4,  # Initial + 3 more
        wave_interval: float = 4.0,  # Time between waves
        seed: int = 42
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.terrain = WatershedTerrain(terrain_config or WatershedConfig(seed=seed))
        self.source_point = source_point
        self.particles_per_wave = particles_per_wave
        self.num_waves = num_waves
        self.wave_interval = wave_interval

        # Results storage
        self.all_trajectories = []
        self.all_times = []
        self.wave_start_times = []
        self.saturation_history = []

    def _create_wave_particles(self, wave_idx: int) -> torch.Tensor:
        """Create initial particle states for a wave."""
        states = []

        for i in range(self.particles_per_wave):
            # Slight random offset from source
            angle = 2 * np.pi * i / self.particles_per_wave + np.random.uniform(-0.2, 0.2)
            radius = np.random.uniform(0.5, 2.0)

            x = self.source_point[0] + radius * np.cos(angle)
            y = self.source_point[1] + radius * np.sin(angle)
            z = self.terrain.get_elevation(x, y)

            # Water mass (slightly varies)
            mass = 1.0 + np.random.uniform(-0.2, 0.2)

            states.append([x, y, z, mass])

        return torch.tensor(states, dtype=torch.float32, device=device).view(-1)

    def run_wave(
        self,
        wave_idx: int,
        t_start: float,
        t_duration: float = 5.0,
        num_steps: int = 60
    ) -> np.ndarray:
        """Run a single wave of water particles."""

        print(f"  Wave {wave_idx + 1}: Starting at t={t_start:.1f}")

        # Create particles
        initial_state = self._create_wave_particles(wave_idx)

        # Create dynamics with current terrain state
        dynamics = WaterParticleDynamics(
            terrain=self.terrain,
            num_particles=self.particles_per_wave,
            base_water_depth=0.1 + 0.05 * wave_idx,  # More water each wave
            dispersion_strength=0.3
        )

        t_span = torch.linspace(0, t_duration, num_steps, device=device)

        with torch.no_grad():
            trajectories = odeint(
                dynamics,
                initial_state,
                t_span,
                method='euler',
                options={'step_size': 0.05}
            )

        # Reshape: [num_steps, particles, 4]
        trajectories = trajectories.view(num_steps, self.particles_per_wave, 4).cpu().numpy()

        # Store
        self.all_trajectories.append(trajectories)
        self.all_times.append(t_span.cpu().numpy() + t_start)
        self.wave_start_times.append(t_start)

        # Record saturation state
        self.saturation_history.append(self.terrain.water_depth.copy())

        return trajectories

    def run(self) -> Dict:
        """Run complete multi-wave simulation."""

        print("=" * 70)
        print("Watershed Multi-Wave Simulation")
        print("=" * 70)
        print(f"Source point: {self.source_point}")
        print(f"Particles per wave: {self.particles_per_wave}")
        print(f"Number of waves: {self.num_waves}")
        print()

        total_time = 0.0

        for wave_idx in range(self.num_waves):
            self.run_wave(
                wave_idx=wave_idx,
                t_start=total_time,
                t_duration=self.wave_interval,
                num_steps=50
            )
            total_time += self.wave_interval

            # Print saturation info
            avg_saturation = np.mean(self.terrain.water_depth)
            max_saturation = np.max(self.terrain.water_depth)
            print(f"    Avg water depth: {avg_saturation:.4f}, Max: {max_saturation:.4f}")

        print("\nSimulation complete!")

        return {
            'trajectories': self.all_trajectories,
            'times': self.all_times,
            'wave_starts': self.wave_start_times,
            'terrain': self.terrain,
            'saturation_history': self.saturation_history
        }

    def visualize(self, save_path: Optional[str] = None):
        """Create comprehensive visualization."""

        fig = plt.figure(figsize=(20, 16))

        # Color scheme for waves
        wave_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # 1. 3D terrain with all trajectories
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self._plot_3d_terrain_with_trajectories(ax1, wave_colors)

        # 2. Top-down view with flow paths
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_topdown_flow(ax2, wave_colors)

        # 3. Saturation evolution
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_saturation_evolution(ax3)

        # 4. Velocity analysis
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_velocity_analysis(ax4, wave_colors)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def _plot_3d_terrain_with_trajectories(self, ax, wave_colors):
        """Plot 3D terrain with all wave trajectories."""

        # Terrain mesh
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

        ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.5, linewidth=0)

        # Plot trajectories for each wave
        for wave_idx, trajectories in enumerate(self.all_trajectories):
            color = wave_colors[wave_idx % len(wave_colors)]

            for p in range(self.particles_per_wave):
                traj = trajectories[:, p, :]
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2] + 0.5,
                       color=color, linewidth=0.8, alpha=0.6)

        # Mark source
        source_z = self.terrain.get_elevation(*self.source_point)
        ax.scatter(*self.source_point, source_z + 3, color='blue', s=200,
                  marker='o', edgecolor='white', linewidth=2, label='Source')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain with Multi-Wave Water Flow', fontweight='bold')
        ax.view_init(elev=30, azim=45)

    def _plot_topdown_flow(self, ax, wave_colors):
        """Plot top-down view with all flow paths."""

        # Surface type background
        surface_colors = ['#808080', '#8B7355', '#4169E1', '#228B22']  # road, soil, channel, veg
        cmap = LinearSegmentedColormap.from_list('surface', surface_colors, N=4)

        ax.imshow(self.terrain.surface_type, origin='lower', cmap=cmap,
                 extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.6)

        # Water accumulation overlay
        water_overlay = np.ma.masked_where(self.terrain.water_depth < 0.001,
                                            self.terrain.water_depth)
        ax.imshow(water_overlay, origin='lower', cmap='Blues',
                 extent=[0, self.terrain.width, 0, self.terrain.height],
                 alpha=0.7, vmin=0, vmax=np.max(self.terrain.water_depth))

        # Plot trajectories
        for wave_idx, trajectories in enumerate(self.all_trajectories):
            color = wave_colors[wave_idx % len(wave_colors)]

            for p in range(self.particles_per_wave):
                traj = trajectories[:, p, :]
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1, alpha=0.5)

                # End point
                ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=20, alpha=0.7)

        # Source
        ax.scatter(*self.source_point, color='blue', s=200, marker='o',
                  edgecolor='white', linewidth=2, zorder=20, label='Source')

        # Legend for surface types
        legend_elements = [
            mpatches.Patch(facecolor='#808080', label='Road'),
            mpatches.Patch(facecolor='#8B7355', label='Soil'),
            mpatches.Patch(facecolor='#4169E1', label='Channel'),
            mpatches.Patch(facecolor='#228B22', label='Vegetation'),
        ]
        for i, color in enumerate(wave_colors[:self.num_waves]):
            legend_elements.append(mpatches.Patch(facecolor=color, label=f'Wave {i+1}'))

        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Top-Down View: Water Flow Paths & Accumulation', fontweight='bold')

    def _plot_saturation_evolution(self, ax):
        """Plot how saturation changes across waves."""

        # Calculate stats for each wave
        wave_labels = [f'After Wave {i+1}' for i in range(len(self.saturation_history))]

        total_water = [np.sum(h) for h in self.saturation_history]
        max_depth = [np.max(h) for h in self.saturation_history]
        wet_area = [np.sum(h > 0.001) for h in self.saturation_history]

        x = np.arange(len(wave_labels))
        width = 0.25

        bars1 = ax.bar(x - width, total_water, width, label='Total Water', color='blue', alpha=0.7)
        bars2 = ax.bar(x, [m * 100 for m in max_depth], width, label='Max Depth (×100)', color='red', alpha=0.7)
        bars3 = ax.bar(x + width, [w / 10 for w in wet_area], width, label='Wet Area (÷10)', color='green', alpha=0.7)

        ax.set_xlabel('Wave')
        ax.set_ylabel('Value')
        ax.set_title('Saturation Evolution Across Waves', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(wave_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotation
        ax.text(0.02, 0.98,
               'As water accumulates:\n• Infiltration decreases\n• Surface flow increases\n• Velocity increases',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _plot_velocity_analysis(self, ax, wave_colors):
        """Analyze velocity changes across waves and surface types."""

        # Calculate mean velocities for each wave
        for wave_idx, trajectories in enumerate(self.all_trajectories):
            times = self.all_times[wave_idx]

            velocities = []
            for t_idx in range(1, len(times)):
                dt = times[t_idx] - times[t_idx - 1]
                if dt > 0:
                    # Mean velocity across all particles
                    dx = trajectories[t_idx, :, 0] - trajectories[t_idx-1, :, 0]
                    dy = trajectories[t_idx, :, 1] - trajectories[t_idx-1, :, 1]
                    v = np.mean(np.sqrt(dx**2 + dy**2)) / dt
                    velocities.append(v)

            color = wave_colors[wave_idx % len(wave_colors)]
            ax.plot(times[1:] - self.wave_start_times[wave_idx], velocities,
                   color=color, linewidth=2, label=f'Wave {wave_idx + 1}')

        ax.set_xlabel('Time (within wave)')
        ax.set_ylabel('Mean Velocity')
        ax.set_title('Velocity Evolution: Later Waves Flow Faster', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotation
        ax.text(0.98, 0.02,
               'Manning\'s Eq: V = (1/n) × R^(2/3) × S^(1/2)\n'
               'n_eff decreases with saturation → V increases',
               transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def create_frame_sequence(self, num_frames: int = 16, save_path: Optional[str] = None):
        """Create frame sequence showing all waves."""

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()

        # Combine all times
        all_times_flat = []
        for times in self.all_times:
            all_times_flat.extend(times)
        all_times_flat = np.array(all_times_flat)

        frame_times = np.linspace(all_times_flat.min(), all_times_flat.max(), num_frames)
        wave_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        surface_colors = ['#808080', '#8B7355', '#4169E1', '#228B22']
        cmap = LinearSegmentedColormap.from_list('surface', surface_colors, N=4)

        for frame_idx, t_target in enumerate(frame_times):
            ax = axes[frame_idx]

            # Background
            ax.imshow(self.terrain.surface_type, origin='lower', cmap=cmap,
                     extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.5)

            # Elevation contours
            ax.contour(self.terrain.elevation, levels=10, colors='brown', alpha=0.3,
                      extent=[0, self.terrain.width, 0, self.terrain.height])

            # Plot active waves
            for wave_idx, (trajectories, times) in enumerate(zip(self.all_trajectories, self.all_times)):
                if times[0] <= t_target <= times[-1]:
                    # Find closest time index
                    t_idx = np.argmin(np.abs(times - t_target))
                    color = wave_colors[wave_idx % len(wave_colors)]

                    # Plot trails
                    trail_start = max(0, t_idx - 10)
                    for p in range(self.particles_per_wave):
                        traj = trajectories[trail_start:t_idx+1, p, :]
                        if len(traj) > 1:
                            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.8, alpha=0.4)

                        # Current position
                        ax.scatter(trajectories[t_idx, p, 0], trajectories[t_idx, p, 1],
                                  color=color, s=15, alpha=0.8)

                elif t_target > times[-1]:
                    # Wave completed - show final positions faded
                    color = wave_colors[wave_idx % len(wave_colors)]
                    for p in range(self.particles_per_wave):
                        ax.scatter(trajectories[-1, p, 0], trajectories[-1, p, 1],
                                  color=color, s=10, alpha=0.3)

            # Source
            ax.scatter(*self.source_point, color='blue', s=100, marker='o',
                      edgecolor='white', linewidth=1, zorder=20)

            ax.set_xlim(0, self.terrain.width)
            ax.set_ylim(0, self.terrain.height)
            ax.set_title(f't = {t_target:.1f}', fontsize=11, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        plt.suptitle('Watershed Multi-Wave Simulation: Time Sequence',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def create_video(self, save_path: str, fps: int = 12):
        """Create animation video."""

        print(f"Creating video: {save_path}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Setup backgrounds
        surface_colors = ['#808080', '#8B7355', '#4169E1', '#228B22']
        cmap = LinearSegmentedColormap.from_list('surface', surface_colors, N=4)

        ax1.imshow(self.terrain.surface_type, origin='lower', cmap=cmap,
                  extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.5)
        ax1.contour(self.terrain.elevation, levels=10, colors='brown', alpha=0.3,
                   extent=[0, self.terrain.width, 0, self.terrain.height])
        ax1.scatter(*self.source_point, color='blue', s=150, marker='o',
                   edgecolor='white', linewidth=2, zorder=20, label='Source')
        ax1.set_xlim(0, self.terrain.width)
        ax1.set_ylim(0, self.terrain.height)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend(loc='upper right')

        # Right panel: saturation map (will update)
        water_img = ax2.imshow(np.zeros_like(self.terrain.water_depth), origin='lower',
                              cmap='Blues', extent=[0, self.terrain.width, 0, self.terrain.height],
                              vmin=0, vmax=0.5)
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        plt.colorbar(water_img, ax=ax2, label='Water Depth')

        wave_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Particle markers
        particle_plots = []
        trail_plots = []
        for wave_idx in range(self.num_waves):
            color = wave_colors[wave_idx % len(wave_colors)]
            scatter = ax1.scatter([], [], color=color, s=20, alpha=0.8, label=f'Wave {wave_idx+1}')
            particle_plots.append(scatter)
            trails, = ax1.plot([], [], color=color, linewidth=0.5, alpha=0.3)
            trail_plots.append(trails)

        time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                            fontsize=12, fontweight='bold', verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        title = fig.suptitle('', fontsize=14, fontweight='bold')

        # Combine all time points
        all_times_flat = []
        for times in self.all_times:
            all_times_flat.extend(times)
        t_min, t_max = min(all_times_flat), max(all_times_flat)

        num_frames = int((t_max - t_min) * fps)
        frame_times = np.linspace(t_min, t_max, num_frames)

        def animate(frame):
            t_target = frame_times[frame]

            # Update water map based on current waves
            cumulative_water = np.zeros_like(self.terrain.water_depth)

            for wave_idx, (trajectories, times) in enumerate(zip(self.all_trajectories, self.all_times)):
                color = wave_colors[wave_idx % len(wave_colors)]

                if times[0] <= t_target <= times[-1]:
                    t_idx = np.argmin(np.abs(times - t_target))

                    # Current positions
                    positions = trajectories[t_idx, :, :2]
                    particle_plots[wave_idx].set_offsets(positions)

                    # Add to water map
                    for p in range(self.particles_per_wave):
                        x, y = int(positions[p, 0]), int(positions[p, 1])
                        if 0 <= x < self.terrain.width and 0 <= y < self.terrain.height:
                            cumulative_water[y, x] += 0.1

                elif t_target > times[-1]:
                    # Show final positions
                    positions = trajectories[-1, :, :2]
                    particle_plots[wave_idx].set_offsets(positions)

                    for p in range(self.particles_per_wave):
                        x, y = int(positions[p, 0]), int(positions[p, 1])
                        if 0 <= x < self.terrain.width and 0 <= y < self.terrain.height:
                            cumulative_water[y, x] += 0.1
                else:
                    # Wave not started
                    particle_plots[wave_idx].set_offsets(np.empty((0, 2)))

            # Smooth water accumulation
            cumulative_water = gaussian_filter(cumulative_water, sigma=2)
            water_img.set_array(cumulative_water)
            water_img.set_clim(0, max(0.1, cumulative_water.max()))

            time_text.set_text(f'Time: {t_target:.2f}')

            # Determine active wave
            active_wave = 0
            for wave_idx, times in enumerate(self.all_times):
                if times[0] <= t_target <= times[-1]:
                    active_wave = wave_idx + 1

            title.set_text(f'Watershed Multi-Wave Simulation (Wave {active_wave} of {self.num_waves})')

            return particle_plots + [water_img, time_text, title]

        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        plt.close(fig)

        print(f"Video saved: {save_path}")
        return save_path


def main():
    """Run the watershed simulation."""

    print("=" * 70)
    print("WATERSHED qODE SIMULATION")
    print("Multi-Wave Water Dispersion with Hydrological Equations")
    print("=" * 70)
    print()
    print("Hydrological Equations Used:")
    print("  1. Manning's Velocity: V = (1/n) × R^(2/3) × S^(1/2)")
    print("  2. Green-Ampt Infiltration: f = K × (1 + ψΔθ/F)")
    print("  3. Saturation Factor: S = 1 - exp(-water/max)")
    print("  4. Effective Roughness: n_eff = n × (1 - 0.6×S)")
    print()

    # Configuration
    terrain_config = WatershedConfig(
        width=120,
        height=120,
        max_elevation=25.0,
        num_hills=5,
        road_width=4,
        road_spacing=25,
        channel_width=3,
        seed=42
    )

    # Create simulation
    sim = WatershedSimulation(
        terrain_config=terrain_config,
        source_point=(25, 100),  # Upper-left area
        particles_per_wave=25,
        num_waves=4,  # Initial + 3 more waves
        wave_interval=4.0,
        seed=42
    )

    # Run simulation
    results = sim.run()

    # Output directory
    output_dir = os.path.dirname(__file__)

    # Create visualizations
    print("\nCreating visualizations...")

    # 1. Main visualization
    sim.visualize(save_path=os.path.join(output_dir, 'watershed_visualization.png'))

    # 2. Frame sequence
    sim.create_frame_sequence(num_frames=16,
                             save_path=os.path.join(output_dir, 'watershed_time_sequence.png'))

    # 3. Video
    sim.create_video(save_path=os.path.join(output_dir, 'watershed_multiwave.gif'), fps=10)

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    print()
    print("Key findings:")
    print("  - Later waves flow faster due to surface saturation")
    print("  - Roads channel water quickly (low Manning's n)")
    print("  - Soil infiltration decreases as it saturates")
    print("  - Water accumulates in channels and low areas")
    print()
    print("Generated files:")
    print(f"  - {os.path.join(output_dir, 'watershed_visualization.png')}")
    print(f"  - {os.path.join(output_dir, 'watershed_time_sequence.png')}")
    print(f"  - {os.path.join(output_dir, 'watershed_multiwave.gif')}")

    return sim


if __name__ == "__main__":
    sim = main()
