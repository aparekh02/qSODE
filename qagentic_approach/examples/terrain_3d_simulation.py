#!/usr/bin/env python3
"""
3D Terrain qODE Simulation with Particle Trajectory Optimization
================================================================

Features:
- 3D terrain with hills and valleys (elevation map)
- Buildings as 3D obstacles with height
- Optimal trajectory finding using gradient-based differential operations
- Particle tracing from start point with dispersion
- Full 3D visualization of particle movement
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


@dataclass
class TerrainConfig:
    """Configuration for 3D terrain generation."""
    width: int = 100
    height: int = 100
    max_elevation: float = 30.0  # Maximum hill height
    num_hills: int = 4
    hill_spread: float = 15.0
    building_height_range: Tuple[float, float] = (10.0, 25.0)
    seed: int = 42


class Terrain3D:
    """
    3D Terrain with elevation map and building obstacles.

    Features:
    - Procedural hill/valley generation
    - Buildings as 3D rectangular obstacles
    - Elevation gradients for trajectory optimization
    """

    def __init__(self, config: TerrainConfig = None):
        self.config = config or TerrainConfig()
        np.random.seed(self.config.seed)

        self.width = self.config.width
        self.height = self.config.height

        # Initialize grids
        self.elevation = np.zeros((self.height, self.width))
        self.building_mask = np.zeros((self.height, self.width), dtype=bool)
        self.building_heights = np.zeros((self.height, self.width))
        self.medium_type = np.zeros((self.height, self.width), dtype=int)  # 0=ground, 1=street, 2=building

        # Generate terrain
        self._generate_hills()
        self._generate_streets()
        self._generate_buildings()

        # Compute gradients
        self._compute_gradients()

    def _generate_hills(self):
        """Generate rolling hills using Gaussian peaks."""
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # Add multiple hill peaks
        for _ in range(self.config.num_hills):
            cx = np.random.uniform(10, self.width - 10)
            cy = np.random.uniform(10, self.height - 10)
            amplitude = np.random.uniform(0.5, 1.0) * self.config.max_elevation
            spread = np.random.uniform(0.8, 1.2) * self.config.hill_spread

            hill = amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * spread**2))
            self.elevation += hill

        # Add some valleys (negative gaussians)
        for _ in range(2):
            cx = np.random.uniform(20, self.width - 20)
            cy = np.random.uniform(20, self.height - 20)
            amplitude = np.random.uniform(0.2, 0.4) * self.config.max_elevation
            spread = np.random.uniform(0.5, 1.0) * self.config.hill_spread

            valley = -amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * spread**2))
            self.elevation += valley

        # Smooth the terrain
        self.elevation = gaussian_filter(self.elevation, sigma=2)

        # Ensure non-negative elevation
        self.elevation = np.maximum(self.elevation, 0)

    def _generate_streets(self):
        """Generate street grid (lower elevation, flat)."""
        street_spacing = 20
        street_width = 3

        for i in range(0, self.width, street_spacing):
            x_start = max(0, i - street_width // 2)
            x_end = min(self.width, i + street_width // 2 + 1)
            self.medium_type[:, x_start:x_end] = 1
            # Streets are at lower elevation
            self.elevation[:, x_start:x_end] *= 0.3

        for j in range(0, self.height, street_spacing):
            y_start = max(0, j - street_width // 2)
            y_end = min(self.height, j + street_width // 2 + 1)
            self.medium_type[y_start:y_end, :] = 1
            self.elevation[y_start:y_end, :] *= 0.3

    def _generate_buildings(self):
        """Generate 3D buildings as obstacles."""
        street_spacing = 20

        for block_x in range(street_spacing // 2, self.width - street_spacing // 2, street_spacing):
            for block_y in range(street_spacing // 2, self.height - street_spacing // 2, street_spacing):
                # Random number of buildings per block
                num_buildings = np.random.randint(1, 4)

                for _ in range(num_buildings):
                    # Building dimensions
                    bw = np.random.randint(4, 10)
                    bh = np.random.randint(4, 10)

                    # Position within block
                    bx = block_x - street_spacing // 2 + 4 + np.random.randint(0, max(1, street_spacing - bw - 8))
                    by = block_y - street_spacing // 2 + 4 + np.random.randint(0, max(1, street_spacing - bh - 8))

                    bx = np.clip(bx, 0, self.width - bw)
                    by = np.clip(by, 0, self.height - bh)

                    # Building height
                    b_height = np.random.uniform(*self.config.building_height_range)

                    # Mark building area
                    self.building_mask[by:by+bh, bx:bx+bw] = True
                    self.building_heights[by:by+bh, bx:bx+bw] = b_height
                    self.medium_type[by:by+bh, bx:bx+bw] = 2

    def _compute_gradients(self):
        """Compute terrain gradients for trajectory optimization."""
        # Gradient in x and y directions
        self.grad_x = np.gradient(self.elevation, axis=1)
        self.grad_y = np.gradient(self.elevation, axis=0)

        # Combined elevation (terrain + buildings)
        self.total_height = self.elevation + self.building_heights
        self.total_grad_x = np.gradient(self.total_height, axis=1)
        self.total_grad_y = np.gradient(self.total_height, axis=0)

    def get_elevation(self, x: float, y: float) -> float:
        """Get terrain elevation at a point."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.elevation[iy, ix]

    def get_total_height(self, x: float, y: float) -> float:
        """Get total height (terrain + building) at a point."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.total_height[iy, ix]

    def get_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Get terrain gradient at a point."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.total_grad_x[iy, ix], self.total_grad_y[iy, ix]

    def is_building(self, x: float, y: float) -> bool:
        """Check if point is inside a building."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))
        return self.building_mask[iy, ix]

    def get_speed_multiplier(self, x: float, y: float) -> float:
        """Get movement speed multiplier based on terrain."""
        ix = int(np.clip(x, 0, self.width - 1))
        iy = int(np.clip(y, 0, self.height - 1))

        medium = self.medium_type[iy, ix]
        if medium == 2:  # Building
            return 0.05  # Very slow (obstacle)
        elif medium == 1:  # Street
            return 1.0  # Fast
        else:  # Open ground
            # Slower on steep terrain
            slope = np.sqrt(self.grad_x[iy, ix]**2 + self.grad_y[iy, ix]**2)
            return max(0.3, 1.0 - slope * 0.1)


class ParticleDynamics3D(nn.Module):
    """
    3D Particle dynamics with trajectory optimization.

    Uses differential operations to find optimal paths:
    - Gradient descent on terrain (avoid uphill)
    - Building avoidance (repulsive potential)
    - Goal attraction
    - Dispersion from initial point
    """

    def __init__(
        self,
        terrain: Terrain3D,
        num_particles: int = 20,
        goal: Optional[Tuple[float, float]] = None,
        dispersion_strength: float = 0.5,
        goal_strength: float = 0.3,
        terrain_strength: float = 0.4,
        building_repulsion: float = 2.0
    ):
        super().__init__()

        self.terrain = terrain
        self.num_particles = num_particles
        self.goal = goal or (terrain.width * 0.8, terrain.height * 0.8)

        self.dispersion_strength = dispersion_strength
        self.goal_strength = goal_strength
        self.terrain_strength = terrain_strength
        self.building_repulsion = building_repulsion

        self.eps = 1e-6

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute particle velocities using differential operations.

        State: [num_particles, 4] -> [x, y, z, phase]
        """
        state = state.view(self.num_particles, 4)

        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        phase = state[:, 3]

        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dz = torch.zeros_like(z)
        dphase = torch.ones_like(phase) * 0.2

        for i in range(self.num_particles):
            xi, yi, zi = x[i].item(), y[i].item(), z[i].item()

            # 1. Terrain gradient descent (move downhill / along contours)
            grad_x, grad_y = self.terrain.get_gradient(xi, yi)
            terrain_force_x = -self.terrain_strength * grad_x
            terrain_force_y = -self.terrain_strength * grad_y

            # 2. Building repulsion (avoid obstacles)
            building_force_x, building_force_y = self._compute_building_repulsion(xi, yi)

            # 3. Goal attraction
            goal_dx = self.goal[0] - xi
            goal_dy = self.goal[1] - yi
            goal_dist = np.sqrt(goal_dx**2 + goal_dy**2) + self.eps
            goal_force_x = self.goal_strength * goal_dx / goal_dist
            goal_force_y = self.goal_strength * goal_dy / goal_dist

            # 4. Dispersion from other particles
            disp_force_x, disp_force_y = 0.0, 0.0
            for j in range(self.num_particles):
                if i != j:
                    xj, yj = x[j].item(), y[j].item()
                    diff_x = xi - xj
                    diff_y = yi - yj
                    dist = np.sqrt(diff_x**2 + diff_y**2) + self.eps

                    # Repulsive dispersion (inverse square)
                    if dist < 15:  # Only nearby particles
                        force_mag = self.dispersion_strength / (dist**2)
                        disp_force_x += force_mag * diff_x / dist
                        disp_force_y += force_mag * diff_y / dist

            # 5. Speed modulation based on terrain
            speed_mult = self.terrain.get_speed_multiplier(xi, yi)

            # Combine all forces
            total_dx = speed_mult * (terrain_force_x + building_force_x + goal_force_x + disp_force_x)
            total_dy = speed_mult * (terrain_force_y + building_force_y + goal_force_y + disp_force_y)

            # Z follows terrain height
            target_z = self.terrain.get_elevation(xi + total_dx * 0.1, yi + total_dy * 0.1)
            dz_val = (target_z - zi) * 2.0  # Smoothly adjust to terrain

            dx[i] = total_dx
            dy[i] = total_dy
            dz[i] = dz_val

            # Phase evolution
            dphase[i] = 0.2 + 0.1 * np.sqrt(total_dx**2 + total_dy**2)

        dstate = torch.stack([dx, dy, dz, dphase], dim=1)
        return dstate.view(-1)

    def _compute_building_repulsion(self, x: float, y: float) -> Tuple[float, float]:
        """Compute repulsive force from nearby buildings."""
        force_x, force_y = 0.0, 0.0
        search_radius = 10

        for dx in range(-search_radius, search_radius + 1, 2):
            for dy in range(-search_radius, search_radius + 1, 2):
                check_x = x + dx
                check_y = y + dy

                if 0 <= check_x < self.terrain.width and 0 <= check_y < self.terrain.height:
                    if self.terrain.is_building(check_x, check_y):
                        dist = np.sqrt(dx**2 + dy**2) + self.eps
                        if dist < search_radius:
                            force_mag = self.building_repulsion / (dist**2)
                            force_x += force_mag * (-dx) / dist
                            force_y += force_mag * (-dy) / dist

        return force_x, force_y


class Simulation3D:
    """
    Full 3D simulation with particle trajectory optimization and visualization.
    """

    def __init__(
        self,
        terrain_config: TerrainConfig = None,
        num_particles: int = 15,
        start_point: Tuple[float, float] = (15, 15),
        goal_point: Tuple[float, float] = (85, 85),
        seed: int = 42
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.terrain = Terrain3D(terrain_config or TerrainConfig(seed=seed))
        self.num_particles = num_particles
        self.start_point = start_point
        self.goal_point = goal_point

        # Initialize dynamics
        self.dynamics = ParticleDynamics3D(
            terrain=self.terrain,
            num_particles=num_particles,
            goal=goal_point,
            dispersion_strength=0.8,
            goal_strength=0.4,
            terrain_strength=0.5,
            building_repulsion=3.0
        )

        # Results storage
        self.trajectories = None
        self.times = None

    def _initialize_particles(self) -> torch.Tensor:
        """Initialize particles at start point with slight dispersion."""
        states = []

        for i in range(self.num_particles):
            # Small random offset from start point
            angle = 2 * np.pi * i / self.num_particles
            radius = np.random.uniform(1, 3)

            x = self.start_point[0] + radius * np.cos(angle)
            y = self.start_point[1] + radius * np.sin(angle)
            z = self.terrain.get_elevation(x, y)
            phase = np.random.uniform(0, 2 * np.pi)

            states.append([x, y, z, phase])

        return torch.tensor(states, dtype=torch.float32, device=device).view(-1)

    def run(self, t_end: float = 15.0, num_steps: int = 150) -> dict:
        """Run the simulation."""
        print(f"Running 3D terrain simulation...")
        print(f"  Particles: {self.num_particles}")
        print(f"  Start: {self.start_point}")
        print(f"  Goal: {self.goal_point}")

        t_span = torch.linspace(0, t_end, num_steps, device=device)
        initial_state = self._initialize_particles()

        with torch.no_grad():
            trajectories = odeint(
                self.dynamics,
                initial_state,
                t_span,
                method='euler',
                options={'step_size': 0.05}
            )

        # Reshape: [num_steps, num_particles, 4]
        trajectories = trajectories.view(num_steps, self.num_particles, 4).cpu().numpy()

        self.trajectories = trajectories
        self.times = t_span.cpu().numpy()

        print("Simulation complete!")

        return {
            'trajectories': trajectories,
            'times': self.times,
            'terrain': self.terrain
        }

    def visualize_3d(self, save_path: Optional[str] = None):
        """Create comprehensive 3D visualization."""
        if self.trajectories is None:
            raise ValueError("Run simulation first")

        fig = plt.figure(figsize=(20, 16))

        # 1. 3D terrain with trajectories
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self._plot_3d_terrain(ax1)

        # 2. Top-down view with trajectory traces
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_topdown_trajectories(ax2)

        # 3. Elevation profile along mean trajectory
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_elevation_profile(ax3)

        # 4. Particle dispersion over time
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_dispersion(ax4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def _plot_3d_terrain(self, ax):
        """Plot 3D terrain surface with buildings and particle trajectories."""
        # Create mesh grid
        x = np.arange(0, self.terrain.width, 2)
        y = np.arange(0, self.terrain.height, 2)
        X, Y = np.meshgrid(x, y)

        # Get elevation for mesh
        Z = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.terrain.elevation[
                    min(int(Y[i, j]), self.terrain.height - 1),
                    min(int(X[i, j]), self.terrain.width - 1)
                ]

        # Plot terrain surface
        ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, linewidth=0)

        # Plot buildings as 3D boxes
        self._add_buildings_3d(ax)

        # Plot particle trajectories
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, self.num_particles))
        for p in range(self.num_particles):
            traj = self.trajectories[:, p, :]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2] + 1,  # +1 to float above terrain
                   color=colors[p], linewidth=1.5, alpha=0.8)

            # Start point
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2] + 1,
                      color='green', s=50, marker='o', zorder=10)
            # End point
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2] + 1,
                      color='red', s=50, marker='*', zorder=10)

        # Mark goal
        goal_z = self.terrain.get_elevation(*self.goal_point)
        ax.scatter(*self.goal_point, goal_z + 5, color='gold', s=200, marker='*',
                  edgecolor='black', linewidth=1, zorder=20, label='Goal')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain with Particle Trajectories', fontsize=12, fontweight='bold')
        ax.view_init(elev=35, azim=45)

    def _add_buildings_3d(self, ax):
        """Add 3D building boxes to the plot."""
        # Find building regions and plot as boxes
        visited = np.zeros_like(self.terrain.building_mask)

        for y in range(0, self.terrain.height, 2):
            for x in range(0, self.terrain.width, 2):
                if self.terrain.building_mask[y, x] and not visited[y, x]:
                    # Find building extent
                    bx_start, bx_end = x, x
                    by_start, by_end = y, y

                    while bx_end < self.terrain.width - 1 and self.terrain.building_mask[y, bx_end + 1]:
                        bx_end += 1
                    while by_end < self.terrain.height - 1 and self.terrain.building_mask[by_end + 1, x]:
                        by_end += 1

                    # Mark as visited
                    visited[by_start:by_end+1, bx_start:bx_end+1] = True

                    # Get building height
                    b_height = self.terrain.building_heights[y, x]
                    base_z = self.terrain.elevation[y, x]

                    if b_height > 0:
                        # Draw building as simple box (top face)
                        vertices = [
                            [bx_start, by_start, base_z + b_height],
                            [bx_end + 1, by_start, base_z + b_height],
                            [bx_end + 1, by_end + 1, base_z + b_height],
                            [bx_start, by_end + 1, base_z + b_height],
                        ]

                        # Create polygon for top face
                        verts = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
                        collection = Poly3DCollection(verts, alpha=0.7, facecolor='#8B4513',
                                                     edgecolor='#5C3317', linewidth=0.5)
                        ax.add_collection3d(collection)

    def _plot_topdown_trajectories(self, ax):
        """Plot top-down view with full trajectory traces."""
        # Plot terrain elevation as background
        im = ax.imshow(self.terrain.elevation, origin='lower', cmap='terrain',
                      extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.7)

        # Overlay buildings
        building_overlay = np.ma.masked_where(~self.terrain.building_mask,
                                               self.terrain.building_heights)
        ax.imshow(building_overlay, origin='lower', cmap='Reds',
                 extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.8)

        # Plot all particle trajectories
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, self.num_particles))
        for p in range(self.num_particles):
            traj = self.trajectories[:, p, :]
            ax.plot(traj[:, 0], traj[:, 1], color=colors[p], linewidth=1.5, alpha=0.8)

            # Start and end markers
            ax.scatter(traj[0, 0], traj[0, 1], color='green', s=80, marker='o',
                      edgecolor='white', linewidth=1, zorder=10)
            ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=80, marker='s',
                      edgecolor='white', linewidth=1, zorder=10)

        # Mark start and goal
        ax.scatter(*self.start_point, color='lime', s=200, marker='o',
                  edgecolor='black', linewidth=2, zorder=20, label='Start')
        ax.scatter(*self.goal_point, color='gold', s=200, marker='*',
                  edgecolor='black', linewidth=2, zorder=20, label='Goal')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Top-Down View: Particle Traces (Start→Finish)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')

        # Add colorbar for elevation
        plt.colorbar(im, ax=ax, label='Elevation', shrink=0.8)

    def _plot_elevation_profile(self, ax):
        """Plot elevation profile along particle trajectories."""
        # Calculate mean trajectory
        mean_traj = np.mean(self.trajectories, axis=1)  # [num_steps, 4]

        # Plot elevation along trajectory
        ax.fill_between(self.times, 0, mean_traj[:, 2], alpha=0.3, color='brown',
                       label='Terrain Elevation')
        ax.plot(self.times, mean_traj[:, 2], 'b-', linewidth=2, label='Mean Particle Z')

        # Plot individual particle Z trajectories
        for p in range(min(5, self.num_particles)):  # Show first 5
            ax.plot(self.times, self.trajectories[:, p, 2], '--', alpha=0.4,
                   linewidth=1, label=f'Particle {p+1}' if p < 3 else None)

        ax.set_xlabel('Time')
        ax.set_ylabel('Elevation (Z)')
        ax.set_title('Elevation Profile Along Trajectory', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_dispersion(self, ax):
        """Plot particle dispersion over time."""
        # Calculate centroid and dispersion at each timestep
        centroids = np.mean(self.trajectories[:, :, :2], axis=1)  # [num_steps, 2]

        dispersions = []
        for t in range(len(self.times)):
            positions = self.trajectories[t, :, :2]
            centroid = centroids[t]
            distances = np.sqrt(np.sum((positions - centroid)**2, axis=1))
            dispersions.append(np.std(distances))

        ax.plot(self.times, dispersions, 'b-', linewidth=2, label='Dispersion (std)')
        ax.fill_between(self.times, 0, dispersions, alpha=0.3)

        # Mark key phases
        ax.axvline(x=self.times[len(self.times)//4], color='g', linestyle='--',
                  alpha=0.5, label='Early spread')
        ax.axvline(x=self.times[len(self.times)//2], color='orange', linestyle='--',
                  alpha=0.5, label='Mid journey')
        ax.axvline(x=self.times[3*len(self.times)//4], color='r', linestyle='--',
                  alpha=0.5, label='Convergence')

        ax.set_xlabel('Time')
        ax.set_ylabel('Dispersion')
        ax.set_title('Particle Dispersion Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def create_frame_sequence(self, num_frames: int = 12, save_path: Optional[str] = None):
        """Create a sequence of frames showing time-by-time movement."""
        if self.trajectories is None:
            raise ValueError("Run simulation first")

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()

        frame_indices = np.linspace(0, len(self.times) - 1, num_frames, dtype=int)
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, self.num_particles))

        for idx, (ax, frame_idx) in enumerate(zip(axes, frame_indices)):
            t = self.times[frame_idx]

            # Plot terrain
            ax.imshow(self.terrain.elevation, origin='lower', cmap='terrain',
                     extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.6)

            # Plot buildings
            building_overlay = np.ma.masked_where(~self.terrain.building_mask,
                                                   self.terrain.building_heights)
            ax.imshow(building_overlay, origin='lower', cmap='Reds',
                     extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.7)

            # Plot trajectory trails up to current frame
            for p in range(self.num_particles):
                traj = self.trajectories[:frame_idx+1, p, :]
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1], color=colors[p], linewidth=1, alpha=0.5)

                # Current position
                ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[p], s=40,
                          edgecolor='white', linewidth=0.5, zorder=10)

            # Mark start and goal
            ax.scatter(*self.start_point, color='lime', s=100, marker='o',
                      edgecolor='black', linewidth=1, zorder=20)
            ax.scatter(*self.goal_point, color='gold', s=100, marker='*',
                      edgecolor='black', linewidth=1, zorder=20)

            ax.set_xlim(0, self.terrain.width)
            ax.set_ylim(0, self.terrain.height)
            ax.set_title(f'Time = {t:.2f}', fontsize=11, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        plt.suptitle('qODE 3D Terrain: Time-by-Time Particle Movement',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def create_video(self, save_path: str, fps: int = 15):
        """Create animated video of the simulation."""
        from matplotlib.animation import FuncAnimation, PillowWriter

        if self.trajectories is None:
            raise ValueError("Run simulation first")

        print(f"Creating video: {save_path}")

        fig = plt.figure(figsize=(16, 7))

        # Left: Top-down view
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xlim(0, self.terrain.width)
        ax1.set_ylim(0, self.terrain.height)

        # Plot static terrain
        ax1.imshow(self.terrain.elevation, origin='lower', cmap='terrain',
                  extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.6)
        building_overlay = np.ma.masked_where(~self.terrain.building_mask,
                                               self.terrain.building_heights)
        ax1.imshow(building_overlay, origin='lower', cmap='Reds',
                  extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.7)

        # Mark start and goal
        ax1.scatter(*self.start_point, color='lime', s=150, marker='o',
                   edgecolor='black', linewidth=2, zorder=20, label='Start')
        ax1.scatter(*self.goal_point, color='gold', s=200, marker='*',
                   edgecolor='black', linewidth=2, zorder=20, label='Goal')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')

        # Right: 3D view
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        # Plot terrain surface (static)
        x = np.arange(0, self.terrain.width, 3)
        y = np.arange(0, self.terrain.height, 3)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.terrain.elevation[
                    min(int(Y[i, j]), self.terrain.height - 1),
                    min(int(X[i, j]), self.terrain.width - 1)
                ]
        ax2.plot_surface(X, Y, Z, cmap='terrain', alpha=0.5, linewidth=0)
        ax2.view_init(elev=30, azim=45)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        colors = plt.cm.plasma(np.linspace(0.2, 0.9, self.num_particles))

        # Initialize particle markers and trails
        particles_2d = []
        particles_3d = []
        trails_2d = []
        trails_3d = []

        for p in range(self.num_particles):
            # 2D markers
            marker, = ax1.plot([], [], 'o', color=colors[p], markersize=8,
                              markeredgecolor='white', markeredgewidth=0.5)
            particles_2d.append(marker)
            trail, = ax1.plot([], [], '-', color=colors[p], linewidth=1.5, alpha=0.6)
            trails_2d.append(trail)

            # 3D markers
            marker3d, = ax2.plot([], [], [], 'o', color=colors[p], markersize=6)
            particles_3d.append(marker3d)
            trail3d, = ax2.plot([], [], [], '-', color=colors[p], linewidth=1, alpha=0.6)
            trails_3d.append(trail3d)

        time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                            fontsize=14, fontweight='bold', verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        title = fig.suptitle('', fontsize=14, fontweight='bold')

        trail_length = 30

        def init():
            for p in range(self.num_particles):
                particles_2d[p].set_data([], [])
                trails_2d[p].set_data([], [])
                particles_3d[p].set_data([], [])
                particles_3d[p].set_3d_properties([])
                trails_3d[p].set_data([], [])
                trails_3d[p].set_3d_properties([])
            time_text.set_text('')
            return particles_2d + trails_2d + particles_3d + trails_3d + [time_text]

        def animate(frame):
            t = self.times[frame]
            trail_start = max(0, frame - trail_length)

            for p in range(self.num_particles):
                x, y, z = self.trajectories[frame, p, :3]

                # Update 2D
                particles_2d[p].set_data([x], [y])
                trails_2d[p].set_data(
                    self.trajectories[trail_start:frame+1, p, 0],
                    self.trajectories[trail_start:frame+1, p, 1]
                )

                # Update 3D
                particles_3d[p].set_data([x], [y])
                particles_3d[p].set_3d_properties([z + 1])
                trails_3d[p].set_data(
                    self.trajectories[trail_start:frame+1, p, 0],
                    self.trajectories[trail_start:frame+1, p, 1]
                )
                trails_3d[p].set_3d_properties(self.trajectories[trail_start:frame+1, p, 2] + 1)

            time_text.set_text(f'Time: {t:.2f}')
            title.set_text(f'qODE 3D Particle Trajectory Optimization (Frame {frame+1}/{len(self.times)})')

            return particles_2d + trails_2d + particles_3d + trails_3d + [time_text, title]

        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=len(self.times), interval=1000/fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        plt.close(fig)

        print(f"Video saved: {save_path}")
        return save_path


def main():
    """Run the 3D terrain simulation example."""

    print("=" * 70)
    print("qODE 3D Terrain Simulation with Particle Trajectory Optimization")
    print("=" * 70)
    print()

    # Configuration
    terrain_config = TerrainConfig(
        width=100,
        height=100,
        max_elevation=25.0,
        num_hills=5,
        hill_spread=12.0,
        building_height_range=(8.0, 20.0),
        seed=42
    )

    # Create simulation
    sim = Simulation3D(
        terrain_config=terrain_config,
        num_particles=15,
        start_point=(10, 10),
        goal_point=(90, 90),
        seed=42
    )

    # Run simulation
    results = sim.run(t_end=15.0, num_steps=150)

    # Output directory
    output_dir = os.path.dirname(__file__)

    # Create visualizations
    print("\nCreating visualizations...")

    # 1. Main 3D visualization
    sim.visualize_3d(save_path=os.path.join(output_dir, '3d_terrain_visualization.png'))

    # 2. Frame sequence
    sim.create_frame_sequence(num_frames=12,
                             save_path=os.path.join(output_dir, '3d_time_sequence.png'))

    # 3. Video animation
    sim.create_video(save_path=os.path.join(output_dir, '3d_particle_trajectory.gif'), fps=12)

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  - {os.path.join(output_dir, '3d_terrain_visualization.png')}")
    print(f"  - {os.path.join(output_dir, '3d_time_sequence.png')}")
    print(f"  - {os.path.join(output_dir, '3d_particle_trajectory.gif')}")

    return sim


if __name__ == "__main__":
    sim = main()
