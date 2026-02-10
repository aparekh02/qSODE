#!/usr/bin/env python3
"""
Watershed Comparison: Flat Ground vs Hilltop
=============================================

Compares water dispersion behavior starting from:
1. FLAT GROUND - Low elevation area
2. HILLTOP - High elevation peak

Shows differences in:
- Flow velocity (gravity-driven acceleration)
- Dispersion patterns
- Path taken through terrain
- Time to reach drainage channel
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# TERRAIN GENERATION
# =============================================================================

class ComparisonTerrain:
    """
    Terrain with distinct flat area and prominent hill for comparison.
    """

    def __init__(self, width: int = 120, height: int = 120, seed: int = 42):
        np.random.seed(seed)
        self.width = width
        self.height = height

        # Grids
        self.elevation = np.zeros((height, width))
        self.surface_type = np.ones((height, width))  # 1 = soil default

        # Generate terrain with clear flat and hill areas
        self._generate_terrain()
        self._generate_roads()
        self._generate_channel()
        self._compute_gradients()

        # Find key locations
        self.hilltop = self._find_hilltop()
        self.flat_ground = self._find_flat_ground()

        print(f"Hilltop location: {self.hilltop} (elevation: {self.get_elevation(*self.hilltop):.1f})")
        print(f"Flat ground location: {self.flat_ground} (elevation: {self.get_elevation(*self.flat_ground):.1f})")

    def _generate_terrain(self):
        """Generate terrain with prominent hill and flat valley."""
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)

        # Base slope toward channel (bottom-right)
        base = 5 + 0.08 * (self.width - X + self.height - Y)
        self.elevation = base

        # Add prominent hill in upper-left quadrant
        hill_cx, hill_cy = 30, 90
        hill_amplitude = 35
        hill_spread = 18
        hill = hill_amplitude * np.exp(-((X - hill_cx)**2 + (Y - hill_cy)**2) / (2 * hill_spread**2))
        self.elevation += hill

        # Add secondary smaller hills
        for cx, cy, amp, spread in [(70, 70, 20, 12), (90, 40, 15, 10), (50, 50, 12, 10)]:
            h = amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * spread**2))
            self.elevation += h

        # Create flat valley area in lower-right
        valley_cx, valley_cy = 85, 25
        valley_spread = 20
        valley_depth = 8
        valley = -valley_depth * np.exp(-((X - valley_cx)**2 + (Y - valley_cy)**2) / (2 * valley_spread**2))
        self.elevation += valley

        # Smooth
        self.elevation = gaussian_filter(self.elevation, sigma=2)
        self.elevation = np.maximum(self.elevation, 0)

    def _generate_roads(self):
        """Generate road grid."""
        spacing = 30
        width = 3

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
        """Compute terrain gradients."""
        self.grad_y, self.grad_x = np.gradient(self.elevation)
        self.slope = np.sqrt(self.grad_x**2 + self.grad_y**2)

    def _find_hilltop(self) -> Tuple[int, int]:
        """Find the highest point (hilltop)."""
        idx = np.unravel_index(np.argmax(self.elevation), self.elevation.shape)
        return (idx[1], idx[0])  # (x, y)

    def _find_flat_ground(self) -> Tuple[int, int]:
        """Find a flat, low elevation area."""
        # Look for area with low slope and low elevation
        # Focus on lower-right quadrant
        search_region = self.elevation[10:50, 60:110]
        slope_region = self.slope[10:50, 60:110]

        # Combined score: low elevation + low slope
        score = search_region + slope_region * 10
        idx = np.unravel_index(np.argmin(score), score.shape)

        return (idx[1] + 60, idx[0] + 10)  # (x, y)

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
        manning_values = {0: 0.013, 1: 0.035, 2: 0.025}  # road, soil, channel
        return manning_values.get(surface, 0.035)


# =============================================================================
# WATER DYNAMICS
# =============================================================================

class WaterDynamics(torch.nn.Module):
    """Water particle dynamics with Manning's equation."""

    def __init__(self, terrain: ComparisonTerrain, num_particles: int):
        super().__init__()
        self.terrain = terrain
        self.num_particles = num_particles
        self.eps = 1e-6

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        state = state.view(self.num_particles, 3)  # [x, y, z]

        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dz = torch.zeros_like(z)

        for i in range(self.num_particles):
            xi, yi = x[i].item(), y[i].item()

            # Boundary check
            if xi < 2 or xi > self.terrain.width - 3 or yi < 2 or yi > self.terrain.height - 3:
                continue

            # Terrain properties
            slope = self.terrain.get_slope(xi, yi)
            n = self.terrain.get_manning_n(xi, yi)
            flow_x, flow_y = self.terrain.get_flow_direction(xi, yi)
            surface = self.terrain.get_surface_type(xi, yi)

            # Manning's velocity: V = (1/n) * R^(2/3) * S^(1/2)
            h = 0.1  # Water depth
            if slope > 0.001 and n > 0:
                V = (1.0 / n) * (h ** (2/3)) * np.sqrt(slope)
            else:
                V = 0.1

            # Boost on roads and channels
            if surface == 0:  # Road
                V *= 1.5
            elif surface == 2:  # Channel
                V *= 2.0

            # Flow direction
            flow_mag = np.sqrt(flow_x**2 + flow_y**2) + self.eps
            vx = V * flow_x / flow_mag
            vy = V * flow_y / flow_mag

            # Particle dispersion
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

            # Z follows terrain
            target_z = self.terrain.get_elevation(xi + vx * 0.1, yi + vy * 0.1)
            dz[i] = (target_z - z[i].item()) * 3.0

        return torch.stack([dx, dy, dz], dim=1).view(-1)


# =============================================================================
# COMPARISON SIMULATION
# =============================================================================

class ComparisonSimulation:
    """
    Run and compare water flow from hilltop vs flat ground.
    """

    def __init__(self, num_particles: int = 30, num_waves: int = 3, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.terrain = ComparisonTerrain(width=120, height=120, seed=seed)
        self.num_particles = num_particles
        self.num_waves = num_waves

        # Source points
        self.hilltop_source = self.terrain.hilltop
        self.flat_source = self.terrain.flat_ground

        # Results
        self.hilltop_trajectories = []
        self.flat_trajectories = []
        self.hilltop_times = []
        self.flat_times = []

    def _create_particles(self, source: Tuple[float, float]) -> torch.Tensor:
        """Create particles at source point."""
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
                       t_duration: float = 6.0, num_steps: int = 80) -> List[np.ndarray]:
        """Run multi-wave simulation from a source point."""

        print(f"\n  Running {name} simulation...")
        print(f"    Source: {source}")
        print(f"    Elevation: {self.terrain.get_elevation(*source):.1f}")

        all_trajectories = []
        all_times = []

        dynamics = WaterDynamics(self.terrain, self.num_particles)

        for wave in range(self.num_waves):
            t_start = wave * t_duration
            print(f"    Wave {wave + 1}: t={t_start:.1f} to t={t_start + t_duration:.1f}")

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

        return all_trajectories, all_times

    def run(self):
        """Run both simulations."""
        print("=" * 70)
        print("WATERSHED COMPARISON: Hilltop vs Flat Ground")
        print("=" * 70)

        # Run hilltop simulation
        self.hilltop_trajectories, self.hilltop_times = self.run_simulation(
            self.hilltop_source, "HILLTOP"
        )

        # Run flat ground simulation
        self.flat_trajectories, self.flat_times = self.run_simulation(
            self.flat_source, "FLAT GROUND"
        )

        print("\nSimulations complete!")

    def calculate_statistics(self):
        """Calculate comparison statistics."""
        stats = {'hilltop': {}, 'flat': {}}

        for name, trajectories, source in [
            ('hilltop', self.hilltop_trajectories, self.hilltop_source),
            ('flat', self.flat_trajectories, self.flat_source)
        ]:
            all_velocities = []
            all_distances = []
            total_elevation_drop = 0

            for wave_traj in trajectories:
                # Velocities
                for t in range(1, len(wave_traj)):
                    dx = wave_traj[t, :, 0] - wave_traj[t-1, :, 0]
                    dy = wave_traj[t, :, 1] - wave_traj[t-1, :, 1]
                    v = np.sqrt(dx**2 + dy**2) / 0.075  # dt ≈ 0.075
                    all_velocities.extend(v)

                # Distance traveled
                for p in range(self.num_particles):
                    dist = np.sum(np.sqrt(
                        np.diff(wave_traj[:, p, 0])**2 +
                        np.diff(wave_traj[:, p, 1])**2
                    ))
                    all_distances.append(dist)

                # Elevation drop
                start_z = wave_traj[0, :, 2].mean()
                end_z = wave_traj[-1, :, 2].mean()
                total_elevation_drop += start_z - end_z

            stats[name] = {
                'start_elevation': self.terrain.get_elevation(*source),
                'mean_velocity': np.mean(all_velocities),
                'max_velocity': np.max(all_velocities),
                'mean_distance': np.mean(all_distances),
                'total_elevation_drop': total_elevation_drop / self.num_waves
            }

        return stats

    def visualize(self, save_path: Optional[str] = None):
        """Create comprehensive comparison visualization."""

        fig = plt.figure(figsize=(20, 16))

        # Calculate statistics
        stats = self.calculate_statistics()

        # 1. 3D terrain with both trajectories
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self._plot_3d_comparison(ax1)

        # 2. Top-down view
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_topdown_comparison(ax2)

        # 3. Velocity comparison over time
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_velocity_comparison(ax3)

        # 4. Statistics comparison
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_statistics(ax4, stats)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def _plot_3d_comparison(self, ax):
        """Plot 3D terrain with both trajectory sets."""

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

        ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, linewidth=0)

        # Hilltop trajectories (red/orange)
        colors_hill = plt.cm.Reds(np.linspace(0.4, 0.9, self.num_waves))
        for wave_idx, traj in enumerate(self.hilltop_trajectories):
            for p in range(0, self.num_particles, 3):  # Every 3rd particle
                ax.plot(traj[:, p, 0], traj[:, p, 1], traj[:, p, 2] + 0.5,
                       color=colors_hill[wave_idx], linewidth=1, alpha=0.7)

        # Flat ground trajectories (blue/cyan)
        colors_flat = plt.cm.Blues(np.linspace(0.4, 0.9, self.num_waves))
        for wave_idx, traj in enumerate(self.flat_trajectories):
            for p in range(0, self.num_particles, 3):
                ax.plot(traj[:, p, 0], traj[:, p, 1], traj[:, p, 2] + 0.5,
                       color=colors_flat[wave_idx], linewidth=1, alpha=0.7)

        # Mark sources
        hill_z = self.terrain.get_elevation(*self.hilltop_source)
        flat_z = self.terrain.get_elevation(*self.flat_source)

        ax.scatter(*self.hilltop_source, hill_z + 3, color='red', s=200,
                  marker='^', edgecolor='white', linewidth=2, label='Hilltop', zorder=10)
        ax.scatter(*self.flat_source, flat_z + 3, color='blue', s=200,
                  marker='o', edgecolor='white', linewidth=2, label='Flat Ground', zorder=10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain: Hilltop (Red) vs Flat Ground (Blue)', fontweight='bold')
        ax.legend(loc='upper left')
        ax.view_init(elev=25, azim=45)

    def _plot_topdown_comparison(self, ax):
        """Plot top-down view with both flow paths."""

        # Elevation background
        im = ax.imshow(self.terrain.elevation, origin='lower', cmap='terrain',
                      extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.7)

        # Contour lines
        ax.contour(self.terrain.elevation, levels=15, colors='brown', alpha=0.4,
                  extent=[0, self.terrain.width, 0, self.terrain.height], linewidths=0.5)

        # Hilltop trajectories
        for wave_idx, traj in enumerate(self.hilltop_trajectories):
            alpha = 0.4 + 0.2 * wave_idx / self.num_waves
            for p in range(self.num_particles):
                ax.plot(traj[:, p, 0], traj[:, p, 1], color='red', linewidth=0.8, alpha=alpha)
            # End points
            ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color='darkred', s=10, alpha=0.5)

        # Flat ground trajectories
        for wave_idx, traj in enumerate(self.flat_trajectories):
            alpha = 0.4 + 0.2 * wave_idx / self.num_waves
            for p in range(self.num_particles):
                ax.plot(traj[:, p, 0], traj[:, p, 1], color='blue', linewidth=0.8, alpha=alpha)
            ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color='darkblue', s=10, alpha=0.5)

        # Sources
        ax.scatter(*self.hilltop_source, color='red', s=300, marker='^',
                  edgecolor='white', linewidth=2, zorder=20, label='Hilltop Start')
        ax.scatter(*self.flat_source, color='blue', s=300, marker='o',
                  edgecolor='white', linewidth=2, zorder=20, label='Flat Ground Start')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Top-Down View: Flow Paths Comparison', fontweight='bold')
        ax.legend(loc='upper right')
        plt.colorbar(im, ax=ax, label='Elevation', shrink=0.8)

    def _plot_velocity_comparison(self, ax):
        """Plot velocity over time for both scenarios."""

        # Calculate velocities over time
        def calc_velocities(trajectories, times):
            all_v = []
            all_t = []
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

        # Smoothed trend lines
        if len(hill_v) > 10:
            hill_v_smooth = gaussian_filter(hill_v, sigma=3)
            flat_v_smooth = gaussian_filter(flat_v, sigma=3)
            ax.plot(hill_t, hill_v_smooth, 'r--', linewidth=3, alpha=0.9, label='Hilltop (smoothed)')
            ax.plot(flat_t, flat_v_smooth, 'b--', linewidth=3, alpha=0.9, label='Flat (smoothed)')

        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Velocity')
        ax.set_title('Velocity Comparison Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotation
        ax.text(0.98, 0.98,
               'Hilltop: High initial velocity\n(steep gradient)\n\n'
               'Flat Ground: Lower, steadier\nvelocity',
               transform=ax.transAxes, fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _plot_statistics(self, ax, stats):
        """Plot comparison statistics."""
        ax.axis('off')

        # Create comparison table
        text = "COMPARISON STATISTICS\n"
        text += "=" * 50 + "\n\n"

        text += f"{'Metric':<25} {'Hilltop':>12} {'Flat Ground':>12}\n"
        text += "-" * 50 + "\n"

        text += f"{'Start Elevation':<25} {stats['hilltop']['start_elevation']:>12.1f} {stats['flat']['start_elevation']:>12.1f}\n"
        text += f"{'Mean Velocity':<25} {stats['hilltop']['mean_velocity']:>12.2f} {stats['flat']['mean_velocity']:>12.2f}\n"
        text += f"{'Max Velocity':<25} {stats['hilltop']['max_velocity']:>12.2f} {stats['flat']['max_velocity']:>12.2f}\n"
        text += f"{'Mean Distance Traveled':<25} {stats['hilltop']['mean_distance']:>12.1f} {stats['flat']['mean_distance']:>12.1f}\n"
        text += f"{'Avg Elevation Drop':<25} {stats['hilltop']['total_elevation_drop']:>12.1f} {stats['flat']['total_elevation_drop']:>12.1f}\n"

        text += "\n" + "=" * 50 + "\n\n"

        # Velocity ratio
        v_ratio = stats['hilltop']['mean_velocity'] / max(stats['flat']['mean_velocity'], 0.01)
        text += f"Velocity Ratio (Hill/Flat): {v_ratio:.2f}x\n\n"

        text += "KEY DIFFERENCES:\n"
        text += "-" * 50 + "\n"
        text += "• Hilltop: Gravity-driven acceleration\n"
        text += "  - Higher initial potential energy\n"
        text += "  - Steeper gradients → faster flow\n"
        text += "  - Water spreads radially then channels\n\n"
        text += "• Flat Ground: Diffusion-dominated\n"
        text += "  - Lower potential energy\n"
        text += "  - Gentle gradients → slower flow\n"
        text += "  - Water spreads more uniformly\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=11, family='monospace', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title('Quantitative Comparison', fontweight='bold', pad=20)

    def create_frame_sequence(self, num_frames: int = 12, save_path: Optional[str] = None):
        """Create side-by-side frame sequence."""

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # Combine times
        all_times = []
        for t in self.hilltop_times:
            all_times.extend(t)
        t_min, t_max = min(all_times), max(all_times)
        frame_times = np.linspace(t_min, t_max, num_frames)

        for frame_idx, t_target in enumerate(frame_times):
            ax = axes.flatten()[frame_idx]

            # Background
            ax.imshow(self.terrain.elevation, origin='lower', cmap='terrain',
                     extent=[0, self.terrain.width, 0, self.terrain.height], alpha=0.6)
            ax.contour(self.terrain.elevation, levels=10, colors='brown', alpha=0.3,
                      extent=[0, self.terrain.width, 0, self.terrain.height], linewidths=0.5)

            # Plot active particles for hilltop
            for wave_idx, (traj, times) in enumerate(zip(self.hilltop_trajectories, self.hilltop_times)):
                if times[0] <= t_target <= times[-1]:
                    t_idx = np.argmin(np.abs(times - t_target))
                    # Trails
                    trail_start = max(0, t_idx - 8)
                    for p in range(self.num_particles):
                        ax.plot(traj[trail_start:t_idx+1, p, 0], traj[trail_start:t_idx+1, p, 1],
                               color='red', linewidth=0.6, alpha=0.4)
                    ax.scatter(traj[t_idx, :, 0], traj[t_idx, :, 1], color='red', s=15, alpha=0.8)
                elif t_target > times[-1]:
                    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color='darkred', s=8, alpha=0.4)

            # Plot active particles for flat ground
            for wave_idx, (traj, times) in enumerate(zip(self.flat_trajectories, self.flat_times)):
                if times[0] <= t_target <= times[-1]:
                    t_idx = np.argmin(np.abs(times - t_target))
                    trail_start = max(0, t_idx - 8)
                    for p in range(self.num_particles):
                        ax.plot(traj[trail_start:t_idx+1, p, 0], traj[trail_start:t_idx+1, p, 1],
                               color='blue', linewidth=0.6, alpha=0.4)
                    ax.scatter(traj[t_idx, :, 0], traj[t_idx, :, 1], color='blue', s=15, alpha=0.8)
                elif t_target > times[-1]:
                    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], color='darkblue', s=8, alpha=0.4)

            # Sources
            ax.scatter(*self.hilltop_source, color='red', s=100, marker='^',
                      edgecolor='white', linewidth=1, zorder=20)
            ax.scatter(*self.flat_source, color='blue', s=100, marker='o',
                      edgecolor='white', linewidth=1, zorder=20)

            ax.set_xlim(0, self.terrain.width)
            ax.set_ylim(0, self.terrain.height)
            ax.set_title(f't = {t_target:.1f}', fontsize=11, fontweight='bold')

        plt.suptitle('Hilltop (Red ▲) vs Flat Ground (Blue ●): Time Sequence',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def create_video(self, save_path: str, fps: int = 12):
        """Create comparison animation."""

        print(f"Creating video: {save_path}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Setup both axes
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

        # Particle scatter plots
        hill_scatter = ax1.scatter([], [], color='red', s=20, alpha=0.8)
        flat_scatter = ax2.scatter([], [], color='blue', s=20, alpha=0.8)

        # Trail lines
        hill_trails = [ax1.plot([], [], 'r-', linewidth=0.5, alpha=0.3)[0] for _ in range(self.num_particles)]
        flat_trails = [ax2.plot([], [], 'b-', linewidth=0.5, alpha=0.3)[0] for _ in range(self.num_particles)]

        # Time and velocity text
        time_text = fig.suptitle('', fontsize=14, fontweight='bold')
        hill_vel_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                                 va='top', bbox=dict(facecolor='white', alpha=0.8))
        flat_vel_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=10,
                                 va='top', bbox=dict(facecolor='white', alpha=0.8))

        # Time range
        all_times = []
        for t in self.hilltop_times + self.flat_times:
            all_times.extend(t)
        t_min, t_max = min(all_times), max(all_times)

        num_frames = int((t_max - t_min) * fps)
        frame_times = np.linspace(t_min, t_max, num_frames)

        trail_length = 15

        def animate(frame):
            t_target = frame_times[frame]

            # Update hilltop
            hill_positions = np.empty((0, 2))
            hill_vel = 0
            for wave_idx, (traj, times) in enumerate(zip(self.hilltop_trajectories, self.hilltop_times)):
                if times[0] <= t_target <= times[-1]:
                    t_idx = np.argmin(np.abs(times - t_target))
                    hill_positions = traj[t_idx, :, :2]

                    # Velocity
                    if t_idx > 0:
                        dt = times[t_idx] - times[t_idx - 1]
                        dx = traj[t_idx, :, 0] - traj[t_idx-1, :, 0]
                        dy = traj[t_idx, :, 1] - traj[t_idx-1, :, 1]
                        hill_vel = np.mean(np.sqrt(dx**2 + dy**2)) / dt

                    # Trails
                    trail_start = max(0, t_idx - trail_length)
                    for p in range(min(self.num_particles, len(hill_trails))):
                        hill_trails[p].set_data(traj[trail_start:t_idx+1, p, 0],
                                               traj[trail_start:t_idx+1, p, 1])
                elif t_target > times[-1]:
                    hill_positions = traj[-1, :, :2]

            hill_scatter.set_offsets(hill_positions)
            hill_vel_text.set_text(f'Velocity: {hill_vel:.2f}')

            # Update flat ground
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

            time_text.set_text(f'Time: {t_target:.2f}  |  Hilltop vs Flat Ground Comparison')

            return [hill_scatter, flat_scatter, time_text, hill_vel_text, flat_vel_text] + hill_trails + flat_trails

        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        plt.close(fig)

        print(f"Video saved: {save_path}")
        return save_path


def main():
    """Run the comparison simulation."""

    print("=" * 70)
    print("WATERSHED COMPARISON: Hilltop vs Flat Ground")
    print("=" * 70)
    print()

    # Create simulation
    sim = ComparisonSimulation(
        num_particles=30,
        num_waves=3,
        seed=42
    )

    # Run both simulations
    sim.run()

    # Output directory
    output_dir = os.path.dirname(__file__)

    # Calculate and print statistics
    stats = sim.calculate_statistics()

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Hilltop':>12} {'Flat Ground':>12}")
    print("-" * 50)
    print(f"{'Start Elevation':<25} {stats['hilltop']['start_elevation']:>12.1f} {stats['flat']['start_elevation']:>12.1f}")
    print(f"{'Mean Velocity':<25} {stats['hilltop']['mean_velocity']:>12.2f} {stats['flat']['mean_velocity']:>12.2f}")
    print(f"{'Max Velocity':<25} {stats['hilltop']['max_velocity']:>12.2f} {stats['flat']['max_velocity']:>12.2f}")
    print(f"{'Mean Distance':<25} {stats['hilltop']['mean_distance']:>12.1f} {stats['flat']['mean_distance']:>12.1f}")

    v_ratio = stats['hilltop']['mean_velocity'] / max(stats['flat']['mean_velocity'], 0.01)
    print(f"\nHilltop flows {v_ratio:.1f}x faster than flat ground!")

    # Create visualizations
    print("\nCreating visualizations...")

    sim.visualize(save_path=os.path.join(output_dir, 'comparison_visualization.png'))
    sim.create_frame_sequence(num_frames=12, save_path=os.path.join(output_dir, 'comparison_time_sequence.png'))
    sim.create_video(save_path=os.path.join(output_dir, 'comparison_animation.gif'), fps=10)

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)

    return sim


if __name__ == "__main__":
    sim = main()
