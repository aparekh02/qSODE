"""
Visualization and video generation for qODE simulations.

Provides:
- Static plots (trajectories, phases, distances)
- Animated visualizations
- MP4 video generation
- Wave field visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Optional, Tuple, List
import warnings

from ..simulation.simulator import SimulationResults, WaveSimulator


@dataclass
class VideoConfig:
    """
    Configuration for video generation.

    Attributes:
        fps: Frames per second
        dpi: Resolution (dots per inch)
        figsize: Figure size (width, height)
        trail_length: Number of frames to show in trail
        wave_radius: Base radius for wave markers
        pulse_amplitude: Amplitude of wave pulsing effect
        show_wave_field: Whether to show interference pattern
        codec: Video codec ('mp4' or 'gif')
        bitrate: Video bitrate for mp4
    """
    fps: int = 15
    dpi: int = 100
    figsize: Tuple[int, int] = (12, 10)
    trail_length: int = 30
    wave_radius: float = 2.0
    pulse_amplitude: float = 0.5
    show_wave_field: bool = False
    codec: str = 'mp4'
    bitrate: int = 2000


class Visualizer:
    """
    Visualization engine for qODE simulations.

    Parameters:
        simulator: WaveSimulator instance (must have results)
        or
        results: SimulationResults object directly
    """

    def __init__(
        self,
        simulator: Optional[WaveSimulator] = None,
        results: Optional[SimulationResults] = None
    ):
        if simulator is not None and simulator.results is not None:
            self.results = simulator.results
            self.environment = simulator.environment
        elif results is not None:
            self.results = results
            self.environment = results.environment
        else:
            raise ValueError("Must provide either simulator with results or results directly")

        # Wave colors
        self.wave_colors = plt.cm.tab10(np.linspace(0, 1, len(self.results.waves)))

    def plot_trajectories(
        self,
        ax: Optional[plt.Axes] = None,
        show_environment: bool = True,
        show_legend: bool = True,
        title: str = "Wave Trajectories in Urban Environment"
    ) -> plt.Axes:
        """Plot wave trajectories on the environment."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Plot environment
        if show_environment and self.environment:
            self.environment.visualize(ax, show_legend=False)

        # Plot trajectories
        for w_idx, wave in enumerate(self.results.waves):
            traj = self.results.positions[:, w_idx, :]
            ax.plot(traj[:, 0], traj[:, 1], '-', color=self.wave_colors[w_idx],
                    linewidth=2, label=f'Wave {w_idx + 1}')
            # Start point
            ax.scatter(traj[0, 0], traj[0, 1], color=self.wave_colors[w_idx], s=100,
                       marker='o', edgecolor='white', zorder=5)
            # End point
            ax.scatter(traj[-1, 0], traj[-1, 1], color=self.wave_colors[w_idx], s=150,
                       marker='*', edgecolor='white', zorder=5)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)

        if show_legend:
            ax.legend(loc='upper left')

        return ax

    def plot_positions_over_time(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "Wave Positions Over Time"
    ) -> plt.Axes:
        """Plot X and Y positions over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for w_idx, wave in enumerate(self.results.waves):
            ax.plot(self.results.times, self.results.positions[:, w_idx, 0], '-',
                    color=self.wave_colors[w_idx], label=f'Wave {w_idx + 1} X')
            ax.plot(self.results.times, self.results.positions[:, w_idx, 1], '--',
                    color=self.wave_colors[w_idx], alpha=0.7, label=f'Wave {w_idx + 1} Y')

        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_phases(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "Quantum Phase Evolution"
    ) -> plt.Axes:
        """Plot phase evolution over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for w_idx, wave in enumerate(self.results.waves):
            ax.plot(self.results.times, self.results.phases[:, w_idx], '-',
                    color=self.wave_colors[w_idx], linewidth=2, label=f'Wave {w_idx + 1}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Phase (radians)')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_inter_wave_distances(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "Inter-Wave Distances (Interaction Strength)"
    ) -> plt.Axes:
        """Plot distances between wave pairs over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        num_waves = len(self.results.waves)
        distances = self.results.get_inter_wave_distances()

        pair_idx = 0
        for i in range(num_waves):
            for j in range(i + 1, num_waves):
                ax.plot(self.results.times, distances[:, pair_idx], '-',
                        linewidth=1.5, label=f'Wave {i+1} - Wave {j+1}')
                pair_idx += 1

        ax.set_xlabel('Time')
        ax.set_ylabel('Distance')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_summary(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create a comprehensive 4-panel summary plot."""
        fig = plt.figure(figsize=(16, 12))

        # 1. Trajectories
        ax1 = fig.add_subplot(2, 2, 1)
        self.plot_trajectories(ax1)

        # 2. Positions over time
        ax2 = fig.add_subplot(2, 2, 2)
        self.plot_positions_over_time(ax2)

        # 3. Phases
        ax3 = fig.add_subplot(2, 2, 3)
        self.plot_phases(ax3)

        # 4. Inter-wave distances
        ax4 = fig.add_subplot(2, 2, 4)
        self.plot_inter_wave_distances(ax4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Summary plot saved to {save_path}")

        return fig

    def plot_wave_field(
        self,
        time_idx: int = -1,
        resolution: int = 50,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the combined wave field (interference pattern) at a specific time.

        Parameters:
            time_idx: Time index (-1 for final)
            resolution: Grid resolution for field calculation
            save_path: Path to save the figure
        """
        if time_idx < 0:
            time_idx = len(self.results.times) + time_idx

        # Create grid
        x = np.linspace(0, self.environment.width, resolution)
        y = np.linspace(0, self.environment.height, resolution)
        X, Y = np.meshgrid(x, y)

        # Calculate wave field (superposition)
        wave_field = np.zeros_like(X)
        num_waves = len(self.results.waves)

        for w_idx in range(num_waves):
            wx, wy = self.results.positions[time_idx, w_idx]
            phase = self.results.phases[time_idx, w_idx]

            # Distance from wave center
            dist = np.sqrt((X - wx)**2 + (Y - wy)**2)

            # Wave amplitude with decay and phase
            amplitude = np.exp(-dist / 20) * np.cos(dist / 5 - phase)
            wave_field += amplitude

        # Smooth the field
        wave_field = gaussian_filter(wave_field, sigma=1)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Environment with wave positions
        ax1 = axes[0]
        if self.environment:
            self.environment.visualize(ax1, show_legend=True)

        for w_idx in range(num_waves):
            wx, wy = self.results.positions[time_idx, w_idx]
            ax1.scatter(wx, wy, color=self.wave_colors[w_idx], s=200,
                       marker='o', edgecolor='white', linewidth=2,
                       label=f'Wave {w_idx + 1}', zorder=10)
            # Draw wave circles
            for r in [5, 10, 15]:
                circle = plt.Circle((wx, wy), r, fill=False,
                                   color=self.wave_colors[w_idx], alpha=0.3)
                ax1.add_patch(circle)

        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'Wave Positions at t={self.results.times[time_idx]:.2f}')
        ax1.legend(loc='upper right')

        # Right: Wave field intensity
        ax2 = axes[1]
        im = ax2.imshow(wave_field, origin='lower', cmap='RdBu_r',
                       extent=[0, self.environment.width, 0, self.environment.height],
                       vmin=-2, vmax=2)
        plt.colorbar(im, ax=ax2, label='Wave Amplitude')

        for w_idx in range(num_waves):
            wx, wy = self.results.positions[time_idx, w_idx]
            ax2.scatter(wx, wy, color='black', s=100, marker='x', linewidth=2)

        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title(f'Wave Field (Interference Pattern) at t={self.results.times[time_idx]:.2f}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Wave field plot saved to {save_path}")

        return fig

    def create_video(
        self,
        save_path: str,
        config: Optional[VideoConfig] = None,
        show_progress: bool = True
    ) -> str:
        """
        Create an MP4 video of the simulation.

        Parameters:
            save_path: Output path (e.g., 'simulation.mp4' or 'simulation.gif')
            config: Video configuration
            show_progress: Print progress messages

        Returns:
            Path to saved video
        """
        config = config or VideoConfig()

        if show_progress:
            print(f"Creating video: {save_path}")
            print(f"  FPS: {config.fps}, Frames: {len(self.results.times)}")

        fig, ax = plt.subplots(figsize=config.figsize)

        # Setup environment background
        if self.environment:
            cmap = self.environment.get_colormap()
            ax.imshow(self.environment.medium_map, cmap=cmap, origin='lower',
                     extent=[0, self.environment.width, 0, self.environment.height],
                     alpha=0.7)

        # Set axis limits
        ax.set_xlim(0, self.environment.width if self.environment else 100)
        ax.set_ylim(0, self.environment.height if self.environment else 100)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        num_waves = len(self.results.waves)

        # Initialize wave markers and trails
        circles = []
        trails = []
        phase_indicators = []

        for w_idx in range(num_waves):
            # Wave marker (circle)
            circle = plt.Circle((0, 0), config.wave_radius,
                               color=self.wave_colors[w_idx],
                               alpha=0.9, zorder=10)
            ax.add_patch(circle)
            circles.append(circle)

            # Trail line
            trail, = ax.plot([], [], '-', color=self.wave_colors[w_idx],
                            alpha=0.6, linewidth=2, zorder=5)
            trails.append(trail)

            # Phase indicator (outer ring)
            phase_ring = plt.Circle((0, 0), config.wave_radius * 1.5,
                                   fill=False, color=self.wave_colors[w_idx],
                                   linewidth=2, alpha=0.5, zorder=9)
            ax.add_patch(phase_ring)
            phase_indicators.append(phase_ring)

        # Time display
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=14, verticalalignment='top',
                           fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Info panel
        info_text = ax.text(0.02, 0.85, '', transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           family='monospace',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Legend
        legend_elements = [patches.Patch(facecolor=self.wave_colors[i],
                          label=f'Wave {i+1}') for i in range(num_waves)]
        ax.legend(handles=legend_elements, loc='upper right')

        title = ax.set_title('qODE Urban Wave Simulation', fontsize=14, fontweight='bold')

        def init():
            """Initialize animation."""
            for circle in circles:
                circle.center = (0, 0)
            for trail in trails:
                trail.set_data([], [])
            for ring in phase_indicators:
                ring.center = (0, 0)
            time_text.set_text('')
            info_text.set_text('')
            return circles + trails + phase_indicators + [time_text, info_text]

        def animate(frame):
            """Update animation for each frame."""
            t = self.results.times[frame]

            # Build info string
            info_lines = []

            for w_idx in range(num_waves):
                x, y = self.results.positions[frame, w_idx]
                phase = self.results.phases[frame, w_idx]

                # Update circle position
                circles[w_idx].center = (x, y)

                # Pulsing effect based on phase
                pulse = config.wave_radius + config.pulse_amplitude * np.sin(phase)
                circles[w_idx].set_radius(pulse)

                # Update phase indicator ring
                phase_indicators[w_idx].center = (x, y)
                ring_radius = config.wave_radius * 1.5 + np.abs(np.sin(phase)) * 2
                phase_indicators[w_idx].set_radius(ring_radius)

                # Update trail
                trail_start = max(0, frame - config.trail_length)
                trails[w_idx].set_data(
                    self.results.positions[trail_start:frame+1, w_idx, 0],
                    self.results.positions[trail_start:frame+1, w_idx, 1]
                )

                # Get medium and velocity
                if self.environment:
                    medium = self.environment.get_medium_name(x, y)
                else:
                    medium = "N/A"

                if frame > 0:
                    dx = x - self.results.positions[frame-1, w_idx, 0]
                    dy = y - self.results.positions[frame-1, w_idx, 1]
                    dt = self.results.times[frame] - self.results.times[frame-1]
                    vel = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
                else:
                    vel = 0

                info_lines.append(f"W{w_idx+1}: ({x:5.1f},{y:5.1f}) v={vel:4.1f} [{medium[:6]}]")

            time_text.set_text(f'Time: {t:.2f}')
            info_text.set_text('\n'.join(info_lines))

            return circles + trails + phase_indicators + [time_text, info_text]

        # Create animation
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(self.results.times),
            interval=1000/config.fps,
            blit=True
        )

        # Save video
        if save_path.endswith('.gif') or config.codec == 'gif':
            writer = PillowWriter(fps=config.fps)
        else:
            try:
                writer = FFMpegWriter(fps=config.fps, bitrate=config.bitrate)
            except Exception:
                warnings.warn("FFmpeg not available, falling back to GIF")
                save_path = save_path.replace('.mp4', '.gif')
                writer = PillowWriter(fps=config.fps)

        if show_progress:
            print(f"  Rendering frames...")

        anim.save(save_path, writer=writer, dpi=config.dpi)
        plt.close(fig)

        if show_progress:
            print(f"  Video saved to: {save_path}")

        return save_path

    def create_detailed_video(
        self,
        save_path: str,
        config: Optional[VideoConfig] = None,
        show_progress: bool = True
    ) -> str:
        """
        Create a detailed multi-panel video showing all aspects of the simulation.

        Panels:
        - Main: Wave trajectories on environment
        - Right-top: Positions over time
        - Right-bottom: Inter-wave distances

        Parameters:
            save_path: Output path
            config: Video configuration
            show_progress: Print progress

        Returns:
            Path to saved video
        """
        config = config or VideoConfig()
        config.figsize = (18, 10)

        if show_progress:
            print(f"Creating detailed video: {save_path}")

        fig = plt.figure(figsize=config.figsize)

        # Layout: main panel left, two panels right
        ax_main = fig.add_axes([0.05, 0.1, 0.55, 0.85])
        ax_pos = fig.add_axes([0.65, 0.55, 0.32, 0.4])
        ax_dist = fig.add_axes([0.65, 0.08, 0.32, 0.4])

        # Setup main panel (environment)
        if self.environment:
            cmap = self.environment.get_colormap()
            ax_main.imshow(self.environment.medium_map, cmap=cmap, origin='lower',
                          extent=[0, self.environment.width, 0, self.environment.height],
                          alpha=0.7)

        ax_main.set_xlim(0, self.environment.width if self.environment else 100)
        ax_main.set_ylim(0, self.environment.height if self.environment else 100)
        ax_main.set_xlabel('X Position')
        ax_main.set_ylabel('Y Position')
        ax_main.set_title('qODE Wave Propagation', fontsize=14, fontweight='bold')

        # Setup position panel
        ax_pos.set_xlim(self.results.times[0], self.results.times[-1])
        pos_min = self.results.positions.min() - 5
        pos_max = self.results.positions.max() + 5
        ax_pos.set_ylim(pos_min, pos_max)
        ax_pos.set_xlabel('Time')
        ax_pos.set_ylabel('Position')
        ax_pos.set_title('Positions Over Time')
        ax_pos.grid(True, alpha=0.3)

        # Setup distance panel
        distances = self.results.get_inter_wave_distances()
        ax_dist.set_xlim(self.results.times[0], self.results.times[-1])
        ax_dist.set_ylim(0, distances.max() * 1.1)
        ax_dist.set_xlabel('Time')
        ax_dist.set_ylabel('Distance')
        ax_dist.set_title('Inter-Wave Distances')
        ax_dist.grid(True, alpha=0.3)

        num_waves = len(self.results.waves)

        # Main panel elements
        circles = []
        trails = []
        for w_idx in range(num_waves):
            circle = plt.Circle((0, 0), config.wave_radius,
                               color=self.wave_colors[w_idx], alpha=0.9, zorder=10)
            ax_main.add_patch(circle)
            circles.append(circle)

            trail, = ax_main.plot([], [], '-', color=self.wave_colors[w_idx],
                                 alpha=0.6, linewidth=2, zorder=5)
            trails.append(trail)

        # Position panel lines
        pos_lines_x = []
        pos_lines_y = []
        for w_idx in range(num_waves):
            line_x, = ax_pos.plot([], [], '-', color=self.wave_colors[w_idx],
                                 linewidth=1.5, label=f'W{w_idx+1} X')
            line_y, = ax_pos.plot([], [], '--', color=self.wave_colors[w_idx],
                                 linewidth=1.5, alpha=0.7)
            pos_lines_x.append(line_x)
            pos_lines_y.append(line_y)
        ax_pos.legend(loc='upper right', fontsize=8)

        # Distance panel lines
        dist_lines = []
        pair_idx = 0
        for i in range(num_waves):
            for j in range(i + 1, num_waves):
                line, = ax_dist.plot([], [], '-', linewidth=1.5,
                                    label=f'W{i+1}-W{j+1}')
                dist_lines.append(line)
                pair_idx += 1
        ax_dist.legend(loc='upper right', fontsize=8)

        # Time marker lines
        time_line_pos, = ax_pos.plot([], [], 'k--', alpha=0.5, linewidth=1)
        time_line_dist, = ax_dist.plot([], [], 'k--', alpha=0.5, linewidth=1)

        # Time text
        time_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                                fontsize=14, verticalalignment='top', fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        def init():
            for circle in circles:
                circle.center = (0, 0)
            for trail in trails:
                trail.set_data([], [])
            for line in pos_lines_x + pos_lines_y + dist_lines:
                line.set_data([], [])
            time_line_pos.set_data([], [])
            time_line_dist.set_data([], [])
            time_text.set_text('')
            return circles + trails + pos_lines_x + pos_lines_y + dist_lines + [time_line_pos, time_line_dist, time_text]

        def animate(frame):
            t = self.results.times[frame]
            times_so_far = self.results.times[:frame+1]

            # Update main panel
            for w_idx in range(num_waves):
                x, y = self.results.positions[frame, w_idx]
                phase = self.results.phases[frame, w_idx]

                circles[w_idx].center = (x, y)
                pulse = config.wave_radius + config.pulse_amplitude * np.sin(phase)
                circles[w_idx].set_radius(pulse)

                trail_start = max(0, frame - config.trail_length)
                trails[w_idx].set_data(
                    self.results.positions[trail_start:frame+1, w_idx, 0],
                    self.results.positions[trail_start:frame+1, w_idx, 1]
                )

                # Update position lines
                pos_lines_x[w_idx].set_data(times_so_far, self.results.positions[:frame+1, w_idx, 0])
                pos_lines_y[w_idx].set_data(times_so_far, self.results.positions[:frame+1, w_idx, 1])

            # Update distance lines
            for idx, line in enumerate(dist_lines):
                line.set_data(times_so_far, distances[:frame+1, idx])

            # Update time markers
            time_line_pos.set_data([t, t], [pos_min, pos_max])
            time_line_dist.set_data([t, t], [0, distances.max() * 1.1])

            time_text.set_text(f'Time: {t:.2f}')

            return circles + trails + pos_lines_x + pos_lines_y + dist_lines + [time_line_pos, time_line_dist, time_text]

        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(self.results.times),
            interval=1000/config.fps,
            blit=True
        )

        if save_path.endswith('.gif') or config.codec == 'gif':
            writer = PillowWriter(fps=config.fps)
        else:
            try:
                writer = FFMpegWriter(fps=config.fps, bitrate=config.bitrate)
            except Exception:
                warnings.warn("FFmpeg not available, falling back to GIF")
                save_path = save_path.replace('.mp4', '.gif')
                writer = PillowWriter(fps=config.fps)

        if show_progress:
            print(f"  Rendering frames...")

        anim.save(save_path, writer=writer, dpi=config.dpi)
        plt.close(fig)

        if show_progress:
            print(f"  Video saved to: {save_path}")

        return save_path
