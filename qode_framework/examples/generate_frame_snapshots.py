#!/usr/bin/env python3
"""
Generate individual frame snapshots at different time points
to visualize the time-by-time wave movement during inferencing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from qode_framework import UrbanEnvironment, WaveSimulator, SimulationConfig

def generate_snapshots():
    """Generate frame snapshots at multiple time points."""

    print("Generating time-by-time frame snapshots...")

    # Create environment and simulator
    env = UrbanEnvironment(width=100, height=100, seed=42)

    config = SimulationConfig(
        t_start=0.0,
        t_end=10.0,
        num_steps=100,
        latent_dim=32,
        seed=42
    )

    initial_positions = [
        [0.1, 0.1],
        [0.9, 0.3],
        [0.5, 0.5],
        [0.3, 0.75],
        [0.8, 0.8],
    ]

    simulator = WaveSimulator(
        environment=env,
        num_waves=5,
        config=config,
        initial_positions=initial_positions
    )

    results = simulator.run(verbose=False)

    # Time points to capture
    time_points = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Wave colors
    wave_colors = plt.cm.tab10(np.linspace(0, 1, 5))

    # Create figure with subplots for all time points
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    # Environment colormap
    colors_map = ['#90EE90', '#808080', '#8B4513', '#4169E1', '#228B22']
    cmap = LinearSegmentedColormap.from_list('urban', colors_map, N=5)

    for idx, t_target in enumerate(time_points):
        ax = axes[idx]

        # Find closest time index
        t_idx = np.argmin(np.abs(results.times - t_target))
        actual_t = results.times[t_idx]

        # Draw environment
        ax.imshow(env.medium_map, cmap=cmap, origin='lower',
                 extent=[0, env.width, 0, env.height], alpha=0.7)

        # Draw wave trails (previous positions)
        trail_length = min(20, t_idx)
        for w_idx in range(5):
            if trail_length > 0:
                trail_start = max(0, t_idx - trail_length)
                ax.plot(
                    results.positions[trail_start:t_idx+1, w_idx, 0],
                    results.positions[trail_start:t_idx+1, w_idx, 1],
                    '-', color=wave_colors[w_idx], alpha=0.5, linewidth=2
                )

        # Draw current wave positions
        for w_idx in range(5):
            x, y = results.positions[t_idx, w_idx]
            phase = results.phases[t_idx, w_idx]

            # Pulsing circle based on phase
            radius = 2 + 0.5 * np.sin(phase)
            circle = plt.Circle((x, y), radius, color=wave_colors[w_idx],
                               alpha=0.9, zorder=10)
            ax.add_patch(circle)

            # Phase indicator ring
            ring = plt.Circle((x, y), radius * 1.5, fill=False,
                            color=wave_colors[w_idx], linewidth=2, alpha=0.5)
            ax.add_patch(ring)

            # Calculate velocity
            if t_idx > 0:
                dx = x - results.positions[t_idx-1, w_idx, 0]
                dy = y - results.positions[t_idx-1, w_idx, 1]
                dt = results.times[t_idx] - results.times[t_idx-1]
                vel = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
            else:
                vel = 0

            # Get medium
            medium = env.get_medium_name(x, y)

        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_title(f'Time = {actual_t:.1f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # Use last subplot for legend
    ax_legend = axes[11]
    ax_legend.axis('off')

    # Create legend
    from matplotlib.patches import Patch, Circle
    legend_elements = [
        Patch(facecolor=wave_colors[i], label=f'Wave {i+1}')
        for i in range(5)
    ]
    legend_elements.extend([
        Patch(facecolor='#90EE90', label='Open Space'),
        Patch(facecolor='#808080', label='Street'),
        Patch(facecolor='#8B4513', label='Building'),
        Patch(facecolor='#4169E1', label='Water'),
    ])
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=12)
    ax_legend.set_title('Legend', fontsize=14, fontweight='bold')

    plt.suptitle('qODE Wave Propagation: Time-by-Time Inferencing',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'time_snapshots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also create a detailed info panel
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.axis('off')

    info_text = "qODE INFERENCING: TIME-BY-TIME WAVE MOVEMENT\n"
    info_text += "=" * 60 + "\n\n"

    for t_target in [0.0, 2.5, 5.0, 7.5, 10.0]:
        t_idx = np.argmin(np.abs(results.times - t_target))
        actual_t = results.times[t_idx]

        info_text += f"TIME = {actual_t:.2f}\n"
        info_text += "-" * 40 + "\n"

        for w_idx in range(5):
            x, y = results.positions[t_idx, w_idx]
            phase = results.phases[t_idx, w_idx]
            medium = env.get_medium_name(x, y)

            if t_idx > 0:
                dx = x - results.positions[t_idx-1, w_idx, 0]
                dy = y - results.positions[t_idx-1, w_idx, 1]
                dt = results.times[t_idx] - results.times[t_idx-1]
                vel = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
            else:
                vel = 0

            info_text += f"  Wave {w_idx+1}: pos=({x:5.1f}, {y:5.1f})  "
            info_text += f"vel={vel:5.2f}  phase={phase:5.2f}  [{medium}]\n"

        info_text += "\n"

    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
            fontsize=11, family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    info_path = os.path.join(os.path.dirname(__file__), 'time_info.png')
    plt.savefig(info_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {info_path}")

    plt.show()

    return output_path, info_path

if __name__ == "__main__":
    generate_snapshots()
