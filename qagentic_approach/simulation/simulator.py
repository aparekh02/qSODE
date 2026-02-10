"""
Wave simulator for qODE simulations.

Main simulation engine that orchestrates:
- Wave initialization
- ODE integration
- Result collection
"""

import numpy as np
import torch
from torchdiffeq import odeint
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from ..core.environment import BaseEnvironment
from ..core.dynamics import qODEDynamics
from ..core.wave import WaveState, WaveFront


# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class SimulationConfig:
    """
    Configuration for qODE simulation.

    Attributes:
        t_start: Simulation start time
        t_end: Simulation end time
        num_steps: Number of time steps to record
        latent_dim: Dimension of latent state vectors
        ode_method: ODE solver method ('euler', 'rk4', 'dopri5')
        ode_step_size: Step size for fixed-step methods
        self_dynamics_scale: Scaling for self-dynamics term
        interaction_scale: Scaling for interaction terms
        refraction_scale: Scaling for refraction effects
        seed: Random seed for reproducibility
    """
    t_start: float = 0.0
    t_end: float = 10.0
    num_steps: int = 100
    latent_dim: int = 32
    ode_method: str = 'euler'
    ode_step_size: float = 0.01
    self_dynamics_scale: float = 0.5
    interaction_scale: float = 0.3
    refraction_scale: float = 0.5
    seed: int = 42


@dataclass
class SimulationResults:
    """
    Container for simulation results.

    Attributes:
        waves: List of WaveFront objects with full histories
        times: Array of time points
        positions: Array of positions [num_steps, num_waves, 2]
        phases: Array of phases [num_steps, num_waves]
        latent_states: Array of latent states [num_steps, num_waves, latent_dim]
        config: Simulation configuration used
        environment: Environment used
    """
    waves: List[WaveFront]
    times: np.ndarray
    positions: np.ndarray
    phases: np.ndarray
    latent_states: np.ndarray
    config: SimulationConfig
    environment: Optional[BaseEnvironment] = None

    def get_wave(self, wave_id: int) -> WaveFront:
        """Get wave by ID."""
        for wave in self.waves:
            if wave.wave_id == wave_id:
                return wave
        raise ValueError(f"Wave {wave_id} not found")

    def get_positions_at_time(self, t: float) -> np.ndarray:
        """Get all wave positions at specific time."""
        idx = np.argmin(np.abs(self.times - t))
        return self.positions[idx]

    def get_inter_wave_distances(self) -> np.ndarray:
        """
        Compute inter-wave distances over time.

        Returns:
            Array of shape [num_steps, num_pairs] where num_pairs = n*(n-1)/2
        """
        num_waves = len(self.waves)
        num_pairs = num_waves * (num_waves - 1) // 2
        distances = np.zeros((len(self.times), num_pairs))

        pair_idx = 0
        for i in range(num_waves):
            for j in range(i + 1, num_waves):
                distances[:, pair_idx] = np.sqrt(
                    (self.positions[:, i, 0] - self.positions[:, j, 0])**2 +
                    (self.positions[:, i, 1] - self.positions[:, j, 1])**2
                )
                pair_idx += 1

        return distances

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        records = []

        for t_idx, t in enumerate(self.times):
            for w_idx, wave in enumerate(self.waves):
                x, y = self.positions[t_idx, w_idx]
                phase = self.phases[t_idx, w_idx]

                # Calculate velocity
                if t_idx > 0:
                    dx = x - self.positions[t_idx-1, w_idx, 0]
                    dy = y - self.positions[t_idx-1, w_idx, 1]
                    dt = self.times[t_idx] - self.times[t_idx-1]
                    velocity = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0.0
                else:
                    velocity = 0.0

                # Get medium
                if self.environment:
                    medium = self.environment.get_medium_name(x, y)
                else:
                    medium = "Unknown"

                records.append({
                    'Time': round(t, 3),
                    'Wave': f'Wave {wave.wave_id + 1}',
                    'X': round(x, 2),
                    'Y': round(y, 2),
                    'Phase': round(phase, 3),
                    'Medium': medium,
                    'Velocity': round(velocity, 3)
                })

        return pd.DataFrame(records)

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        df = self.to_dataframe()
        summary = {
            'num_waves': len(self.waves),
            'duration': self.config.t_end - self.config.t_start,
            'num_steps': len(self.times),
            'waves': {}
        }

        for wave in self.waves:
            wave_data = df[df['Wave'] == f'Wave {wave.wave_id + 1}']
            summary['waves'][f'Wave {wave.wave_id + 1}'] = {
                'x_range': [float(wave_data['X'].min()), float(wave_data['X'].max())],
                'y_range': [float(wave_data['Y'].min()), float(wave_data['Y'].max())],
                'avg_velocity': float(wave_data['Velocity'].mean()),
                'mediums_visited': wave_data['Medium'].unique().tolist(),
            }

        return summary


class WaveSimulator:
    """
    Main simulation engine for quantum ODE wave propagation.

    Orchestrates:
    - Wave initialization
    - ODE integration
    - Result collection

    Parameters:
        environment: Environment for wave propagation
        num_waves: Number of wave fronts to simulate
        config: Simulation configuration (optional)
        initial_positions: Custom initial positions as list of [x, y] (optional)
    """

    def __init__(
        self,
        environment: BaseEnvironment,
        num_waves: int = 5,
        config: Optional[SimulationConfig] = None,
        initial_positions: Optional[List[List[float]]] = None
    ):
        self.environment = environment
        self.num_waves = num_waves
        self.config = config or SimulationConfig()

        # Set seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Store custom initial positions
        self._custom_positions = initial_positions

        # Initialize dynamics model
        self.dynamics = qODEDynamics(
            num_waves=num_waves,
            latent_dim=self.config.latent_dim,
            environment=environment,
            self_dynamics_scale=self.config.self_dynamics_scale,
            interaction_scale=self.config.interaction_scale,
            refraction_scale=self.config.refraction_scale
        ).to(device)

        # Initialize wave fronts and initial state
        self.waves: List[WaveFront] = []
        self.initial_state = self._initialize_waves()

        # Results storage
        self.results: Optional[SimulationResults] = None

    def _get_initial_positions(self) -> List[List[float]]:
        """Get initial positions for waves."""
        if self._custom_positions is not None:
            return self._custom_positions[:self.num_waves]

        # Default diverse positions across the environment
        default_positions = [
            [0.1, 0.1],   # Street intersection area
            [0.9, 0.3],   # Building district
            [0.5, 0.5],   # Center (park area)
            [0.3, 0.75],  # Near river
            [0.8, 0.8],   # Upper area
            [0.2, 0.6],   # Mid-left
            [0.7, 0.4],   # Mid-right
            [0.4, 0.2],   # Lower-mid
        ]
        return default_positions[:self.num_waves]

    def _initialize_waves(self) -> torch.Tensor:
        """Initialize wave fronts."""
        positions = self._get_initial_positions()
        states = []

        for i, (px, py) in enumerate(positions):
            # Create WaveFront object
            initial_pos = np.array([
                px * self.environment.width,
                py * self.environment.height
            ])
            initial_phase = np.random.uniform(0, 2 * np.pi)

            wave = WaveFront(
                wave_id=i,
                initial_position=initial_pos,
                initial_phase=initial_phase,
                aggressiveness=0.5
            )
            self.waves.append(wave)

            # Create latent state tensor
            h = torch.randn(self.config.latent_dim, device=device) * 0.1

            # Encode position in first two dimensions
            h[0] = (px * self.environment.width - self.environment.width / 2) / (self.environment.width / 20)
            h[1] = (py * self.environment.height - self.environment.height / 2) / (self.environment.height / 20)

            # Phase tensor
            phase = torch.tensor([initial_phase], device=device)

            state = torch.cat([h, phase])
            states.append(state)

        return torch.stack(states).view(-1)

    def run(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        num_steps: Optional[int] = None,
        verbose: bool = True
    ) -> SimulationResults:
        """
        Run the qODE simulation.

        Parameters:
            t_start: Override config start time
            t_end: Override config end time
            num_steps: Override config num_steps
            verbose: Print progress messages

        Returns:
            SimulationResults object
        """
        t_start = t_start or self.config.t_start
        t_end = t_end or self.config.t_end
        num_steps = num_steps or self.config.num_steps

        t_span = torch.linspace(t_start, t_end, num_steps, device=device)

        if verbose:
            print(f"Running qODE simulation from t={t_start} to t={t_end}...")
            print(f"Number of waves: {self.num_waves}")
            print(f"Latent dimension: {self.config.latent_dim}")

        # Reset dynamics state
        self.dynamics.reset_state()

        # Solve ODE
        with torch.no_grad():
            if self.config.ode_method == 'euler':
                options = {'step_size': self.config.ode_step_size}
            else:
                options = {}

            trajectories = odeint(
                self.dynamics,
                self.initial_state,
                t_span,
                method=self.config.ode_method,
                options=options
            )

        # Parse results
        trajectories = trajectories.view(num_steps, self.num_waves, self.config.latent_dim + 1)

        # Extract positions and build wave histories
        positions = []
        times = t_span.cpu().numpy()

        for t_idx in range(num_steps):
            t = times[t_idx]
            t_positions = []

            for w_idx in range(self.num_waves):
                h = trajectories[t_idx, w_idx, :self.config.latent_dim]
                phase = trajectories[t_idx, w_idx, -1].item()

                # Decode position
                pos = self.dynamics.position_decoder(h).detach().cpu().numpy()
                x = pos[0] * self.environment.width / 20 + self.environment.width / 2
                y = pos[1] * self.environment.height / 20 + self.environment.height / 2
                x = np.clip(x, 1, self.environment.width - 2)
                y = np.clip(y, 1, self.environment.height - 2)

                t_positions.append([x, y])

                # Calculate velocity
                if t_idx > 0:
                    prev_pos = positions[t_idx - 1][w_idx]
                    dt = times[t_idx] - times[t_idx - 1]
                    velocity = np.array([
                        (x - prev_pos[0]) / dt,
                        (y - prev_pos[1]) / dt
                    ])
                else:
                    velocity = np.zeros(2)

                # Get medium
                medium = self.environment.get_medium_name(x, y)

                # Create wave state
                state = WaveState(
                    time=t,
                    position=np.array([x, y]),
                    velocity=velocity,
                    phase=phase,
                    medium=medium,
                    latent_state=h.detach().cpu().numpy()
                )
                self.waves[w_idx].add_state(state)

            positions.append(t_positions)

        positions = np.array(positions)
        phases = trajectories[:, :, -1].detach().cpu().numpy()
        latent_states = trajectories[:, :, :self.config.latent_dim].detach().cpu().numpy()

        # Create results object
        self.results = SimulationResults(
            waves=self.waves,
            times=times,
            positions=positions,
            phases=phases,
            latent_states=latent_states,
            config=self.config,
            environment=self.environment
        )

        if verbose:
            print("Simulation complete!")

        return self.results

    def reset(self):
        """Reset simulator for a new run."""
        self.waves = []
        self.initial_state = self._initialize_waves()
        self.results = None
