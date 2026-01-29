"""
Wave state and wave front representations for qODE simulations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class WaveState:
    """
    Captures the complete state of a wave front at a specific time.

    Attributes:
        time: Current simulation time
        position: 2D position [x, y] in environment coordinates
        velocity: 2D velocity vector [vx, vy]
        amplitude: Wave amplitude (intensity)
        phase: Quantum phase (radians)
        coherence: Measure of wave coherence (0-1)
        latent_state: Full latent state vector (optional)
        medium: Current medium type name
        metadata: Additional custom data
    """
    time: float
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    amplitude: float = 1.0
    phase: float = 0.0
    coherence: float = 1.0
    latent_state: Optional[np.ndarray] = None
    medium: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.asarray(self.position)
        self.velocity = np.asarray(self.velocity)
        if self.latent_state is not None:
            self.latent_state = np.asarray(self.latent_state)

    def distance_to(self, other: 'WaveState') -> float:
        """Calculate Euclidean distance to another wave state."""
        return np.linalg.norm(self.position - other.position)

    def phase_difference(self, other: 'WaveState') -> float:
        """Calculate phase difference with another wave (wrapped to [-pi, pi])."""
        diff = self.phase - other.phase
        return np.arctan2(np.sin(diff), np.cos(diff))

    def speed(self) -> float:
        """Calculate current speed."""
        return np.linalg.norm(self.velocity)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'time': self.time,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'amplitude': self.amplitude,
            'phase': self.phase,
            'coherence': self.coherence,
            'medium': self.medium,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WaveState':
        """Create from dictionary."""
        return cls(
            time=data['time'],
            position=np.array(data['position']),
            velocity=np.array(data['velocity']),
            amplitude=data['amplitude'],
            phase=data['phase'],
            coherence=data.get('coherence', 1.0),
            medium=data.get('medium', 'Unknown'),
            metadata=data.get('metadata', {}),
        )


@dataclass
class WaveFront:
    """
    Represents a single wave front with its complete trajectory history.

    Attributes:
        wave_id: Unique identifier for this wave
        initial_position: Starting position [x, y]
        initial_phase: Starting quantum phase
        aggressiveness: Interaction aggressiveness parameter (0-1)
        history: List of WaveState snapshots over time
        color: Visualization color (RGB tuple or name)
    """
    wave_id: int
    initial_position: np.ndarray
    initial_phase: float = 0.0
    aggressiveness: float = 0.5
    history: List[WaveState] = field(default_factory=list)
    color: Optional[str] = None

    def __post_init__(self):
        """Initialize and set default color if not provided."""
        self.initial_position = np.asarray(self.initial_position)
        if self.color is None:
            # Default colors for first 10 waves
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            self.color = colors[self.wave_id % len(colors)]

    def add_state(self, state: WaveState):
        """Add a state snapshot to history."""
        self.history.append(state)

    @property
    def current_state(self) -> Optional[WaveState]:
        """Get the most recent state."""
        return self.history[-1] if self.history else None

    @property
    def trajectory(self) -> np.ndarray:
        """Get position trajectory as [N, 2] array."""
        if not self.history:
            return np.array([]).reshape(0, 2)
        return np.array([s.position for s in self.history])

    @property
    def times(self) -> np.ndarray:
        """Get time array."""
        if not self.history:
            return np.array([])
        return np.array([s.time for s in self.history])

    @property
    def phases(self) -> np.ndarray:
        """Get phase trajectory."""
        if not self.history:
            return np.array([])
        return np.array([s.phase for s in self.history])

    @property
    def velocities(self) -> np.ndarray:
        """Get velocity trajectory as [N, 2] array."""
        if not self.history:
            return np.array([]).reshape(0, 2)
        return np.array([s.velocity for s in self.history])

    def get_state_at_time(self, t: float) -> Optional[WaveState]:
        """Get state at specific time (interpolated if between snapshots)."""
        if not self.history:
            return None

        times = self.times
        idx = np.searchsorted(times, t)

        if idx == 0:
            return self.history[0]
        if idx >= len(times):
            return self.history[-1]

        # Linear interpolation
        t0, t1 = times[idx-1], times[idx]
        alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0.0

        s0, s1 = self.history[idx-1], self.history[idx]

        return WaveState(
            time=t,
            position=s0.position * (1 - alpha) + s1.position * alpha,
            velocity=s0.velocity * (1 - alpha) + s1.velocity * alpha,
            amplitude=s0.amplitude * (1 - alpha) + s1.amplitude * alpha,
            phase=s0.phase + alpha * (s1.phase - s0.phase),
            coherence=s0.coherence * (1 - alpha) + s1.coherence * alpha,
            medium=s1.medium,
        )

    def total_distance_traveled(self) -> float:
        """Calculate total distance traveled."""
        if len(self.history) < 2:
            return 0.0
        traj = self.trajectory
        return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

    def average_velocity(self) -> float:
        """Calculate average velocity magnitude."""
        if len(self.history) < 2:
            return 0.0
        total_dist = self.total_distance_traveled()
        total_time = self.times[-1] - self.times[0]
        return total_dist / total_time if total_time > 0 else 0.0

    def get_mediums_visited(self) -> List[str]:
        """Get list of unique mediums visited."""
        return list(set(s.medium for s in self.history))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'wave_id': self.wave_id,
            'initial_position': self.initial_position.tolist(),
            'initial_phase': self.initial_phase,
            'aggressiveness': self.aggressiveness,
            'color': self.color,
            'history': [s.to_dict() for s in self.history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WaveFront':
        """Create from dictionary."""
        wave = cls(
            wave_id=data['wave_id'],
            initial_position=np.array(data['initial_position']),
            initial_phase=data['initial_phase'],
            aggressiveness=data['aggressiveness'],
            color=data.get('color'),
        )
        wave.history = [WaveState.from_dict(s) for s in data['history']]
        return wave
