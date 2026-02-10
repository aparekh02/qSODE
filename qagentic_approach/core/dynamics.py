"""
Neural ODE dynamics for qODE simulations.

Contains:
- InteractionNetwork: Learns interaction intensity k(i,j)
- SelfDynamicsNetwork: Models intrinsic wave evolution
- QuantumPhaseMixer: Handles quantum interference effects
- qODEDynamics: Main ODE function combining all components
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .environment import BaseEnvironment


class InteractionNetwork(nn.Module):
    """
    Neural network to compute interaction intensity k(i,j) between wave fronts.

    Based on Social ODE: learns how much wave j affects wave i beyond distance.

    Parameters:
        latent_dim: Dimension of latent state vectors
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        # Input: concatenation of states [h_i, h_j, dh_i/dt, dh_j/dt]
        input_dim = latent_dim * 4

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor,
                dh_i: torch.Tensor, dh_j: torch.Tensor) -> torch.Tensor:
        """
        Compute interaction intensity.

        Args:
            h_i: Latent state of wave i
            h_j: Latent state of wave j
            dh_i: Velocity of wave i
            dh_j: Velocity of wave j

        Returns:
            Interaction vector k(i,j)
        """
        combined = torch.cat([h_i, h_j, dh_i, dh_j], dim=-1)
        return self.network(combined)


class SelfDynamicsNetwork(nn.Module):
    """
    Neural network for self-dynamics f_θ(h_i(t)).

    Captures intrinsic wave evolution independent of other waves.

    Parameters:
        latent_dim: Dimension of latent state vectors
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute self-dynamics term.

        Args:
            h: Latent state vector

        Returns:
            Self-dynamics velocity contribution
        """
        return self.network(h)


class QuantumPhaseMixer(nn.Module):
    """
    Quantum-inspired phase mixing for wave interference.

    Introduces quantum coherence and superposition-like effects.

    Parameters:
        latent_dim: Dimension of latent state vectors
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()

        # Phase rotation parameters (learnable)
        self.phase_weights = nn.Parameter(torch.randn(latent_dim) * 0.1)

        # Coherence mixing network
        self.coherence_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor,
                phase_diff: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum-inspired interference between waves.

        Args:
            h_i: Latent state of wave i
            h_j: Latent state of wave j
            phase_diff: Phase difference between waves (scalar or tensor)

        Returns:
            Interference contribution to dynamics
        """
        # Phase-modulated interaction (constructive/destructive interference)
        phase_factor = torch.cos(phase_diff * self.phase_weights)

        # Coherence factor (how well waves can interfere)
        coherence = self.coherence_net(torch.cat([h_i, h_j], dim=-1))

        # Quantum interference term
        interference = phase_factor * coherence * (h_j - h_i)

        return interference


class qODEDynamics(nn.Module):
    """
    Main quantum ODE dynamics module.

    Combines:
    - Self-dynamics (intrinsic evolution)
    - Distance-based coupling (1/||h_i - h_j||)
    - Learned interaction intensity k(i,j)
    - Quantum phase mixing
    - Environment-based speed modulation

    Equation:
    dh_i/dt = f_θ(h_i) + Σ_j [1/||h_i - h_j|| * k(i,j) * a_i * Q(phase)]

    Parameters:
        num_waves: Number of wave fronts
        latent_dim: Dimension of latent state vectors
        environment: Environment for speed modulation (optional)
        self_dynamics_scale: Scaling factor for self-dynamics term
        interaction_scale: Scaling factor for interaction terms
        refraction_scale: Scaling factor for refraction effects
    """

    def __init__(
        self,
        num_waves: int,
        latent_dim: int = 32,
        environment: Optional[BaseEnvironment] = None,
        self_dynamics_scale: float = 0.5,
        interaction_scale: float = 0.3,
        refraction_scale: float = 0.5
    ):
        super().__init__()

        self.num_waves = num_waves
        self.latent_dim = latent_dim
        self.environment = environment
        self.self_dynamics_scale = self_dynamics_scale
        self.interaction_scale = interaction_scale
        self.refraction_scale = refraction_scale

        # Neural network components
        self.self_dynamics = SelfDynamicsNetwork(latent_dim)
        self.interaction_net = InteractionNetwork(latent_dim)
        self.quantum_mixer = QuantumPhaseMixer(latent_dim)

        # Learnable aggressiveness parameters per wave
        self.aggressiveness = nn.Parameter(torch.ones(num_waves) * 0.5)

        # Small epsilon for numerical stability
        self.eps = 1e-6

        # Position decoder (latent -> physical coordinates)
        self.position_decoder = nn.Linear(latent_dim, 2)
        self._init_position_decoder()

        # State tracking for velocity estimation
        self.prev_state = None
        self.prev_t = None

    def _init_position_decoder(self):
        """Initialize position decoder with identity-like mapping."""
        with torch.no_grad():
            self.position_decoder.weight.zero_()
            self.position_decoder.weight[0, 0] = 1.0  # x = h[0]
            self.position_decoder.weight[1, 1] = 1.0  # y = h[1]
            self.position_decoder.bias.zero_()

    def reset_state(self):
        """Reset internal state (call before new simulation)."""
        self.prev_state = None
        self.prev_t = None

    def _estimate_velocity(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """Estimate dh/dt from finite differences."""
        if self.prev_state is None or self.prev_t is None:
            # Initial estimate: use self-dynamics
            vel = torch.zeros_like(h)
            for i in range(self.num_waves):
                vel[i] = self.self_dynamics(h[i])
            return vel

        dt = t - self.prev_t
        if dt < self.eps:
            dt = 0.01

        return (h - self.prev_state) / dt

    def _get_environment_effects(self, positions: torch.Tensor, h: torch.Tensor,
                                  device: torch.device, dtype: torch.dtype):
        """Get environment speed multipliers and refraction gradients."""
        num_waves = positions.shape[0]
        speed_mults = torch.ones(num_waves, device=device, dtype=dtype)
        refractions = torch.zeros(num_waves, 2, device=device, dtype=dtype)

        if self.environment is None:
            return speed_mults, refractions

        for i in range(num_waves):
            pos = positions[i].detach().cpu().numpy()

            # Scale positions to environment grid
            x = pos[0] * self.environment.width / 20 + self.environment.width / 2
            y = pos[1] * self.environment.height / 20 + self.environment.height / 2
            x = np.clip(x, 0, self.environment.width - 1)
            y = np.clip(y, 0, self.environment.height - 1)

            speed_mults[i] = self.environment.get_speed_multiplier(x, y)

            # Refraction gradient (waves bend toward faster mediums)
            grad_x, grad_y = self.environment.get_speed_gradient(x, y)
            refractions[i, 0] = grad_x * 5
            refractions[i, 1] = grad_y * 5

        return speed_mults, refractions

    def forward(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative dh/dt for all waves.

        Args:
            t: Current time
            state: Tensor of shape [num_waves * (latent_dim + 1)]
                   Contains latent state + phase for each wave

        Returns:
            Derivative tensor of same shape
        """
        device = state.device
        dtype = state.dtype

        # Reshape state: [num_waves, latent_dim + 1]
        state_reshaped = state.view(self.num_waves, self.latent_dim + 1)
        h = state_reshaped[:, :self.latent_dim]  # Latent states
        phases = state_reshaped[:, -1]  # Phases

        # Estimate velocities
        dh_estimated = self._estimate_velocity(h, float(t))

        # Initialize derivatives
        dh_dt = torch.zeros_like(h)
        dphase_dt = torch.ones(self.num_waves, device=device, dtype=dtype) * 0.1

        # Get physical positions for environment interaction
        positions = self.position_decoder(h)

        # Get environment effects
        speed_mults, refractions = self._get_environment_effects(
            positions, h, device, dtype
        )

        for i in range(self.num_waves):
            # Self-dynamics term
            self_term = self.self_dynamics(h[i])

            # Interaction terms
            interaction_sum = torch.zeros_like(h[i])

            for j in range(self.num_waves):
                if i == j:
                    continue

                # Distance-based coupling (clamped for stability)
                dist = torch.norm(h[i] - h[j]) + self.eps
                distance_factor = torch.clamp(1.0 / dist, max=2.0)

                # Interaction intensity k(i,j)
                k_ij = self.interaction_net(h[i], h[j], dh_estimated[i], dh_estimated[j])

                # Phase difference for quantum effects
                phase_diff = phases[i] - phases[j]

                # Quantum phase mixing
                quantum_term = self.quantum_mixer(h[i], h[j], phase_diff)

                # Combined interaction with aggressiveness
                aggressiveness_i = torch.sigmoid(self.aggressiveness[i])
                interaction = distance_factor * k_ij * aggressiveness_i + quantum_term

                interaction_sum = interaction_sum + interaction

            # Combine all terms with environment modulation
            dh_dt[i] = speed_mults[i] * (
                self_term * self.self_dynamics_scale +
                interaction_sum * self.interaction_scale
            )

            # Add refraction effect to position dimensions
            if self.latent_dim >= 2:
                dh_dt[i, :2] = dh_dt[i, :2] + self.refraction_scale * refractions[i]

            # Phase evolution based on interactions
            dphase_dt[i] = 0.1 + 0.05 * torch.sum(torch.abs(interaction_sum))

        # Update stored state for velocity estimation
        self.prev_state = h.clone().detach()
        self.prev_t = float(t)

        # Combine state derivative
        dstate_dt = torch.cat([dh_dt, dphase_dt.unsqueeze(1)], dim=1)

        return dstate_dt.view(-1)

    def decode_positions(self, latent_states: torch.Tensor) -> torch.Tensor:
        """
        Decode latent states to physical positions.

        Args:
            latent_states: Tensor of shape [..., latent_dim]

        Returns:
            Positions of shape [..., 2]
        """
        return self.position_decoder(latent_states)

    def get_aggressiveness_values(self) -> np.ndarray:
        """Get current aggressiveness values as numpy array."""
        with torch.no_grad():
            return torch.sigmoid(self.aggressiveness).cpu().numpy()
