"""
Main self-evolving simulation loop.

Orchestrates: run simulation -> extract metrics -> agent analysis ->
orchestrator synthesis -> parameter update -> repeat until convergence.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .parameter_space import ParameterSet
from .metrics import SimulationMetrics, MetricsExtractor
from .history import EvolutionHistory, IterationRecord
from ..agents.hydrology_agent import HydrologyAgent
from ..agents.surface_agent import SurfaceAgent
from ..agents.quantum_soil_agent import QuantumSoilAgent
from ..agents.orchestrator import Orchestrator


@dataclass
class EvolutionConfig:
    """Configuration for the evolution process."""
    max_iterations: int = 10
    convergence_threshold: float = 0.01
    convergence_window: int = 3
    learning_rate: float = 0.5
    params_to_evolve: Optional[List[str]] = None  # None = all
    use_claude_arbitration: bool = True
    claude_model: str = "claude-sonnet-4-20250514"
    sim_particles: int = 40
    sim_duration: float = 8.0
    sim_steps: int = 120
    seed: int = 42
    verbose: bool = True
    save_history: bool = True
    history_path: str = "evolution_history.json"


class EvolutionLoop:
    """
    Main self-evolving simulation loop.

    Each iteration:
    1. Run simulation with current ParameterSet
    2. Extract metrics
    3. Each agent analyzes and suggests changes
    4. Orchestrator synthesizes into new ParameterSet
    5. Check convergence
    """

    def __init__(self, config: EvolutionConfig, api_key: str,
                 initial_params: Optional[ParameterSet] = None,
                 sim_factory=None):
        """
        Args:
            config: Evolution configuration
            api_key: Anthropic API key
            initial_params: Starting parameters (defaults if None)
            sim_factory: Callable(params, config) -> simulation object with .run() and attributes
                         needed by MetricsExtractor. If None, must be set before evolve().
        """
        self.config = config
        self.api_key = api_key
        self.params = initial_params or ParameterSet()
        self.history = EvolutionHistory()
        self.sim_factory = sim_factory

        # Initialize agents
        self.agents = [
            HydrologyAgent(api_key=api_key, model=config.claude_model),
            SurfaceAgent(api_key=api_key, model=config.claude_model),
            QuantumSoilAgent(api_key=api_key, model=config.claude_model),
        ]

        self.orchestrator = Orchestrator(
            api_key=api_key,
            model=config.claude_model,
            learning_rate=config.learning_rate,
            use_claude_arbitration=config.use_claude_arbitration,
        )

    def evolve(self) -> Tuple[ParameterSet, EvolutionHistory]:
        """
        Run the full evolution loop.

        Returns:
            (final_params, history)
        """
        if self.sim_factory is None:
            raise RuntimeError("sim_factory must be set before calling evolve()")

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"\n{'=' * 60}")
                print(f"EVOLUTION ITERATION {iteration + 1}/{self.config.max_iterations}")
                print(f"{'=' * 60}")

            # Step 1: Run simulation
            if self.config.verbose:
                print("  Running simulation...")
            sim = self.sim_factory(self.params, self.config)
            sim.run(t_duration=self.config.sim_duration,
                    num_steps=self.config.sim_steps)

            # Step 2: Extract metrics
            metrics = MetricsExtractor.extract(sim)
            metrics_dict = metrics.to_dict()

            if self.config.verbose:
                print(f"  Velocity ratio (road/soil): {metrics.velocity_ratio_road_soil:.2f}")
                print(f"  Mass loss: {metrics.mass_loss_fraction * 100:.1f}%")
                print(f"  Total infiltration: {metrics.total_infiltration:.2f}")

            # Step 3: Agent analysis
            suggestions = []
            for agent in self.agents:
                if self.config.verbose:
                    print(f"  Consulting {agent.role} agent...")
                try:
                    suggestion = agent.analyze(
                        sim_metrics=metrics_dict,
                        current_params=self.params,
                        iteration=iteration,
                    )
                    suggestions.append(suggestion)
                    if self.config.verbose:
                        n_changes = len(suggestion.parameter_changes)
                        print(f"    -> {n_changes} parameter suggestion(s), "
                              f"priority={suggestion.priority:.2f}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"    -> Error: {e}")

            # Step 4: Orchestrator synthesis
            if self.config.verbose:
                print("  Orchestrator synthesizing...")
            new_params = self.orchestrator.synthesize(suggestions, self.params)

            # Filter to only evolve specified parameters
            if self.config.params_to_evolve:
                final_dict = self.params.to_dict()
                new_dict = new_params.to_dict()
                for k in self.config.params_to_evolve:
                    if k in new_dict:
                        final_dict[k] = new_dict[k]
                new_params = ParameterSet.from_dict(final_dict)

            # Step 5: Compute improvement
            improvement = self._compute_improvement(metrics)

            # Step 6: Record
            suggestion_dicts = [s.to_dict() for s in suggestions]
            record = IterationRecord(
                iteration=iteration,
                parameters=self.params.to_dict(),
                metrics=metrics_dict,
                agent_suggestions=suggestion_dicts,
                orchestrator_decision=new_params.to_dict(),
                improvement_score=improvement,
            )
            self.history.add(record)

            # Print diff
            if self.config.verbose:
                diff = self.params.diff(new_params)
                if diff:
                    print("  Parameter changes:")
                    for name, (old, new) in diff.items():
                        print(f"    {name}: {old:.4f} -> {new:.4f}")
                else:
                    print("  No parameter changes.")

            # Step 7: Update
            self.params = new_params

            # Step 8: Convergence check
            if self.history.has_converged(
                window=self.config.convergence_window,
                threshold=self.config.convergence_threshold,
            ):
                if self.config.verbose:
                    print(f"\nConverged after {iteration + 1} iterations.")
                break

        # Save history
        if self.config.save_history:
            self.history.save(self.config.history_path)
            if self.config.verbose:
                print(f"\nHistory saved to {self.config.history_path}")

        return self.params, self.history

    def _compute_improvement(self, current_metrics: SimulationMetrics) -> float:
        """Compute improvement score vs previous iteration."""
        if not self.history.records:
            return 0.0

        prev_metrics = self.history.records[-1].metrics

        # Multi-objective targets
        targets = {
            "velocity_ratio_road_soil": 2.5,
            "mass_loss_fraction": 0.30,
        }

        current_error = sum(
            abs(current_metrics.to_dict().get(k, 0) - v) / max(v, 1e-6)
            for k, v in targets.items()
        )
        prev_error = sum(
            abs(prev_metrics.get(k, 0) - v) / max(v, 1e-6)
            for k, v in targets.items()
        )

        return prev_error - current_error  # Positive = improvement
