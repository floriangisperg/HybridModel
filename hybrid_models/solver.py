"""
Solver configuration and utilities for hybrid models.

This module provides classes and functions to configure and manage
ODE solvers for hybrid modeling.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import diffrax


@dataclass
class SolverConfig:
    """
    Configuration for ODE solvers with sensible defaults for hybrid modeling.
    """
    # Solver instance
    solver: Any = field(default_factory=lambda: diffrax.Tsit5())

    # Step size control
    rtol: float = 1e-3
    atol: float = 1e-6
    max_steps: int = 100000
    dt0: float = 0.1

    # Additional solver options
    extra_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_training(cls) -> 'SolverConfig':
        """
        Create a solver configuration optimized for training.

        During training, we can use more relaxed tolerances for speed.
        """
        return cls(
            rtol=1e-2,
            atol=1e-4,
            max_steps=500000
        )

    @classmethod
    def for_evaluation(cls) -> 'SolverConfig':
        """
        Create a solver configuration optimized for final evaluation.

        During evaluation, we use tighter tolerances for accuracy.
        """
        return cls(
            rtol=1e-4,
            atol=1e-8,
            max_steps=1000000
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary for solve function.
        """
        result = {
            'solver': self.solver,
            'rtol': self.rtol,
            'atol': self.atol,
            'max_steps': self.max_steps,
            'dt0': self.dt0
        }
        # Add any extra options
        result.update(self.extra_options)
        return result


def solve_for_dataset(model, dataset, solver_config: Optional[SolverConfig] = None):
    """
    Solve the model for a given dataset using specified solver configuration.

    Args:
        model: The hybrid model to solve
        dataset: Dataset containing initial conditions, times, and inputs
        solver_config: Optional solver configuration (uses default if None)

    Returns:
        Solution dictionary containing time points and state variables
    """
    # Use default training config if none provided
    if solver_config is None:
        solver_config = SolverConfig.for_training()

    # Extract solver parameters as dictionary
    solver_params = solver_config.to_dict()

    # Solve the ODE system
    solution = model.solve(
        initial_state=dataset['initial_state'],
        t_span=(dataset['times'][0], dataset['times'][-1]),
        evaluation_times=dataset['times'],
        args={
            'time_dependent_inputs': dataset.get('time_dependent_inputs', {}),
            'static_inputs': dataset.get('static_inputs', {})
        },
        **solver_params
    )

    return solution