"""
Solver configuration and utilities for hybrid models.

This module provides classes and functions to configure and manage
ODE solvers for hybrid modeling.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union, Callable
import diffrax
import jax
import jax.numpy as jnp


@dataclass
class SolverConfig:
    """
    Configuration for ODE solvers with extensive customization options.
    """
    # Solver selection
    solver_type: str = "tsit5"  # Options: "tsit5", "dopri5", "euler", "heun", etc.

    # Step size control
    step_size_controller: str = "pid"  # Options: "pid", "adaptive", "constant"
    rtol: float = 1e-3
    atol: float = 1e-6
    dt0: float = 0.1
    max_steps: int = 100000

    # Fixed step size (for constant controller)
    dt: float = 0.1

    # Additional solver options
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def get_solver(self) -> diffrax.AbstractSolver:
        """
        Get the appropriate solver instance based on configuration.
        """
        solvers = {
            "tsit5": diffrax.Tsit5(),
            "dopri5": diffrax.Dopri5(),
            "dopri8": diffrax.Dopri8(),
            "euler": diffrax.Euler(),
            "heun": diffrax.Heun(),
            "midpoint": diffrax.Midpoint(),
        }

        # Get solver or default to Tsit5
        return solvers.get(self.solver_type.lower(), diffrax.Tsit5())

    def get_step_size_controller(self) -> diffrax.AbstractStepSizeController:
        """
        Get the appropriate step size controller based on configuration.
        """
        if self.step_size_controller.lower() == "pid":
            return diffrax.PIDController(rtol=self.rtol, atol=self.atol)
        elif self.step_size_controller.lower() == "adaptive":
            return diffrax.AdaptiveStepSizeController(rtol=self.rtol, atol=self.atol)
        elif self.step_size_controller.lower() == "constant":
            return diffrax.ConstantStepSize()
        else:
            # Default to PID controller
            return diffrax.PIDController(rtol=self.rtol, atol=self.atol)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary for solve function.

        Returns parameters that match what HybridODESystem.solve() expects.
        """
        # Create dictionary with basic parameters
        result = {
            'solver': self.get_solver(),
            'rtol': self.rtol,
            'atol': self.atol,
            'dt0': self.dt0,
            'max_steps': self.max_steps,
            'stepsize_controller': self.get_step_size_controller()
        }

        # If using constant step size, add dt
        if self.step_size_controller.lower() == "constant":
            result['dt'] = self.dt

        # Add any extra options
        result.update(self.extra_options)
        return result

    @classmethod
    def for_training(cls, solver_type: str = "tsit5") -> 'SolverConfig':
        """
        Create a solver configuration optimized for training.

        During training, we can use more relaxed tolerances for speed.
        """
        return cls(
            solver_type=solver_type,
            step_size_controller="pid",
            rtol=1e-2,
            atol=1e-4,
            max_steps=500000
        )

    @classmethod
    def for_evaluation(cls, solver_type: str = "dopri5") -> 'SolverConfig':
        """
        Create a solver configuration optimized for high-accuracy evaluation.

        This creates a higher-accuracy solver configuration with tighter tolerances,
        which can be used when you specifically want more accurate results than
        the default training solver. Note that by default, the training solver
        is used for evaluation to maintain consistency.

        Args:
            solver_type: Type of solver to use (default: "dopri5" for higher accuracy)

        Returns:
            SolverConfig with high-accuracy settings
        """
        return cls(
            solver_type=solver_type,
            step_size_controller="pid",
            rtol=1e-4,
            atol=1e-8,
            max_steps=1000000
        )

    @classmethod
    def fixed_step(cls, dt: float = 0.1, solver_type: str = "tsit5") -> 'SolverConfig':
        """
        Create a configuration with fixed step size.

        This is useful for ensuring deterministic results or when the ODE
        is stiff and adaptive step sizes are causing issues.
        """
        return cls(
            solver_type=solver_type,
            step_size_controller="constant",
            dt=dt,
            max_steps=1000000
        )

    def __str__(self) -> str:
        """
        Return a human-readable description of the solver configuration.
        """
        return (
            f"Solver: {self.solver_type.upper()}, "
            f"Controller: {self.step_size_controller}, "
            f"rtol: {self.rtol}, atol: {self.atol}, "
            f"max_steps: {self.max_steps}"
        )

def solve_for_dataset(model, dataset, solver_config: Optional[SolverConfig] = None):
    """
    Solve the model for a given dataset using the specified solver configuration.

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