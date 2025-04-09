"""Tests for the solver configuration module."""
import pytest
import jax
import jax.numpy as jnp
import diffrax
from hybrid_models.solver import SolverConfig, solve_for_dataset


def test_solver_config_initialization():
    """Test that the SolverConfig initializes with default values."""
    config = SolverConfig()

    # Check default values
    assert config.solver_type == "tsit5"
    assert config.step_size_controller == "pid"
    assert config.rtol == 1e-3
    assert config.atol == 1e-6
    assert config.max_steps == 100000


def test_solver_config_factory_methods():
    """Test the factory methods for creating specific solver configurations."""
    # Training config
    training_config = SolverConfig.for_training()
    assert training_config.rtol == 1e-2  # Less strict for training
    assert training_config.solver_type == "tsit5"

    # Evaluation config
    eval_config = SolverConfig.for_evaluation()
    assert eval_config.rtol == 1e-4  # More strict for evaluation
    assert eval_config.solver_type == "dopri5"

    # Fixed step config
    fixed_config = SolverConfig.fixed_step(dt=0.05)
    assert fixed_config.step_size_controller == "constant"
    assert fixed_config.dt == 0.05


def test_get_solver():
    """Test that the correct solver is returned based on configuration."""
    # Test tsit5
    config = SolverConfig(solver_type="tsit5")
    solver = config.get_solver()
    assert isinstance(solver, diffrax.Tsit5)

    # Test dopri5
    config = SolverConfig(solver_type="dopri5")
    solver = config.get_solver()
    assert isinstance(solver, diffrax.Dopri5)

    # Test euler
    config = SolverConfig(solver_type="euler")
    solver = config.get_solver()
    assert isinstance(solver, diffrax.Euler)

    # Test unknown solver type (should default to Tsit5)
    config = SolverConfig(solver_type="unknown_solver")
    solver = config.get_solver()
    assert isinstance(solver, diffrax.Tsit5)


def test_get_step_size_controller():
    """Test that the correct step size controller is returned."""
    # Test PID controller
    config = SolverConfig(step_size_controller="pid", rtol=1e-4, atol=1e-6)
    controller = config.get_step_size_controller()
    assert isinstance(controller, diffrax.PIDController)
    assert controller.rtol == 1e-4
    assert controller.atol == 1e-6

    # Test adaptive controller (which now uses PIDController under the hood)
    config = SolverConfig(step_size_controller="adaptive")
    controller = config.get_step_size_controller()
    assert isinstance(controller, diffrax.PIDController)  # Now we expect a PIDController for 'adaptive' too

    # Test constant controller
    config = SolverConfig(step_size_controller="constant")
    controller = config.get_step_size_controller()
    assert isinstance(controller, diffrax.ConstantStepSize)

    # Test unknown controller (should default to PID)
    config = SolverConfig(step_size_controller="unknown")
    controller = config.get_step_size_controller()
    assert isinstance(controller, diffrax.PIDController)

def test_to_dict():
    """Test conversion of SolverConfig to a dictionary for solve function."""
    config = SolverConfig(
        solver_type="dopri5",
        step_size_controller="pid",
        rtol=1e-4,
        atol=1e-8,
        dt0=0.05,
        max_steps=50000
    )

    config_dict = config.to_dict()

    # Check that all necessary keys are present
    assert "solver" in config_dict
    assert "rtol" in config_dict
    assert "atol" in config_dict
    assert "dt0" in config_dict
    assert "max_steps" in config_dict
    assert "stepsize_controller" in config_dict

    # Check values
    assert isinstance(config_dict["solver"], diffrax.Dopri5)
    assert config_dict["rtol"] == 1e-4
    assert config_dict["atol"] == 1e-8
    assert config_dict["dt0"] == 0.05
    assert config_dict["max_steps"] == 50000
    assert isinstance(config_dict["stepsize_controller"], diffrax.PIDController)


def test_solver_config_string_representation():
    """Test the string representation of SolverConfig."""
    config = SolverConfig(
        solver_type="dopri5",
        step_size_controller="pid",
        rtol=1e-4,
        atol=1e-8
    )

    config_str = str(config)

    # Check that the string contains important info
    assert "DOPRI5" in config_str
    assert "pid" in config_str
    # Check for 0.0001 instead of 1e-4 in the string
    assert "0.0001" in config_str
    assert "1e-08" in config_str  # or "1e-8"


class MockModel:
    """A minimal mock model for testing solve_for_dataset."""

    def solve(self, initial_state, t_span, evaluation_times, args, **kwargs):
        """Mock solve method."""
        # Return a simple dictionary with expected keys
        return {
            'times': evaluation_times,
            'X': jnp.ones_like(evaluation_times),
            'P': jnp.ones_like(evaluation_times) * 2.0
        }


def test_solve_for_dataset():
    """Test the solve_for_dataset function with a mock model."""
    # Create a mock dataset
    dataset = {
        'initial_state': {'X': 1.0, 'P': 0.0},
        'times': jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    }

    # Create a mock model
    model = MockModel()

    # Create solver config
    solver_config = SolverConfig(
        solver_type="tsit5",
        rtol=1e-3,
        atol=1e-6
    )

    # Call solve_for_dataset
    solution = solve_for_dataset(model, dataset, solver_config)

    # Check solution
    assert 'times' in solution
    assert 'X' in solution
    assert 'P' in solution
    assert jnp.array_equal(solution['times'], dataset['times'])
    assert jnp.all(solution['X'] == 1.0)
    assert jnp.all(solution['P'] == 2.0)

    # Test with default solver config
    solution_default = solve_for_dataset(model, dataset)
    assert 'times' in solution_default
    assert 'X' in solution_default
    assert 'P' in solution_default