"""Tests for edge cases in the ODE solver component."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from hybrid_models import (
    HybridODESystem,
    ConfigurableNN,
    create_initial_random_key
)


@pytest.fixture
def simple_model():
    """Create a simple hybrid model for testing edge cases."""

    # Define simple mechanistic components
    def x_growth(inputs):
        return 0.1 * inputs['X']  # Simple exponential growth

    def p_formation(inputs):
        return 0.05 * inputs['X']  # Linear product formation from X

    mechanistic_components = {
        'X': x_growth,
        'P': p_formation
    }

    # No neural networks for this simple model
    nn_replacements = {}

    # State variables
    state_names = ['X', 'P']

    # Return model
    return HybridODESystem(
        mechanistic_components=mechanistic_components,
        nn_replacements=nn_replacements,
        state_names=state_names
    )


@pytest.fixture
def stiff_model():
    """Create a stiff ODE system to test solver robustness."""

    # Define a stiff system with fast and slow dynamics
    def fast_component(inputs):
        return -1000.0 * inputs['Y']  # Fast decay

    def slow_component(inputs):
        return 0.1 * inputs['X']  # Slow growth

    mechanistic_components = {
        'X': slow_component,
        'Y': fast_component
    }

    nn_replacements = {}
    state_names = ['X', 'Y']

    return HybridODESystem(
        mechanistic_components=mechanistic_components,
        nn_replacements=nn_replacements,
        state_names=state_names
    )


@pytest.fixture
def model_with_nn():
    """Create a model with neural network components."""

    # Define mechanistic components
    def x_growth(inputs):
        return inputs['growth_rate'] * inputs['X']

    mechanistic_components = {
        'X': x_growth
    }

    # Create a simple neural network
    key = create_initial_random_key(42)
    nn = ConfigurableNN(
        norm_params={},
        input_features=['X'],
        hidden_dims=[4],
        key=key
    )

    nn_replacements = {
        'growth_rate': nn
    }

    state_names = ['X']

    return HybridODESystem(
        mechanistic_components=mechanistic_components,
        nn_replacements=nn_replacements,
        state_names=state_names
    )


def test_zero_initial_condition(simple_model):
    """Test solving with zero initial conditions."""
    initial_state = {'X': 0.0, 'P': 0.0}
    times = jnp.linspace(0, 10, 11)

    solution = simple_model.solve(
        initial_state=initial_state,
        t_span=(times[0], times[-1]),
        evaluation_times=times,
        args={}
    )

    # With zero initial conditions and this model, everything should stay at zero
    assert jnp.allclose(solution['X'], jnp.zeros_like(times))
    assert jnp.allclose(solution['P'], jnp.zeros_like(times))


def test_missing_initial_condition(simple_model):
    """Test handling of missing initial condition."""
    # Missing P in initial state
    initial_state = {'X': 1.0}
    times = jnp.linspace(0, 10, 11)

    # This should raise a ValueError
    with pytest.raises(ValueError):
        simple_model.solve(
            initial_state=initial_state,
            t_span=(times[0], times[-1]),
            evaluation_times=times,
            args={}
        )


def test_stiff_system(stiff_model):
    """Test solving a stiff ODE system."""
    initial_state = {'X': 1.0, 'Y': 1.0}
    times = jnp.linspace(0, 10, 11)

    solution = stiff_model.solve(
        initial_state=initial_state,
        t_span=(times[0], times[-1]),
        evaluation_times=times,
        args={},
        rtol=1e-6,
        atol=1e-8,
        max_steps=10000
    )

    # Y should decay very rapidly to near zero
    assert solution['Y'][1] < 0.1  # Should be small after first step

    # X should grow slowly
    assert solution['X'][-1] > initial_state['X']

    # Check Y remains stable at near zero
    assert jnp.all(solution['Y'][1:] < 0.01)


def test_evaluation_times_outside_span(simple_model):
    """Test with evaluation times outside the span."""
    initial_state = {'X': 1.0, 'P': 0.0}
    t_span = (0.0, 10.0)
    evaluation_times = jnp.array([-1.0, 0.0, 5.0, 10.0, 11.0])

    # This should adjust the t_span to include all evaluation times
    # or raise an error - check the intended behavior
    with pytest.raises(Exception):
        solution = simple_model.solve(
            initial_state=initial_state,
            t_span=t_span,
            evaluation_times=evaluation_times,
            args={}
        )


def test_time_dependent_inputs(simple_model):
    """Test with time-dependent inputs that vary significantly."""
    initial_state = {'X': 1.0, 'P': 0.0}
    times = jnp.linspace(0, 10, 21)  # 0 to 10 in 0.5 steps

    # Create oscillating temperature
    temp = 37.0 + 5.0 * jnp.sin(times)  # Temperature oscillating ±5°C

    # Create step function for feed
    feed = jnp.zeros_like(times)
    feed = feed.at[10:].set(1.0)  # Feed starts at t=5

    args = {
        'time_dependent_inputs': {
            'temp': (times, temp),
            'feed': (times, feed)
        }
    }

    solution = simple_model.solve(
        initial_state=initial_state,
        t_span=(times[0], times[-1]),
        evaluation_times=times,
        args=args
    )

    # Basic checks
    assert solution['X'].shape == times.shape
    assert solution['P'].shape == times.shape

    # X and P should be monotonically increasing
    assert jnp.all(jnp.diff(solution['X']) >= 0)
    assert jnp.all(jnp.diff(solution['P']) >= 0)


def test_discontinuous_inputs(simple_model):
    """Test with discontinuous inputs that might cause solver issues."""
    initial_state = {'X': 1.0, 'P': 0.0}
    times = jnp.linspace(0, 10, 101)  # 0 to 10 in 0.1 steps

    # Create square wave for temp (alternating each time step)
    temp = jnp.ones_like(times) * 30.0
    temp = temp.at[::2].set(45.0)  # Alternating 30°C and 45°C

    args = {
        'time_dependent_inputs': {
            'temp': (times, temp)
        }
    }

    solution = simple_model.solve(
        initial_state=initial_state,
        t_span=(times[0], times[-1]),
        evaluation_times=times,
        args=args,
        rtol=1e-4,
        atol=1e-6,
        max_steps=50000
    )

    # Solution should still be computed successfully
    assert solution['X'].shape == times.shape
    assert solution['P'].shape == times.shape


def test_long_integration(simple_model):
    """Test solving over a long time period."""
    initial_state = {'X': 1.0, 'P': 0.0}
    times = jnp.linspace(0, 1000, 101)  # 0 to 1000 with 100 steps

    solution = simple_model.solve(
        initial_state=initial_state,
        t_span=(times[0], times[-1]),
        evaluation_times=times,
        args={},
        max_steps=100000  # Increase max_steps for long integration
    )

    # For exponential growth with rate 0.1, X should be approximately e^(0.1*t)
    expected_final_x = initial_state['X'] * jnp.exp(0.1 * times[-1])

    # Use a very loose tolerance due to approximation
    assert jnp.abs(solution['X'][-1] / expected_final_x - 1.0) < 0.1


def test_coupled_neural_network(model_with_nn):
    """Test a model with neural network component that affects ODE behavior."""
    initial_state = {'X': 1.0}
    times = jnp.linspace(0, 10, 21)

    solution = model_with_nn.solve(
        initial_state=initial_state,
        t_span=(times[0], times[-1]),
        evaluation_times=times,
        args={}
    )

    # Check that solution exists
    assert 'X' in solution
    assert solution['X'].shape == times.shape

    # Growth should happen (but we can't predict exactly how much with a random NN)
    assert solution['X'][-1] != initial_state['X']


def test_very_small_timesteps(simple_model):
    """Test solving with very small timesteps."""
    initial_state = {'X': 1.0, 'P': 0.0}
    times = jnp.linspace(0, 0.001, 11)  # Very small time range

    solution = simple_model.solve(
        initial_state=initial_state,
        t_span=(times[0], times[-1]),
        evaluation_times=times,
        args={}
    )

    # Solution should still work with small timesteps
    assert solution['X'].shape == times.shape
    assert solution['P'].shape == times.shape


def test_widely_differing_scales(simple_model):
    """Test with very different scales for state variables."""
    initial_state = {'X': 1e-6, 'P': 1e6}  # Very different scales
    times = jnp.linspace(0, 10, 11)

    solution = simple_model.solve(
        initial_state=initial_state,
        t_span=(times[0], times[-1]),
        evaluation_times=times,
        args={},
        rtol=1e-8,  # Tighter tolerance for different scales
        atol=1e-10
    )

    # Check that both variables are tracked correctly
    assert jnp.isclose(solution['X'][0], initial_state['X'])
    assert jnp.isclose(solution['P'][0], initial_state['P'])
    assert solution['X'][-1] > initial_state['X']  # X should grow
    assert solution['P'][-1] > initial_state['P']  # P should grow