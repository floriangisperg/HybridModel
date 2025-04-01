"""Common test fixtures for the hybrid models tests."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from hybrid_models import HybridModelBuilder, create_initial_random_key


@pytest.fixture
def random_key():
    """Provide a consistent random key for tests."""
    return create_initial_random_key(42)


@pytest.fixture
def norm_params():
    """Sample normalization parameters for testing."""
    return {
        'X_mean': 5.0,
        'X_std': 2.0,
        'P_mean': 1.0,
        'P_std': 0.5,
        'temp_mean': 37.0,
        'temp_std': 5.0,
    }


@pytest.fixture
def simple_dataset():
    """Simple dataset for testing models."""
    # Create a simple time series
    times = jnp.linspace(0, 10, 21)  # 0 to 10 in steps of 0.5

    # Simple growth curve for X (biomass)
    X = 1.0 + 0.5 * times

    # Simple product formation for P
    P = 0.1 * times ** 2

    # Control inputs
    temp = 37.0 * jnp.ones_like(times)
    feed = 0.1 * times

    return {
        'X_true': X,
        'P_true': P,
        'times': times,
        'initial_state': {
            'X': float(X[0]),
            'P': float(P[0])
        },
        'time_dependent_inputs': {
            'temp': (times, temp),
            'feed': (times, feed),
            'inductor_switch': (times, jnp.ones_like(times)),  # Always on
        }
    }


@pytest.fixture
def simple_hybrid_model(norm_params, random_key):
    """Create a simple hybrid model for testing."""
    builder = HybridModelBuilder()

    # Set normalization parameters
    builder.set_normalization_params(norm_params)

    # Add state variables
    builder.add_state('X')  # Biomass
    builder.add_state('P')  # Product

    # Simple mechanistic growth model
    def biomass_ode(inputs):
        X = inputs['X']
        mu = inputs['growth_rate']  # Will be replaced by neural network
        return mu * X

    # Simple mechanistic product formation
    def product_ode(inputs):
        X = inputs['X']
        qp = inputs['product_rate']  # Will be replaced by neural network
        inductor_switch = inputs.get('inductor_switch', 0.0)
        return qp * X * inductor_switch

    # Add mechanistic components
    builder.add_mechanistic_component('X', biomass_ode)
    builder.add_mechanistic_component('P', product_ode)

    # Split key for two neural networks
    key1, key2 = jax.random.split(random_key)

    # Replace growth rate with neural network
    builder.replace_with_nn(
        name='growth_rate',
        input_features=['X', 'temp'],
        hidden_dims=[4, 4],  # Small network for testing
        key=key1
    )

    # Replace product formation rate with neural network
    builder.replace_with_nn(
        name='product_rate',
        input_features=['X', 'P', 'temp'],
        hidden_dims=[4, 4],  # Small network for testing
        output_activation=jax.nn.softplus,  # Ensure non-negative rate
        key=key2
    )

    # Build and return the model
    return builder.build()