"""Tests for the ConfigurableNN module."""
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from hybrid_models import ConfigurableNN


def test_configurable_nn_initialization(random_key, norm_params):
    """Test that the ConfigurableNN initializes correctly."""
    # Create a neural network
    input_features = ['X', 'P', 'temp']
    hidden_dims = [16, 8]
    nn = ConfigurableNN(
        norm_params=norm_params,
        input_features=input_features,
        hidden_dims=hidden_dims,
        key=random_key
    )

    # Check that the network properties are set correctly
    assert nn.input_features == input_features
    assert nn.norm_params == norm_params

    # Check that the layers have the correct structure
    assert len(nn.layers) >= 5  # Input layer, 2 hidden layers with activations, output layer

    # The first layer should be a Linear layer with input_dim = len(input_features)
    assert isinstance(nn.layers[0], eqx.nn.Linear)
    assert nn.layers[0].in_features == len(input_features)
    assert nn.layers[0].out_features == hidden_dims[0]


def test_configurable_nn_forward_pass(random_key, norm_params):
    """Test the forward pass of ConfigurableNN."""
    # Create a neural network
    nn = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X', 'temp'],
        hidden_dims=[8, 4],
        key=random_key
    )

    # Create some input data
    inputs = {
        'X': jnp.array(10.0),  # Above mean (5.0)
        'temp': jnp.array(32.0),  # Below mean (37.0)
        'extra_feature': jnp.array(1.0)  # Should be ignored
    }

    # Perform forward pass
    output = nn(inputs)

    # Check output shape and type
    assert output.shape == ()  # Scalar output
    assert isinstance(output, jnp.ndarray)


def test_configurable_nn_normalization(random_key):
    """Test that normalization is applied correctly."""
    # Create normalization parameters
    norm_params = {
        'X_mean': 5.0,
        'X_std': 2.0,
        'temp_mean': 37.0,
        'temp_std': 5.0,
    }

    # Create two identical networks with the same initialization
    key1, key2 = jax.random.split(random_key)
    nn1 = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X', 'temp'],
        hidden_dims=[4],
        key=key1
    )
    nn2 = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X', 'temp'],
        hidden_dims=[4],
        key=key1  # Use the same key for identical initialization
    )

    # Test with non-normalized vs. normalized-equivalent inputs
    # X = 7.0 should be normalized to (7.0 - 5.0) / 2.0 = 1.0
    # temp = 42.0 should be normalized to (42.0 - 37.0) / 5.0 = 1.0
    inputs1 = {
        'X': jnp.array(7.0),  # Raw value
        'temp': jnp.array(42.0)  # Raw value
    }

    inputs2 = {
        'X': jnp.array(1.0),  # Already normalized
        'temp': jnp.array(1.0)  # Already normalized
    }

    # Save original norm_params
    original_norm_params = nn2.norm_params

    # Temporarily modify nn2 to skip normalization by setting empty norm_params
    nn2 = eqx.tree_at(lambda nn: nn.norm_params, nn2, {})

    # The outputs should be different because nn1 will normalize inputs but nn2 won't
    output1 = nn1(inputs1)
    output2 = nn2(inputs2)

    # They should be approximately equal since we're feeding equivalent inputs
    assert jnp.allclose(output1, output2, rtol=1e-5)


def test_configurable_nn_output_activation(random_key, norm_params):
    """Test that output activation is applied correctly."""
    # Create neural networks with different output activations
    nn_linear = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X'],
        hidden_dims=[4],
        output_activation=None,  # Linear output
        key=random_key
    )

    key2 = jax.random.split(random_key)[0]
    nn_relu = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X'],
        hidden_dims=[4],
        output_activation=jax.nn.relu,  # ReLU output
        key=key2  # Different key to avoid comparison issues
    )

    # Create negative input data that would result in negative outputs with linear activation
    # Use transformed keys to force negative outputs in the linear case
    key_transformed = jax.random.fold_in(random_key, 0)
    nn_linear = eqx.tree_at(
        lambda nn: nn.layers[-1].weight,
        nn_linear,
        -jnp.abs(jax.random.normal(key_transformed, (1, 4)))  # Ensure negative weights
    )
    nn_linear = eqx.tree_at(
        lambda nn: nn.layers[-1].bias,
        nn_linear,
        jnp.array([-1.0])  # Negative bias
    )

    # Copy the weights to the ReLU network to ensure comparable behavior
    nn_relu = eqx.tree_at(lambda nn: nn.layers[-2].weight, nn_relu, nn_linear.layers[-1].weight)
    nn_relu = eqx.tree_at(lambda nn: nn.layers[-2].bias, nn_relu, nn_linear.layers[-1].bias)

    inputs = {'X': jnp.array(5.0)}

    # Get outputs
    linear_output = nn_linear(inputs)
    relu_output = nn_relu(inputs)

    # Linear output should be able to go negative, ReLU output should be >= 0
    assert relu_output >= 0

    # If linear_output is negative, relu_output should be 0
    if linear_output < 0:
        assert jnp.allclose(relu_output, 0.0)