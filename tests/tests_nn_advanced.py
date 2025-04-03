"""Advanced tests for the neural network module."""
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from hybrid_models import ConfigurableNN, create_initial_random_key


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


def test_nn_differentiation(random_key, norm_params):
    """Test that the neural network is differentiable."""
    nn = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X', 'temp'],
        hidden_dims=[8, 8],
        key=random_key
    )

    inputs = {
        'X': jnp.array(5.0),
        'temp': jnp.array(37.0)
    }

    # Define a function to differentiate
    def f(nn_params, inputs):
        nn_local = eqx.tree_at(lambda n: n.layers, nn, nn_params)
        return nn_local(inputs)

    # Get gradient
    grad_f = jax.grad(lambda p: f(p, inputs))

    # Compute the gradient
    gradients = grad_f(nn.layers)

    # Check that we got gradients for all layers
    for i, layer in enumerate(gradients):
        if isinstance(layer, eqx.nn.Linear):
            assert layer.weight is not None
            assert layer.bias is not None

            # Check that at least some gradients are non-zero
            assert jnp.any(layer.weight != 0)
            assert jnp.any(layer.bias != 0)


def test_nn_jit_compatibility(random_key, norm_params):
    """Test that the neural network works with JAX JIT compilation."""
    nn = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X', 'temp'],
        hidden_dims=[8, 8],
        key=random_key
    )

    # Create a JIT-compiled version of the forward pass
    @jax.jit
    def jitted_forward(nn, X, temp):
        inputs = {'X': X, 'temp': temp}
        return nn(inputs)

    # Test with different inputs
    result1 = jitted_forward(nn, jnp.array(5.0), jnp.array(37.0))
    result2 = jitted_forward(nn, jnp.array(6.0), jnp.array(38.0))

    # Results should be different for different inputs
    assert result1 != result2

    # But the function should be stable for the same inputs
    result1_again = jitted_forward(nn, jnp.array(5.0), jnp.array(37.0))
    assert result1 == result1_again


def test_nn_batch_inputs(random_key, norm_params):
    """Test the neural network with batched inputs."""
    nn = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X', 'temp'],
        hidden_dims=[8, 8],
        key=random_key
    )

    # Create a batch of inputs
    batch_inputs = {
        'X': jnp.array([3.0, 5.0, 7.0]),
        'temp': jnp.array([35.0, 37.0, 39.0])
    }

    # Original implementation expects scalar inputs, should raise error or
    # would need to be modified to support batched inputs
    with pytest.raises(Exception):
        nn(batch_inputs)


def test_nn_with_missing_features(random_key, norm_params):
    """Test neural network behavior with missing input features."""
    nn = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X', 'temp', 'pH'],  # pH is not in norm_params
        hidden_dims=[8],
        key=random_key
    )

    # Input with all features
    inputs_complete = {
        'X': jnp.array(5.0),
        'temp': jnp.array(37.0),
        'pH': jnp.array(7.0)
    }

    # Input missing pH
    inputs_missing_ph = {
        'X': jnp.array(5.0),
        'temp': jnp.array(37.0)
    }

    # Complete inputs should work
    result_complete = nn(inputs_complete)

    # Missing inputs should raise KeyError
    with pytest.raises(KeyError):
        nn(inputs_missing_ph)


def test_nn_with_extra_features(random_key, norm_params):
    """Test neural network behavior with extra input features."""
    nn = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X', 'temp'],
        hidden_dims=[8],
        key=random_key
    )

    # Input with extra features
    inputs_extra = {
        'X': jnp.array(5.0),
        'temp': jnp.array(37.0),
        'pH': jnp.array(7.0),
        'DO': jnp.array(80.0)
    }

    # Should ignore extra features
    result = nn(inputs_extra)

    # Compare with only required features
    inputs_required = {
        'X': jnp.array(5.0),
        'temp': jnp.array(37.0)
    }

    result_required = nn(inputs_required)

    # Results should be identical
    assert result == result_required


def test_nn_output_range(random_key, norm_params):
    """Test that output activations constrain the output range appropriately."""
    # NN with sigmoid activation (0 to 1)
    nn_sigmoid = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X'],
        hidden_dims=[8],
        output_activation=jax.nn.sigmoid,
        key=random_key
    )

    # NN with relu activation (>= 0)
    key2 = jax.random.fold_in(random_key, 1)
    nn_relu = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X'],
        hidden_dims=[8],
        output_activation=jax.nn.relu,
        key=key2
    )

    # NN with softplus activation (> 0)
    key3 = jax.random.fold_in(random_key, 2)
    nn_softplus = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X'],
        hidden_dims=[8],
        output_activation=jax.nn.softplus,
        key=key3
    )

    # Test with various input values
    input_values = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])

    for x in input_values:
        inputs = {'X': x}

        # Sigmoid should be between 0 and 1
        sigmoid_output = nn_sigmoid(inputs)
        assert 0.0 <= sigmoid_output <= 1.0

        # ReLU should be >= 0
        relu_output = nn_relu(inputs)
        assert relu_output >= 0.0

        # Softplus should be > 0
        softplus_output = nn_softplus(inputs)
        assert softplus_output > 0.0


def test_nn_normalization_impact(random_key, norm_params):
    """Test the impact of normalization on neural network output."""
    # Create two identical NNs
    nn1 = ConfigurableNN(
        norm_params=norm_params,
        input_features=['X', 'temp'],
        hidden_dims=[8],
        key=random_key
    )

    # Same NN but with different normalization params
    different_norm_params = {
        'X_mean': 10.0,  # Different from norm_params
        'X_std': 5.0,
        'temp_mean': 30.0,
        'temp_std': 10.0
    }

    nn2 = ConfigurableNN(
        norm_params=different_norm_params,
        input_features=['X', 'temp'],
        hidden_dims=[8],
        key=random_key  # Same key for identical weights
    )

    # Same NN but without normalization
    nn3 = ConfigurableNN(
        norm_params={},  # Empty dict means no normalization
        input_features=['X', 'temp'],
        hidden_dims=[8],
        key=random_key
    )

    # Input values
    inputs = {
        'X': jnp.array(5.0),
        'temp': jnp.array(37.0)
    }

    # Get outputs
    output1 = nn1(inputs)
    output2 = nn2(inputs)
    output3 = nn3(inputs)

    # Outputs should differ due to different normalization
    assert output1 != output2
    assert output1 != output3
    assert output2 != output3