"""Tests for the HybridModelBuilder class."""
import pytest
import jax
import jax.numpy as jnp
from hybrid_models import (
    HybridModelBuilder,
    HybridODESystem,
    ConfigurableNN,
    create_initial_random_key
)


@pytest.fixture
def random_key():
    """Create a reproducible random key for testing."""
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


def test_builder_initialization():
    """Test builder initialization."""
    builder = HybridModelBuilder()
    assert len(builder.state_names) == 0
    assert len(builder.mechanistic_components) == 0
    assert len(builder.nn_replacements) == 0
    assert len(builder.norm_params) == 0


def test_add_state():
    """Test adding state variables."""
    builder = HybridModelBuilder()
    builder.add_state('X')
    builder.add_state('P')

    assert 'X' in builder.state_names
    assert 'P' in builder.state_names
    assert len(builder.state_names) == 2

    # Test adding duplicate state
    builder.add_state('X')
    assert len(builder.state_names) == 2  # Should not add duplicates


def test_add_mechanistic_component():
    """Test adding mechanistic components."""
    builder = HybridModelBuilder()

    def dummy_component(inputs):
        return inputs['X'] * 0.5

    builder.add_mechanistic_component('growth_rate', dummy_component)

    assert 'growth_rate' in builder.mechanistic_components
    assert builder.mechanistic_components['growth_rate'] is dummy_component


def test_set_normalization_params(norm_params):
    """Test setting normalization parameters."""
    builder = HybridModelBuilder()
    builder.set_normalization_params(norm_params)

    assert builder.norm_params == norm_params


def test_replace_with_nn(norm_params, random_key):
    """Test replacing a component with a neural network."""
    builder = HybridModelBuilder()
    builder.set_normalization_params(norm_params)

    # Replace growth_rate with NN
    builder.replace_with_nn(
        name='growth_rate',
        input_features=['X', 'temp'],
        hidden_dims=[16, 8],
        key=random_key
    )

    assert 'growth_rate' in builder.nn_replacements
    assert isinstance(builder.nn_replacements['growth_rate'], ConfigurableNN)
    assert builder.nn_replacements['growth_rate'].input_features == ['X', 'temp']

    # Check that NN has the right number of layers
    layers = builder.nn_replacements['growth_rate'].layers
    # Input layer, 2 hidden layers with activations each, output layer
    assert len(layers) >= 5


def test_replace_with_nn_custom_activation(norm_params, random_key):
    """Test replacing a component with NN using custom activation."""
    builder = HybridModelBuilder()
    builder.set_normalization_params(norm_params)

    # Use softplus activation for output
    builder.replace_with_nn(
        name='product_rate',
        input_features=['X', 'P'],
        hidden_dims=[8],
        output_activation=jax.nn.softplus,
        key=random_key
    )

    assert 'product_rate' in builder.nn_replacements
    nn = builder.nn_replacements['product_rate']

    # Last layer should be the activation function
    assert nn.layers[-1] is jax.nn.softplus


def test_build(norm_params, random_key):
    """Test building the hybrid model."""
    builder = HybridModelBuilder()
    builder.set_normalization_params(norm_params)

    # Add states
    builder.add_state('X')
    builder.add_state('P')

    # Add mechanistic components
    def biomass_ode(inputs):
        return inputs['X'] * inputs['growth_rate']

    def product_ode(inputs):
        return inputs['X'] * inputs['product_rate']

    builder.add_mechanistic_component('X', biomass_ode)
    builder.add_mechanistic_component('P', product_ode)

    # Replace components with NNs
    key1, key2 = jax.random.split(random_key)

    builder.replace_with_nn(
        name='growth_rate',
        input_features=['X', 'temp'],
        hidden_dims=[8],
        key=key1
    )

    builder.replace_with_nn(
        name='product_rate',
        input_features=['X', 'P'],
        hidden_dims=[8],
        output_activation=jax.nn.softplus,
        key=key2
    )

    # Build the model
    model = builder.build()

    # Check model structure
    assert isinstance(model, HybridODESystem)
    assert model.state_names == ['X', 'P']
    assert 'X' in model.mechanistic_components
    assert 'P' in model.mechanistic_components
    assert 'growth_rate' in model.nn_replacements
    assert 'product_rate' in model.nn_replacements


def test_build_empty_model():
    """Test building an empty model."""
    builder = HybridModelBuilder()
    model = builder.build()

    assert isinstance(model, HybridODESystem)
    assert len(model.state_names) == 0
    assert len(model.mechanistic_components) == 0
    assert len(model.nn_replacements) == 0