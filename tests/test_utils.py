"""Tests for the utility functions."""
import pytest
import jax
import jax.numpy as jnp
from hybrid_models.utils import (
    normalize_data,
    combine_normalization_params,
    calculate_rate,
    create_initial_random_key
)


def test_normalize_data():
    """Test data normalization function."""
    # Create sample data
    data = {
        'X': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'Y': jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    }

    # Normalize data
    normalized_data, norm_params = normalize_data(data)

    # Check normalization parameters
    assert 'X_mean' in norm_params
    assert 'X_std' in norm_params
    assert 'Y_mean' in norm_params
    assert 'Y_std' in norm_params

    assert jnp.isclose(norm_params['X_mean'], 3.0)
    assert jnp.isclose(norm_params['X_std'], jnp.std(data['X']))
    assert jnp.isclose(norm_params['Y_mean'], 30.0)
    assert jnp.isclose(norm_params['Y_std'], jnp.std(data['Y']))

    # Check normalized data
    for key in data:
        # Mean should be close to 0
        assert jnp.abs(jnp.mean(normalized_data[key])) < 1e-5

        # Standard deviation should be close to 1
        assert jnp.abs(jnp.std(normalized_data[key]) - 1.0) < 1e-5


def test_normalize_data_with_zero_std():
    """Test normalization with zero standard deviation."""
    # Create data with zero variance
    data = {
        'X': jnp.array([5.0, 5.0, 5.0, 5.0, 5.0])
    }

    # Normalize data
    normalized_data, norm_params = normalize_data(data)

    # Check that std is not zero (should be set to at least 1e-8)
    assert norm_params['X_std'] >= 1e-8

    # All normalized values should be zero
    assert jnp.allclose(normalized_data['X'], jnp.zeros_like(data['X']))


def test_combine_normalization_params():
    """Test combining normalization parameters from multiple datasets."""
    # Create sample normalization parameters
    params1 = {
        'X_mean': 1.0,
        'X_std': 2.0,
        'Y_mean': 10.0,
        'Y_std': 20.0
    }

    params2 = {
        'X_mean': 3.0,
        'X_std': 4.0,
        'Z_mean': 30.0,
        'Z_std': 40.0
    }

    # Combine parameters
    combined = combine_normalization_params([params1, params2])

    # Check combined parameters
    assert jnp.isclose(combined['X_mean'], 2.0)  # (1.0 + 3.0) / 2
    assert jnp.isclose(combined['X_std'], 3.0)  # (2.0 + 4.0) / 2
    assert jnp.isclose(combined['Y_mean'], 10.0)
    assert jnp.isclose(combined['Y_std'], 20.0)
    assert jnp.isclose(combined['Z_mean'], 30.0)
    assert jnp.isclose(combined['Z_std'], 40.0)


def test_calculate_rate():
    """Test calculation of rates of change."""
    # Create sample data
    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = jnp.array([10.0, 15.0, 25.0, 40.0, 60.0])

    # Calculate rates
    rates = calculate_rate(times, values)

    # Check rates
    # Expected: [5.0, 10.0, 15.0, 20.0, 20.0] (last value copied from previous)
    expected_rates = jnp.array([5.0, 10.0, 15.0, 20.0, 20.0])
    assert jnp.allclose(rates, expected_rates)


def test_calculate_rate_uneven_times():
    """Test rate calculation with uneven time steps."""
    # Create sample data with uneven time steps
    times = jnp.array([0.0, 2.0, 3.0, 7.0])
    values = jnp.array([10.0, 20.0, 25.0, 45.0])

    # Calculate rates
    rates = calculate_rate(times, values)

    # Check rates
    # Expected: [5.0, 5.0, 5.0, 5.0]
    expected_rates = jnp.array([5.0, 5.0, 5.0, 5.0])
    assert jnp.allclose(rates, expected_rates)


def test_create_initial_random_key():
    """Test creating initial random keys."""
    # Create keys with different seeds
    key1 = create_initial_random_key(0)
    key2 = create_initial_random_key(1)
    key1_duplicate = create_initial_random_key(0)

    # Check that keys are different with different seeds
    assert not jnp.array_equal(key1, key2)

    # Check that keys are the same with the same seed
    assert jnp.array_equal(key1, key1_duplicate)

    # Check key shape
    assert key1.shape == (2,)