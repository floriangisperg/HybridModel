"""Tests for the loss module."""
import pytest
import jax
import jax.numpy as jnp
from hybrid_models.loss import (
    MSE, MAE, WeightedMSE, create_hybrid_model_loss, mse_loss, mae_loss
)


def test_mse_loss_metric():
    """Test the MSE loss metric."""
    y_pred = jnp.array([1.0, 2.0, 3.0])
    y_true = jnp.array([1.0, 3.0, 4.0])

    # Expected: ((0)² + (1)² + (1)²) / 3 = 0.6666...
    expected = jnp.mean(jnp.square(y_pred - y_true))

    loss = MSE.compute(y_pred, y_true)
    assert jnp.isclose(loss, expected)
    assert MSE.name() == "mse"


def test_mae_loss_metric():
    """Test the MAE loss metric."""
    y_pred = jnp.array([1.0, 2.0, 3.0])
    y_true = jnp.array([1.0, 3.0, 4.0])

    # Expected: (|0| + |1| + |1|) / 3 = 0.6666...
    expected = jnp.mean(jnp.abs(y_pred - y_true))

    loss = MAE.compute(y_pred, y_true)
    assert jnp.isclose(loss, expected)
    assert MAE.name() == "mae"


def test_weighted_mse_loss_metric():
    """Test the WeightedMSE loss metric."""
    y_pred = jnp.array([1.0, 2.0, 3.0])
    y_true = jnp.array([1.0, 3.0, 4.0])
    weights = jnp.array([0.5, 1.0, 2.0])

    # Expected: (0.5*(0)² + 1.0*(1)² + 2.0*(1)²) / 3 = 1.0
    squared_errors = jnp.square(y_pred - y_true)
    expected = jnp.mean(weights * squared_errors)

    loss = WeightedMSE.compute(y_pred, y_true, weights)
    assert jnp.isclose(loss, expected)
    assert WeightedMSE.name() == "weighted_mse"


class MockModel:
    """Mock model for testing loss functions."""

    def solve(self, initial_state, t_span, evaluation_times, args={}, **kwargs):
        """Mock solve method that returns predefined solutions."""
        return {
            'times': evaluation_times,
            'X': jnp.array([1.0, 2.0, 3.0]),
            'P': jnp.array([0.1, 0.2, 0.3])
        }


def test_create_hybrid_model_loss():
    """Test creating a hybrid model loss function."""
    model = MockModel()
    datasets = [
        {
            'initial_state': {'X': 1.0, 'P': 0.1},
            'times': jnp.array([0.0, 1.0, 2.0]),
            'X_true': jnp.array([1.0, 2.0, 3.0]),
            'P_true': jnp.array([0.2, 0.3, 0.4])
        }
    ]

    # Create loss function with MSE metric
    loss_fn = create_hybrid_model_loss(
        state_names=['X', 'P'],
        loss_metric=MSE
    )

    # Calculate loss
    loss, aux = loss_fn(model, datasets)

    # Expected X loss: ((0)² + (0)² + (0)²) / 3 = 0
    # Expected P loss: ((0.1)² + (0.1)² + (0.1)²) / 3 = 0.01
    # Total loss: 0 + 0.01 = 0.01
    expected_x_loss = 0.0
    expected_p_loss = 0.01
    expected_total = expected_x_loss + expected_p_loss

    assert jnp.isclose(loss, expected_total)
    assert len(aux) == 2
    assert jnp.isclose(aux[0], expected_x_loss)
    assert jnp.isclose(aux[1], expected_p_loss)


def test_mse_loss_convenience():
    """Test the MSE loss convenience function."""
    model = MockModel()
    datasets = [
        {
            'initial_state': {'X': 1.0, 'P': 0.1},
            'times': jnp.array([0.0, 1.0, 2.0]),
            'X_true': jnp.array([1.0, 2.0, 3.0]),
            'P_true': jnp.array([0.2, 0.3, 0.4])
        }
    ]

    # Use convenience function
    loss, aux = mse_loss(model, datasets, state_names=['X', 'P'])

    # Expected X loss: ((0)² + (0)² + (0)²) / 3 = 0
    # Expected P loss: ((0.1)² + (0.1)² + (0.1)²) / 3 = 0.01
    # Total loss: 0 + 0.01 = 0.01
    expected_x_loss = 0.0
    expected_p_loss = 0.01
    expected_total = expected_x_loss + expected_p_loss

    assert jnp.isclose(loss, expected_total)
    assert len(aux) == 2
    assert jnp.isclose(aux[0], expected_x_loss)
    assert jnp.isclose(aux[1], expected_p_loss)


def test_mse_loss_with_weights():
    """Test MSE loss with component weights."""
    model = MockModel()
    datasets = [
        {
            'initial_state': {'X': 1.0, 'P': 0.1},
            'times': jnp.array([0.0, 1.0, 2.0]),
            'X_true': jnp.array([1.0, 2.0, 3.0]),
            'P_true': jnp.array([0.2, 0.3, 0.4])
        }
    ]

    # Create loss function with weights
    loss_fn = create_hybrid_model_loss(
        state_names=['X', 'P'],
        loss_metric=MSE,
        component_weights={'X': 1.0, 'P': 2.0}
    )

    # Calculate loss
    loss, aux = loss_fn(model, datasets)

    # Expected X loss: ((0)² + (0)² + (0)²) / 3 = 0
    # Expected P loss: ((0.1)² + (0.1)² + (0.1)²) / 3 = 0.01
    # Total loss: 1.0*0 + 2.0*0.01 = 0.02
    expected_x_loss = 0.0
    expected_p_loss = 0.01
    expected_total = 1.0 * expected_x_loss + 2.0 * expected_p_loss

    assert jnp.isclose(loss, expected_total)
    assert len(aux) == 2
    assert jnp.isclose(aux[0], expected_x_loss)
    assert jnp.isclose(aux[1], expected_p_loss)


def test_automatic_state_detection():
    """Test that states are automatically detected from the solution."""
    model = MockModel()
    datasets = [
        {
            'initial_state': {'X': 1.0, 'P': 0.1},
            'times': jnp.array([0.0, 1.0, 2.0]),
            'X_true': jnp.array([1.0, 2.0, 3.0]),
            'P_true': jnp.array([0.2, 0.3, 0.4])
        }
    ]

    # Create loss function without specifying state_names
    loss_fn = create_hybrid_model_loss(loss_metric=MSE)

    # Calculate loss
    loss, aux = loss_fn(model, datasets)

    # Should still calculate loss for both X and P
    expected_x_loss = 0.0
    expected_p_loss = 0.01
    expected_total = expected_x_loss + expected_p_loss

    assert jnp.isclose(loss, expected_total)


def test_custom_solve_function():
    """Test using a custom solve function."""

    def custom_solve(model, dataset):
        """Custom solve function for testing."""
        return {
            'times': dataset['times'],
            'X': jnp.array([1.1, 2.1, 3.1]),
            'P': jnp.array([0.15, 0.25, 0.35])
        }

    model = MockModel()  # Will not be used since we provide custom_solve
    datasets = [
        {
            'initial_state': {'X': 1.0, 'P': 0.1},
            'times': jnp.array([0.0, 1.0, 2.0]),
            'X_true': jnp.array([1.0, 2.0, 3.0]),
            'P_true': jnp.array([0.2, 0.3, 0.4])
        }
    ]

    # Create loss function with custom solve function
    loss_fn = create_hybrid_model_loss(
        solve_fn=custom_solve,
        state_names=['X', 'P'],
        loss_metric=MSE
    )

    # Calculate loss
    loss, aux = loss_fn(model, datasets)

    # Expected X loss: ((0.1)² + (0.1)² + (0.1)²) / 3 = 0.01
    # Expected P loss: ((0.05)² + (0.05)² + (0.05)²) / 3 = 0.0025
    # Total loss: 0.01 + 0.0025 = 0.0125
    expected_x_loss = 0.01
    expected_p_loss = 0.0025
    expected_total = expected_x_loss + expected_p_loss

    assert jnp.isclose(loss, expected_total, rtol=1e-4)
    assert len(aux) == 2
    assert jnp.isclose(aux[0], expected_x_loss, rtol=1e-4)
    assert jnp.isclose(aux[1], expected_p_loss, rtol=1e-4)


def test_regularization():
    """Test adding regularization to the loss function."""

    def reg_function(model):
        """Simple regularization function for testing."""
        return 0.5

    model = MockModel()
    datasets = [
        {
            'initial_state': {'X': 1.0, 'P': 0.1},
            'times': jnp.array([0.0, 1.0, 2.0]),
            'X_true': jnp.array([1.0, 2.0, 3.0]),
            'P_true': jnp.array([0.2, 0.3, 0.4])
        }
    ]

    # Create loss function with regularization
    loss_fn = create_hybrid_model_loss(
        state_names=['X', 'P'],
        loss_metric=MSE,
        regularization=reg_function,
        reg_strength=0.1
    )

    # Calculate loss
    loss, aux = loss_fn(model, datasets)

    # Expected X loss: ((0)² + (0)² + (0)²) / 3 = 0
    # Expected P loss: ((0.1)² + (0.1)² + (0.1)²) / 3 = 0.01
    # Reg term: 0.1 * 0.5 = 0.05
    # Total loss: 0 + 0.01 + 0.05 = 0.06
    expected_x_loss = 0.0
    expected_p_loss = 0.01
    expected_reg = 0.5
    expected_total = expected_x_loss + expected_p_loss + 0.1 * expected_reg

    assert jnp.isclose(loss, expected_total)
    assert len(aux) == 3  # X, P, and regularization
    assert jnp.isclose(aux[0], expected_x_loss)
    assert jnp.isclose(aux[1], expected_p_loss)
    assert jnp.isclose(aux[2], expected_reg)
