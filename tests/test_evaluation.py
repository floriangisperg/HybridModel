"""Tests for the evaluation module."""
import pytest
import jax
import jax.numpy as jnp
from hybrid_models.evaluation import calculate_metrics, evaluate_hybrid_model


def test_calculate_metrics_perfect_prediction():
    """Test metrics calculation with perfect predictions."""
    y_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    metrics = calculate_metrics(y_true, y_pred)

    # Check all metrics
    assert metrics['mse'] == 0.0
    assert metrics['rmse'] == 0.0
    assert metrics['mae'] == 0.0
    assert metrics['r2'] == 1.0


def test_calculate_metrics_imperfect_prediction():
    """Test metrics calculation with imperfect predictions."""
    y_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = jnp.array([1.1, 2.2, 2.8, 4.1, 4.9])

    metrics = calculate_metrics(y_true, y_pred)

    # Check metrics are within expected ranges
    assert 0.0 < metrics['mse'] < 0.1
    assert 0.0 < metrics['rmse'] < 0.3
    assert 0.0 < metrics['mae'] < 0.2
    assert 0.8 < metrics['r2'] < 1.0


def test_calculate_metrics_bad_prediction():
    """Test metrics calculation with poor predictions."""
    y_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Inverse relationship

    metrics = calculate_metrics(y_true, y_pred)

    # Check metrics reflect poor performance
    assert metrics['mse'] >= 8.0  # Changed from > to >= since MSE is exactly 8.0
    assert metrics['rmse'] > 2.8
    assert metrics['mae'] > 2.4
    assert metrics['r2'] < 0.0  # Negative R² indicates worse than mean prediction


def test_calculate_metrics_constant_prediction():
    """Test metrics calculation with constant predictions."""
    y_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = jnp.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Mean value

    metrics = calculate_metrics(y_true, y_pred)

    # Check metrics reflect mean prediction
    assert metrics['mse'] == 2.0
    assert jnp.isclose(metrics['rmse'], jnp.sqrt(2.0))
    assert jnp.isclose(metrics['mae'], 1.2, rtol=1e-5)  # Use isclose instead of == for floating point
    assert jnp.isclose(metrics['r2'], 0.0)  # R² = 0 for mean prediction


def test_calculate_metrics_constant_truth():
    """Test metrics calculation with constant ground truth."""
    y_true = jnp.array([3.0, 3.0, 3.0, 3.0, 3.0])
    y_pred = jnp.array([2.0, 2.5, 3.0, 3.5, 4.0])

    metrics = calculate_metrics(y_true, y_pred)

    # When variance is zero, R² is undefined and should be set to 0
    assert metrics['r2'] == 0.0


def test_evaluate_hybrid_model():
    """Test evaluating a hybrid model on multiple datasets."""
    # Create a mock model with a simple solve function
    class MockModel:
        def solve(self, **kwargs):
            # Always return the same predictions
            return {
                'times': jnp.array([0.0, 1.0, 2.0]),
                'X': jnp.array([1.0, 2.0, 3.0]),
                'P': jnp.array([0.1, 0.4, 0.9])
            }

    # Create a mock solve function
    def mock_solve_fn(model, dataset):
        return model.solve()

    # Create mock datasets
    datasets = [
        {
            'X_true': jnp.array([1.0, 2.0, 3.0]),
            'P_true': jnp.array([0.1, 0.4, 0.9])
        },
        {
            'X_true': jnp.array([1.1, 1.9, 3.1]),
            'P_true': jnp.array([0.2, 0.3, 1.0])
        }
    ]

    # Evaluate model
    results = evaluate_hybrid_model(MockModel(), datasets, mock_solve_fn)

    # Check results structure
    assert 'dataset_0' in results
    assert 'dataset_1' in results
    assert 'overall' in results

    # Check metrics for each dataset
    for i in range(2):
        assert 'X' in results[f'dataset_{i}']
        assert 'P' in results[f'dataset_{i}']

        # First dataset should have perfect metrics for both X and P
        if i == 0:
            assert results[f'dataset_{i}']['X']['mse'] == 0.0
            assert results[f'dataset_{i}']['P']['mse'] == 0.0
        else:
            # Second dataset should have some error
            assert results[f'dataset_{i}']['X']['mse'] > 0.0
            assert results[f'dataset_{i}']['P']['mse'] > 0.0

    # Check overall metrics
    assert 'X' in results['overall']
    assert 'P' in results['overall']
    assert results['overall']['X']['mse'] > 0.0  # Should have some error due to second dataset
    assert results['overall']['P']['mse'] > 0.0


def test_evaluate_hybrid_model_single_dataset():
    """Test evaluating a hybrid model on a single dataset."""
    # Create a mock model with a simple solve function
    class MockModel:
        def solve(self, **kwargs):
            # Return slightly different predictions from ground truth
            return {
                'times': jnp.array([0.0, 1.0, 2.0]),
                'X': jnp.array([1.1, 2.1, 3.1])
            }

    # Create a mock solve function
    def mock_solve_fn(model, dataset):
        return model.solve()

    # Create mock dataset with only X (no P)
    datasets = [
        {
            'X_true': jnp.array([1.0, 2.0, 3.0])
            # Intentionally no P_true
        }
    ]

    # Evaluate model
    results = evaluate_hybrid_model(MockModel(), datasets, mock_solve_fn)

    # Check results
    assert 'dataset_0' in results
    assert 'X' in results['dataset_0']
    assert 'P' not in results['dataset_0']  # P should not be evaluated

    assert results['dataset_0']['X']['mse'] > 0.0
    assert jnp.isclose(results['dataset_0']['X']['mse'], 0.01)  # (0.1)² = 0.01