"""Enhanced tests for the evaluation module."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from hybrid_models.evaluation import calculate_metrics, evaluate_hybrid_model


def test_calculate_metrics_nan_handling():
    """Test metrics calculation with NaN values."""
    y_true = jnp.array([1.0, 2.0, jnp.nan, 4.0, 5.0])
    y_pred = jnp.array([1.1, 2.2, 3.3, 4.4, 5.5])

    # This should handle NaNs by filtering them out
    # If your implementation doesn't already handle NaNs, this test will guide that enhancement
    metrics = calculate_metrics(y_true, y_pred)

    # Check that metrics are computed despite NaNs
    assert jnp.isfinite(metrics['mse'])
    assert jnp.isfinite(metrics['rmse'])
    assert jnp.isfinite(metrics['mae'])
    assert jnp.isfinite(metrics['r2'])


def test_calculate_metrics_empty_arrays():
    """Test metrics calculation with empty arrays."""
    y_true = jnp.array([])
    y_pred = jnp.array([])

    # Handle the empty array case
    # This might raise a ValueError or return NaNs depending on desired behavior
    with pytest.raises(Exception):
        metrics = calculate_metrics(y_true, y_pred)


def test_r2_score_edge_cases():
    """Test R² score calculation in edge cases."""
    # All predictions exactly match true values
    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.0, 3.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert jnp.isclose(metrics['r2'], 1.0)

    # Predictions are mean of true values (R² should be 0)
    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([2.0, 2.0, 2.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert jnp.isclose(metrics['r2'], 0.0)

    # Predictions completely wrong (R² can be negative)
    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([3.0, 2.0, 1.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics['r2'] < 0.0


def test_evaluate_hybrid_model_with_missing_outputs():
    """Test evaluation when some datasets are missing certain outputs."""

    # Create a mock model
    class MockModel:
        def solve(self, **kwargs):
            return {
                'times': jnp.array([0.0, 1.0, 2.0]),
                'X': jnp.array([1.0, 2.0, 3.0]),
                'P': jnp.array([0.1, 0.4, 0.9])
            }

    # Mock solve function
    def mock_solve_fn(model, dataset):
        return model.solve()

    # Create datasets with different available outputs
    datasets = [
        {
            'X_true': jnp.array([1.0, 2.0, 3.0]),
            # No P_true
        },
        {
            # No X_true
            'P_true': jnp.array([0.2, 0.3, 1.0])
        }
    ]

    # Evaluate model
    results = evaluate_hybrid_model(MockModel(), datasets, mock_solve_fn)

    # Check that only available outputs were evaluated
    assert 'X' in results['dataset_0']
    assert 'P' not in results['dataset_0']
    assert 'X' not in results['dataset_1']
    assert 'P' in results['dataset_1']

    # Overall metrics should have both X and P
    assert 'X' in results['overall']
    assert 'P' in results['overall']


def test_evaluate_hybrid_model_with_different_length_outputs():
    """Test evaluation with outputs of different lengths."""

    # Create a mock model that returns outputs with length matching the dataset
    class VariableLengthModel:
        def solve(self, initial_state, t_span, evaluation_times, args):
            times = evaluation_times
            X = jnp.ones_like(times)
            P = jnp.ones_like(times) * 0.5
            return {
                'times': times,
                'X': X,
                'P': P
            }

    # Mock solve function
    def mock_solve_fn(model, dataset):
        return model.solve(
            initial_state={},
            t_span=(0, 1),
            evaluation_times=dataset['times'],
            args={}
        )

    # Create datasets with different lengths
    datasets = [
        {
            'times': jnp.array([0.0, 1.0, 2.0]),
            'X_true': jnp.array([1.0, 2.0, 3.0]),
            'P_true': jnp.array([0.1, 0.4, 0.9])
        },
        {
            'times': jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            'X_true': jnp.array([1.0, 1.5, 2.0, 2.5, 3.0]),
            'P_true': jnp.array([0.1, 0.2, 0.4, 0.6, 0.9])
        }
    ]

    # Evaluate model
    results = evaluate_hybrid_model(VariableLengthModel(), datasets, mock_solve_fn)

    # Check that all datasets were evaluated
    assert 'dataset_0' in results
    assert 'dataset_1' in results
    assert 'overall' in results

    # Check that both variables were evaluated for all datasets
    for i in range(2):
        assert 'X' in results[f'dataset_{i}']
        assert 'P' in results[f'dataset_{i}']