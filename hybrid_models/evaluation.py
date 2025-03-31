import jax
import jax.numpy as jnp
from typing import Dict, List, Any
from jaxtyping import Array, Float


def calculate_metrics(y_true: Float[Array, "N"], y_pred: Float[Array, "N"]) -> Dict[str, float]:
    """Calculate error metrics for model evaluation."""
    # Mean Squared Error
    mse = jnp.mean(jnp.square(y_pred - y_true))

    # Root Mean Squared Error
    rmse = jnp.sqrt(mse)

    # Mean Absolute Error
    mae = jnp.mean(jnp.abs(y_pred - y_true))

    # R² (coefficient of determination)
    ss_total = jnp.sum(jnp.square(y_true - jnp.mean(y_true)))
    ss_residual = jnp.sum(jnp.square(y_true - y_pred))

    # Handle edge case where ss_total is zero or very small
    r2 = jnp.where(ss_total > 1e-10,
                   1 - ss_residual / ss_total,
                   0.0)

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def evaluate_hybrid_model(model: Any, datasets: List[Dict], solve_fn: callable) -> Dict:
    """
    Evaluate a hybrid model on multiple datasets.

    Args:
        model: The hybrid model to evaluate
        datasets: List of datasets for evaluation
        solve_fn: Function to solve the ODE system and get predictions

    Returns:
        Dictionary of evaluation metrics
    """
    results = {}

    # Overall metrics collections
    all_metrics = {}

    for i, dataset in enumerate(datasets):
        # Get predictions
        predictions = solve_fn(model, dataset)

        # Calculate metrics for each state variable
        dataset_metrics = {}
        for state_name in predictions.keys():
            if state_name != 'times' and f"{state_name}_true" in dataset:
                y_true = dataset[f"{state_name}_true"]
                y_pred = predictions[state_name]

                # Calculate metrics
                state_metrics = calculate_metrics(y_true, y_pred)
                dataset_metrics[state_name] = state_metrics

                # Add to overall metrics
                if state_name not in all_metrics:
                    all_metrics[state_name] = {
                        'true': [],
                        'pred': []
                    }

                all_metrics[state_name]['true'].extend(y_true)
                all_metrics[state_name]['pred'].extend(y_pred)

        # Store metrics for this dataset
        results[f"dataset_{i}"] = dataset_metrics

    # Calculate overall metrics
    overall_metrics = {}
    for state_name, values in all_metrics.items():
        y_true = jnp.array(values['true'])
        y_pred = jnp.array(values['pred'])
        overall_metrics[state_name] = calculate_metrics(y_true, y_pred)

    results['overall'] = overall_metrics

    return results