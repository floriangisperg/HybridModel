"""
Loss functions for hybrid models.

This module provides standardized loss functions and metrics for training hybrid models,
with support for multiple state variables, component-wise loss tracking, and customization.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from jaxtyping import Array, Float, PyTree


class LossMetric:
    """Base class for loss metrics."""

    @staticmethod
    def compute(y_pred, y_true):
        """Compute the loss between predicted and true values."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def name():
        """Return the name of the loss metric."""
        raise NotImplementedError("Subclasses must implement this method")


class MSE(LossMetric):
    """Mean Squared Error loss metric."""

    @staticmethod
    def compute(y_pred, y_true):
        """Compute the mean squared error."""
        return jnp.mean(jnp.square(y_pred - y_true))

    @staticmethod
    def name():
        return "mse"


class RelativeMSE(LossMetric):
    """Relative Mean Squared Error loss metric."""

    @staticmethod
    def compute(y_pred, y_true):
        """Compute the relative mean squared error."""
        return jnp.mean(jnp.square((y_pred - y_true) / (y_true + 1e-8)))

    @staticmethod
    def name():
        return "relative_mse"


class MAE(LossMetric):
    """Mean Absolute Error loss metric."""

    @staticmethod
    def compute(y_pred, y_true):
        """Compute the mean absolute error."""
        return jnp.mean(jnp.abs(y_pred - y_true))

    @staticmethod
    def name():
        return "mae"


class WeightedMSE(LossMetric):
    """Weighted Mean Squared Error loss metric."""

    @staticmethod
    def compute(y_pred, y_true, weights=None):
        """Compute the weighted mean squared error."""
        if weights is None:
            return MSE.compute(y_pred, y_true)

        squared_errors = jnp.square(y_pred - y_true)
        return jnp.mean(weights * squared_errors)

    @staticmethod
    def name():
        return "weighted_mse"


def create_hybrid_model_loss(
        solve_fn: Optional[Callable] = None,
        state_names: Optional[List[str]] = None,
        loss_metric: LossMetric = MSE,
        component_weights: Optional[Dict[str, float]] = None,
        regularization: Optional[Callable] = None,
        reg_strength: float = 0.0,
        solve_kwargs: Optional[Dict] = None
):
    """
    Create a loss function for hybrid model training.

    Args:
        solve_fn: Function to solve the model for a dataset. If None, assumes model.solve is available.
        state_names: Names of state variables to use in loss calculation. If None, uses all available states.
        loss_metric: Loss metric to use for calculating errors.
        component_weights: Optional weights for each state component in the loss.
        regularization: Optional regularization function that takes the model and returns a regularization term.
        reg_strength: Strength of the regularization term.
        solve_kwargs: Additional keyword arguments to pass to the solve function.

    Returns:
        A loss function that takes (model, datasets) and returns (loss_value, aux)
    """

    solve_kwargs = solve_kwargs or {}

    def loss_function(model, datasets):
        """Loss function for hybrid model training.

        Args:
            model: The hybrid model to evaluate
            datasets: List of datasets containing ground truth values

        Returns:
            Tuple of (total_loss, component_losses)
        """
        total_loss = 0.0
        component_losses = {}

        # Initialize component losses
        if state_names:
            for state in state_names:
                component_losses[state] = 0.0

        for dataset in datasets:
            # Get predictions
            if solve_fn is not None:
                solution = solve_fn(model, dataset)
            else:
                # Default to using model's solve method with additional kwargs
                solution = model.solve(
                    initial_state=dataset['initial_state'],
                    t_span=(dataset['times'][0], dataset['times'][-1]),
                    evaluation_times=dataset['times'],
                    args={
                        'time_dependent_inputs': dataset.get('time_dependent_inputs', {}),
                        'static_inputs': dataset.get('static_inputs', {})
                    },
                    **solve_kwargs
                )

            # Determine which states to use in loss calculation
            states_to_use = state_names if state_names else [
                k for k in solution.keys() if k != 'times' and f"{k}_true" in dataset
            ]

            # Calculate loss for each state
            dataset_loss = 0.0

            for state in states_to_use:
                true_key = f"{state}_true"
                if true_key in dataset and state in solution:
                    y_pred = solution[state]
                    y_true = dataset[true_key]

                    # Apply weights if provided
                    weight = component_weights.get(state, 1.0) if component_weights else 1.0

                    # Calculate component loss
                    if hasattr(loss_metric, 'compute'):
                        # Class-based metric
                        component_loss = loss_metric.compute(y_pred, y_true)
                    else:
                        # Function-based metric
                        component_loss = loss_metric(y_pred, y_true)

                    # Apply weight
                    weighted_loss = weight * component_loss

                    # Add to total loss
                    dataset_loss += weighted_loss

                    # Track component loss
                    if state in component_losses:
                        component_losses[state] += component_loss

            # Add dataset loss to total
            total_loss += dataset_loss

        # Calculate average loss
        n_datasets = len(datasets)
        avg_loss = total_loss / n_datasets

        # Calculate average component losses
        avg_component_losses = {k: v / n_datasets for k, v in component_losses.items()}

        # Apply regularization if provided
        if regularization is not None and reg_strength > 0:
            reg_term = regularization(model)
            avg_loss = avg_loss + reg_strength * reg_term
            avg_component_losses['regularization'] = reg_term

        # Return average loss and auxiliary output (component losses as tuple)
        return avg_loss, tuple(avg_component_losses.values())

    return loss_function


def mse_loss(model, datasets, state_names=None, weights=None, solve_kwargs=None):
    """
    Convenience function for creating an MSE loss function.

    Args:
        model: The model being trained
        datasets: The datasets for training
        state_names: Optional list of state names to include in loss
        weights: Optional weights for each state
        solve_kwargs: Additional keyword arguments to pass to the solve function

    Returns:
        Loss value and component losses
    """
    component_weights = dict(zip(state_names, weights)) if weights and state_names else None
    loss_fn = create_hybrid_model_loss(
        state_names=state_names,
        loss_metric=MSE,
        component_weights=component_weights,
        solve_kwargs=solve_kwargs
    )

    return loss_fn(model, datasets)


def mae_loss(model, datasets, state_names=None, weights=None, solve_kwargs=None):
    """
    Convenience function for creating an MAE loss function.

    Args:
        model: The model being trained
        datasets: The datasets for training
        state_names: Optional list of state names to include in loss
        weights: Optional weights for each state
        solve_kwargs: Additional keyword arguments to pass to the solve function

    Returns:
        Loss value and component losses
    """
    component_weights = dict(zip(state_names, weights)) if weights and state_names else None
    loss_fn = create_hybrid_model_loss(
        state_names=state_names,
        loss_metric=MAE,
        component_weights=component_weights,
        solve_kwargs=solve_kwargs
    )

    return loss_fn(model, datasets)