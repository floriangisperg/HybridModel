"""
Visualization utilities for hybrid models.

This module provides functions for visualizing hybrid model training and prediction results.
"""

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List, Any, Optional, Tuple, Callable


def plot_parity(model: Any,
                datasets: List[Dict],
                solve_fn: Callable,
                state_names: List[str],
                dataset_type: str = "Dataset",
                output_dir: str = "results",
                figsize: Tuple[int, int] = (10, 10),
                labels: Optional[Dict[str, str]] = None,
                include_r2: bool = True):
    """
    Create parity plots (predicted vs actual) for each state variable.

    Args:
        model: Trained model
        datasets: List of datasets containing true values
        solve_fn: Function to solve the model for predictions
        state_names: List of state variable names to plot
        dataset_type: String describing the dataset type (e.g., "Training", "Test")
        output_dir: Directory to save plots
        figsize: Figure size as (width, height) in inches
        labels: Optional dictionary mapping state names to display labels
        include_r2: Whether to include R² value on the plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use default labels if none provided
    if labels is None:
        labels = {state: state for state in state_names}

    # For each state variable, create a parity plot
    for state in state_names:
        # Collect all predicted and actual values across datasets
        y_true_all = []
        y_pred_all = []

        # Get predictions for each dataset
        for i, dataset in enumerate(datasets):
            true_key = f"{state}_true"
            if true_key in dataset:
                # Get true values
                y_true = dataset[true_key]

                # Get predictions
                solution = solve_fn(model, dataset)
                if state in solution:
                    y_pred = solution[state]

                    # Add to collections
                    y_true_all.extend(y_true)
                    y_pred_all.extend(y_pred)

        # Convert to arrays
        y_true_all = jnp.array(y_true_all)
        y_pred_all = jnp.array(y_pred_all)

        # Create the parity plot
        plt.figure(figsize=figsize)

        # Plot points
        plt.scatter(y_true_all, y_pred_all, alpha=0.6)

        # Add parity line
        max_val = max(jnp.max(y_true_all), jnp.max(y_pred_all))
        min_val = min(jnp.min(y_true_all), jnp.min(y_pred_all))
        # Add a small margin
        margin = (max_val - min_val) * 0.1
        plt.plot([min_val - margin, max_val + margin],
                 [min_val - margin, max_val + margin],
                 'k--', label='Parity Line')

        # Calculate R² if requested
        if include_r2 and len(y_true_all) > 0:
            from hybrid_models.evaluation import calculate_metrics
            metrics = calculate_metrics(y_true_all, y_pred_all)
            r2 = metrics['r2']
            rmse = metrics['rmse']
            plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                     transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top')

        # Labels and title
        plt.xlabel(f'Measured {labels.get(state, state)}')
        plt.ylabel(f'Predicted {labels.get(state, state)}')
        plt.title(f'Parity Plot for {labels.get(state, state)} - {dataset_type} Data')

        # Equal aspect ratio and grid
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'parity_{state}_{dataset_type.lower()}.png'))
        plt.close()


def plot_combined_parity(model: Any,
                         train_datasets: List[Dict],
                         test_datasets: Optional[List[Dict]],
                         solve_fn: Callable,
                         state_names: List[str],
                         output_dir: str = "results",
                         figsize: Tuple[int, int] = (10, 10),
                         labels: Optional[Dict[str, str]] = None):
    """
    Create combined parity plots with training and test data shown separately.

    Args:
        model: Trained model
        train_datasets: List of training datasets
        test_datasets: Optional list of test datasets
        solve_fn: Function to solve the model for predictions
        state_names: List of state variable names to plot
        output_dir: Directory to save plots
        figsize: Figure size as (width, height) in inches
        labels: Optional dictionary mapping state names to display labels
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use default labels if none provided
    if labels is None:
        labels = {state: state for state in state_names}

    # For each state variable, create a combined parity plot
    for state in state_names:
        # Collect all predicted and actual values
        train_true = []
        train_pred = []
        test_true = []
        test_pred = []

        # Process training datasets
        for dataset in train_datasets:
            true_key = f"{state}_true"
            if true_key in dataset:
                # Get true values
                y_true = dataset[true_key]

                # Get predictions
                solution = solve_fn(model, dataset)
                if state in solution:
                    y_pred = solution[state]

                    # Add to collections
                    train_true.extend(y_true)
                    train_pred.extend(y_pred)

        # Process test datasets if available
        if test_datasets:
            for dataset in test_datasets:
                true_key = f"{state}_true"
                if true_key in dataset:
                    # Get true values
                    y_true = dataset[true_key]

                    # Get predictions
                    solution = solve_fn(model, dataset)
                    if state in solution:
                        y_pred = solution[state]

                        # Add to collections
                        test_true.extend(y_true)
                        test_pred.extend(y_pred)

        # Convert to arrays
        train_true = jnp.array(train_true)
        train_pred = jnp.array(train_pred)
        test_true = jnp.array(test_true) if test_true else jnp.array([])
        test_pred = jnp.array(test_pred) if test_pred else jnp.array([])

        # Create the combined parity plot
        plt.figure(figsize=figsize)

        # Plot training points
        if len(train_true) > 0:
            plt.scatter(train_true, train_pred, alpha=0.6, color='blue', label='Training')

            # Calculate metrics for training data
            from hybrid_models.evaluation import calculate_metrics
            train_metrics = calculate_metrics(train_true, train_pred)
            train_r2 = train_metrics['r2']
            train_rmse = train_metrics['rmse']

        # Plot test points
        if len(test_true) > 0:
            plt.scatter(test_true, test_pred, alpha=0.6, color='red', label='Test')

            # Calculate metrics for test data
            from hybrid_models.evaluation import calculate_metrics
            test_metrics = calculate_metrics(test_true, test_pred)
            test_r2 = test_metrics['r2']
            test_rmse = test_metrics['rmse']

        # Add parity line
        all_true = jnp.concatenate([train_true, test_true]) if len(test_true) > 0 else train_true
        all_pred = jnp.concatenate([train_pred, test_pred]) if len(test_pred) > 0 else train_pred

        max_val = max(jnp.max(all_true), jnp.max(all_pred))
        min_val = min(jnp.min(all_true), jnp.min(all_pred))
        # Add a small margin
        margin = (max_val - min_val) * 0.1
        plt.plot([min_val - margin, max_val + margin],
                 [min_val - margin, max_val + margin],
                 'k--', label='Parity Line')

        # Add metrics to the plot
        text = ""
        if len(train_true) > 0:
            text += f'Training: R² = {train_r2:.4f}, RMSE = {train_rmse:.4f}\n'
        if len(test_true) > 0:
            text += f'Test: R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}'

        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top')

        # Labels and title
        plt.xlabel(f'Measured {labels.get(state, state)}')
        plt.ylabel(f'Predicted {labels.get(state, state)}')
        plt.title(f'Parity Plot for {labels.get(state, state)}')

        # Equal aspect ratio and grid
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'parity_{state}_combined.png'))
        plt.close()

        """
Visualization utilities for hybrid models.

This module provides functions for visualizing hybrid model training and prediction results.
"""


import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List, Any, Optional, Tuple, Callable


def plot_training_history(history: Dict[str, List],
                          output_dir: str = "results",
                          filename: str = "training_loss.png",
                          component_losses: bool = True,
                          component_names: Optional[List[str]] = None,
                          validation_history: Optional[Dict[str, List]] = None,
                          test_history: Optional[Dict[str, List]] = None,
                          figsize: Tuple[int, int] = (10, 6)):
    """
    Plot training history including loss curves.

    Args:
        history: Training history dictionary with 'loss' and optional 'aux' keys
        output_dir: Directory to save plots
        filename: Base filename for the main loss plot
        component_losses: Whether to plot component losses if available
        component_names: List of names for the component losses
        validation_history: Optional validation loss history
        test_history: Optional test loss history
        figsize: Figure size as (width, height) in inches
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot main loss
    plt.figure(figsize=figsize)
    plt.plot(history['loss'], 'b-', label='Training')

    # Add validation loss if provided
    if validation_history and 'loss' in validation_history:
        plt.plot(validation_history['loss'], 'g-', label='Validation')

    # Add test loss if provided
    if test_history and 'loss' in test_history:
        plt.plot(test_history['loss'], 'r-', label='Test')

    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if validation_history or test_history:
        plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

    # Plot component losses if available and requested
    if component_losses and 'aux' in history and history['aux']:
        # Check if aux is a list of tuples
        if isinstance(history['aux'][0], tuple):
            plt.figure(figsize=figsize)

            # Create a separate line for each component in the aux tuple
            for i in range(len(history['aux'][0])):
                component_values = [aux[i] for aux in history['aux']]

                # Use provided component names or default
                label = component_names[i] if component_names and i < len(component_names) else f'Component {i + 1}'
                plt.plot(component_values, label=label)

            plt.title('Component Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"component_losses.png"))
            plt.close()


def plot_state_predictions(model: Any,
                           datasets: List[Dict],
                           solve_fn: Callable,
                           state_names: List[str],
                           dataset_type: str = "Dataset",
                           output_dir: str = "results",
                           figsize: Tuple[int, int] = (10, 12),
                           labels: Optional[Dict[str, str]] = None):
    """
    Plot predictions against measured data for multiple state variables.

    Args:
        model: Trained model to generate predictions
        datasets: List of datasets containing true values
        solve_fn: Function to solve the model for predictions
        state_names: List of state variable names to plot
        dataset_type: String describing the dataset type (e.g., "Training", "Test")
        output_dir: Directory to save plots
        figsize: Figure size as (width, height) in inches
        labels: Optional dictionary mapping state names to display labels
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use default labels if none provided
    if labels is None:
        labels = {state: state for state in state_names}

    # Plot predictions for each dataset
    for i, dataset in enumerate(datasets):
        solution = solve_fn(model, dataset)

        # Create a figure with subplots for each state
        fig, axs = plt.subplots(len(state_names), 1, figsize=figsize)

        # Handle the case where there's only one state (axs is not a list)
        if len(state_names) == 1:
            axs = [axs]

        # Plot each state
        for j, state in enumerate(state_names):
            true_key = f"{state}_true"
            if true_key in dataset and state in solution:
                axs[j].plot(dataset['times'], dataset[true_key], 'bo-', label='Measured')
                axs[j].plot(solution['times'], solution[state], 'r-', label='Predicted')
                axs[j].set_title(f'{dataset_type} {i + 1}: {labels.get(state, state)}')
                axs[j].set_xlabel('Time')
                axs[j].set_ylabel(labels.get(state, state))
                axs[j].legend()
                axs[j].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_type.lower()}_dataset_{i + 1}_predictions.png'))
        plt.close()


def plot_all_results(model: Any,
                     train_datasets: List[Dict],
                     test_datasets: List[Dict],
                     history: Dict,
                     solve_fn: Callable,
                     state_names: List[str],
                     output_dir: str = "results",
                     state_labels: Optional[Dict[str, str]] = None,
                     component_names: Optional[List[str]] = None,
                     validation_history: Optional[Dict] = None,
                     test_history: Optional[Dict] = None,
                     include_parity_plots: bool = True):
    """
    Plot training history and predictions for both training and test datasets.

    Args:
        model: Trained model
        train_datasets: List of training datasets
        test_datasets: List of test datasets
        history: Training history dictionary
        solve_fn: Function to solve the model for predictions
        state_names: List of state variable names to plot
        output_dir: Directory to save plots
        state_labels: Optional dictionary mapping state names to display labels
        component_names: Optional list of names for the component losses
        validation_history: Optional validation loss history
        test_history: Optional test loss history
        include_parity_plots: Whether to include parity plots
    """
    # Plot training history
    plot_training_history(
        history,
        output_dir,
        component_names=component_names,
        validation_history=validation_history,
        test_history=test_history
    )

    # Plot predictions for training datasets
    plot_state_predictions(
        model,
        train_datasets,
        solve_fn,
        state_names,
        "Training",
        output_dir,
        labels=state_labels
    )

    # Plot predictions for test datasets if available
    if test_datasets:
        plot_state_predictions(
            model,
            test_datasets,
            solve_fn,
            state_names,
            "Test",
            output_dir,
            labels=state_labels
        )

    # Plot parity plots if requested
    if include_parity_plots:
        # Plot separate parity plots for training and test data
        plot_parity(
            model,
            train_datasets,
            solve_fn,
            state_names,
            dataset_type="Training",
            output_dir=output_dir,
            labels=state_labels
        )

        if test_datasets:
            plot_parity(
                model,
                test_datasets,
                solve_fn,
                state_names,
                dataset_type="Test",
                output_dir=output_dir,
                labels=state_labels
            )

        # Plot combined parity plots
        plot_combined_parity(
            model,
            train_datasets,
            test_datasets,
            solve_fn,
            state_names,
            output_dir=output_dir,
            labels=state_labels
        )