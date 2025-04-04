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
                label = component_names[i] if component_names and i < len(component_names) else f'Component {i+1}'
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
                axs[j].set_title(f'{dataset_type} {i+1}: {labels.get(state, state)}')
                axs[j].set_xlabel('Time')
                axs[j].set_ylabel(labels.get(state, state))
                axs[j].legend()
                axs[j].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_type.lower()}_dataset_{i+1}_predictions.png'))
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
                    test_history: Optional[Dict] = None):
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