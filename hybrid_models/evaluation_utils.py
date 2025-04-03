"""
Evaluation utilities for hybrid models.

This module extends the basic evaluation functionality with reporting, aggregation,
and model comparison capabilities.
"""
import os

import jax.numpy as jnp
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from .evaluation import calculate_metrics
import pandas as pd
import numpy as np


def evaluate_model_performance(model: Any,
                              datasets: List[Dict],
                              solve_fn: Callable,
                              state_names: Optional[List[str]] = None,
                              dataset_type: str = "Dataset",
                              verbose: bool = True,
                              save_metrics: bool = False,
                              output_dir: str = "results",
                              metrics_filename: str = "model_metrics.txt") -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate model performance on multiple datasets and state variables.

    Args:
        model: The model to evaluate
        datasets: List of datasets for evaluation
        solve_fn: Function to solve the model and get predictions
        state_names: Optional list of state variables to evaluate (if None, discovers from datasets)
        dataset_type: String to identify dataset type in output
        verbose: Whether to print evaluation results
        save_metrics: Whether to save metrics to a text file
        output_dir: Directory to save metrics
        metrics_filename: Filename for the metrics text file

    Returns:
        Nested dictionary of evaluation metrics by dataset and state variable
    """
    evaluation = {}

    # Create output directory if saving metrics
    if save_metrics:
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = open(os.path.join(output_dir, metrics_filename), 'w')
        metrics_file.write(f"=== {dataset_type} Evaluation Metrics ===\n\n")

    # If state_names not provided, discover from datasets
    if state_names is None:
        state_names = []
        for dataset in datasets:
            for key in dataset:
                if key.endswith('_true'):
                    state_name = key[:-5]  # Remove '_true' suffix
                    if state_name not in state_names:
                        state_names.append(state_name)

    # Evaluate each dataset
    for i, dataset in enumerate(datasets):
        # Get predictions
        solution = solve_fn(model, dataset)

        # Calculate metrics for each state
        dataset_metrics = {}

        # Write dataset header to file if saving
        if save_metrics:
            metrics_file.write(f"{dataset_type} Dataset {i+1}:\n")

        for state_name in state_names:
            true_key = f"{state_name}_true"
            if true_key in dataset and state_name in solution:
                y_true = dataset[true_key]
                y_pred = solution[state_name]

                # Calculate metrics
                state_metrics = calculate_metrics(y_true, y_pred)
                dataset_metrics[state_name] = state_metrics

                # Print results if verbose
                if verbose:
                    print(f"{dataset_type} {i+1} - {state_name}: "
                         f"R²: {state_metrics['r2']:.4f}, "
                         f"RMSE: {state_metrics['rmse']:.4f}, "
                         f"MAE: {state_metrics['mae']:.4f}")

                # Write to file if saving
                if save_metrics:
                    metrics_file.write(f"  {state_name}:\n")
                    for metric_name, value in state_metrics.items():
                        metrics_file.write(f"    {metric_name}: {value:.6f}\n")

        # Store metrics for this dataset
        evaluation[f"{dataset_type}_{i}"] = dataset_metrics

        # Add spacing in file
        if save_metrics:
            metrics_file.write("\n")

    # Calculate aggregate metrics across all datasets
    if len(datasets) > 1:
        aggregate_metrics = aggregate_evaluation_results(evaluation)
        evaluation['aggregate'] = aggregate_metrics

        if verbose:
            print("\nAggregate metrics:")
            for state_name, metrics in aggregate_metrics.items():
                print(f"{state_name}: R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

        # Write aggregate metrics to file
        if save_metrics:
            metrics_file.write("=== Aggregate Metrics ===\n")
            for state_name, metrics in aggregate_metrics.items():
                metrics_file.write(f"{state_name}:\n")
                for metric_name, value in metrics.items():
                    metrics_file.write(f"  {metric_name}: {value:.6f}\n")
            metrics_file.write("\n")

    # Close file if saving
    if save_metrics:
        # Add timestamp
        from datetime import datetime
        metrics_file.write(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        metrics_file.close()
        if verbose:
            print(f"Metrics saved to {os.path.join(output_dir, metrics_filename)}")

    return evaluation


def aggregate_evaluation_results(evaluation: Dict[str, Dict[str, Dict[str, float]]],
                               method: str = 'mean') -> Dict[str, Dict[str, float]]:
    """
    Aggregate evaluation results across multiple datasets.

    Args:
        evaluation: Nested dictionary of evaluation metrics
        method: Aggregation method ('mean', 'median', or 'weighted')

    Returns:
        Dictionary of aggregated metrics by state variable
    """
    # Collect all metrics by state variable
    metrics_by_state = {}

    for dataset_key, dataset_metrics in evaluation.items():
        # Skip the aggregate key if it exists
        if dataset_key == 'aggregate':
            continue

        for state_name, metrics in dataset_metrics.items():
            if state_name not in metrics_by_state:
                metrics_by_state[state_name] = {}

            for metric_name, value in metrics.items():
                if metric_name not in metrics_by_state[state_name]:
                    metrics_by_state[state_name][metric_name] = []

                metrics_by_state[state_name][metric_name].append(value)

    # Aggregate metrics
    aggregated = {}
    for state_name, metrics in metrics_by_state.items():
        aggregated[state_name] = {}

        for metric_name, values in metrics.items():
            if method == 'mean':
                aggregated[state_name][metric_name] = float(np.mean(values))
            elif method == 'median':
                aggregated[state_name][metric_name] = float(np.median(values))
            else:
                # Default to mean
                aggregated[state_name][metric_name] = float(np.mean(values))

    return aggregated


def create_metrics_summary(evaluation: Dict[str, Dict[str, Dict[str, float]]],
                          format_type: str = 'dataframe') -> Union[pd.DataFrame, Dict]:
    """
    Create a summary of evaluation metrics in different formats.

    Args:
        evaluation: Nested dictionary of evaluation metrics
        format_type: Output format ('dataframe', 'dict', or 'flat_dict')

    Returns:
        Summary of metrics in the specified format
    """
    if format_type == 'dataframe':
        # Create a list of records for DataFrame
        records = []

        for dataset_key, dataset_metrics in evaluation.items():
            for state_name, metrics in dataset_metrics.items():
                record = {
                    'Dataset': dataset_key,
                    'State': state_name
                }
                # Add all metrics
                record.update(metrics)
                records.append(record)

        # Convert to DataFrame
        return pd.DataFrame(records)

    elif format_type == 'flat_dict':
        # Create a flattened dictionary
        flat_dict = {}

        for dataset_key, dataset_metrics in evaluation.items():
            for state_name, metrics in dataset_metrics.items():
                for metric_name, value in metrics.items():
                    key = f"{dataset_key}.{state_name}.{metric_name}"
                    flat_dict[key] = value

        return flat_dict

    else:
        # Return the original nested dictionary
        return evaluation


def compare_models(models: List[Any],
                  model_names: List[str],
                  datasets: List[Dict],
                  solve_fn: Callable,
                  state_names: Optional[List[str]] = None,
                  dataset_type: str = "Dataset") -> pd.DataFrame:
    """
    Compare multiple models on the same datasets.

    Args:
        models: List of models to compare
        model_names: List of names for each model
        datasets: List of datasets for evaluation
        solve_fn: Function to solve models and get predictions
        state_names: Optional list of state variables to evaluate
        dataset_type: String to identify dataset type

    Returns:
        DataFrame with comparison metrics
    """
    comparison_records = []

    # Evaluate each model
    for model, model_name in zip(models, model_names):
        evaluation = evaluate_model_performance(
            model, datasets, solve_fn, state_names, dataset_type, verbose=False
        )

        # Use aggregate results if available, otherwise use first dataset
        if 'aggregate' in evaluation:
            results = evaluation['aggregate']
        else:
            dataset_key = next(iter(evaluation))
            results = evaluation[dataset_key]

        # Create records for each state variable
        for state_name, metrics in results.items():
            record = {
                'Model': model_name,
                'State': state_name
            }
            # Add metrics
            record.update(metrics)
            comparison_records.append(record)

    # Convert to DataFrame
    return pd.DataFrame(comparison_records)