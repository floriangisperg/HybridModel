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


def evaluate_model_performance(
    model: Any,
    datasets: List[Dict],
    solve_fn: Callable,
    state_names: Optional[List[str]] = None,
    dataset_type: str = "Dataset",
    verbose: bool = True,
    save_metrics: bool = False,
    output_dir: str = "results",
    metrics_filename: str = "model_metrics.txt",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate model performance on multiple datasets and state variables.
    Calculates per-dataset metrics and overall metrics based on all data points combined.

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
        Nested dictionary containing per-dataset metrics and overall metrics.
        The 'overall' key contains metrics calculated on concatenated data.
    """
    evaluation_results = {}
    all_predictions_by_state: Dict[str, List[jnp.ndarray]] = {}
    all_true_values_by_state: Dict[str, List[jnp.ndarray]] = {}

    # --- Determine state names if not provided ---
    if state_names is None:
        state_names = set()
        for dataset in datasets:
            for key in dataset:
                if key.endswith("_true"):
                    state_name = key[:-5]
                    state_names.add(state_name)
        state_names = sorted(list(state_names))  # Ensure consistent order
        print(f"Discovered state names for evaluation: {state_names}")

    # Initialize collection dictionaries
    for state in state_names:
        all_predictions_by_state[state] = []
        all_true_values_by_state[state] = []

    # --- Setup Saving ---
    metrics_file = None
    if save_metrics:
        os.makedirs(output_dir, exist_ok=True)
        metrics_filepath = os.path.join(output_dir, metrics_filename)
        metrics_file = open(metrics_filepath, "w")
        metrics_file.write(f"=== {dataset_type} Evaluation Metrics ===\n\n")
        print(f"Saving metrics to: {metrics_filepath}")

    # --- Evaluate each dataset ---
    for i, dataset in enumerate(datasets):
        dataset_key = f"{dataset_type}_{i+1}"  # More informative key
        if verbose:
            print(f"Evaluating {dataset_key}...")

        try:
            # Get predictions
            solution = solve_fn(model, dataset)
        except Exception as e:
            print(f"ERROR solving model for {dataset_key}: {e}")
            if metrics_file:
                metrics_file.write(f"{dataset_key}: ERROR - Could not solve model.\n\n")
            continue  # Skip to the next dataset

        dataset_metrics = {}
        if metrics_file:
            metrics_file.write(f"{dataset_key}:\n")

        for state_name in state_names:
            true_key = f"{state_name}_true"
            if true_key in dataset and state_name in solution:
                y_true = dataset[true_key]
                y_pred = solution[state_name]

                # Ensure consistent shapes (sometimes solvers might return slightly different lengths)
                min_len = min(len(y_true), len(y_pred))
                if len(y_true) != len(y_pred):
                    print(
                        f"Warning: Mismatched lengths for {state_name} in {dataset_key} ({len(y_true)} vs {len(y_pred)}). Truncating."
                    )
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

                if len(y_true) == 0:
                    print(
                        f"Warning: No overlapping data points for {state_name} in {dataset_key}."
                    )
                    continue  # Skip metric calculation if no points

                # Calculate metrics for this specific dataset
                try:
                    state_metrics = calculate_metrics(y_true, y_pred)
                    dataset_metrics[state_name] = state_metrics

                    # Store data for overall calculation
                    all_true_values_by_state[state_name].append(y_true)
                    all_predictions_by_state[state_name].append(y_pred)

                    # Print/Save per-dataset results
                    if verbose:
                        print(
                            f"  {dataset_key} - {state_name}: "
                            f"R²={state_metrics.get('r2', float('nan')):.4f}, "
                            f"RMSE={state_metrics.get('rmse', float('nan')):.4f}, "
                            f"MAE={state_metrics.get('mae', float('nan')):.4f}"
                        )
                    if metrics_file:
                        metrics_file.write(f"  {state_name}:\n")
                        for metric_name, value in state_metrics.items():
                            metrics_file.write(f"    {metric_name}: {value:.6f}\n")

                except Exception as calc_e:
                    print(
                        f"ERROR calculating metrics for {state_name} in {dataset_key}: {calc_e}"
                    )
                    if metrics_file:
                        metrics_file.write(
                            f"  {state_name}: ERROR calculating metrics.\n"
                        )

            elif true_key not in dataset:
                if verbose:
                    print(
                        f"  {dataset_key}: Skipping {state_name} (no '{true_key}' found)"
                    )
            elif state_name not in solution:
                if verbose:
                    print(
                        f"  {dataset_key}: Skipping {state_name} (not found in model solution)"
                    )

        evaluation_results[dataset_key] = dataset_metrics
        if metrics_file:
            metrics_file.write("\n")

    # --- Calculate OVERALL metrics based on concatenated data ---
    overall_metrics = {}
    print("\nCalculating overall metrics (all data points combined)...")
    if metrics_file:
        metrics_file.write("=== Overall Metrics (All Data Combined) ===\n")

    for state_name in state_names:
        # Concatenate all true values and predictions for this state
        if all_true_values_by_state[state_name]:  # Check if any data was collected
            y_true_all = jnp.concatenate(all_true_values_by_state[state_name])
            y_pred_all = jnp.concatenate(all_predictions_by_state[state_name])

            if len(y_true_all) > 0:
                try:
                    # Calculate metrics once on the combined data
                    state_overall_metrics = calculate_metrics(y_true_all, y_pred_all)
                    overall_metrics[state_name] = state_overall_metrics

                    if verbose:
                        print(
                            f"Overall - {state_name}: "
                            f"R²={state_overall_metrics.get('r2', float('nan')):.4f}, "
                            f"RMSE={state_overall_metrics.get('rmse', float('nan')):.4f}, "
                            f"MAE={state_overall_metrics.get('mae', float('nan')):.4f}"
                        )
                    if metrics_file:
                        metrics_file.write(f"{state_name}:\n")
                        for metric_name, value in state_overall_metrics.items():
                            metrics_file.write(f"  {metric_name}: {value:.6f}\n")
                except Exception as overall_calc_e:
                    print(
                        f"ERROR calculating overall metrics for {state_name}: {overall_calc_e}"
                    )
                    if metrics_file:
                        metrics_file.write(
                            f"{state_name}: ERROR calculating overall metrics.\n"
                        )
            else:
                if verbose:
                    print(
                        f"Overall - {state_name}: No data points to calculate metrics."
                    )
                if metrics_file:
                    metrics_file.write(f"{state_name}: No data points.\n")
        else:
            if verbose:
                print(
                    f"Overall - {state_name}: No data collected for this state across datasets."
                )
            if metrics_file:
                metrics_file.write(f"{state_name}: No data collected.\n")

    # Add overall metrics to the results dictionary
    evaluation_results["overall"] = overall_metrics

    # --- Cleanup Saving ---
    if metrics_file:
        metrics_file.write("\n")
        # Add timestamp
        from datetime import datetime

        metrics_file.write(
            f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        metrics_file.close()
        if verbose:
            print(f"Metrics saved successfully.")

    return evaluation_results


def create_metrics_summary(
    evaluation: Dict[str, Dict[str, Dict[str, float]]], format_type: str = "dataframe"
) -> Union[pd.DataFrame, Dict]:
    """
    Create a summary of evaluation metrics in different formats.

    Args:
        evaluation: Nested dictionary of evaluation metrics
        format_type: Output format ('dataframe', 'dict', or 'flat_dict')

    Returns:
        Summary of metrics in the specified format
    """
    if format_type == "dataframe":
        # Create a list of records for DataFrame
        records = []

        for dataset_key, dataset_metrics in evaluation.items():
            for state_name, metrics in dataset_metrics.items():
                record = {"Dataset": dataset_key, "State": state_name}
                # Add all metrics
                record.update(metrics)
                records.append(record)

        # Convert to DataFrame
        return pd.DataFrame(records)

    elif format_type == "flat_dict":
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


def compare_models(
    models: List[Any],
    model_names: List[str],
    datasets: List[Dict],
    solve_fn: Callable,
    state_names: Optional[List[str]] = None,
    dataset_type: str = "Dataset",
) -> pd.DataFrame:
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
        if "aggregate" in evaluation:
            results = evaluation["aggregate"]
        else:
            dataset_key = next(iter(evaluation))
            results = evaluation[dataset_key]

        # Create records for each state variable
        for state_name, metrics in results.items():
            record = {"Model": model_name, "State": state_name}
            # Add metrics
            record.update(metrics)
            comparison_records.append(record)

    # Convert to DataFrame
    return pd.DataFrame(comparison_records)
