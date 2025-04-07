"""
Model persistence utilities for hybrid models.

This module provides functionality to save and load hybrid models and their configurations.
"""
import os
import json
import pickle
from typing import Dict, Any, Tuple, Optional
import equinox as eqx
import jax
import jax.numpy as jnp


def save_model(model, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save a hybrid model to a file.

    Args:
        model: The hybrid model to save
        filepath: Path to save the model
        metadata: Optional dictionary with additional model metadata

    Returns:
        Path to the saved model file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Prepare metadata
    if metadata is None:
        metadata = {}

    # Use equinox serialization
    eqx.tree_serialise_leaves(filepath, model)

    # Save metadata separately if provided
    if metadata:
        metadata_path = f"{filepath}.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    return filepath


def load_model(filepath: str, model_template=None) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Load a hybrid model from a file.

    Args:
        filepath: Path to the saved model
        model_template: Optional template model with the same structure

    Returns:
        Tuple of (loaded_model, metadata_dict)
    """
    # Load model using equinox
    model = eqx.tree_deserialise_leaves(filepath, model_template)

    # Try to load metadata if it exists
    metadata = None
    metadata_path = f"{filepath}.meta.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    return model, metadata


def save_training_results(output_dir: str, model,
                          training_history: Dict,
                          model_config: Dict,
                          norm_params: Dict,
                          solver_config: Dict = None,
                          metrics: Dict = None) -> Dict[str, str]:
    """
    Save complete training results including model, history, and configurations.

    Args:
        output_dir: Directory to save all files
        model: Trained model
        training_history: Dictionary with training history
        model_config: Model configuration
        norm_params: Normalization parameters
        solver_config: Optional solver configuration
        metrics: Optional evaluation metrics

    Returns:
        Dictionary with paths to all saved files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "model.eqx")
    metadata = {
        "model_config": model_config,
        "norm_params": norm_params
    }
    save_model(model, model_path, metadata)

    # Save training history
    history_path = os.path.join(output_dir, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(training_history, f)

    # Save normalization parameters
    norm_path = os.path.join(output_dir, "normalization_params.json")
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f, indent=2)

    # Save other configurations if provided
    result_paths = {
        "model": model_path,
        "history": history_path,
        "norm_params": norm_path
    }

    if solver_config:
        solver_path = os.path.join(output_dir, "solver_config.json")
        with open(solver_path, 'w') as f:
            json.dump(solver_config, f, indent=2)
        result_paths["solver_config"] = solver_path

    if metrics:
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        result_paths["metrics"] = metrics_path

    return result_paths


def load_training_results(output_dir: str, model_template=None) -> Dict[str, Any]:
    """
    Load complete training results.

    Args:
        output_dir: Directory with saved files
        model_template: Optional template model with the same structure

    Returns:
        Dictionary with loaded results
    """
    results = {}

    # Load model and metadata
    model_path = os.path.join(output_dir, "model.eqx")
    if os.path.exists(model_path):
        model, metadata = load_model(model_path, model_template)
        results["model"] = model
        results["metadata"] = metadata

    # Load training history
    history_path = os.path.join(output_dir, "training_history.pkl")
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            results["training_history"] = pickle.load(f)

    # Load normalization parameters
    norm_path = os.path.join(output_dir, "normalization_params.json")
    if os.path.exists(norm_path):
        with open(norm_path, 'r') as f:
            results["norm_params"] = json.load(f)

    # Load solver configuration
    solver_path = os.path.join(output_dir, "solver_config.json")
    if os.path.exists(solver_path):
        with open(solver_path, 'r') as f:
            results["solver_config"] = json.load(f)

    # Load evaluation metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results["metrics"] = json.load(f)

    return results