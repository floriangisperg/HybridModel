import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any
from jaxtyping import Array, Float, PyTree


def normalize_data(data: Dict[str, Float[Array, "..."]]) -> Tuple[Dict[str, Float[Array, "..."]], Dict[str, float]]:
    """
    Normalize data using standard scaling.

    Args:
        data: Dictionary of data arrays

    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    normalized_data = {}
    norm_params = {}

    for key, values in data.items():
        # Calculate mean and std
        mean_val = float(jnp.mean(values))
        std_val = float(jnp.std(values))

        # Store normalization parameters
        norm_params[f"{key}_mean"] = mean_val
        norm_params[f"{key}_std"] = max(std_val, 1e-8)  # Avoid division by zero

        # Normalize data
        normalized_data[key] = (values - mean_val) / norm_params[f"{key}_std"]

    return normalized_data, norm_params


def combine_normalization_params(params_list: List[Dict]) -> Dict:
    """
    Combine normalization parameters from multiple datasets.

    Args:
        params_list: List of normalization parameter dictionaries

    Returns:
        Combined normalization parameters
    """
    combined_params = {}

    # Get all unique keys
    all_keys = set()
    for params in params_list:
        all_keys.update(params.keys())

    # Average parameters with the same keys
    for key in all_keys:
        values = [params[key] for params in params_list if key in params]
        if values:
            combined_params[key] = sum(values) / len(values)

    return combined_params


def calculate_rate(times: Float[Array, "N"], values: Float[Array, "N"]) -> Float[Array, "N"]:
    """
    Calculate rate of change (derivative) of values.

    Args:
        times: Time points
        values: Values at those time points

    Returns:
        Array of rates of change
    """
    # Initialize rates array
    rates = jnp.zeros_like(values)

    # Calculate rates using forward differences
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if dt > 0:
            rates = rates.at[i].set((values[i + 1] - values[i]) / dt)

    # For the last point, use the previous rate
    rates = rates.at[-1].set(rates[-2] if len(rates) > 1 else 0.0)

    return rates


def create_initial_random_key(seed: int = 0) -> Array:
    """
    Create an initial random key for JAX.

    Args:
        seed: Random seed

    Returns:
        JAX random key
    """
    return jax.random.PRNGKey(seed)