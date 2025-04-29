import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any
from jaxtyping import Array, Float, PyTree


def normalize_data(
    data: Dict[str, Float[Array, "..."]],
) -> Tuple[Dict[str, Float[Array, "..."]], Dict[str, float]]:
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


def calculate_rate(
    times: Float[Array, "N"], values: Float[Array, "N"]
) -> Float[Array, "N"]:
    """
    Calculate rate of change (derivative) of values using forward differences.
    Handles potential non-uniform time steps.

    Args:
        times: Time points (must be sorted).
        values: Values at those time points.

    Returns:
        Array of rates of change, same length as inputs.
    """
    # Ensure float types for division
    times = times.astype(jnp.float64)
    values = values.astype(jnp.float64)

    if len(times) != len(values):
        raise ValueError("Length of times and values must be equal.")
    if len(times) < 2:
        return jnp.zeros_like(values)  # Cannot calculate rate for less than 2 points

    # Calculate differences
    dt = jnp.diff(times)
    dv = jnp.diff(values)

    # Calculate rates, handle zero dt to avoid division by zero
    # Where dt is near zero, forward fill the rate
    rates_diff = jnp.where(dt > 1e-9, dv / dt, 0.0)  # Calculate where possible

    # Pad the rates array to match the original length
    # Use forward fill: repeat the last calculated rate for the last point
    rates = jnp.concatenate([rates_diff, jnp.array([rates_diff[-1]])])

    # Handle cases where the *first* dt was zero (less common if times are sorted)
    rates = jnp.where(jnp.isnan(rates), 0.0, rates)  # Replace potential NaNs with 0

    # Address the issue where the first few dt might be zero
    mask_zero_dt = jnp.concatenate(
        [dt <= 1e-9, jnp.array([False])]
    )  # Mask including last point
    first_valid_rate_index = jnp.argmax(~mask_zero_dt)  # Find first non-zero dt index

    # Forward fill from the first valid rate for initial zero-dt points
    rates = jnp.where(
        (jnp.arange(len(rates)) < first_valid_rate_index) & mask_zero_dt,
        rates[first_valid_rate_index],
        rates,
    )

    return rates


# **** ADDED: Linear interpolation function (using jnp.interp) ****
@jax.jit
def interp_linear(t: Float, xp: Float[Array, "N"], fp: Float[Array, "N"]) -> Float:
    """
    JIT-compatible linear interpolation. Equivalent to jnp.interp.
    Handles extrapolation by repeating boundary values.

    Args:
        t: The time point(s) at which to evaluate.
        xp: The array of time points for the data (must be increasing).
        fp: The array of values corresponding to xp.

    Returns:
        The interpolated value(s) at time t.
    """
    # Ensure t is an array for consistent broadcasting with jnp.interp
    t_arr = jnp.asarray(t)
    # jnp.interp is JIT-compatible and handles extrapolation
    return jnp.interp(t_arr, xp, fp)


def create_initial_random_key(seed: int = 0) -> Array:
    """
    Create an initial random key for JAX.

    Args:
        seed: Random seed

    Returns:
        JAX random key
    """
    return jax.random.PRNGKey(seed)
