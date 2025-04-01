"""Utilities for profiling and performance optimization."""
import time
import functools
import jax
import jax.numpy as jnp
from typing import Dict, List, Callable, Any, Optional, Tuple


class TimingStats:
    """Class to collect and analyze timing statistics."""

    def __init__(self):
        self.times = {}
        self.call_counts = {}

    def add_timing(self, name: str, duration: float):
        """Add a timing measurement."""
        if name not in self.times:
            self.times[name] = []
            self.call_counts[name] = 0

        self.times[name].append(duration)
        self.call_counts[name] += 1

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all timing measurements."""
        summary = {}

        for name, times in self.times.items():
            times_array = jnp.array(times)
            summary[name] = {
                'mean': float(jnp.mean(times_array)),
                'median': float(jnp.median(times_array)),
                'min': float(jnp.min(times_array)),
                'max': float(jnp.max(times_array)),
                'total': float(jnp.sum(times_array)),
                'calls': self.call_counts[name]
            }

        return summary

    def print_summary(self):
        """Print summary statistics for all timing measurements."""
        summary = self.get_summary()

        print("\n--- Timing Summary ---")
        print(f"{'Function':<25} {'Calls':<10} {'Total (s)':<12} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 85)

        # Sort by total time
        for name, stats in sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True):
            print(f"{name:<25} {stats['calls']:<10} {stats['total']:<12.4f} {stats['mean'] * 1000:<12.2f} "
                  f"{stats['min'] * 1000:<12.2f} {stats['max'] * 1000:<12.2f}")


# Create a global timing stats object
global_timing_stats = TimingStats()


def timed(name: Optional[str] = None):
    """Decorator to time a function."""

    def decorator(func):
        func_name = name if name else func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            global_timing_stats.add_timing(func_name, duration)
            return result

        return wrapper

    return decorator


def profile_model(model: Any, datasets: List[Dict], solve_fn: Callable, n_repeats: int = 3) -> Tuple[Dict, TimingStats]:
    """
    Profile the performance of a model on given datasets.

    Args:
        model: The model to profile
        datasets: List of datasets to use for profiling
        solve_fn: Function to solve the model (typically model.solve or a wrapper)
        n_repeats: Number of times to repeat the profiling

    Returns:
        Tuple of (results, timing_stats)
    """
    # Create a new timing stats object for this profiling session
    timing_stats = TimingStats()

    # Define timed solve function
    @timed("model_solve")
    def timed_solve(model, dataset):
        return solve_fn(model, dataset)

    results = {}

    # Profile each dataset
    for i, dataset in enumerate(datasets):
        dataset_results = []

        # Perform multiple runs for consistent timing
        for repeat in range(n_repeats):
            # Clear JAX compilation cache for clean measurement of first compile
            if repeat == 0:
                # We can't directly clear JAX's cache, but we can force recompilation
                # by using a new model object with same parameters
                if hasattr(model, 'state_names'):
                    # For HybridODESystem
                    profiled_model = type(model)(
                        mechanistic_components=model.mechanistic_components,
                        nn_replacements=model.nn_replacements,
                        state_names=model.state_names
                    )
                else:
                    # For other models, use the original
                    profiled_model = model
            else:
                profiled_model = model

            # Time the solve function
            start_time = time.time()
            solution = timed_solve(profiled_model, dataset)
            duration = time.time() - start_time

            # Record timing
            timing_stats.add_timing(f"dataset_{i}_total", duration)

            # Store results
            dataset_results.append({
                'solution': solution,
                'duration': duration
            })

        # Store average results
        results[f"dataset_{i}"] = {
            'mean_duration': sum(r['duration'] for r in dataset_results) / len(dataset_results),
            'solution': dataset_results[0]['solution']  # Just store the first solution
        }

    return results, timing_stats


def optimize_jit_ode_system(ode_system: Any) -> Any:
    """
    Optimize JIT compilation for a HybridODESystem by explicitly JIT-compiling
    the ODE function with appropriate static arguments.

    Args:
        ode_system: The HybridODESystem instance to optimize

    Returns:
        Optimized HybridODESystem instance
    """
    # Get the original ode_function
    original_ode_function = ode_system.ode_function

    # Create a JIT-compiled version with static_argnums for the args dictionary
    @functools.partial(jax.jit, static_argnames=['args'])
    def jitted_ode_function(t, y, args):
        return original_ode_function(t, y, args)

    # Replace the ODE function with the JIT-compiled version
    ode_system.ode_function = jitted_ode_function

    return ode_system


def optimize_batch_processing(model: Any, datasets: List[Dict]) -> Callable:
    """
    Create an optimized batch processing function for multiple datasets.

    Args:
        model: The model to optimize
        datasets: List of datasets to process

    Returns:
        A function that processes all datasets in an optimized batch
    """

    # Define a function to process a single dataset
    def process_dataset(model, dataset):
        solution = model.solve(
            initial_state=dataset['initial_state'],
            t_span=(dataset['times'][0], dataset['times'][-1]),
            evaluation_times=dataset['times'],
            args={'time_dependent_inputs': dataset.get('time_dependent_inputs', {})}
        )
        return solution

    # JIT-compile the function
    jitted_process = jax.jit(process_dataset, static_argnums=(0, 1))

    # Define a function to process all datasets in sequence
    def process_all_datasets():
        results = []
        for dataset in datasets:
            result = jitted_process(model, dataset)
            results.append(result)
        return results

    return process_all_datasets


def get_jax_memory_usage():
    """Get current JAX memory usage statistics."""
    try:
        return jax.live_arrays()
    except:
        # Fallback for older JAX versions
        return {"Memory usage information not available": "Upgrade to a newer JAX version"}