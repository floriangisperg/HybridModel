#!/usr/bin/env python
"""Benchmark specific optimizations for hybrid models."""
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import jax
import jax.numpy as jnp

# Import from your framework
from hybrid_models import (
    HybridModelBuilder,
    train_hybrid_model,
    ConfigurableNN
)

# Import optimization modules
from hybrid_models.parallel_processing import (
    parallel_loss_function,
    parallel_dataset_loss,
    process_dataset_with_timing
)
from hybrid_models.optimized_nn import (
    optimize_nn_components,
    create_optimized_ode_function
)

# Import bioprocess example
from bioprocess import (
    load_bioprocess_data,
    define_bioprocess_model,
    prepare_bioprocess_dataset,
    bioprocess_loss_function
)


def benchmark_nn_forward_pass(model, inputs, n_repeats=1000):
    """Benchmark neural network forward pass speed."""
    # Extract a neural network
    if not model.nn_replacements:
        print("No neural networks found in model")
        return None, None

    # Get the first neural network
    nn_name = list(model.nn_replacements.keys())[0]
    nn = model.nn_replacements[nn_name]

    # Create optimized version
    from hybrid_models.optimized_nn import JITOptimizedNN
    optimized_nn = JITOptimizedNN(nn)

    # Benchmark original
    start_time = time.time()
    for _ in range(n_repeats):
        _ = nn(inputs)
    original_time = time.time() - start_time

    # Benchmark optimized
    # First call to initialize JIT compilation
    _ = optimized_nn(inputs)

    start_time = time.time()
    for _ in range(n_repeats):
        _ = optimized_nn(inputs)
    optimized_time = time.time() - start_time

    # Calculate speedup
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')

    print(f"Neural Network Forward Pass ({n_repeats} iterations):")
    print(f"  Original: {original_time:.4f} seconds")
    print(f"  Optimized: {optimized_time:.4f} seconds")
    print(f"  Speedup: {speedup:.2f}x")

    return original_time, optimized_time


def benchmark_parallel_loss(model, datasets, n_repeats=10):
    """Benchmark parallel loss calculation."""

    # Sequential loss
    def sequential_loss(model, datasets):
        total_loss = 0.0
        for dataset in datasets:
            loss = process_dataset_with_timing(model, dataset, profile=False)
            if loss is not None:
                total_loss += loss
        return total_loss / max(1, len(datasets))

    # Benchmark sequential
    start_time = time.time()
    for _ in range(n_repeats):
        sequential_loss(model, datasets)
    sequential_time = time.time() - start_time

    # Benchmark parallel
    start_time = time.time()
    for _ in range(n_repeats):
        parallel_dataset_loss(
            model=model,
            datasets=datasets,
            process_dataset=lambda m, d: process_dataset_with_timing(m, d, profile=False)
        )
    parallel_time = time.time() - start_time

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')

    print(f"Loss Calculation ({n_repeats} iterations across {len(datasets)} datasets):")
    print(f"  Sequential: {sequential_time:.4f} seconds")
    print(f"  Parallel: {parallel_time:.4f} seconds")
    print(f"  Speedup: {speedup:.2f}x")

    return sequential_time, parallel_time


def benchmark_ode_function(model, dataset, n_repeats=100):
    """Benchmark ODE function performance."""
    # Create test inputs
    t = 0.0
    y = jnp.array([dataset['initial_state'][name] for name in model.state_names])
    args = {'time_dependent_inputs': dataset['time_dependent_inputs']}

    # Get original ODE function
    original_ode_function = model.ode_function

    # Create optimized ODE function
    optimized_ode_function = create_optimized_ode_function(model)

    # Benchmark original
    start_time = time.time()
    for _ in range(n_repeats):
        _ = original_ode_function(t, y, args)
    original_time = time.time() - start_time

    # Benchmark optimized
    start_time = time.time()
    for _ in range(n_repeats):
        _ = optimized_ode_function(t, y, args)
    optimized_time = time.time() - start_time

    # Calculate speedup
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')

    print(f"ODE Function Evaluation ({n_repeats} iterations):")
    print(f"  Original: {original_time:.4f} seconds")
    print(f"  Optimized: {optimized_time:.4f} seconds")
    print(f"  Speedup: {speedup:.2f}x")

    return original_time, optimized_time


def benchmark_real_model(data_file='Train_data_masked.xlsx', max_runs=2):
    """Benchmark with the real bioprocess model."""
    print("\nBenchmarking with real bioprocess model:")

    # Load data
    try:
        runs = load_bioprocess_data(data_file, max_runs=max_runs)
        datasets = prepare_bioprocess_dataset(runs)
        model = define_bioprocess_model(runs[0]['norm_params'])

        # Verify model structure
        print(f"Model states: {model.state_names}")
        print(f"Neural networks: {list(model.nn_replacements.keys())}")

        # Profile model inference first
        example_dataset = datasets[0]

        # Run each benchmark
        nn_times = benchmark_nn_forward_pass(
            model=model,
            inputs={
                'X': 1.0,
                'P': 0.5,
                'temp': 37.0,
                'feed': 0.1,
                'inductor_mass': 10.0,
                'inductor_switch': 1.0
            }
        )

        ode_times = benchmark_ode_function(
            model=model,
            dataset=example_dataset
        )

        parallel_times = benchmark_parallel_loss(
            model=model,
            datasets=datasets,
            n_repeats=3  # Lower number for real model
        )

        # Return results
        return {
            'nn_times': nn_times,
            'ode_times': ode_times,
            'parallel_times': parallel_times
        }

    except Exception as e:
        print(f"Error benchmarking real model: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_benchmark_results(results, output_dir="optimization_benchmarks"):
    """Plot benchmark results."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Plot NN forward pass results
    if 'nn_times' in results and results['nn_times'][0] is not None:
        orig_time, opt_time = results['nn_times']

        plt.figure(figsize=(10, 6))
        labels = ['Original', 'JIT-Optimized']
        times = [orig_time, opt_time]
        plt.bar(labels, times, color=['blue', 'green'])
        plt.title('Neural Network Forward Pass Performance')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add speedup annotations
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        plt.text(1, opt_time + 0.05 * max(times),
                 f"{speedup:.2f}x faster",
                 ha='center', va='bottom')

        plt.savefig(f"{output_dir}/nn_forward_performance.png")
        plt.close()

    # Plot ODE function results
    if 'ode_times' in results and results['ode_times'][0] is not None:
        orig_time, opt_time = results['ode_times']

        plt.figure(figsize=(10, 6))
        labels = ['Original', 'Optimized']
        times = [orig_time, opt_time]
        plt.bar(labels, times, color=['blue', 'green'])
        plt.title('ODE Function Performance')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add speedup annotations
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        plt.text(1, opt_time + 0.05 * max(times),
                 f"{speedup:.2f}x faster",
                 ha='center', va='bottom')

        plt.savefig(f"{output_dir}/ode_function_performance.png")
        plt.close()

    # Plot parallel loss results
    if 'parallel_times' in results and results['parallel_times'][0] is not None:
        seq_time, par_time = results['parallel_times']

        plt.figure(figsize=(10, 6))
        labels = ['Sequential', 'Parallel']
        times = [seq_time, par_time]
        plt.bar(labels, times, color=['blue', 'green'])
        plt.title('Loss Calculation Performance')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add speedup annotations
        speedup = seq_time / par_time if par_time > 0 else float('inf')
        plt.text(1, par_time + 0.05 * max(times),
                 f"{speedup:.2f}x faster",
                 ha='center', va='bottom')

        plt.savefig(f"{output_dir}/parallel_loss_performance.png")
        plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Benchmark specific hybrid model optimizations')
    parser.add_argument('--real', action='store_true', help='Benchmark with real bioprocess model')
    parser.add_argument('--output', default='optimization_benchmarks', help='Output directory')
    args = parser.parse_args()

    # Print JAX information
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Number of devices: {jax.device_count()}")

    # Run benchmarks
    if args.real:
        results = benchmark_real_model()
    else:
        print("No benchmark type specified.")
        return

    # Plot results
    if results:
        plot_benchmark_results(results, args.output)
        print(f"\nBenchmark results saved to {args.output}/")


if __name__ == "__main__":
    main()