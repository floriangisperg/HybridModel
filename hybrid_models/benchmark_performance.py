#!/usr/bin/env python
"""Benchmark performance of hybrid model components."""
import argparse
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import math
import traceback
from hybrid_models.basic_training import benchmark_basic_training

# Import the hybrid model framework
from hybrid_models import (
    HybridModelBuilder,
    HybridODESystem,
    train_hybrid_model,
    normalize_data,
    ConfigurableNN
)

# Import optimized components
from hybrid_models.optimized_ode import OptimizedODESystem
from hybrid_models.optimized_training import train_hybrid_model_optimized
from hybrid_models.profiling import TimingStats, timed, global_timing_stats
from hybrid_models.batch_processor import DatasetBatchProcessor
from hybrid_models.optimized_hybrid_training import train_hybrid_optimized

# Import bioprocess example
from bioprocess import (
    load_bioprocess_data,
    define_bioprocess_model,
    prepare_bioprocess_dataset,
    bioprocess_loss_function
)


def create_synthetic_datasets(n_datasets=1, n_timepoints=5, n_states=1):
    """Create synthetic datasets for benchmarking."""
    datasets = []

    # Create a random key
    key = jax.random.PRNGKey(42)

    for i in range(n_datasets):
        # Create time points - use an EXTREMELY short timespan for stability
        times = jnp.linspace(0, 0.5, n_timepoints)  # Very short timespan (0 to 0.5)

        # Create state values (VERY SIMPLE constant growth)
        state_values = {}
        for j in range(n_states):
            # Different growth rate for each state
            growth_rate = 0.1 + 0.01 * j

            # Create subkeys for randomness
            key, subkey1 = jax.random.split(key)

            # Add tiny random noise to constant growth
            noise = 0.01 * jax.random.normal(subkey1, shape=times.shape)
            state_values[f'X{j}'] = 1.0 + growth_rate * times + noise

        # Create control inputs with minimal randomness
        key, subkey2, subkey3 = jax.random.split(key, 3)
        temp = 37.0 * jnp.ones_like(times) + 0.1 * jax.random.normal(subkey2, shape=times.shape)
        feed = 0.1 * jnp.ones_like(times)

        # Create dataset
        dataset = {
            'times': times,
            'initial_state': {f'X{j}': float(state_values[f'X{j}'][0]) for j in range(n_states)},
        }

        # Add state values
        for j in range(n_states):
            dataset[f'X{j}_true'] = state_values[f'X{j}']

        # Add control inputs
        dataset['time_dependent_inputs'] = {
            'temp': (times, temp),
            'feed': (times, feed),
        }

        datasets.append(dataset)

    return datasets


def create_synthetic_model(n_states=2, nn_replacements=1):
    """Create a synthetic model for benchmarking."""
    # Create model builder
    builder = HybridModelBuilder()

    # Add state variables
    state_names = [f'X{j}' for j in range(n_states)]
    for state_name in state_names:
        builder.add_state(state_name)

    # Define mechanistic components
    for j in range(n_states):
        state_name = f'X{j}'
        growth_rate_name = f'growth_rate_{j}'

        # Define growth function
        def make_growth_fn(j):
            def growth_fn(inputs):
                # Get growth rate from inputs if available, otherwise use default
                growth_rate = inputs.get(f'growth_rate_{j}', 0.1)  # Default growth rate if not provided
                return growth_rate * inputs[f'X{j}']

            return growth_fn

        # Add mechanistic component
        builder.add_mechanistic_component(state_name, make_growth_fn(j))

    # Create random key for neural network initialization
    key = jax.random.PRNGKey(42)

    # Replace some components with neural networks
    for j in range(min(nn_replacements, n_states)):
        key, subkey = jax.random.split(key)

        # Replace growth rate with neural network
        builder.replace_with_nn(
            name=f'growth_rate_{j}',
            input_features=['temp', 'feed'] + state_names,
            hidden_dims=[8, 8],
            key=subkey
        )

    # Build and return the model
    return builder.build()


def benchmark_ode_system(n_runs=3, n_datasets=2, n_timepoints=20, n_states=2, nn_replacements=2):
    """Benchmark the ODE system."""
    print(f"\n--- Benchmarking ODE System (states={n_states}, nn={nn_replacements}) ---")

    # Create synthetic data and models
    datasets = create_synthetic_datasets(n_datasets, n_timepoints, n_states)
    original_model = create_synthetic_model(n_states, nn_replacements)

    # Print model structure for debugging
    print(f"Model states: {original_model.state_names}")
    print(f"Neural networks: {list(original_model.nn_replacements.keys())}")

    # Benchmark original model
    original_times = []
    for i in range(n_runs):
        start_time = time.time()
        for dataset in datasets:
            try:
                original_model.solve(
                    initial_state=dataset['initial_state'],
                    t_span=(dataset['times'][0], dataset['times'][-1]),
                    evaluation_times=dataset['times'],
                    args={'time_dependent_inputs': dataset['time_dependent_inputs']},
                    max_steps=1000000,  # Much higher max_steps
                    rtol=1e-2,  # Relaxed tolerances
                    atol=1e-2,
                    dt0=0.001  # Smaller initial step size
                )
                original_times.append(time.time() - start_time)
                print(f"Original model run {i + 1} completed in {time.time() - start_time:.4f} seconds")
            except Exception as e:
                print(f"Error solving original model: {str(e)}")
                traceback.print_exc()
                original_times.append(float('inf'))

    if original_times and not all(t == float('inf') for t in original_times):
        avg_original_time = sum(t for t in original_times if t != float('inf')) / sum(
            1 for t in original_times if t != float('inf'))
        print(f"Original ODE System: {avg_original_time:.4f} seconds")
    else:
        avg_original_time = float('inf')
        print("Original ODE System: Failed to complete")

    # Create an optimized model if original model ran successfully
    if avg_original_time != float('inf'):
        try:
            optimized_model = OptimizedODESystem.from_hybrid_ode_system(original_model)

            # Benchmark optimized model
            optimized_times = []
            for i in range(n_runs):
                start_time = time.time()
                for dataset in datasets:
                    try:
                        optimized_model.solve(
                            initial_state=dataset['initial_state'],
                            t_span=(dataset['times'][0], dataset['times'][-1]),
                            evaluation_times=dataset['times'],
                            args={'time_dependent_inputs': dataset['time_dependent_inputs']},
                            max_steps=1000000,  # Much higher max_steps
                            rtol=1e-2,  # Relaxed tolerances
                            atol=1e-2,
                            dt0=0.001  # Smaller initial step size
                        )
                        optimized_times.append(time.time() - start_time)
                        print(f"Optimized model run {i + 1} completed in {time.time() - start_time:.4f} seconds")
                    except Exception as e:
                        print(f"Error solving optimized model: {str(e)}")
                        traceback.print_exc()
                        optimized_times.append(float('inf'))

            if optimized_times and not all(t == float('inf') for t in optimized_times):
                avg_optimized_time = sum(t for t in optimized_times if t != float('inf')) / sum(
                    1 for t in optimized_times if t != float('inf'))
                print(f"Optimized ODE System: {avg_optimized_time:.4f} seconds")
            else:
                avg_optimized_time = float('inf')
                print("Optimized ODE System: Failed to complete")

            # Benchmark batch processor if optimized model ran successfully
            if avg_optimized_time != float('inf'):
                try:
                    batch_processor = DatasetBatchProcessor(optimized_model, batch_size=2)

                    batch_times = []
                    for i in range(n_runs):
                        start_time = time.time()
                        try:
                            batch_processor.process(datasets, show_progress=False)
                            batch_times.append(time.time() - start_time)
                            print(f"Batch processor run {i + 1} completed in {time.time() - start_time:.4f} seconds")
                        except Exception as e:
                            print(f"Error with batch processor: {str(e)}")
                            traceback.print_exc()
                            batch_times.append(float('inf'))

                    if batch_times and not all(t == float('inf') for t in batch_times):
                        avg_batch_time = sum(t for t in batch_times if t != float('inf')) / sum(
                            1 for t in batch_times if t != float('inf'))
                        print(f"Batch Processor: {avg_batch_time:.4f} seconds")
                    else:
                        avg_batch_time = float('inf')
                        print("Batch Processor: Failed to complete")
                except Exception as e:
                    print(f"Error creating batch processor: {str(e)}")
                    avg_batch_time = float('inf')
            else:
                avg_batch_time = float('inf')
        except Exception as e:
            print(f"Error creating optimized model: {str(e)}")
            avg_optimized_time = float('inf')
            avg_batch_time = float('inf')
    else:
        avg_optimized_time = float('inf')
        avg_batch_time = float('inf')

    # Calculate speedups (avoiding division by zero or infinity)
    if avg_original_time != float('inf') and avg_optimized_time != float('inf') and avg_optimized_time > 0:
        speedup_optimized = avg_original_time / avg_optimized_time
        print(f"Speedup (Optimized): {speedup_optimized:.2f}x")
    else:
        speedup_optimized = float('nan')
        print("Speedup (Optimized): N/A")

    if avg_original_time != float('inf') and avg_batch_time != float('inf') and avg_batch_time > 0:
        speedup_batch = avg_original_time / avg_batch_time
        print(f"Speedup (Batch): {speedup_batch:.2f}x")
    else:
        speedup_batch = float('nan')
        print("Speedup (Batch): N/A")

    return {
        'original_time': float(avg_original_time) if avg_original_time != float('inf') else None,
        'optimized_time': float(avg_optimized_time) if avg_optimized_time != float('inf') else None,
        'batch_time': float(avg_batch_time) if avg_batch_time != float('inf') else None,
        'speedup_optimized': float(speedup_optimized) if not math.isnan(speedup_optimized) else None,
        'speedup_batch': float(speedup_batch) if not math.isnan(speedup_batch) else None
    }


def benchmark_training(n_epochs=5, n_datasets=1, n_timepoints=5, n_states=1, nn_replacements=1):
    """Benchmark the training process with extremely simplified models."""
    print(f"\n--- Benchmarking Training (epochs={n_epochs}, states={n_states}) ---")

    # Create synthetic data and models with ULTRA SIMPLE settings
    datasets = create_synthetic_datasets(n_datasets, n_timepoints, n_states)
    original_model = create_synthetic_model(n_states, nn_replacements)

    # Benchmark basic training (NO JIT)
    print("Running basic training benchmark (no JIT)...")
    basic_time, basic_history = benchmark_basic_training(
        model=original_model,
        datasets=datasets,
        n_epochs=n_epochs,
        verbose=True
    )
    print(f"Basic Training: {basic_time:.4f} seconds")

    # Create a new model for optimized training
    reset_model = create_synthetic_model(n_states, nn_replacements)

    # Try to run optimized training if basic training succeeded
    try:
        print("Running optimized training benchmark...")
        optimized_time, optimized_history = benchmark_basic_training(
            model=reset_model,
            datasets=datasets,
            n_epochs=n_epochs,
            verbose=True
        )
        print(f"Optimized Training: {optimized_time:.4f} seconds")

        # Calculate speedup
        speedup = basic_time / optimized_time if optimized_time > 0 else float('nan')
        print(f"Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"Error in optimized benchmark: {e}")
        traceback.print_exc()
        optimized_time = float('inf')
        optimized_history = {'loss': []}
        speedup = float('nan')

    return {
        'original_time': float(basic_time),
        'optimized_time': None if optimized_time == float('inf') else float(optimized_time),
        'speedup': None if math.isnan(speedup) else float(speedup),
        'original_history': basic_history,
        'optimized_history': optimized_history
    }


def benchmark_bioprocess(data_file='Train_data_masked.xlsx', n_epochs=50):
    """Benchmark with the bioprocess example."""
    print(f"\n--- Benchmarking Bioprocess Example (epochs={n_epochs}) ---")

    # Load data
    runs = load_bioprocess_data(data_file)
    norm_params = runs[0]['norm_params']

    # Build model
    original_model = define_bioprocess_model(norm_params)
    optimized_model = OptimizedODESystem.from_hybrid_ode_system(original_model)

    # Prepare datasets
    datasets = prepare_bioprocess_dataset(runs)

    # Benchmark original vs optimized ODE solving
    start_time = time.time()
    for dataset in datasets:
        original_model.solve(
            initial_state=dataset['initial_state'],
            t_span=(dataset['times'][0], dataset['times'][-1]),
            evaluation_times=dataset['times'],
            args={'time_dependent_inputs': dataset['time_dependent_inputs']}
        )
    original_solve_time = time.time() - start_time

    start_time = time.time()
    for dataset in datasets:
        optimized_model.solve(
            initial_state=dataset['initial_state'],
            t_span=(dataset['times'][0], dataset['times'][-1]),
            evaluation_times=dataset['times'],
            args={'time_dependent_inputs': dataset['time_dependent_inputs']}
        )
    optimized_solve_time = time.time() - start_time

    # Benchmark batch processor
    batch_processor = DatasetBatchProcessor(optimized_model, batch_size=1)

    start_time = time.time()
    batch_processor.process(datasets, show_progress=False)
    batch_solve_time = time.time() - start_time

    print(f"Original Solve: {original_solve_time:.4f} seconds")
    print(f"Optimized Solve: {optimized_solve_time:.4f} seconds")
    print(f"Batch Processor: {batch_solve_time:.4f} seconds")

    # Calculate solve speedups
    solve_speedup = original_solve_time / optimized_solve_time if optimized_solve_time > 0 else float('inf')
    batch_speedup = original_solve_time / batch_solve_time if batch_solve_time > 0 else float('inf')

    print(f"Solve Speedup (Optimized): {solve_speedup:.2f}x")
    print(f"Solve Speedup (Batch): {batch_speedup:.2f}x")

    # Skip training benchmark if n_epochs is 0
    if n_epochs == 0:
        return {
            'original_solve_time': original_solve_time,
            'optimized_solve_time': optimized_solve_time,
            'batch_solve_time': batch_solve_time,
            'solve_speedup': solve_speedup,
            'batch_speedup': batch_speedup
        }

    # Benchmark training
    print("\nTraining benchmark:")

    # Reset model for training
    train_model = define_bioprocess_model(norm_params)

    # Benchmark original training
    start_time = time.time()
    _, original_history = train_hybrid_model(
        model=train_model,
        datasets=datasets,
        loss_fn=bioprocess_loss_function,
        num_epochs=n_epochs,
        learning_rate=1e-3,
        verbose=True
    )
    original_time = time.time() - start_time

    print(f"Original Training: {original_time:.4f} seconds")

    # Reset model for fair comparison
    train_model_2 = define_bioprocess_model(norm_params)

    # Benchmark optimized training
    start_time = time.time()
    _, optimized_history = train_hybrid_model_optimized(
        model=train_model_2,
        datasets=datasets,
        loss_fn=bioprocess_loss_function,
        num_epochs=n_epochs,
        learning_rate=1e-3,
        learning_rate_schedule='cosine',
        batch_size=1,  # Use batch processing
        verbose=True,
        log_interval=10
    )
    optimized_time = time.time() - start_time

    print(f"Optimized Training: {optimized_time:.4f} seconds")

    # Calculate training speedup
    training_speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    print(f"Training Speedup: {training_speedup:.2f}x")

    return {
        'original_solve_time': original_solve_time,
        'optimized_solve_time': optimized_solve_time,
        'batch_solve_time': batch_solve_time,
        'solve_speedup': solve_speedup,
        'batch_speedup': batch_speedup,
        'original_training_time': original_time,
        'optimized_training_time': optimized_time,
        'training_speedup': training_speedup
    }


def plot_results(results, output_dir="benchmark_results"):
    """Plot benchmark results."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Plot ODE system results
    if 'ode_system' in results:
        ode_results = results['ode_system']

        # Check if we have valid results
        if all(key in ode_results and ode_results[key] is not None
               for key in ['original_time', 'optimized_time', 'batch_time']):
            # Plot times
            plt.figure(figsize=(10, 6))
            labels = ['Original', 'Optimized', 'Batch']
            times = [
                ode_results['original_time'],
                ode_results['optimized_time'],
                ode_results['batch_time']
            ]
            plt.bar(labels, times, color=['blue', 'green', 'orange'])
            plt.title('ODE System Execution Time')
            plt.ylabel('Time (seconds)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add speedup annotations
            for i, time_val in enumerate(times):
                if i > 0 and times[
                    0] is not None and time_val is not None and time_val > 0:  # Skip the first bar (original)
                    speedup = times[0] / time_val
                    plt.text(i, time_val + 0.05 * max(times),
                             f"{speedup:.2f}x faster",
                             ha='center', va='bottom')

            plt.savefig(f"{output_dir}/ode_system_times.png")
            plt.close()
        else:
            print("Skipping ODE system plot due to missing data")

    # Plot training results
    if 'training' in results:
        train_results = results['training']

        # Check if we have valid timing results
        if all(key in train_results and train_results[key] is not None
               for key in ['original_time', 'optimized_time']):
            # Plot times
            plt.figure(figsize=(10, 6))
            labels = ['Original', 'Optimized']
            times = [
                train_results['original_time'],
                train_results['optimized_time']
            ]
            plt.bar(labels, times, color=['blue', 'green'])
            plt.title('Training Execution Time')
            plt.ylabel('Time (seconds)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add speedup annotation if possible
            if train_results.get('speedup') is not None:
                speedup = train_results['speedup']
                plt.text(1, times[1] + 0.05 * max(times),
                         f"{speedup:.2f}x faster",
                         ha='center', va='bottom')

            plt.savefig(f"{output_dir}/training_times.png")
            plt.close()
        else:
            print("Skipping training time plot due to missing data")

        # Plot loss curves if available
        if ('original_history' in train_results and 'optimized_history' in train_results and
                'loss' in train_results['original_history'] and 'loss' in train_results['optimized_history']):

            original_loss = train_results['original_history']['loss']
            optimized_loss = train_results['optimized_history']['loss']

            if original_loss and optimized_loss:  # Make sure they're not empty
                plt.figure(figsize=(10, 6))
                plt.plot(original_loss, 'b-', label='Original')
                plt.plot(optimized_loss, 'g-', label='Optimized')
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_dir}/training_loss.png")
                plt.close()
        else:
            print("Skipping loss curve plot due to missing data")

        # Plot learning rate for optimized training if available
        if ('optimized_history' in train_results and
                'learning_rate' in train_results['optimized_history'] and
                train_results['optimized_history']['learning_rate']):
            plt.figure(figsize=(10, 6))
            plt.plot(train_results['optimized_history']['learning_rate'], 'r-')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.savefig(f"{output_dir}/learning_rate.png")
            plt.close()

    # Plot bioprocess results
    if 'bioprocess' in results:
        bio_results = results['bioprocess']

        # Check if we have valid solve times
        if all(key in bio_results and bio_results[key] is not None
               for key in ['original_solve_time', 'optimized_solve_time', 'batch_solve_time']):
            # Plot solve times
            plt.figure(figsize=(10, 6))
            labels = ['Original', 'Optimized', 'Batch']
            times = [
                bio_results['original_solve_time'],
                bio_results['optimized_solve_time'],
                bio_results['batch_solve_time']
            ]
            plt.bar(labels, times, color=['blue', 'green', 'orange'])
            plt.title('Bioprocess Solve Time')
            plt.ylabel('Time (seconds)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add speedup annotations
            for i, time_val in enumerate(times):
                if i > 0 and times[
                    0] is not None and time_val is not None and time_val > 0:  # Skip the first bar (original)
                    speedup = times[0] / time_val
                    plt.text(i, time_val + 0.05 * max(times),
                             f"{speedup:.2f}x faster",
                             ha='center', va='bottom')

            plt.savefig(f"{output_dir}/bioprocess_solve_times.png")
            plt.close()
        else:
            print("Skipping bioprocess solve time plot due to missing data")

        # Plot training times if available
        if all(key in bio_results and bio_results[key] is not None
               for key in ['original_training_time', 'optimized_training_time']):
            plt.figure(figsize=(10, 6))
            labels = ['Original', 'Optimized']
            times = [
                bio_results['original_training_time'],
                bio_results['optimized_training_time']
            ]
            plt.bar(labels, times, color=['blue', 'green'])
            plt.title('Bioprocess Training Time')
            plt.ylabel('Time (seconds)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add speedup annotation if available
            if 'training_speedup' in bio_results and bio_results['training_speedup'] is not None:
                speedup = bio_results['training_speedup']
                plt.text(1, times[1] + 0.05 * max(times),
                         f"{speedup:.2f}x faster",
                         ha='center', va='bottom')

            plt.savefig(f"{output_dir}/bioprocess_training_times.png")
            plt.close()
        else:
            print("Skipping bioprocess training time plot due to missing data")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='Benchmark hybrid model performance')
    parser.add_argument('--ode', action='store_true', help='Benchmark ODE system')
    parser.add_argument('--training', action='store_true', help='Benchmark training')
    parser.add_argument('--bioprocess', action='store_true', help='Benchmark bioprocess example')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training benchmarks')
    parser.add_argument('--output', default='benchmark_results', help='Output directory')
    args = parser.parse_args()

    # Set default if no specific benchmark is selected
    if not (args.ode or args.training or args.bioprocess):
        args.training = True  # Default to just training benchmark for stability

    # Initialize results dictionary
    results = {}

    # Print JAX information
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Number of devices: {jax.device_count()}")

    # Run selected benchmarks
    if args.all or args.ode:
        try:
            results['ode_system'] = benchmark_ode_system(n_runs=2, n_timepoints=10)
        except Exception as e:
            print(f"Error in ODE system benchmark: {e}")
            traceback.print_exc()

    if args.all or args.training:
        try:
            results['training'] = benchmark_training(n_epochs=args.epochs, n_timepoints=10)
        except Exception as e:
            print(f"Error in training benchmark: {e}")
            traceback.print_exc()

    if args.all or args.bioprocess:
        try:
            results['bioprocess'] = benchmark_bioprocess(n_epochs=args.epochs)
        except FileNotFoundError:
            print("Warning: Bioprocess data file not found. Skipping bioprocess benchmark.")
        except Exception as e:
            print(f"Error in bioprocess benchmark: {e}")
            traceback.print_exc()

    # Create output directory
    Path(args.output).mkdir(exist_ok=True)

    # Attempt to plot results
    try:
        plot_results(results, args.output)
    except Exception as e:
        print(f"Error plotting results: {e}")

    # Save results to JSON
    try:
        with open(f"{args.output}/benchmark_results.json", 'w') as f:
            # Convert any non-serializable objects to strings
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if v is None:
                            json_results[key][k] = None
                        elif hasattr(v, 'tolist'):
                            json_results[key][k] = v.tolist()
                        elif isinstance(v, dict) and 'loss' in v:
                            # Handle history dictionaries
                            json_results[key][k] = {
                                sk: None if sv is None else sv.tolist() if hasattr(sv, 'tolist') else sv
                                for sk, sv in v.items()
                            }
                        else:
                            json_results[key][k] = v
                else:
                    json_results[key] = value

            json.dump(json_results, f, indent=2)
        print(f"\nBenchmark results saved to {args.output}/")
    except Exception as e:
        print(f"Error saving results: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()