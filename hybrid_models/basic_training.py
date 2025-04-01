"""Basic training implementation with NO JIT for maximum compatibility."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
from typing import Dict, List, Tuple, Callable, Any, Optional


def train_basic(
        model: Any,
        datasets: List[Dict],
        loss_fn: Callable,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        verbose: bool = True
):
    """
    Train a model with absolutely no JIT compilation for maximum compatibility.

    Args:
        model: The model to train
        datasets: List of datasets for training
        loss_fn: Loss function that takes (model, datasets) and returns (loss_value, aux)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        verbose: Whether to print training progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    # Split model into trainable and static parts
    model_trainable, model_static = eqx.partition(model, eqx.is_array)
    params = model_trainable

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Set up history
    history = {'loss': [], 'aux': []}
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        # Calculate loss and gradients
        def loss_wrapper(p):
            full_model = eqx.combine(p, model_static)
            try:
                loss_value, aux = loss_fn(full_model, datasets)
                # Handle NaN or infinite values
                if jnp.isnan(loss_value) or jnp.isinf(loss_value):
                    return 1.0e10, None
                return loss_value, aux
            except Exception as e:
                if verbose:
                    print(f"Error in loss calculation: {e}")
                return 1.0e10, None

        try:
            (loss_value, aux), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # Record history
            history['loss'].append(float(loss_value))
            history['aux'].append(aux)

            # Print progress
            if verbose and (epoch % 5 == 0 or epoch == num_epochs - 1):
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch}/{num_epochs}, Loss: {float(loss_value):.4f}, Time: {elapsed_time:.2f}s")

        except Exception as e:
            if verbose:
                print(f"Error during epoch {epoch}: {e}")
            # Continue to next epoch

    # Return trained model
    trained_model = eqx.combine(params, model_static)
    return trained_model, history


def benchmark_basic_training(
        model: Any,
        datasets: List[Dict],
        n_epochs: int = 10,
        verbose: bool = True
):
    """
    Benchmark basic training with extremely simplified loss function.

    Args:
        model: Model to train
        datasets: List of datasets
        n_epochs: Number of epochs
        verbose: Whether to print progress

    Returns:
        Training time
    """

    # Define super simple loss function
    # Update simple_loss in basic_training.py
    def simple_loss(model, datasets):
        total_loss = 0.0
        for dataset in datasets:
            try:
                # Add simple ODE solving back with generous tolerances
                solution = model.solve(
                    initial_state=dataset['initial_state'],
                    t_span=(dataset['times'][0], dataset['times'][-1]),
                    evaluation_times=dataset['times'],
                    args={'time_dependent_inputs': dataset['time_dependent_inputs']},
                    max_steps=1000000,  # Very high
                    rtol=1e-1,  # Very relaxed
                    atol=1e-1
                )

                # Calculate loss normally
                for state_name in model.state_names:
                    true_key = f"{state_name}_true"
                    if true_key in dataset:
                        loss = jnp.mean(jnp.square(solution[state_name] - dataset[true_key]))
                        total_loss += loss
            except Exception as e:
                print(f"Error in ODE solution: {e}")
                return jnp.array(1.0e6), None

        return total_loss / max(1, len(datasets)), None

    # Time training
    start_time = time.time()
    _, history = train_basic(
        model=model,
        datasets=datasets,
        loss_fn=simple_loss,
        num_epochs=n_epochs,
        verbose=verbose
    )
    duration = time.time() - start_time

    return duration, history