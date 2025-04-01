"""Optimized training for hybrid models that maintains precision."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
from typing import List, Dict, Tuple, Callable, Any, Optional
from .parallel_processing import parallel_loss_function
from .optimized_nn import optimize_nn_components


def train_hybrid_optimized(
        model: Any,
        datasets: List[Dict],
        loss_fn: Callable,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        learning_rate_schedule: str = 'cosine',
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 1e-5,
        use_parallel: bool = True,
        max_parallel_workers: Optional[int] = None,
        verbose: bool = True,
        log_interval: int = 10
):
    """
    Train a hybrid model with optimized components while maintaining precision.

    This optimized training function:
    1. Uses parallel processing for datasets when possible
    2. Optimizes neural network components with JIT compilation
    3. Maintains the high-precision ODE solver requirements

    Args:
        model: The hybrid model to train
        datasets: List of datasets for training
        loss_fn: Loss function that takes (model, datasets) and returns (loss_value, aux)
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        learning_rate_schedule: Schedule type ('constant', 'cosine', 'exponential')
        early_stopping_patience: Number of epochs to wait without improvement
        early_stopping_min_delta: Minimum change to qualify as improvement
        use_parallel: Whether to use parallel processing for datasets
        max_parallel_workers: Maximum number of parallel workers
        verbose: Whether to print training progress
        log_interval: Interval for logging progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    if verbose:
        print("Preparing optimized training...")
        print(f"Model states: {model.state_names}")
        print(f"Neural networks: {list(model.nn_replacements.keys())}")

    # Optimize neural network components
    optimized_model = optimize_nn_components(model)

    if verbose:
        print("Neural network components optimized for JIT compilation")

    # Split model into trainable and static parts
    model_trainable, model_static = eqx.partition(optimized_model, eqx.is_array)
    params = model_trainable

    # Initialize learning rate schedule
    if learning_rate_schedule == 'cosine':
        schedule_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=num_epochs,
            alpha=0.1
        )
    elif learning_rate_schedule == 'exponential':
        schedule_fn = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=num_epochs // 10,
            decay_rate=0.9
        )
    else:  # constant
        schedule_fn = lambda _: learning_rate

    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_fn)
    )
    opt_state = optimizer.init(params)

    # Setup for early stopping
    best_loss = float('inf')
    best_params = params
    patience_counter = 0

    # Prepare loss function with parallel processing if requested
    if use_parallel:
        # Create a wrapper loss function that uses parallel processing
        def parallel_wrapper_loss(full_model, datasets):
            return parallel_loss_function(
                model=full_model,
                datasets=datasets,
                max_workers=max_parallel_workers
            )

        actual_loss_fn = parallel_wrapper_loss
        if verbose:
            print(f"Using parallel processing for {len(datasets)} datasets")
    else:
        actual_loss_fn = loss_fn

    # Setup training history
    history = {'loss': [], 'aux': [], 'learning_rate': []}
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Calculate loss and gradients
        def loss_wrapper(p):
            full_model = eqx.combine(p, model_static)
            try:
                loss_value, aux = actual_loss_fn(full_model, datasets)
                # Handle NaN or infinite values
                if jnp.isnan(loss_value) or jnp.isinf(loss_value):
                    if verbose:
                        print(f"Warning: NaN or Inf loss detected in epoch {epoch}")
                    return jnp.array(1.0e10), None
                return loss_value, aux
            except Exception as e:
                if verbose:
                    print(f"Error in loss calculation: {e}")
                return jnp.array(1.0e10), None

        try:
            # Calculate gradients
            (loss_value, aux), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # Get current learning rate
            current_lr = schedule_fn(epoch)

            # Record history
            history['loss'].append(float(loss_value))
            history['aux'].append(aux)
            history['learning_rate'].append(float(current_lr))

            # Print progress
            epoch_time = time.time() - epoch_start
            if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
                elapsed_time = time.time() - start_time
                avg_time_per_epoch = elapsed_time / (epoch + 1)
                est_remaining = avg_time_per_epoch * (num_epochs - epoch - 1)

                print(f"Epoch {epoch}/{num_epochs}, "
                      f"Loss: {float(loss_value):.4f}, "
                      f"LR: {current_lr:.6f}, "
                      f"Epoch time: {epoch_time:.2f}s, "
                      f"Est. remaining: {est_remaining:.2f}s")

            # Early stopping check
            if early_stopping_patience is not None:
                if loss_value < best_loss - early_stopping_min_delta:
                    best_loss = loss_value
                    best_params = jax.tree_util.tree_map(
                        lambda x: x.copy() if hasattr(x, 'copy') else x,
                        params
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        except Exception as e:
            if verbose:
                print(f"Error during epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()

            # Skip to next epoch
            continue

    # Use best parameters if early stopping was used
    if early_stopping_patience is not None and patience_counter < early_stopping_patience:
        trained_model = eqx.combine(best_params, model_static)
        if verbose:
            print(f"Using best model with loss: {best_loss:.4f}")
    else:
        trained_model = eqx.combine(params, model_static)

    # Print training summary
    if verbose:
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Initial loss: {history['loss'][0]:.4f}")
        print(f"Final loss: {history['loss'][-1]:.4f}")

    return trained_model, history