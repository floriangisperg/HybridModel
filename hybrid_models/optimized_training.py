"""Optimized training functionality for hybrid models."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, List, Tuple, Callable, Any, Optional
from functools import partial
import time
import math
from jaxtyping import Array, Float, PyTree
from .profiling import timed, TimingStats


@timed("train_hybrid_model_optimized")
def train_hybrid_model_optimized(
        model: Any,
        datasets: List[Dict],
        loss_fn: Callable,
        num_epochs: int = 1000,
        learning_rate: float = 1e-3,
        learning_rate_schedule: str = 'cosine',
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 1e-5,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        log_interval: int = 50
):
    """
    Train a hybrid model with optimized JAX compilation.

    Args:
        model: The hybrid model to train
        datasets: List of datasets for training
        loss_fn: Loss function that takes (model, datasets) and returns (loss_value, aux)
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate for the optimizer
        learning_rate_schedule: Schedule type ('constant', 'cosine', 'exponential')
        early_stopping_patience: Number of epochs to wait without improvement before stopping
        early_stopping_min_delta: Minimum change to qualify as improvement
        batch_size: Number of datasets to process at once (None = use all)
        verbose: Whether to print training progress
        log_interval: How often to log progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    # Create timing stats
    timing_stats = TimingStats()

    # Split model into trainable and static parts
    model_trainable, model_static = eqx.partition(model, eqx.is_array)
    params = model_trainable

    # Initialize learning rate schedule
    if learning_rate_schedule == 'cosine':
        # Cosine decay learning rate schedule
        schedule_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=num_epochs,
            alpha=0.1  # Minimum learning rate factor
        )
    elif learning_rate_schedule == 'exponential':
        # Exponential decay learning rate schedule
        schedule_fn = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=num_epochs // 10,
            decay_rate=0.9,
            staircase=False
        )
    else:
        # Constant learning rate
        schedule_fn = lambda _: learning_rate

    # Initialize optimizer - use Adam with learning rate schedule
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adam(learning_rate=schedule_fn)
    )
    opt_state = optimizer.init(params)

    # Define loss function with improved robustness
    @partial(jax.jit, static_argnums=(2,))
    @timed("loss_evaluation")
    def loss_with_safety(p, s, datasets):
        """Loss function with error handling."""
        try:
            full_model = eqx.combine(p, s)
            loss_value, aux = loss_fn(full_model, datasets)
            # Replace NaN or infinite values with large finite values
            loss_value = jnp.nan_to_num(loss_value, nan=1.0e10, posinf=1.0e10, neginf=-1.0e10)
            return loss_value, aux
        except Exception as e:
            # If an error occurs, return a high loss
            return jnp.array(1.0e10), None

    # Define JIT-compiled update step
    @partial(jax.jit, static_argnums=(2,))
    @timed("update_step")
    def update_step(params, opt_state, model_static, datasets, step):
        """Perform one optimization step."""

        def loss_wrapper(p):
            loss_value, aux = loss_with_safety(p, model_static, datasets)
            return loss_value, aux

        (loss_value, aux), grads = jax.value_and_grad(
            loss_wrapper, has_aux=True
        )(params)

        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)

        return params_new, opt_state_new, loss_value, aux

    # Batch processing support
    if batch_size is None or batch_size >= len(datasets):
        # Process all datasets at once (original behavior)
        batched_datasets = [datasets]
    else:
        # Split datasets into batches
        batched_datasets = [
            datasets[i:i + batch_size]
            for i in range(0, len(datasets), batch_size)
        ]

    # Setup for early stopping
    best_loss = float('inf')
    best_params = params
    patience_counter = 0

    # Training loop
    history = {'loss': [], 'aux': [], 'learning_rate': []}
    start_time = time.time()

    for epoch in range(num_epochs):
        # Track epoch loss
        epoch_loss = 0.0
        epoch_aux = None

        # Get current learning rate
        current_lr = schedule_fn(epoch)
        history['learning_rate'].append(float(current_lr))

        # Process each batch
        for batch_idx, batch in enumerate(batched_datasets):
            # Update parameters
            start_batch = time.time()
            params, opt_state, loss_value, aux = update_step(
                params, opt_state, model_static, batch, jnp.array(epoch)
            )
            batch_time = time.time() - start_batch

            # Add to epoch totals
            epoch_loss += float(loss_value) * (len(batch) / len(datasets))
            if epoch_aux is None and aux is not None:
                epoch_aux = aux

            # Detailed logging for large batches
            if verbose and len(batched_datasets) > 1 and (epoch % log_interval == 0):
                print(f"  Batch {batch_idx + 1}/{len(batched_datasets)}, "
                      f"Loss: {float(loss_value):.4f}, Time: {batch_time:.2f}s")

        # Record history
        history['loss'].append(float(epoch_loss))
        history['aux'].append(epoch_aux)

        # Print progress
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            est_time_remaining = avg_time_per_epoch * (num_epochs - epoch - 1)

            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Elapsed: {elapsed_time:.2f}s, "
                  f"Est. remaining: {est_time_remaining:.2f}s")

        # Early stopping check
        if early_stopping_patience is not None:
            if epoch_loss < best_loss - early_stopping_min_delta:
                best_loss = epoch_loss
                best_params = jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, params)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Use the best parameters if early stopping was used
    if early_stopping_patience is not None:
        trained_model = eqx.combine(best_params, model_static)
        if verbose:
            print(f"Using best model with loss: {best_loss:.4f}")
    else:
        trained_model = eqx.combine(params, model_static)

    # Final timing stats
    if verbose:
        timing_stats.print_summary()

    return trained_model, history