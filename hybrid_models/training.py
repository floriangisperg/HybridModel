import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, List, Tuple, Callable, Any, Optional
from functools import partial
import time
from jaxtyping import Array, Float, PyTree


def train_hybrid_model(
        model: Any,
        datasets: List[Dict],
        loss_fn: Callable,
        num_epochs: int = 1000,
        learning_rate: float = 1e-3,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 1e-5,
        verbose: bool = True
):
    """
    Train a hybrid model.

    Args:
        model: The hybrid model to train
        datasets: List of datasets for training
        loss_fn: Loss function that takes (model, datasets) and returns (loss_value, aux)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        early_stopping_patience: Number of epochs to wait without improvement before stopping
        early_stopping_min_delta: Minimum change to qualify as improvement
        verbose: Whether to print training progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    # Split model into trainable and static parts
    model_trainable, model_static = eqx.partition(model, eqx.is_array)
    params = model_trainable

    # Initialize optimizer - use simple Adam without weight decay
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Define JIT-compiled update step
    @partial(jax.jit, static_argnums=(2,))
    def update_step(params, opt_state, model_static, datasets):
        def loss_wrapper(p):
            full_model = eqx.combine(p, model_static)
            loss_value, aux = loss_fn(full_model, datasets)
            return loss_value, aux

        (loss_value, aux), grads = jax.value_and_grad(
            loss_wrapper, has_aux=True
        )(params)

        updates, opt_state_new = optimizer.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)

        return params_new, opt_state_new, loss_value, aux

    # Setup for early stopping
    best_loss = float('inf')
    best_params = params
    patience_counter = 0

    # Training loop
    history = {'loss': [], 'aux': []}
    start_time = time.time()

    for epoch in range(num_epochs):
        # Update parameters
        params, opt_state, loss_value, aux = update_step(params, opt_state, model_static, datasets)

        # Record history
        history['loss'].append(float(loss_value))
        history['aux'].append(aux)

        # Print progress
        if verbose and (epoch % 50 == 0 or epoch == num_epochs - 1):
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss_value:.4f}, Time: {elapsed_time:.2f}s")

        # Early stopping check
        if early_stopping_patience is not None:
            if loss_value < best_loss - early_stopping_min_delta:
                best_loss = loss_value
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

    return trained_model, history