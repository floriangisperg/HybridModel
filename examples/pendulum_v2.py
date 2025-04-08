# examples/pendulum_example.py

"""
Elastic Pendulum Hybrid Model Example

This example demonstrates using the hybrid_models framework to model an elastic pendulum,
implementing three approaches:
1. Pure mechanistic model with trainable parameters
2. Pure neural network model
3. Hybrid model combining mechanistic knowledge with neural network corrections
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple

# Import hybrid_models framework components
from hybrid_models import (
    HybridModelBuilder,
    train_hybrid_model,
    SolverConfig
)
import diffrax


def generate_pendulum_data(model_type='elastic', noise_level=0.1, n_samples=1000):
    """Generate pendulum data using an ODE solver."""
    # Define time points
    t_span = (0.0, 10.0)
    t = jnp.linspace(t_span[0], t_span[1], n_samples)

    # Initial conditions [x, dx, theta, dtheta]
    y0 = jnp.array([-0.75, 0.0, 1.25, 0.0])

    # Generate data using ideal or elastic pendulum model
    if model_type == 'ideal':
        # Ideal pendulum parameters
        g = 9.81  # gravity
        l = 3.52  # length

        def pendulum_ode(t, y, args):
            x, dx, theta, dtheta = y
            ddx = 0.0
            ddtheta = -g / l * jnp.sin(theta)
            return jnp.array([dx, ddx, dtheta, ddtheta])

    else:  # elastic pendulum
        # Elastic pendulum parameters
        g = 9.81  # gravity
        l0 = 2.25  # natural length
        k_m = 366  # spring constant / mass
        s0 = 1.125  # spring offset

        def pendulum_ode(t, y, args):
            x, dx, theta, dtheta = y
            ddx = (l0 + x) * dtheta ** 2 - k_m * (x + s0) + g * jnp.cos(theta)
            ddtheta = -g / (l0 + x) * jnp.sin(theta) - 2 * dx / (l0 + x) * dtheta
            return jnp.array([dx, ddx, dtheta, ddtheta])

    # Solve ODE using diffrax
    term = diffrax.ODETerm(pendulum_ode)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=0.01,
        y0=y0,
        saveat=saveat,
        max_steps=10000
    )

    # Add noise
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, shape=solution.ys.shape) * noise_level
    noisy_data = solution.ys + noise

    return t, noisy_data


def prepare_pendulum_dataset(t, data):
    """Prepare pendulum data for training with the hybrid_model framework."""
    # Extract initial state
    initial_state = {
        'x': float(data[0, 0]),
        'dx': float(data[0, 1]),
        'theta': float(data[0, 2]),
        'dtheta': float(data[0, 3])
    }

    # Create dataset dictionary
    dataset = {
        'times': t,
        'initial_state': initial_state,
        'time_dependent_inputs': {},
        'static_inputs': {},
        'x_true': data[:, 0],
        'dx_true': data[:, 1],
        'theta_true': data[:, 2],
        'dtheta_true': data[:, 3]
    }

    return dataset


def create_pure_mechanistic_model(g=9.81, initial_length=2.0):
    """Create a pure mechanistic pendulum model with trainable length parameter."""
    builder = HybridModelBuilder()

    # Add states: x, dx, theta, dtheta
    builder.add_state('x')
    builder.add_state('dx')
    builder.add_state('theta')
    builder.add_state('dtheta')

    # Add trainable parameter for pendulum length
    builder.add_trainable_parameter(
        name='l',  # Length of pendulum
        initial_value=initial_length,
        bounds=(0.5, 5.0),  # Reasonable bounds for length
        transform='sigmoid'  # Apply sigmoid to keep within bounds
    )

    # Define mechanistic components with the trainable parameter
    def x_ode(inputs):
        dx = inputs['dx']
        return dx

    def dx_ode(inputs):
        # No acceleration in x for ideal pendulum
        return 0.0

    def theta_ode(inputs):
        dtheta = inputs['dtheta']
        return dtheta

    def dtheta_ode(inputs):
        theta = inputs['theta']
        l = inputs['l']  # Trainable parameter
        return -g / l * jnp.sin(theta)

    # Add mechanistic components
    builder.add_mechanistic_component('x', x_ode)
    builder.add_mechanistic_component('dx', dx_ode)
    builder.add_mechanistic_component('theta', theta_ode)
    builder.add_mechanistic_component('dtheta', dtheta_ode)

    return builder.build()


def create_pure_neural_model(hidden_dims=[40, 20]):
    """Create a pure neural network model for pendulum dynamics."""
    builder = HybridModelBuilder()

    # Add states: x, dx, theta, dtheta
    builder.add_state('x')
    builder.add_state('dx')
    builder.add_state('theta')
    builder.add_state('dtheta')

    # Create neural networks for accelerations
    # Initialize random key
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    # Replace the acceleration terms with neural networks
    builder.replace_with_nn(
        name='dx_ode',
        input_features=['x', 'dx', 'theta', 'dtheta'],
        hidden_dims=hidden_dims,
        key=key1
    )

    builder.replace_with_nn(
        name='dtheta_ode',
        input_features=['x', 'dx', 'theta', 'dtheta'],
        hidden_dims=hidden_dims,
        key=key2
    )

    # Define mechanistic components for the velocity terms
    def x_ode(inputs):
        dx = inputs['dx']
        return dx

    def dx_residual(inputs):
        return inputs['dx_ode']  # Use NN prediction

    def theta_ode(inputs):
        dtheta = inputs['dtheta']
        return dtheta

    def dtheta_residual(inputs):
        return inputs['dtheta_ode']  # Use NN prediction

    # Add components
    builder.add_mechanistic_component('x', x_ode)
    builder.add_mechanistic_component('dx', dx_residual)
    builder.add_mechanistic_component('theta', theta_ode)
    builder.add_mechanistic_component('dtheta', dtheta_residual)

    return builder.build()


def create_hybrid_model(g=9.81, initial_length=2.0, hidden_dims=[40, 20]):
    """Create a hybrid model combining mechanistic knowledge with neural networks."""
    builder = HybridModelBuilder()

    # Add states: x, dx, theta, dtheta
    builder.add_state('x')
    builder.add_state('dx')
    builder.add_state('theta')
    builder.add_state('dtheta')

    # Add trainable parameter for pendulum length
    builder.add_trainable_parameter(
        name='l',  # Length of pendulum
        initial_value=initial_length,
        bounds=(0.5, 5.0),  # Reasonable bounds for length
        transform='sigmoid'  # Apply sigmoid to keep within bounds
    )

    # Create neural networks for residuals/corrections
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    # Replace the residual terms with neural networks
    builder.replace_with_nn(
        name='dx_residual',
        input_features=['x', 'dx', 'theta', 'dtheta'],
        hidden_dims=hidden_dims,
        key=key1
    )

    builder.replace_with_nn(
        name='dtheta_residual',
        input_features=['x', 'dx', 'theta', 'dtheta'],
        hidden_dims=hidden_dims,
        key=key2
    )

    # Define mechanistic components with neural network corrections
    def x_ode(inputs):
        dx = inputs['dx']
        return dx

    def dx_ode(inputs):
        # Neural network correction for x acceleration
        return inputs['dx_residual']

    def theta_ode(inputs):
        dtheta = inputs['dtheta']
        return dtheta

    def dtheta_ode(inputs):
        theta = inputs['theta']
        l = inputs['l']  # Trainable parameter

        # Combine mechanistic term with neural network correction
        mechanistic_term = -g / l * jnp.sin(theta)
        correction = inputs['dtheta_residual']

        return mechanistic_term + correction

    # Add mechanistic components
    builder.add_mechanistic_component('x', x_ode)
    builder.add_mechanistic_component('dx', dx_ode)
    builder.add_mechanistic_component('theta', theta_ode)
    builder.add_mechanistic_component('dtheta', dtheta_ode)

    return builder.build()


def create_mse_loss(weight_theta=2.0):
    """Create a weighted MSE loss function that emphasizes theta."""

    def loss_fn(model, datasets):
        total_loss = 0.0
        component_losses = []

        for dataset in datasets:
            # Solve ODE
            solution = model.solve(
                initial_state=dataset['initial_state'],
                t_span=(dataset['times'][0], dataset['times'][-1]),
                evaluation_times=dataset['times'],
                args={}
            )

            # Calculate loss for each state (with higher weight for theta)
            state_losses = []
            state_weights = {'x': 1.0, 'dx': 1.0, 'theta': weight_theta, 'dtheta': 1.0}

            for state, weight in state_weights.items():
                pred = solution[state]
                true = dataset[f'{state}_true']
                state_loss = jnp.mean(jnp.square(pred - true))
                weighted_loss = weight * state_loss
                state_losses.append(state_loss)
                total_loss += weighted_loss

            component_losses.extend(state_losses)

        return total_loss / len(datasets), tuple(component_losses)

    return loss_fn



def train_and_evaluate_pendulum_model(model_type='hybrid', n_epochs=2000, learning_rate=0.01,
                                      train_ratio=0.7):
    """Train and evaluate a pendulum model with proper train/test split."""
    # Create output directory
    results_dir = f"results_pendulum_{model_type}"
    os.makedirs(results_dir, exist_ok=True)

    # Generate data
    print("Generating pendulum data...")
    t, data = generate_pendulum_data(model_type='elastic', noise_level=0.1, n_samples=1500)

    # Split into training and test sets based on time
    train_size = int(len(t) * train_ratio)

    t_train = t[:train_size]
    data_train = data[:train_size]

    t_test = t[train_size:]
    data_test = data[train_size:]

    # Prepare datasets
    train_dataset = prepare_pendulum_dataset(t_train, data_train)
    test_dataset = prepare_pendulum_dataset(t_test, data_test)

    train_datasets = [train_dataset]

    # Create appropriate model
    print(f"Creating {model_type} model...")
    if model_type == 'mechanistic':
        model = create_pure_mechanistic_model(initial_length=3.0)
    elif model_type == 'neural':
        model = create_pure_neural_model(hidden_dims=[40, 20])
    else:  # hybrid
        model = create_hybrid_model(initial_length=3.0, hidden_dims=[40, 20])

    # Create loss function
    loss_fn = create_mse_loss(weight_theta=2.0)

    # Configure solver
    solver_config = SolverConfig(
        solver_type="dopri5",
        step_size_controller="pid",
        rtol=1e-3,
        atol=1e-6
    )

    # Train model
    print(f"Training {model_type} model for {n_epochs} epochs...")
    trained_model, history = train_hybrid_model(
        model=model,
        datasets=train_datasets,
        loss_fn=loss_fn,
        num_epochs=n_epochs,
        learning_rate=learning_rate,
        early_stopping_patience=n_epochs // 4,
        verbose=True
    )

    # Save loss history
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.capitalize()} Model Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{model_type}_loss.png'))
    plt.close()

    # Evaluate on test data
    test_metrics = evaluate_model(trained_model, test_dataset)

    # Print test metrics
    print(f"\nTest Set Metrics for {model_type.capitalize()} Model:")
    for state, metrics in test_metrics.items():
        print(f"  {state}: MSE = {metrics['mse']:.6f}, R² = {metrics['r2']:.6f}")

    # Save test metrics
    with open(os.path.join(results_dir, f'{model_type}_test_metrics.txt'), 'w') as f:
        f.write(f"Test Set Metrics for {model_type.capitalize()} Model:\n")
        for state, metrics in test_metrics.items():
            f.write(f"  {state}: MSE = {metrics['mse']:.6f}, R² = {metrics['r2']:.6f}\n")

    # Visualize results on both training and test data
    visualize_results_with_test(trained_model, t_train, data_train, t_test, data_test,
                                results_dir, model_type)

    return trained_model, history, test_metrics


def evaluate_model(model, test_dataset):
    """Evaluate model performance on test data."""
    # Solve ODE with trained model
    solution = model.solve(
        initial_state=test_dataset['initial_state'],
        t_span=(test_dataset['times'][0], test_dataset['times'][-1]),
        evaluation_times=test_dataset['times'],
        args={}
    )

    # Calculate metrics for each state
    metrics = {}
    for state in ['x', 'dx', 'theta', 'dtheta']:
        pred = solution[state]
        true = test_dataset[f'{state}_true']

        # Calculate MSE
        mse = jnp.mean(jnp.square(pred - true))

        # Calculate R²
        ss_total = jnp.sum(jnp.square(true - jnp.mean(true)))
        ss_residual = jnp.sum(jnp.square(true - pred))
        r2 = 1 - ss_residual / (ss_total + 1e-10)  # Avoid division by zero

        # Store metrics
        metrics[state] = {
            'mse': float(mse),
            'r2': float(r2)
        }

    return metrics


def visualize_results_with_test(model, t_train, data_train, t_test, data_test,
                                results_dir, model_type):
    """Visualize model results on both training and test data."""
    # Prepare datasets
    train_dataset = prepare_pendulum_dataset(t_train, data_train)
    test_dataset = prepare_pendulum_dataset(t_test, data_test)

    # Solve ODE for training data
    train_solution = model.solve(
        initial_state=train_dataset['initial_state'],
        t_span=(t_train[0], t_train[-1]),
        evaluation_times=t_train,
        args={}
    )

    # Solve ODE for test data (using last state from training as initial state)
    test_solution = model.solve(
        initial_state={
            'x': float(data_train[-1, 0]),
            'dx': float(data_train[-1, 1]),
            'theta': float(data_train[-1, 2]),
            'dtheta': float(data_train[-1, 3])
        },
        t_span=(t_test[0], t_test[-1]),
        evaluation_times=t_test,
        args={}
    )

    # Create full time and predictions arrays
    t_full = jnp.concatenate([t_train, t_test])
    theta_pred_full = jnp.concatenate([train_solution['theta'], test_solution['theta']])
    dtheta_pred_full = jnp.concatenate([train_solution['dtheta'], test_solution['dtheta']])

    # Create full true data arrays
    theta_true_full = jnp.concatenate([data_train[:, 2], data_test[:, 2]])
    dtheta_true_full = jnp.concatenate([data_train[:, 3], data_test[:, 3]])

    # Plot theta vs time
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot(t_train, train_solution['theta'], 'b-', label='Predicted θ (Training)')
    plt.plot(t_train, data_train[:, 2], 'bo', alpha=0.3, label='True θ (Training)')

    # Plot test data with different color
    plt.plot(t_test, test_solution['theta'], 'r-', label='Predicted θ (Test)')
    plt.plot(t_test, data_test[:, 2], 'ro', alpha=0.3, label='True θ (Test)')

    # Add vertical line to indicate train/test split
    plt.axvline(x=t_train[-1], color='k', linestyle='--', alpha=0.7, label='Train/Test Split')

    plt.xlabel('Time')
    plt.ylabel('θ (rad)')
    plt.title(f'{model_type.capitalize()} Model: Pendulum Angle over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{model_type}_theta_vs_time_with_test.png'))
    plt.close()

    # Plot phase portrait
    plt.figure(figsize=(12, 6))

    # Plot training and test predictions
    plt.plot(theta_pred_full, dtheta_pred_full, 'b-', label='Predicted')

    # Plot training data
    plt.plot(data_train[:, 2], data_train[:, 3], 'bo', alpha=0.3, label='True (Training)')

    # Plot test data
    plt.plot(data_test[:, 2], data_test[:, 3], 'ro', alpha=0.3, label='True (Test)')

    plt.xlabel('θ (rad)')
    plt.ylabel('dθ/dt (rad/s)')
    plt.title(f'{model_type.capitalize()} Model: Pendulum Phase Portrait')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{model_type}_phase_portrait_with_test.png'))
    plt.close()

    # If the model has trainable parameters, print them
    if hasattr(model, 'trainable_parameters') and 'l' in model.trainable_parameters:
        # Extract and transform the length parameter
        l_param = model.trainable_parameters['l']
        l_transform = model.parameter_transforms['l']

        if l_transform['transform'] == 'sigmoid' and l_transform['bounds']:
            lower, upper = l_transform['bounds']
            l_value = lower + (upper - lower) * jax.nn.sigmoid(l_param)
        else:
            l_value = l_param

        print(f"Trained pendulum length: {float(l_value):.4f}")

        # Save parameter value to file
        with open(os.path.join(results_dir, f'{model_type}_parameters.txt'), 'w') as f:
            f.write(f"Trained pendulum length: {float(l_value):.4f}\n")
            f.write(f"True elastic pendulum length: 2.25\n")
            f.write(f"True ideal pendulum length: 3.52\n")


def main():
    """Main function to demonstrate pendulum models with train/test split."""
    # Set random seed for reproducibility
    np.random.seed(0)

    # Training parameters
    n_epochs = 2000  # Match the original PyTorch example
    learning_rate = 0.01
    train_ratio = 0.7  # Use 70% of data for training, 30% for testing

    # Train and evaluate different model types
    model_types = ['mechanistic', 'neural', 'hybrid']

    for model_type in model_types:
        print(f"\n===== Training and Evaluating {model_type.upper()} Model =====\n")

        # Train and evaluate model
        trained_model, history, test_metrics = train_and_evaluate_pendulum_model(
            model_type=model_type,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            train_ratio=train_ratio
        )

        print(f"\n===== {model_type.upper()} Model Complete =====\n")

    print("All models trained and evaluated successfully.")

if __name__ == "__main__":
    main()