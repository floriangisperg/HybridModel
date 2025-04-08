"""
Elastic Pendulum Hybrid Model Example

This example demonstrates how to use the hybrid_models framework to model an elastic pendulum
system, combining mechanistic knowledge with neural networks. The example replicates the
approach from the PyTorch implementation but using JAX and the hybrid_models framework.
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# Import hybrid_models components
from hybrid_models import (
    HybridModelBuilder,
    train_hybrid_model,
    SolverConfig,
    MSE,
    create_hybrid_model_loss
)

# Set random seed for reproducibility
KEY = jax.random.PRNGKey(42)


def generate_elastic_pendulum_data(num_points=1000, t_span=(0, 10), noise_level=0.1):
    """
    Generate data for an elastic pendulum system.

    Args:
        num_points: Number of data points
        t_span: Time span (start, end)
        noise_level: Standard deviation of Gaussian noise

    Returns:
        Dictionary with time points and state variables
    """
    # Constants for the elastic pendulum
    g = 9.81      # Acceleration due to gravity
    l0 = 2.25     # Rest length of the pendulum
    k_m = 366     # Spring constant / mass
    s0 = 1.125    # Spring offset

    # Define the ODE function
    def elastic_pendulum_ode(t, y):
        x, dx, theta, dtheta = y

        # Calculate derivatives
        ddx = (l0 + x) * dtheta**2 - k_m * (x + s0) + g * np.cos(theta)
        ddtheta = -g/(l0 + x) * np.sin(theta) - 2 * dx / (l0 + x) * dtheta

        return [dx, ddx, dtheta, ddtheta]

    # Initial conditions [x, dx, theta, dtheta]
    y0 = [-0.75, 0, 1.25, 0]

    # Generate time points
    t = np.linspace(t_span[0], t_span[1], num_points)

    # Solve ODE using SciPy's solve_ivp
    from scipy.integrate import solve_ivp
    sol = solve_ivp(
        elastic_pendulum_ode,
        t_span,
        y0,
        t_eval=t,
        method='RK45',
        rtol=1e-5
    )

    # Extract solution
    x = sol.y[0]
    dx = sol.y[1]
    theta = sol.y[2]
    dtheta = sol.y[3]

    # Add noise
    if noise_level > 0:
        rng = np.random.RandomState(42)
        x += rng.normal(0, noise_level, size=x.shape)
        dx += rng.normal(0, noise_level, size=dx.shape)
        theta += rng.normal(0, noise_level, size=theta.shape)
        dtheta += rng.normal(0, noise_level, size=dtheta.shape)

    # Convert to JAX arrays
    times = jnp.array(t)
    x = jnp.array(x)
    dx = jnp.array(dx)
    theta = jnp.array(theta)
    dtheta = jnp.array(dtheta)

    # Create dataset in the format expected by hybrid_models
    dataset = {
        'times': times,
        'initial_state': {
            'x': float(x[0]),
            'dx': float(dx[0]),
            'theta': float(theta[0]),
            'dtheta': float(dtheta[0])
        },
        'x_true': x,
        'dx_true': dx,
        'theta_true': theta,
        'dtheta_true': dtheta,
        'time_dependent_inputs': {}
    }

    return dataset


def build_ideal_pendulum_model(norm_params={}, key=None):
    """
    Build a simple ideal pendulum model (incomplete knowledge).

    This model assumes a rigid pendulum with no spring, which is an incomplete
    representation of the actual elastic pendulum system.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    builder = HybridModelBuilder()

    # Set normalization parameters (empty for this example)
    builder.set_normalization_params(norm_params)

    # Add state variables
    builder.add_state('x')      # Position
    builder.add_state('dx')     # Velocity (x)
    builder.add_state('theta')  # Angle
    builder.add_state('dtheta') # Angular velocity

    # Define mechanistic components
    def x_ode(inputs):
        dx = inputs['dx']
        return dx

    def dx_ode(inputs):
        # In ideal pendulum, no spring forces
        return 0.0  # This is where the model is incomplete

    def theta_ode(inputs):
        dtheta = inputs['dtheta']
        return dtheta

    def dtheta_ode(inputs):
        theta = inputs['theta']
        g = 9.81
        l = inputs.get('pendulum_length', 2.25)  # Default value for length
        return -g/l * jnp.sin(theta)

    # Add mechanistic components
    builder.add_mechanistic_component('x', x_ode)
    builder.add_mechanistic_component('dx', dx_ode)
    builder.add_mechanistic_component('theta', theta_ode)
    builder.add_mechanistic_component('dtheta', dtheta_ode)

    # We'll learn a parameter for the pendulum length
    # Split the key for random initialization
    key, subkey = jax.random.split(key)
    builder.replace_with_nn(
        name='pendulum_length',
        input_features=[],  # No inputs needed, just learn a constant
        hidden_dims=[1],    # Minimal network to learn a single parameter
        output_activation=jax.nn.softplus,  # Ensure positive length
        key=subkey
    )

    return builder.build()


def build_hybrid_pendulum_model(norm_params={}, key=None):
    """
    Build a hybrid pendulum model combining first principles with neural networks.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    builder = HybridModelBuilder()

    # Set normalization parameters
    builder.set_normalization_params(norm_params)

    # Add state variables
    builder.add_state('x')      # Position
    builder.add_state('dx')     # Velocity (x)
    builder.add_state('theta')  # Angle
    builder.add_state('dtheta') # Angular velocity

    # Define mechanistic components
    def x_ode(inputs):
        dx = inputs['dx']
        return dx

    def dx_ode(inputs):
        # This will include both mechanistic and neural network parts
        theta = inputs['theta']
        g = 9.81
        # Add known part (gravity component)
        dx_mech = g * jnp.cos(theta)
        # Add neural network correction for unknown parts (spring force, etc.)
        dx_nn = inputs['dx_correction']
        return dx_mech + dx_nn

    def theta_ode(inputs):
        dtheta = inputs['dtheta']
        return dtheta

    def dtheta_ode(inputs):
        theta = inputs['theta']
        x = inputs['x']
        dx = inputs['dx']
        g = 9.81
        l = inputs['pendulum_length']

        # Base mechanistic model (ideal pendulum)
        dtheta_mech = -g/l * jnp.sin(theta)

        # Add neural network correction for elastic pendulum effects
        dtheta_correction = inputs['dtheta_correction']

        return dtheta_mech + dtheta_correction

    # Add mechanistic components
    builder.add_mechanistic_component('x', x_ode)
    builder.add_mechanistic_component('dx', dx_ode)
    builder.add_mechanistic_component('theta', theta_ode)
    builder.add_mechanistic_component('dtheta', dtheta_ode)

    # Replace pendulum length with a neural network to learn
    key, subkey1 = jax.random.split(key)
    builder.replace_with_nn(
        name='pendulum_length',
        input_features=[],  # No inputs needed, just learn a constant
        hidden_dims=[1],    # Single parameter network
        output_activation=jax.nn.softplus,  # Ensure positive length
        key=subkey1
    )

    # Add neural network for dx correction
    key, subkey2 = jax.random.split(key)
    builder.replace_with_nn(
        name='dx_correction',
        input_features=['x', 'dx', 'theta', 'dtheta'],
        hidden_dims=[16, 8],
        output_activation=None,  # Linear activation
        key=subkey2
    )

    # Add neural network correction for dtheta
    key, subkey3 = jax.random.split(key)
    builder.replace_with_nn(
        name='dtheta_correction',
        input_features=['x', 'dx', 'theta', 'dtheta'],
        hidden_dims=[16, 8],
        output_activation=None,  # Linear activation
        key=subkey3
    )

    return builder.build()


def build_neural_pendulum_model(norm_params={}, key=None):
    """
    Build a completely neural network-based pendulum model.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    builder = HybridModelBuilder()

    # Set normalization parameters
    builder.set_normalization_params(norm_params)

    # Add state variables
    builder.add_state('x')      # Position
    builder.add_state('dx')     # Velocity (x)
    builder.add_state('theta')  # Angle
    builder.add_state('dtheta') # Angular velocity

    # Define basic ODE components
    def x_ode(inputs):
        dx = inputs['dx']
        return dx

    def dx_ode(inputs):
        # This will be completely data-driven
        return inputs['dx_nn']

    def theta_ode(inputs):
        dtheta = inputs['dtheta']
        return dtheta

    def dtheta_ode(inputs):
        # This will be completely data-driven
        return inputs['dtheta_nn']

    # Add mechanistic components
    builder.add_mechanistic_component('x', x_ode)
    builder.add_mechanistic_component('dx', dx_ode)
    builder.add_mechanistic_component('theta', theta_ode)
    builder.add_mechanistic_component('dtheta', dtheta_ode)

    # Replace dx with neural network
    key, subkey1 = jax.random.split(key)
    builder.replace_with_nn(
        name='dx_nn',
        input_features=['x', 'dx', 'theta', 'dtheta'],
        hidden_dims=[32, 16],
        output_activation=None,  # Linear output
        key=subkey1
    )

    # Replace dtheta with neural network
    key, subkey2 = jax.random.split(key)
    builder.replace_with_nn(
        name='dtheta_nn',
        input_features=['x', 'dx', 'theta', 'dtheta'],
        hidden_dims=[32, 16],
        output_activation=None,  # Linear output
        key=subkey2
    )

    return builder.build()


def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Generate data for the elastic pendulum system
    print("Generating elastic pendulum data...")
    dataset = generate_elastic_pendulum_data(num_points=1000, noise_level=0.05)

    # Create a single dataset for simplicity
    train_dataset = [dataset]  # Wrap in list for training function

    # Print dataset information
    print(f"Dataset size: {len(dataset['times'])}")
    print(f"Initial state: x={dataset['initial_state']['x']}, dx={dataset['initial_state']['dx']}, " 
          f"theta={dataset['initial_state']['theta']}, dtheta={dataset['initial_state']['dtheta']}")

    # Plot the generated data
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(dataset['times'], dataset['theta_true'], label='Theta')
    plt.xlabel('Time')
    plt.ylabel('Angle (rad)')
    plt.title('Pendulum Angle')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(dataset['times'], dataset['x_true'], label='x')
    plt.xlabel('Time')
    plt.ylabel('Position (m)')
    plt.title('Spring Extension')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(dataset['theta_true'], dataset['dtheta_true'])
    plt.xlabel('Theta')
    plt.ylabel('dTheta/dt')
    plt.title('Phase Portrait (Angular)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(dataset['x_true'], dataset['dx_true'])
    plt.xlabel('x')
    plt.ylabel('dx/dt')
    plt.title('Phase Portrait (Position)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/generated_data.png")
    plt.close()

    # Create a solver configuration optimized for training
    training_solver_config = SolverConfig(
        solver_type="rk4",        # Fixed-step solver for training stability
        step_size_controller="constant",
        dt=0.02,
        max_steps=10000
    )

    # Solver for evaluation (more accurate)
    eval_solver_config = SolverConfig(
        solver_type="dopri5",
        step_size_controller="pid",
        rtol=1e-4,
        atol=1e-6,
        max_steps=10000
    )

    # Train the different model types
    for model_type in ['ideal', 'hybrid', 'neural']:
        print(f"\nTraining {model_type} pendulum model...")

        # Build the appropriate model
        if model_type == 'ideal':
            model = build_ideal_pendulum_model(key=KEY)
            num_epochs = 200
            title = "Ideal Pendulum Model (Incomplete Knowledge)"
        elif model_type == 'hybrid':
            model = build_hybrid_pendulum_model(key=KEY)
            num_epochs = 500
            title = "Hybrid Pendulum Model"
        else:  # neural
            model = build_neural_pendulum_model(key=KEY)
            num_epochs = 500
            title = "Neural Pendulum Model"

        # Create a loss function with the training solver configuration
        loss_fn = create_hybrid_model_loss(
            state_names=['x', 'dx', 'theta', 'dtheta'],
            loss_metric=MSE,
            solve_kwargs=training_solver_config.to_dict()
        )

        # Train the model
        trained_model, history = train_hybrid_model(
            model=model,
            datasets=train_dataset,
            loss_fn=loss_fn,
            num_epochs=num_epochs,
            learning_rate=0.01,
            verbose=True
        )

        # Save the trained model
        results_dir = f"results/{model_type}_pendulum"
        os.makedirs(results_dir, exist_ok=True)

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'])
        plt.title(f'Training Loss - {title}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(f"{results_dir}/training_loss.png")
        plt.close()

        # Test the model on the full dataset using the evaluation solver
        solution = trained_model.solve(
            initial_state=dataset['initial_state'],
            t_span=(dataset['times'][0], dataset['times'][-1]),
            evaluation_times=dataset['times'],
            args={},
            **eval_solver_config.to_dict()
        )

        # Calculate MSE
        x_mse = float(jnp.mean((solution['x'] - dataset['x_true'])**2))
        theta_mse = float(jnp.mean((solution['theta'] - dataset['theta_true'])**2))
        total_mse = (x_mse + theta_mse) / 2

        print(f"Evaluation MSE: x={x_mse:.4f}, theta={theta_mse:.4f}, total={total_mse:.4f}")

        # Plot results
        plt.figure(figsize=(15, 10))

        # Theta vs time
        plt.subplot(2, 2, 1)
        plt.plot(dataset['times'], dataset['theta_true'], 'o', markersize=2, label='True')
        plt.plot(solution['times'], solution['theta'], '-', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Theta')
        plt.title(f'Pendulum Angle (MSE: {theta_mse:.4f})')
        plt.legend()
        plt.grid(True)

        # dTheta vs time
        plt.subplot(2, 2, 2)
        plt.plot(dataset['times'], dataset['dtheta_true'], 'o', markersize=2, label='True')
        plt.plot(solution['times'], solution['dtheta'], '-', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('dTheta/dt')
        plt.title('Angular Velocity')
        plt.legend()
        plt.grid(True)

        # X vs time
        plt.subplot(2, 2, 3)
        plt.plot(dataset['times'], dataset['x_true'], 'o', markersize=2, label='True')
        plt.plot(solution['times'], solution['x'], '-', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('X')
        plt.title(f'Spring Extension (MSE: {x_mse:.4f})')
        plt.legend()
        plt.grid(True)

        # Phase portrait (theta vs dtheta)
        plt.subplot(2, 2, 4)
        plt.plot(dataset['theta_true'], dataset['dtheta_true'], 'o', markersize=2, label='True')
        plt.plot(solution['theta'], solution['dtheta'], '-', label='Predicted')
        plt.xlabel('Theta')
        plt.ylabel('dTheta/dt')
        plt.title('Phase Portrait')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"{title} - Total MSE: {total_mse:.4f}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"{results_dir}/pendulum_predictions.png")
        plt.close()

        print(f"Results saved to {results_dir}")

    print("\nAll models trained and evaluated successfully!")


if __name__ == "__main__":
    main()