"""
Astaxanthin Production Hybrid Model Example

This example demonstrates creating a hybrid model for astaxanthin production
where some parameters are replaced by neural networks while keeping the
mechanistic structure intact.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import List, Dict

# Import necessary hybrid_models components
from hybrid_models import (
    HybridModelBuilder,
    ModelConfig,
    NeuralNetworkConfig,
    ExperimentManager,
    SolverConfig
)


# Define the ground truth model for data generation
class GroundTruthModel:
    def __init__(self):
        # Ground truth parameters
        self.mu_m = 0.43      # maximum specific growth rate
        self.K_S = 65.0       # substrate saturation constant
        self.mu_d = 2.10e-3   # specific decay rate
        self.Y_S = 2.58       # substrate yield coefficient
        self.b = 0.236        # growth-independent yield coefficient
        self.k_d = 6.48e-2    # specific consumption rate for astaxanthin
        self.K_p = 2.50       # product saturation constant

    def __call__(self, t, y):
        X, S, P = y

        # Prevent negative values (can occur due to numerical errors)
        X = max(1e-10, X)
        S = max(1e-10, S)
        P = max(1e-10, P)

        # Ground truth dynamics (complex model we assume we don't fully know)
        dX_dt = self.mu_m * S / (S + self.K_S) * X - self.mu_d * X
        dS_dt = -self.Y_S * self.mu_m * S / (S + self.K_S) * X
        dP_dt = self.b * X - self.k_d * X ** 2 * P / (P + self.K_p)

        return np.array([dX_dt, dS_dt, dP_dt])


def generate_data(initial_conditions=None,
                 t_max=168.0,
                 irregular=True,
                 noise_level=0.1):
    """
    Generate in silico experimental data from the ground truth model.
    """
    model = GroundTruthModel()

    experiments = []

    for exp_idx, y0 in enumerate(initial_conditions):
        # Define time points
        if irregular:
            # Generate irregular time points with more samples at the beginning
            base_points = np.linspace(0, t_max, 15)  # 15 base points
            noise = np.random.uniform(-4, 4, size=len(base_points) - 2)  # Add noise to middle points
            t_points = np.sort(np.concatenate([
                [0],  # Always include t=0
                base_points[1:-1] + noise,
                [t_max]  # Always include t_max
            ]))
        else:
            # Regular sampling every 12 hours
            t_points = np.arange(0, t_max + 12, 12)

        # Solve the ODE
        solution = solve_ivp(model, [0, t_max], y0, t_eval=t_points, method='RK45')

        # Extract results
        times = solution.t
        X = solution.y[0]
        S = solution.y[1]
        P = solution.y[2]

        # Add measurement noise
        X_noisy = X * (1 + noise_level * np.random.randn(len(X)))
        S_noisy = S * (1 + noise_level * np.random.randn(len(S)))
        P_noisy = P * (1 + noise_level * np.random.randn(len(P)))

        # Ensure no negative values after adding noise
        X_noisy = np.maximum(X_noisy, 1e-10)
        S_noisy = np.maximum(S_noisy, 1e-10)
        P_noisy = np.maximum(P_noisy, 1e-10)

        # Store the experiment data in the format expected by hybrid_models
        experiments.append({
            'times': jnp.array(times),
            'initial_state': {
                'X': float(y0[0]),
                'S': float(y0[1]),
                'P': float(y0[2])
            },
            'X_true': jnp.array(X_noisy),
            'S_true': jnp.array(S_noisy),
            'P_true': jnp.array(P_noisy)
        })

    return experiments


def main():
    # Set output directory
    output_dir = "examples/results/astaxanthin"
    os.makedirs(output_dir, exist_ok=True)

    # Define initial conditions as specified
    Y0a = [0.1, 10, 0]
    Y0b = [0.2, 5, 0]
    Y0c = [0.05, 14, 0]
    Y0j = [0.1, 12, 0]
    Y0d = [0.15, 7.5, 0]
    Y0e = [0.05, 5, 0]
    Y0f = [0.2, 15, 0]
    Y0l = [0.075, 7.5, 0]

    Y0_train = [Y0a, Y0b, Y0c, Y0j]
    Y0_test = [Y0d, Y0e, Y0f, Y0l]

    # Generate training and test datasets with specified initial conditions
    print("Generating synthetic data...")
    train_datasets = generate_data(initial_conditions=Y0_train, noise_level=0.05)
    test_datasets = generate_data(initial_conditions=Y0_test, noise_level=0.05)

    # Calculate normalization parameters
    print("Calculating normalization parameters...")
    norm_params = {}

    # Collect all values for normalization
    all_X = []
    all_S = []
    all_P = []

    for exp in train_datasets:
        all_X.extend(exp['X_true'])
        all_S.extend(exp['S_true'])
        all_P.extend(exp['P_true'])

    # Calculate mean and std for each state variable
    norm_params['X_mean'] = float(np.mean(all_X))
    norm_params['X_std'] = float(np.std(all_X))
    norm_params['S_mean'] = float(np.mean(all_S))
    norm_params['S_std'] = float(np.std(all_S))
    norm_params['P_mean'] = float(np.mean(all_P))
    norm_params['P_std'] = float(np.std(all_P))

    # Create the hybrid model builder
    print("Creating hybrid model...")
    builder = HybridModelBuilder()

    # Set normalization parameters
    builder.set_normalization_params(norm_params)

    # Add state variables
    builder.add_state('X')  # Biomass
    builder.add_state('S')  # Substrate
    builder.add_state('P')  # Product

    # Add trainable parameters (with positive constraints)
    builder.add_trainable_parameter('k_c', 50.0, bounds=(1.0, 200.0), transform="softplus")
    # Make y_sx a trainable parameter instead of a neural network
    builder.add_trainable_parameter('y_sx', 2.5, bounds=(0.1, 10.0), transform="softplus")

    # Define the simplified mechanistic ODE components
    def biomass_ode(inputs):
        # Extract state variables
        X = inputs['X']
        S = inputs['S']

        # Neural network parameter
        mu_m = inputs['mu_m']  # This will be replaced by a neural network

        # Trainable parameter
        k_c = inputs['k_c']

        # Simplified biomass ODE without decay term
        dX_dt = mu_m * S / (S + k_c) * X
        return dX_dt

    def substrate_ode(inputs):
        # Extract state variables
        X = inputs['X']
        S = inputs['S']

        # Neural network parameter
        mu_m = inputs['mu_m']  # This will be replaced by a neural network

        # Now y_sx is a trainable parameter, not a neural network
        y_sx = inputs['y_sx']

        # Trainable parameter
        k_c = inputs['k_c']

        # Simplified substrate ODE
        dS_dt = -y_sx * mu_m * S / (S + k_c) * X
        return dS_dt

    def product_ode(inputs):
        # Extract state variables
        X = inputs['X']

        # Neural network parameter
        beta = inputs['beta']  # This will be replaced by a neural network

        # Simplified product ODE without consumption term
        dP_dt = beta * X
        return dP_dt

    # Add mechanistic components
    builder.add_mechanistic_component('X', biomass_ode)
    builder.add_mechanistic_component('S', substrate_ode)
    builder.add_mechanistic_component('P', product_ode)

    # Define random keys for neural networks
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Replace parameters with neural networks
    # Note: We use softplus activation to ensure positive values
    builder.replace_with_nn(
        name='mu_m',  # Maximum specific growth rate
        input_features=['X', 'S', 'P'],
        hidden_dims=[16, 8],
        output_activation=jax.nn.soft_sign,  # Ensure positivity
        key=key1
    )

    builder.replace_with_nn(
        name='beta',  # Growth-independent yield coefficient
        input_features=['X', 'S', 'P'],
        hidden_dims=[16, 8],
        output_activation=jax.nn.soft_sign,  # Ensure positivity
        key=key2
    )

    # Build the model
    model = builder.build()

    # Create model configuration for documentation
    model_config = ModelConfig(
        state_names=['X', 'S', 'P'],
        mechanistic_components={
            'X': biomass_ode,
            'S': substrate_ode,
            'P': product_ode
        }
    )

    # Add neural network configurations
    model_config.add_nn(NeuralNetworkConfig(
        name='mu_m',
        input_features=['X', 'S', 'P'],
        hidden_dims=[16, 8],
        output_activation='softplus',
        seed=42
    ))

    # No longer need this config for y_sx

    model_config.add_nn(NeuralNetworkConfig(
        name='beta',
        input_features=['X', 'S', 'P'],
        hidden_dims=[16, 8],
        output_activation='softplus',
        seed=44
    ))

    # Create experiment manager
    experiment = ExperimentManager(
        model=model,
        model_config=model_config,
        norm_params=norm_params,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        output_dir=output_dir,
        experiment_name="astaxanthin_hybrid"
    )

    # Generate model documentation
    print("Generating model documentation...")
    experiment.generate_model_documentation()

    # Define solver configuration
    solver_config = SolverConfig(
        solver_type="dopri5",
        step_size_controller="pid",
        rtol=1e-4,
        atol=1e-6,
        max_steps=10000
    )

    # Train the model with component weights for different states
    print("Training model...")
    trained_model = experiment.train(
        state_names=['X', 'S', 'P'],
        num_epochs=5000,
        learning_rate=3e-3,
        early_stopping_patience=100,
        # Add component weights to give different importance to each state
        component_weights={'X': 2.0, 'S': 1.0, 'P': 1.0},
        # Biomass normal weight, Substrate less weight, Product higher weight
        solver_config=solver_config,
        save_checkpoints=True
    )

    # Evaluate the model
    print("Evaluating model...")
    evaluation_results = experiment.evaluate(
        state_names=['X', 'S', 'P'],
        solver_config=SolverConfig.for_evaluation(),
        verbose=True
    )

    # Visualize results
    print("Generating visualizations...")
    experiment.visualize(
        state_names=['X', 'S', 'P'],
        state_labels={'X': 'Biomass (g/L)', 'S': 'Substrate (g/L)', 'P': 'Product (mg/L)'},
        component_names=['Biomass Loss', 'Substrate Loss', 'Product Loss']
    )

    # Save all results
    print("Saving all results...")
    experiment.save_all_results()

    print(f"Experiment completed! Results saved to {output_dir}")

    return trained_model, train_datasets, test_datasets, experiment


if __name__ == "__main__":
    main()