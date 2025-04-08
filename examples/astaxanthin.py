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
import pandas as pd
from scipy.integrate import solve_ivp
from typing import List, Dict

# Import necessary hybrid_models components
from hybrid_models import (
    HybridModelBuilder,
    ModelConfig,
    NeuralNetworkConfig,
    ExperimentManager,
    SolverConfig,
    VariableRegistry
)
from hybrid_models.data import DatasetManager


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

        # Ground truth dynamics
        dX_dt = self.mu_m * S / (S + self.K_S) * X - self.mu_d * X
        dS_dt = -self.Y_S * self.mu_m * S / (S + self.K_S) * X
        dP_dt = self.b * X - self.k_d * X ** 2 * P / (P + self.K_p)

        return np.array([dX_dt, dS_dt, dP_dt])


def generate_experimental_data(initial_conditions, t_max=168.0, noise_level=0.15):
    """Generate synthetic experimental data from the ground truth model."""
    model = GroundTruthModel()
    all_data = []

    for exp_idx, y0 in enumerate(initial_conditions):
        # Generate irregular time points
        base_points = np.linspace(0, t_max, 15)
        noise = np.random.uniform(-4, 4, size=len(base_points) - 2)
        t_points = np.sort(np.concatenate([
            [0], base_points[1:-1] + noise, [t_max]
        ]))

        # Solve ODE
        solution = solve_ivp(model, [0, t_max], y0, t_eval=t_points, method='RK45')

        # Extract results and add noise
        times = solution.t
        X = solution.y[0] * (1 + noise_level * np.random.randn(len(times)))
        S = solution.y[1] * (1 + noise_level * np.random.randn(len(times)))
        P = solution.y[2] * (1 + noise_level * np.random.randn(len(times)))

        # Ensure positive values
        X = np.maximum(X, 1e-10)
        S = np.maximum(S, 1e-10)
        P = np.maximum(P, 1e-10)

        # Create rows for DataFrame - simple version with just X, S, P and time
        for i in range(len(times)):
            all_data.append({
                'time': times[i],
                'biomass': X[i],
                'substrate': S[i],
                'product': P[i],
                'RunID': f"Exp{exp_idx+1}"
            })

    return pd.DataFrame(all_data)


def main():
    # Set output directory
    output_dir = "examples/results/astaxanthin"
    os.makedirs(output_dir, exist_ok=True)

    # Define initial conditions
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

    # Generate experimental data
    print("Generating synthetic data...")
    df_train = generate_experimental_data(Y0_train, noise_level=0.20)
    df_test = generate_experimental_data(Y0_test, noise_level=0.20)

    # Set up DatasetManager for data handling
    print("Processing data...")
    manager = DatasetManager()

    # Load data with train/test split
    manager.load_from_dataframe(
        df=df_train,
        time_column='time',
        run_id_column='RunID'
    )

    # Define variables using VariableRegistry
    variables = VariableRegistry()

    # State variables (outputs)
    variables.add_state('biomass', internal_name='X', is_output=True)
    variables.add_state('substrate', internal_name='S', is_output=True)
    variables.add_state('product', internal_name='P', is_output=True)

    # Add variables to datasets
    manager.add_variables(variables.to_list(), df_train)

    # Calculate normalization parameters (from training data only)
    manager.calculate_norm_params()

    # Prepare training datasets
    train_datasets = manager.prepare_training_data()

    # Set up test data manager
    test_manager = DatasetManager()
    test_manager.load_from_dataframe(
        df=df_test,
        time_column='time',
        run_id_column='RunID'
    )
    test_manager.add_variables(variables.to_list(), df_test)
    test_datasets = test_manager.prepare_test_data()

    # Create the hybrid model
    print("Creating hybrid model...")
    builder = HybridModelBuilder()

    # Set normalization parameters
    builder.set_normalization_params(manager.norm_params)

    # Add state variables
    builder.add_state('X')
    builder.add_state('S')
    builder.add_state('P')

    # Add the trainable parameter k_c (substrate saturation constant)
    builder.add_trainable_parameter('k_c', 50.0, bounds=(1.0, 200.0), transform="softplus")

    # Add the trainable parameter y_sx (yield coefficient)
    builder.add_trainable_parameter('y_sx', 2.5, bounds=(0.1, 10.0), transform="softplus")

    # Define the mechanistic ODE components
    def biomass_ode(inputs):
        X = inputs['X']
        S = inputs['S']
        mu_m = inputs['mu_m']  # Neural network parameter
        k_c = inputs['k_c']    # Trainable parameter

        # Simplified biomass ODE
        dX_dt = mu_m * S / (S + k_c) * X
        return dX_dt

    def substrate_ode(inputs):
        X = inputs['X']
        S = inputs['S']
        mu_m = inputs['mu_m']  # Neural network parameter
        y_sx = inputs['y_sx']  # Trainable parameter
        k_c = inputs['k_c']    # Trainable parameter

        # Simplified substrate ODE
        dS_dt = -y_sx * mu_m * S / (S + k_c) * X
        return dS_dt

    def product_ode(inputs):
        X = inputs['X']
        beta = inputs['beta']  # Neural network parameter

        # Simplified product ODE
        dP_dt = beta * X
        return dP_dt

    # Add mechanistic components
    builder.add_mechanistic_component('X', biomass_ode)
    builder.add_mechanistic_component('S', substrate_ode)
    builder.add_mechanistic_component('P', product_ode)

    # Define random keys for neural networks
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Replace parameters with neural networks - using only X, S, P as inputs
    builder.replace_with_nn(
        name='mu_m',
        input_features=['X', 'S', 'P'],
        hidden_dims=[16, 8],
        output_activation=jax.nn.soft_sign,
        key=key1
    )

    builder.replace_with_nn(
        name='beta',
        input_features=['X', 'S', 'P'],
        hidden_dims=[16, 8],
        output_activation=jax.nn.soft_sign,
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

    model_config.add_nn(NeuralNetworkConfig(
        name='beta',
        input_features=['X', 'S', 'P'],
        hidden_dims=[16, 8],
        output_activation='softplus',
        seed=43
    ))

    # Create experiment manager
    experiment = ExperimentManager(
        model=model,
        model_config=model_config,
        norm_params=manager.norm_params,
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

    # Train the model
    print("Training model...")
    trained_model = experiment.train(
        state_names=['X', 'S', 'P'],
        num_epochs=5000,
        learning_rate=3e-3,
        early_stopping_patience=500,
        component_weights={'X': 1.0, 'S': 1, 'P': 1},
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