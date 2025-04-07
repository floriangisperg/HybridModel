import jax
import jax.numpy as jnp
import pandas as pd
import os

# Import hybrid modeling framework
from hybrid_models import (
    HybridModelBuilder,
    VariableRegistry,
    MSE,
    RelativeMSE
)

# Import our improved modules
from hybrid_models.experiment import ExperimentManager
from hybrid_models.solver import SolverConfig
from hybrid_models.data import DatasetManager, VariableType


# =============================================
# DATA LOADING AND PREPROCESSING
# =============================================

def load_bioprocess_data(file_path, train_run_ids=None, test_run_ids=None, train_ratio=0.8):
    """
    Load bioprocess experimental data using the new approach.
    """
    # Load data from Excel file
    data = pd.read_excel(file_path)

    # Create dataset manager
    manager = DatasetManager()

    # Load data with train/test split
    manager.load_from_dataframe(
        df=data,
        time_column='feedtimer(h)',
        run_id_column='RunID',
        train_run_ids=train_run_ids,
        test_run_ids=test_run_ids,
        train_ratio=train_ratio
    )

    # Define variables using VariableRegistry
    variables = VariableRegistry()

    # State variables (outputs)
    variables.add_state('CDW(g/L)', internal_name='X')
    variables.add_state('Produktsol(g/L)', internal_name='P')

    # Control variables
    variables.add_control('Temp(°C)', internal_name='temp')
    variables.add_control('InductorMASS(mg)', internal_name='inductor_mass')
    variables.add_control('Inductor(yesno)', internal_name='inductor_switch')
    variables.add_control('Reaktorvolumen(L)', internal_name='reactor_volume')

    # Feed variables (with rate calculation)
    variables.add_feed('Feed(L)', internal_name='feed')
    variables.add_feed('Base(L)', internal_name='base')

    # Add variables to datasets
    manager.add_variables(variables.to_list(), data)

    # Calculate normalization parameters (only from training data)
    manager.calculate_norm_params()

    return manager


# =============================================
# DEFINE BIOPROCESS MODEL
# =============================================

def define_bioprocess_model(norm_params):
    """Define the bioprocess model components."""
    # Create model builder
    builder = HybridModelBuilder()

    # Set normalization parameters
    builder.set_normalization_params(norm_params)

    # Add state variables
    builder.add_state('X')  # Biomass
    builder.add_state('P')  # Product

    # Define dilution rate calculation
    def calculate_dilution_rate(inputs):
        """Calculate dilution rate from feed and base rates."""
        volume = inputs.get('reactor_volume', 1.0)
        feed_rate = inputs.get('feed_rate', 0.0)
        base_rate = inputs.get('base_rate', 0.0)

        # Calculate total flow rate
        total_flow_rate = feed_rate + base_rate

        # Calculate dilution rate (avoid division by zero)
        dilution_rate = jnp.where(volume > 1e-6,
                                  total_flow_rate / volume,
                                  0.0)

        return dilution_rate

    # Define biomass ODE (mechanistic part)
    def biomass_ode(inputs):
        X = inputs['X']
        mu = inputs['growth_rate']  # Will be replaced by neural network

        # Calculate dilution
        dilution_rate = calculate_dilution_rate(inputs)

        # Biomass ODE with dilution
        dXdt = mu * X - dilution_rate * X

        return dXdt

    # Define product ODE (mechanistic part)
    def product_ode(inputs):
        X = inputs['X']
        P = inputs['P']
        vpx = inputs['product_rate']  # Will be replaced by neural network
        inductor_switch = inputs.get('inductor_switch', 0.0)

        # Calculate dilution
        dilution_rate = calculate_dilution_rate(inputs)

        # Product ODE with dilution
        dPdt = vpx * X * inductor_switch - dilution_rate * P

        return dPdt

    # Add mechanistic components
    builder.add_mechanistic_component('X', biomass_ode)
    builder.add_mechanistic_component('P', product_ode)

    # Create random key for neural network initialization
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Replace growth rate with neural network
    builder.replace_with_nn(
        name='growth_rate',
        input_features=['X', 'P', 'temp', 'feed', 'inductor_mass'],
        hidden_dims=[32, 32, 32],
        output_activation=jax.nn.soft_sign,  # Constrains output to [-1, 1]
        key=key1
    )

    # Replace product formation rate with neural network
    builder.replace_with_nn(
        name='product_rate',
        input_features=['X', 'P', 'temp', 'feed', 'inductor_mass'],
        hidden_dims=[32, 32, 32],
        output_activation=jax.nn.softplus,  # Ensure non-negative rate
        key=key2
    )

    # Build and return the model
    return builder.build()


# =============================================
# MAIN FUNCTION
# =============================================

def main():
    """Main function to run the bioprocess hybrid modeling experiment."""
    # Load data
    print("Loading data...")
    data_manager = load_bioprocess_data(
        'testtrain.xlsx',
        train_ratio=0.8
    )

    print(f"Loaded {len(data_manager.train_datasets)} training datasets and "
          f"{len(data_manager.test_datasets)} test datasets")

    # Get normalization parameters
    norm_params = data_manager.norm_params

    # Prepare datasets for training and testing
    train_datasets = data_manager.prepare_training_data()
    test_datasets = data_manager.prepare_test_data()

    # Build model
    print("Building hybrid model...")
    model = define_bioprocess_model(norm_params)

    # Create experiment manager
    experiment = ExperimentManager(
        model=model,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        output_dir="results",
        experiment_name="bioprocess_experiment"
    )

    # Train model
    trained_model = experiment.train(
        state_names=['X', 'P'],
        num_epochs=5000,
        learning_rate=1e-3,
        early_stopping_patience=2500,
        component_weights={'X': 1.0, 'P': 1.0},
        loss_metric=MSE,
        solver_config=SolverConfig.for_training()
    )

    # Evaluate model
    metrics = experiment.evaluate(
        state_names=['X', 'P'],
        solver_config=SolverConfig.for_evaluation()
    )

    # Visualize results
    experiment.visualize(
        state_names=['X', 'P'],
        state_labels={'X': 'Biomass (CDW g/L)', 'P': 'Product (g/L)'},
        component_names=['Biomass Loss', 'Product Loss']
    )

    # Save experiment summary
    experiment.save_results_summary()

    print("Experiment completed successfully!")
    return trained_model, train_datasets, test_datasets


if __name__ == "__main__":
    main()