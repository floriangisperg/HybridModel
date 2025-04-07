"""
Bioprocess hybrid modeling example using the improved framework.

This script demonstrates the use of the enhanced hybrid modeling framework
with explicit control over model architecture, solver configuration, and
comprehensive model documentation and persistence.
"""
import jax
import jax.numpy as jnp
import pandas as pd
import os

# Import core hybrid modeling framework
from hybrid_models import (
    HybridModelBuilder,
    VariableRegistry,
    MSE, RelativeMSE,
    SolverConfig,
    ExperimentManager,
    ModelConfig, NeuralNetworkConfig
)

from hybrid_models.data import DatasetManager, VariableType


# =============================================
# DATA LOADING AND PREPROCESSING
# =============================================

def load_bioprocess_data(file_path, train_run_ids=None, test_run_ids=None, train_ratio=0.8):
    """
    Load bioprocess experimental data.
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

    # Calculate normalization parameters (from training data only)
    manager.calculate_norm_params()

    return manager


# =============================================
# DEFINE BIOPROCESS MODEL
# =============================================

def define_bioprocess_model(
        norm_params,
        ann_config=None  # Allow passing in custom ANN configurations
):
    """
    Define the bioprocess model with customizable neural network architecture.

    Args:
        norm_params: Normalization parameters
        ann_config: Optional dictionary with neural network configurations

    Returns:
        Tuple of (model, model_config)
    """
    # Set default ANN configuration if none provided
    if ann_config is None:
        ann_config = {
            'growth_rate': {
                'hidden_dims': [32, 32, 32],
                'output_activation': 'soft_sign',
                'input_features': ['X', 'P', 'temp', 'feed', 'inductor_mass'],
                'seed': 42
            },
            'product_rate': {
                'hidden_dims': [32, 32, 32],
                'output_activation': 'softplus',
                'input_features': ['X', 'P', 'temp', 'feed', 'inductor_mass'],
                'seed': 43
            }
        }

    # Create model builder
    builder = HybridModelBuilder()

    # Set normalization parameters
    builder.set_normalization_params(norm_params)

    # Add state variables
    builder.add_state('X')  # Biomass
    builder.add_state('P')  # Product

    # Create model config for documentation
    model_config = ModelConfig(
        state_names=['X', 'P'],
        mechanistic_components={}  # Will be filled in below
    )

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

    # Add mechanistic components to model config
    model_config.mechanistic_components['X'] = biomass_ode
    model_config.mechanistic_components['P'] = product_ode

    # Create neural network configurations
    for nn_name, nn_settings in ann_config.items():
        # Create neural network configuration
        nn_config = NeuralNetworkConfig(
            name=nn_name,
            input_features=nn_settings['input_features'],
            hidden_dims=nn_settings['hidden_dims'],
            output_activation=nn_settings['output_activation'],
            seed=nn_settings.get('seed', 0)
        )

        # Add to model config for documentation
        model_config.add_nn(nn_config)

        # Create the neural network in the builder
        builder.replace_with_nn(
            name=nn_name,
            input_features=nn_config.input_features,
            hidden_dims=nn_config.hidden_dims,
            output_activation=nn_config.get_activation_fn(),
            key=nn_config.get_random_key()
        )

    # Build the model
    model = builder.build()

    return model, model_config


# =============================================
# MAIN FUNCTION
# =============================================

def main():
    """Main function to run the bioprocess hybrid modeling experiment."""
    # Configure model architecture
    ann_config = {
        'growth_rate': {
            'hidden_dims': [64, 32, 16],  # Customized architecture
            'output_activation': 'soft_sign',
            'input_features': ['X', 'P', 'temp', 'feed', 'inductor_mass'],
            'seed': 42
        },
        'product_rate': {
            'hidden_dims': [32, 32],  # Different architecture
            'output_activation': 'softplus',
            'input_features': ['X', 'P', 'temp', 'feed', 'inductor_mass'],
            'seed': 43
        }
    }

    # Configure solver
    solver_config = SolverConfig(
        solver_type="tsit5",
        step_size_controller="pid",
        rtol=1e-2,
        atol=1e-4,
        max_steps=500000
    )

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

    # Build model with customized architecture
    print("Building hybrid model...")
    model, model_config = define_bioprocess_model(norm_params, ann_config)

    # Create experiment manager
    experiment = ExperimentManager(
        model=model,
        model_config=model_config,
        norm_params=norm_params,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        output_dir="results",
        experiment_name="bioprocess_experiment"
    )

    # Generate model documentation
    print("Generating model documentation...")
    experiment.generate_model_documentation()

    # Save normalization parameters
    experiment.save_normalization_parameters()

    # Train model with customized settings
    print("Training model...")
    trained_model = experiment.train(
        num_epochs=5000,
        learning_rate=1e-3,
        early_stopping_patience=2500,
        component_weights={'X': 1.0, 'P': 1.5},  # Weight product prediction higher
        loss_metric=MSE,
        solver_config=solver_config,
        save_checkpoints=True
    )



    # Visualize results
    print("Generating visualizations...")
    experiment.visualize(
        state_labels={'X': 'Biomass (CDW g/L)', 'P': 'Product (g/L)'},
        component_names=['Biomass Loss', 'Product Loss'],
    )

    # Save all results
    print("Saving all results...")
    result_paths = experiment.save_all_results()

    print("Experiment completed successfully!")
    print(f"All results saved to {experiment.output_dir}")

    return trained_model, train_datasets, test_datasets, experiment


if __name__ == "__main__":
    main()