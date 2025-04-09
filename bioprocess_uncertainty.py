"""
Bioprocess hybrid modeling example with uncertainty quantification.

This script demonstrates the use of the enhanced hybrid modeling framework
with uncertainty quantification through ensemble modeling.
"""
import jax
import jax.numpy as jnp
import pandas as pd
import os
import matplotlib.pyplot as plt

# Import core hybrid modeling framework
from hybrid_models import (
    HybridModelBuilder,
    VariableRegistry,
    MSE, RelativeMSE,
    SolverConfig,
    ExperimentManager,
    ModelConfig, NeuralNetworkConfig
)

# Import data management components
from hybrid_models.data import DatasetManager, VariableType

# Import the uncertainty module
from hybrid_models.uncertainty import (
    EnsembleModel,
    plot_predictions_with_uncertainty,
    compare_uncertainties
)


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

    return builder, model_config


# =============================================
# MAIN FUNCTION WITH UNCERTAINTY QUANTIFICATION
# =============================================

def main():
    """Main function to run the bioprocess hybrid modeling experiment with uncertainty quantification."""
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
    builder, model_config = define_bioprocess_model(norm_params, ann_config)

    # Create output directory
    output_dir = "results/bioprocess_uncertainty"
    os.makedirs(output_dir, exist_ok=True)

    # Generate model documentation
    print("Generating model documentation...")
    from hybrid_models.model_utils import save_model_description
    save_model_description(
        model=None,  # We don't have a built model yet
        model_config=model_config,
        norm_params=norm_params,
        filepath=os.path.join(output_dir, "model_documentation.txt")
    )

    # Save normalization parameters
    from hybrid_models.model_utils import save_normalization_params
    save_normalization_params(
        norm_params=norm_params,
        filepath=os.path.join(output_dir, "normalization_params.txt")
    )

    # Create an ensemble of models
    print("Creating ensemble of models for uncertainty quantification...")
    ensemble = EnsembleModel.from_builder(
        builder=builder,
        n_models=5,  # Use 5 models for this example (increase for production)
        state_names=['X', 'P'],
        seed=42
    )

    # Create loss function
    from hybrid_models.loss import create_hybrid_model_loss
    loss_fn = create_hybrid_model_loss(
        state_names=['X', 'P'],
        loss_metric=MSE,
        component_weights={'X': 1.0, 'P': 1.5},  # Weight product prediction higher
        solve_kwargs=solver_config.to_dict()
    )

    # Train the ensemble
    print("Training ensemble models...")
    trained_ensemble, histories = ensemble.train(
        datasets=train_datasets,
        loss_fn=loss_fn,
        num_epochs=2500,  # Reduced from 5000 for example
        learning_rate=1e-3,
        early_stopping_patience=250,  # Reduced from 2500 for example
        verbose=True
    )

    # Plot training histories for each model in the ensemble
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(histories):
        plt.plot(history['loss'], label=f'Model {i + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for Ensemble Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "ensemble_training_loss.png"))
    plt.close()

    # Make predictions with uncertainty
    print("Making predictions with uncertainty quantification...")

    # Prepare solver config dictionary for predictions
    solver_kwargs = solver_config.to_dict()

    # Predict for training datasets
    train_predictions = []
    for dataset in train_datasets:
        prediction = trained_ensemble.predict(dataset, solver_kwargs)
        train_predictions.append(prediction)

    # Predict for test datasets
    test_predictions = []
    for dataset in test_datasets:
        prediction = trained_ensemble.predict(dataset, solver_kwargs)
        test_predictions.append(prediction)

    # Visualize predictions with uncertainty
    print("Generating visualizations with uncertainty bands...")

    # Visualize training predictions
    for i, (dataset, prediction) in enumerate(zip(train_datasets, train_predictions)):
        # Plot predictions with uncertainty
        plot_predictions_with_uncertainty(
            prediction=prediction,
            dataset=dataset,
            state_names=['X', 'P'],
            output_dir=output_dir,
            prefix=f"train_{i + 1}_"
        )

        # Compare with true values
        for state in ['X', 'P']:
            compare_uncertainties(
                ensemble_predictions=prediction,
                true_values=dataset,
                state_name=state,
                output_dir=output_dir,
                filename=f"train_{i + 1}_{state}_comparison.png",
                title=f"Training Run {i + 1}: {state}"
            )

    # Visualize test predictions
    for i, (dataset, prediction) in enumerate(zip(test_datasets, test_predictions)):
        # Plot predictions with uncertainty
        plot_predictions_with_uncertainty(
            prediction=prediction,
            dataset=dataset,
            state_names=['X', 'P'],
            output_dir=output_dir,
            prefix=f"test_{i + 1}_"
        )

        # Compare with true values
        for state in ['X', 'P']:
            compare_uncertainties(
                ensemble_predictions=prediction,
                true_values=dataset,
                state_name=state,
                output_dir=output_dir,
                filename=f"test_{i + 1}_{state}_comparison.png",
                title=f"Test Run {i + 1}: {state}"
            )

    # Evaluate ensemble performance
    print("Evaluating ensemble performance...")
    train_metrics = trained_ensemble.evaluate(train_datasets, solver_kwargs)
    test_metrics = trained_ensemble.evaluate(test_datasets, solver_kwargs)

    # Save metrics summary
    with open(os.path.join(output_dir, "ensemble_metrics.txt"), 'w') as f:
        f.write("BIOPROCESS ENSEMBLE MODEL EVALUATION\n")
        f.write("=================================\n\n")

        f.write("TRAINING METRICS\n")
        f.write("--------------\n")
        for state, metrics in train_metrics['overall'].items():
            f.write(f"{state}:\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.6f}\n")

        f.write("\nTEST METRICS\n")
        f.write("-----------\n")
        for state, metrics in test_metrics['overall'].items():
            f.write(f"{state}:\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.6f}\n")

    # Print results summary
    print("\nTraining Performance:")
    for state, metrics in train_metrics['overall'].items():
        print(f"  {state}: R² = {metrics.get('r2', 'N/A'):.4f}, RMSE = {metrics.get('rmse', 'N/A'):.4f}")

    print("\nTest Performance:")
    for state, metrics in test_metrics['overall'].items():
        print(f"  {state}: R² = {metrics.get('r2', 'N/A'):.4f}, RMSE = {metrics.get('rmse', 'N/A'):.4f}")

    print("\nExperiment completed successfully!")
    print(f"All results saved to {output_dir}")

    return trained_ensemble, train_datasets, test_datasets


if __name__ == "__main__":
    main()