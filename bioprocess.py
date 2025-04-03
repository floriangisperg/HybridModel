import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

# Import our framework
from hybrid_models import (
    HybridModelBuilder,
    train_hybrid_model,
    evaluate_hybrid_model,
    calculate_metrics,
    create_initial_random_key
)

# Import our new modules
from hybrid_models.visualization import plot_all_results
from hybrid_models.evaluation_utils import evaluate_model_performance

# Import our data module
from hybrid_models.data import DatasetManager, VariableType


# =============================================
# DATA LOADING AND PREPROCESSING
# =============================================

def load_bioprocess_data(file_path, train_run_ids=None, test_run_ids=None, train_ratio=0.8):
    """
    Load bioprocess experimental data using the new DatasetManager.

    Args:
        file_path: Path to Excel file with data
        train_run_ids: Specific run IDs to use for training (optional)
        test_run_ids: Specific run IDs to use for testing (optional)
        train_ratio: Ratio of runs to use for training if specific IDs not provided

    Returns:
        DatasetManager object with loaded data
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

    # Define variables to load
    variable_definitions = [
        # State variables (output variables)
        ('CDW(g/L)', VariableType.STATE, 'X', True, False),
        ('Produktsol(g/L)', VariableType.STATE, 'P', True, False),

        # Control variables
        ('Temp(°C)', VariableType.CONTROL, 'temp', False, False),
        ('InductorMASS(mg)', VariableType.CONTROL, 'inductor_mass', False, False),
        ('Inductor(yesno)', VariableType.CONTROL, 'inductor_switch', False, False),

        # Feed variables (with rate calculation)
        ('Feed(L)', VariableType.FEED, 'feed', False, True),
        ('Base(L)', VariableType.FEED, 'base', False, True),
        ('Reaktorvolumen(L)', VariableType.CONTROL, 'reactor_volume', False, False),
    ]

    # Add variables to datasets
    manager.add_variables(variable_definitions, data)

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
    key = create_initial_random_key(42)
    key1, key2 = jax.random.split(key)

    # Replace growth rate with neural network
    builder.replace_with_nn(
        name='growth_rate',
        input_features=['X', 'P', 'temp', 'feed', 'inductor_mass', 'inductor_switch'],
        hidden_dims=[8, 8],  # Smaller network
        key=key1
    )

    # Replace product formation rate with neural network
    builder.replace_with_nn(
        name='product_rate',
        input_features=['X', 'P', 'temp', 'feed', 'inductor_mass', 'inductor_switch'],
        hidden_dims=[8, 8],  # Smaller network
        output_activation=jax.nn.softplus,  # Ensure non-negative rate
        key=key2
    )

    # Build and return the model
    return builder.build()


# =============================================
# DEFINE LOSS FUNCTION
# =============================================

def bioprocess_loss_function(model, datasets):
    """Loss function for bioprocess model training."""
    total_loss = 0.0
    total_x_loss = 0.0
    total_p_loss = 0.0

    for dataset in datasets:
        # Get predictions
        solution = model.solve(
            initial_state=dataset['initial_state'],
            t_span=(dataset['times'][0], dataset['times'][-1]),
            evaluation_times=dataset['times'],
            args={
                'time_dependent_inputs': dataset['time_dependent_inputs'],
                'static_inputs': dataset.get('static_inputs', {})
            },
            max_steps=100000,
            rtol=1e-2,  # Slightly relaxed tolerance
            atol=1e-4  # Slightly relaxed tolerance
        )

        # Calculate loss
        X_pred = solution['X']
        P_pred = solution['P']
        X_true = dataset['X_true']
        P_true = dataset['P_true']

        X_loss = jnp.mean(jnp.square(X_pred - X_true))
        P_loss = jnp.mean(jnp.square(P_pred - P_true))

        # Add to total loss
        run_loss = X_loss + P_loss
        total_loss += run_loss
        total_x_loss += X_loss
        total_p_loss += P_loss

    # Return average loss
    n_datasets = len(datasets)
    return total_loss / n_datasets, (total_x_loss / n_datasets, total_p_loss / n_datasets)


# =============================================
# SOLVE MODEL FOR A DATASET
# =============================================

def solve_for_dataset(model, dataset):
    """Solve the model for a given dataset."""
    solution = model.solve(
        initial_state=dataset['initial_state'],
        t_span=(dataset['times'][0], dataset['times'][-1]),
        evaluation_times=dataset['times'],
        args={
            'time_dependent_inputs': dataset['time_dependent_inputs'],
            'static_inputs': dataset.get('static_inputs', {})
        },
        max_steps=100000,
        rtol=1e-2,  # Slightly relaxed tolerance
        atol=1e-4  # Slightly relaxed tolerance
    )

    return solution


# =============================================
# MAIN FUNCTION
# =============================================

def main():
    # Load data with train/test split
    print("Loading data...")
    data_manager = load_bioprocess_data(
        'testtrain.xlsx',
        train_run_ids=None,#[58, 61, 53],
        test_run_ids=None,#[63, 101],
        train_ratio=0.8
    )

    print(f"Loaded {len(data_manager.train_datasets)} training datasets and "
          f"{len(data_manager.test_datasets)} test datasets")

    # Get normalization parameters from training data only
    norm_params = data_manager.norm_params

    # Prepare datasets for training
    train_datasets = data_manager.prepare_training_data()
    test_datasets = data_manager.prepare_test_data()

    # Build model
    print("Building hybrid model...")
    model = define_bioprocess_model(norm_params)

    # Train model with error handling
    print("Training model...")
    try:
        # Split data for validation if enough datasets are available
        if len(train_datasets) > 3:
            # Use last 20% of train_datasets for validation
            split_idx = int(len(train_datasets) * 0.8)
            validation_datasets = train_datasets[split_idx:]
            main_train_datasets = train_datasets[:split_idx]

            trained_model, history, validation_history = train_hybrid_model(
                model=model,
                datasets=main_train_datasets,
                loss_fn=bioprocess_loss_function,
                num_epochs=500,
                learning_rate=1e-3,
                early_stopping_patience=50,
                validation_datasets=validation_datasets
            )
        else:
            # Not enough data for validation split
            trained_model, history = train_hybrid_model(
                model=model,
                datasets=train_datasets,
                loss_fn=bioprocess_loss_function,
                num_epochs=500,
                learning_rate=1e-3,
                early_stopping_patience=50
            )
            validation_history = None

        print("Training complete")
    except Exception as e:
        print(f"Error during training: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nFalling back to returning the untrained model")
        history = {"loss": [], "aux": []}
        validation_history = None
        trained_model = model

    # Use our new visualization module to plot results
    print("Plotting results...")
    plot_all_results(
        model=trained_model,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        history=history,
        solve_fn=solve_for_dataset,
        state_names=['X', 'P'],
        output_dir="bioprocess_results",
        state_labels={'X': 'Biomass (CDW g/L)', 'P': 'Product (g/L)'},
        component_names=['Biomass Loss', 'Product Loss'],
        validation_history=validation_history
    )

    # Use our new evaluation module to evaluate the model
    print("\nEvaluating model...")
    train_evaluation = evaluate_model_performance(
        model=trained_model,
        datasets=train_datasets,
        solve_fn=solve_for_dataset,
        state_names=['X', 'P'],
        dataset_type="Training",
        save_metrics=True,
        output_dir="bioprocess_results",
        metrics_filename="training_metrics.txt"
    )

    if test_datasets:
        test_evaluation = evaluate_model_performance(
            model=trained_model,
            datasets=test_datasets,
            solve_fn=solve_for_dataset,
            state_names=['X', 'P'],
            dataset_type="Test",
            save_metrics=True,
            output_dir="bioprocess_results",
            metrics_filename="test_metrics.txt"
        )

    print("Process complete!")
    return trained_model, train_datasets, test_datasets, history


if __name__ == "__main__":
    main()