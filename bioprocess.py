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

# Import our new data module
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
# PLOT RESULTS
# =============================================

def plot_results(model, train_datasets, test_datasets, history, output_dir="results"):
    """
    Plot training results and predictions for both training and test datasets.

    Args:
        model: Trained hybrid model
        train_datasets: List of training datasets
        test_datasets: List of test datasets
        history: Training history
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # Plot component losses
    plt.figure(figsize=(10, 6))
    x_losses = [aux[0] for aux in history['aux']]
    p_losses = [aux[1] for aux in history['aux']]
    plt.plot(x_losses, 'g-', label='X Loss')
    plt.plot(p_losses, 'r-', label='P Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'component_losses.png'))
    plt.close()

    # Plot predictions for training datasets
    for i, dataset in enumerate(train_datasets):
        solution = solve_for_dataset(model, dataset)

        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Plot biomass (X)
        axs[0].plot(dataset['times'], dataset['X_true'], 'bo-', label='Measured')
        axs[0].plot(solution['times'], solution['X'], 'r-', label='Predicted')
        axs[0].set_title(f'Training Dataset {i + 1}: Biomass (CDW g/L)')
        axs[0].set_xlabel('Time (h)')
        axs[0].set_ylabel('CDW (g/L)')
        axs[0].legend()
        axs[0].grid(True)

        # Plot product (P)
        axs[1].plot(dataset['times'], dataset['P_true'], 'bo-', label='Measured')
        axs[1].plot(solution['times'], solution['P'], 'r-', label='Predicted')
        axs[1].set_title(f'Training Dataset {i + 1}: Product (g/L)')
        axs[1].set_xlabel('Time (h)')
        axs[1].set_ylabel('Product (g/L)')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'train_dataset_{i + 1}_predictions.png'))
        plt.close()

    # Plot predictions for test datasets
    for i, dataset in enumerate(test_datasets):
        solution = solve_for_dataset(model, dataset)

        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Plot biomass (X)
        axs[0].plot(dataset['times'], dataset['X_true'], 'bo-', label='Measured')
        axs[0].plot(solution['times'], solution['X'], 'r-', label='Predicted')
        axs[0].set_title(f'Test Dataset {i + 1}: Biomass (CDW g/L)')
        axs[0].set_xlabel('Time (h)')
        axs[0].set_ylabel('CDW (g/L)')
        axs[0].legend()
        axs[0].grid(True)

        # Plot product (P)
        axs[1].plot(dataset['times'], dataset['P_true'], 'bo-', label='Measured')
        axs[1].plot(solution['times'], solution['P'], 'r-', label='Predicted')
        axs[1].set_title(f'Test Dataset {i + 1}: Product (g/L)')
        axs[1].set_xlabel('Time (h)')
        axs[1].set_ylabel('Product (g/L)')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'test_dataset_{i + 1}_predictions.png'))
        plt.close()


# =============================================
# EVALUATE MODEL
# =============================================

def evaluate_model(model, datasets, dataset_type="training"):
    """
    Evaluate model performance on datasets.

    Args:
        model: Trained hybrid model
        datasets: List of datasets to evaluate
        dataset_type: String to identify dataset type in output

    Returns:
        Dictionary of evaluation metrics
    """
    evaluation = {}

    for i, dataset in enumerate(datasets):
        # Get predictions
        solution = solve_for_dataset(model, dataset)

        # Calculate metrics
        X_metrics = calculate_metrics(dataset['X_true'], solution['X'])
        P_metrics = calculate_metrics(dataset['P_true'], solution['P'])

        evaluation[f"{dataset_type}_dataset_{i}"] = {
            'X': X_metrics,
            'P': P_metrics
        }

        print(f"{dataset_type.capitalize()} Dataset {i + 1}:")
        print(f"  X - R²: {X_metrics['r2']:.4f}, RMSE: {X_metrics['rmse']:.4f}")
        print(f"  P - R²: {P_metrics['r2']:.4f}, RMSE: {P_metrics['rmse']:.4f}")

    return evaluation


# =============================================
# MAIN FUNCTION
# =============================================

def main():
    # Load data with train/test split
    print("Loading data...")
    data_manager = load_bioprocess_data(
        'Train_data_masked.xlsx',
        train_run_ids=[58, 61, 53 ],
        test_run_ids=[63, 101],
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
        trained_model, history = train_hybrid_model(
            model=model,
            datasets=train_datasets,
            loss_fn=bioprocess_loss_function,
            num_epochs=500,
            learning_rate=1e-3,
            early_stopping_patience=50
        )
        print("Training complete")
    except Exception as e:
        print(f"Error during training: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nFalling back to returning the untrained model")
        history = {"loss": [], "aux": []}
        trained_model = model

    # Plot results
    print("Plotting results...")
    plot_results(trained_model, train_datasets, test_datasets, history, "bioprocess_results")

    # Evaluate model on training data
    print("\nEvaluating model on training data...")
    train_evaluation = evaluate_model(trained_model, train_datasets, "training")

    # Evaluate model on test data
    if test_datasets:
        print("\nEvaluating model on test data...")
        test_evaluation = evaluate_model(trained_model, test_datasets, "test")

    print("Process complete!")
    return trained_model, train_datasets, test_datasets, history


if __name__ == "__main__":
    main()