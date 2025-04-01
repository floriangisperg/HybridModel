import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from jaxtyping import Array, Float

# Import our framework (assuming it's installed or in the same directory)
from hybrid_models import (
    HybridModelBuilder,
    train_hybrid_model,
    evaluate_hybrid_model,
    calculate_metrics,
    normalize_data,
    calculate_rate,
    create_initial_random_key,
    # Import new data handling classes and functions
    TimeSeriesData,
    prepare_datasets_for_training,
    VariableType
)


# =============================================
# DATA LOADING AND PREPROCESSING
# =============================================

def load_bioprocess_data(file_path, run_ids=None, max_runs=2):
    """Load bioprocess experimental data using our generic data framework."""
    # Load data from Excel file
    data = pd.read_excel(file_path)

    # Use our new data loading framework
    datasets = TimeSeriesData.load_datasets_from_dataframe(
        df=data,
        time_column='feedtimer(h)',
        run_id_column='RunID',
        max_runs=max_runs
    )

    # Configure each dataset with the correct variable types
    for dataset in datasets:
        # Add state variables (X variables)
        dataset.add_state('CDW(g/L)', 'X')  # Biomass concentration
        dataset.add_state('Produktsol(g/L)', 'P')  # Product concentration

        # Add control variables (W variables)
        dataset.add_control('Temp(°C)', 'temp')  # Temperature

        # Add feed variables (F variables) - calculate rates automatically
        dataset.add_feed('Feed(L)', 'feed')  # Glucose feed
        dataset.add_feed('Base(L)', 'base')  # Base addition

        # Add more control variables
        dataset.add_control('InductorMASS(mg)', 'inductor_mass')
        dataset.add_control('Inductor(yesno)', 'inductor_switch')
        dataset.add_control('Reaktorvolumen(L)', 'reactor_volume')

        # Interpolate to handle missing values
        dataset.interpolate()

    # Calculate normalization parameters for all datasets
    norm_params = TimeSeriesData.calculate_normalization_parameters(datasets)

    # Apply normalization parameters
    TimeSeriesData.apply_normalization_parameters(datasets, norm_params)

    return datasets


# =============================================
# DEFINE BIOPROCESS MODEL
# =============================================

def define_bioprocess_model(norm_params):
    """Define the bioprocess model components."""

    # Create model builder
    builder = HybridModelBuilder()

    # Set normalization parameters
    builder.set_normalization_params(norm_params)

    # Add state variables (X variables in the framework)
    builder.add_state('X')  # Biomass
    builder.add_state('P')  # Product

    # Define dilution rate calculation
    def calculate_dilution_rate(inputs):
        """Calculate dilution rate from feed and base rates."""
        volume = inputs.get('reactor_volume', 1.0)  # Control variable (W)
        feed_rate = inputs.get('feed_rate', 0.0)  # Feed variable rate (F)
        base_rate = inputs.get('base_rate', 0.0)  # Feed variable rate (F)

        # Calculate total flow rate
        total_flow_rate = feed_rate + base_rate

        # Calculate dilution rate (avoid division by zero)
        dilution_rate = jnp.where(volume > 1e-6,
                                  total_flow_rate / volume,
                                  0.0)

        return dilution_rate

    # Define biomass ODE (mechanistic part)
    def biomass_ode(inputs):
        X = inputs['X']  # State variable (X)
        mu = inputs['growth_rate']  # Will be replaced by neural network

        # Calculate dilution
        dilution_rate = calculate_dilution_rate(inputs)

        # Biomass ODE with dilution
        dXdt = mu * X - dilution_rate * X

        return dXdt

    # Define product ODE (mechanistic part)
    def product_ode(inputs):
        X = inputs['X']  # State variable (X)
        P = inputs['P']  # State variable (X)
        vpx = inputs['product_rate']  # Will be replaced by neural network
        inductor_switch = inputs.get('inductor_switch', 0.0)  # Control variable (W)

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
    # Using both state variables (X) and control/feed variables (W, F)
    builder.replace_with_nn(
        name='growth_rate',
        input_features=['X', 'P', 'temp', 'feed', 'inductor_mass', 'inductor_switch'],
        hidden_dims=[8, 8],  # Smaller network
        key=key1
    )

    # Replace product formation rate with neural network
    # Using both state variables (X) and control/feed variables (W, F)
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

def plot_results(model, datasets, history, output_dir="results"):
    """Plot training results and predictions."""
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

    # Plot predictions for each dataset
    for i, dataset in enumerate(datasets):
        # Get predictions
        solution = solve_for_dataset(model, dataset)

        # Create plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Plot biomass (X)
        axs[0].plot(dataset['times'], dataset['X_true'], 'bo-', label='Measured')
        axs[0].plot(solution['times'], solution['X'], 'r-', label='Predicted')
        axs[0].set_title(f'Dataset {i + 1}: Biomass (CDW g/L)')
        axs[0].set_xlabel('Time (h)')
        axs[0].set_ylabel('CDW (g/L)')
        axs[0].legend()
        axs[0].grid(True)

        # Plot product (P)
        axs[1].plot(dataset['times'], dataset['P_true'], 'bo-', label='Measured')
        axs[1].plot(solution['times'], solution['P'], 'r-', label='Predicted')
        axs[1].set_title(f'Dataset {i + 1}: Product (g/L)')
        axs[1].set_xlabel('Time (h)')
        axs[1].set_ylabel('Product (g/L)')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'dataset_{i + 1}_predictions.png'))
        plt.close()


# =============================================
# MAIN FUNCTION
# =============================================

def main():
    # Load data
    print("Loading data...")
    datasets = load_bioprocess_data('Train_data_masked.xlsx')
    print(f"Loaded {len(datasets)} datasets")

    # Get normalization parameters from the first dataset
    norm_params = datasets[0].norm_params

    # Build model
    print("Building hybrid model...")
    model = define_bioprocess_model(norm_params)

    # Prepare datasets for training
    print("Preparing datasets for training...")
    training_datasets = prepare_datasets_for_training(datasets)

    # Train model with error handling
    print("Training model...")
    try:
        trained_model, history = train_hybrid_model(
            model=model,
            datasets=training_datasets,
            loss_fn=bioprocess_loss_function,
            num_epochs=500,  # Reduced for demonstration
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
    plot_results(trained_model, training_datasets, history, "bioprocess_results")

    # Evaluate model
    print("Evaluating model...")
    evaluation = {}

    for i, dataset in enumerate(training_datasets):
        # Get predictions
        solution = solve_for_dataset(trained_model, dataset)

        # Calculate metrics
        X_metrics = calculate_metrics(dataset['X_true'], solution['X'])
        P_metrics = calculate_metrics(dataset['P_true'], solution['P'])

        evaluation[f"dataset_{i}"] = {
            'X': X_metrics,
            'P': P_metrics
        }

        print(f"Dataset {i + 1}:")
        print(f"  X - R²: {X_metrics['r2']:.4f}, RMSE: {X_metrics['rmse']:.4f}")
        print(f"  P - R²: {P_metrics['r2']:.4f}, RMSE: {P_metrics['rmse']:.4f}")

    print("Process complete!")
    return trained_model, training_datasets, history


if __name__ == "__main__":
    main()