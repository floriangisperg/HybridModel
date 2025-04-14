"""
Bioprocess hybrid modeling example using the improved and abstracted framework.

This script defines the specific bioprocess problem configuration (data,
mechanistic equations, NN structures) and utilizes the hybrid_models package's
higher-level abstractions (DatasetManager, ModelConfig, ExperimentManager,
model factory) to build, train, and evaluate the hybrid model.
"""

import jax
import jax.numpy as jnp
import pandas as pd
import os
from typing import Dict, Any

# Import necessary components from the hybrid_models package
from hybrid_models import (
    # Data Handling
    DatasetManager,
    VariableRegistry,
    VariableType,
    # Model Definition & Configuration
    ModelConfig,
    NeuralNetworkConfig,
    # Model Building (assuming factory function exists)
    create_model_from_config,  # Assuming this factory is added to the package
    # Solver
    SolverConfig,
    # Loss
    MSE,
    # Experiment Management
    ExperimentManager,
    # Utilities (if needed, e.g., for activation functions if not handled by factory)
    # utils, # Less likely needed here now
)

# =============================================
# CONFIGURATION SECTION
# =============================================

# --- Experiment Setup ---
MASTER_SEED = 45  # Master seed for reproducible NN initialization
EXPERIMENT_BASE_NAME = "bioprocess_experiment"
OUTPUT_DIR_BASE = "results_refactored"
DATA_FILE = "testtrain.xlsx"

# --- Neural Network Architecture ---
# Define the structure and inputs for each NN component
ANN_CONFIG = {
    "growth_rate": {
        "input_features": ["X", "P", "temp", "feed", "inductor_mass"],
        "hidden_dims": [32, 32, 32],
        "output_activation": "soft_sign",  # String name is fine for NeuralNetworkConfig
    },
    "product_rate": {
        "input_features": ["X", "P", "temp", "feed", "inductor_mass"],
        "hidden_dims": [32, 32, 32],
        "output_activation": "softplus",  # String name is fine for NeuralNetworkConfig
    },
}

# --- Solver Configuration ---
# Define parameters for the ODE solver used during training/evaluation
SOLVER_CONFIG_PARAMS = {
    "solver_type": "tsit5",
    "step_size_controller": "pid",
    "rtol": 1e-2,
    "atol": 1e-4,
    "max_steps": 500000,
}

# --- Training Configuration ---
TRAINING_PARAMS = {
    "num_epochs": 10000,
    "learning_rate": 1e-3,
    "early_stopping_patience": 2500,
    "early_stopping_min_delta": 1e-6,  # Define min delta for early stopping
    "loss_metric": MSE,  # Use the class directly
    "component_weights": {"X": 1.0, "P": 10.0},  # Weights for loss components
    "save_checkpoints": True,
    "verbose": True,
}

# --- Visualization Configuration ---
VISUALIZATION_LABELS = {
    "state_labels": {"X": "Biomass (CDW g/L)", "P": "Product (g/L)"},
    "component_names": [
        "Biomass Loss",
        "Product Loss",
    ],  # Corresponds to loss_fn aux output order
}


# =============================================
# DATA LOADING AND PREPROCESSING
# =============================================


def load_bioprocess_data(
    file_path,
    time_col,
    run_id_col,
    train_ratio=0.8,
    train_run_ids=None,
    test_run_ids=None,
) -> DatasetManager:
    """Loads and preprocesses bioprocess data using DatasetManager and VariableRegistry."""
    print(f"Loading data from: {file_path}")

    manager = DatasetManager()
    manager.load_from_dataframe(
        df=data,
        time_column=time_col,
        run_id_column=run_id_col,
        train_run_ids=train_run_ids,
        test_run_ids=test_run_ids,
        train_ratio=train_ratio,
    )

    # Define variables using the fluent VariableRegistry API
    variables = (
        VariableRegistry()
        .add_state("CDW(g/L)", internal_name="X", is_output=True)
        .add_state("Produktsol(g/L)", internal_name="P", is_output=True)
        .add_control("Temp(°C)", internal_name="temp")
        .add_control("InductorMASS(mg)", internal_name="inductor_mass")
        .add_control("Inductor(yesno)", internal_name="inductor_switch")
        .add_control("Reaktorvolumen(L)", internal_name="reactor_volume")
        .add_feed(
            "Feed(L)", internal_name="feed", calculate_rate=True
        )  # Calculate feed rate
        .add_feed(
            "Base(L)", internal_name="base", calculate_rate=True
        )  # Calculate base rate
    )

    # Add variables to the datasets managed by DatasetManager
    manager.add_variables(
        variables.to_list(), data=data
    )  # Pass data again for variable extraction

    # Calculate normalization parameters based on training data
    manager.calculate_norm_params()

    print(
        f"Loaded {len(manager.train_datasets)} training datasets and "
        f"{len(manager.test_datasets)} test datasets."
    )
    print(
        f"Normalization parameters calculated for: {list(manager.norm_params.keys())}"
    )
    return manager


# =============================================
# MECHANISTIC MODEL COMPONENTS
# =============================================
# These functions define the physics/biology specific to this bioprocess.
# They remain defined here as they are part of the problem specification.


def calculate_dilution_rate(inputs: Dict[str, Any]) -> jax.Array:
    """Calculates the dilution rate D = F/V."""
    volume = inputs.get("reactor_volume", 1.0)  # Default to 1.0 if not provided
    # Use the calculated rates (assuming VariableRegistry added them)
    feed_rate = inputs.get("feed_rate", 0.0)
    base_rate = inputs.get("base_rate", 0.0)
    total_flow_rate = feed_rate + base_rate
    # Avoid division by zero
    dilution_rate = jnp.where(volume > 1e-6, total_flow_rate / volume, 0.0)
    return dilution_rate


def biomass_ode(inputs: Dict[str, Any]) -> jax.Array:
    """ODE component for biomass (X) dynamics: dX/dt = mu*X - D*X."""
    X = inputs["X"]
    mu = inputs["growth_rate"]  # This will be provided by the NN
    dilution_rate = calculate_dilution_rate(inputs)
    dXdt = mu * X - dilution_rate * X
    return dXdt


def product_ode(inputs: Dict[str, Any]) -> jax.Array:
    """ODE component for product (P) dynamics: dP/dt = vpx*X*switch - D*P."""
    X = inputs["X"]
    P = inputs["P"]
    vpx = inputs["product_rate"]  # This will be provided by the NN
    inductor_switch = inputs.get("inductor_switch", 0.0)  # Default to 0 if not provided
    dilution_rate = calculate_dilution_rate(inputs)
    dPdt = vpx * X * inductor_switch - dilution_rate * P
    return dPdt


# =============================================
# MAIN EXECUTION SCRIPT
# =============================================


def main(master_seed: int):
    """Main function to run the bioprocess hybrid modeling experiment."""
    print(f"\n--- Starting Bioprocess Experiment Run ---")
    print(f"Using Master Seed: {master_seed}")

    # --- 1. Data Loading ---
    data_manager = load_bioprocess_data(
        file_path=DATA_FILE,
        time_col="feedtimer(h)",
        run_id_col="RunID",
        train_ratio=0.8,
    )
    norm_params = data_manager.norm_params
    train_datasets = data_manager.prepare_training_data()
    test_datasets = data_manager.prepare_test_data()

    if not train_datasets:
        print("Error: No training data loaded. Exiting.")
        return
    if not test_datasets:
        print("Warning: No test data loaded. Evaluation might be limited.")
        # Optionally use training data for evaluation if needed for testing purposes
        # test_datasets = train_datasets

    # --- 2. Define Model Configuration ---
    print("Defining model configuration...")
    model_config = ModelConfig(
        state_names=["X", "P"],
        # Map the state name to the corresponding mechanistic function
        mechanistic_components={
            "X": biomass_ode,
            "P": product_ode,
        },
        # Create NeuralNetworkConfig objects from the ANN_CONFIG dictionary
        neural_networks=[
            NeuralNetworkConfig(
                name=name,
                input_features=config["input_features"],
                hidden_dims=config["hidden_dims"],
                output_activation=config["output_activation"],
                # Seed info here is primarily for documentation if desired
                # The actual key generation happens in the factory
                seed=master_seed,
            )
            for name, config in ANN_CONFIG.items()
        ],
        # Add trainable parameters here if needed:
        # trainable_parameters={
        #     "param_name": {"initial_value": 1.0, "bounds": (0, 10), "transform": "sigmoid"}
        # }
    )

    # --- 3. Build Model using Factory Function ---
    print("Building hybrid model from configuration...")
    # This assumes create_model_from_config exists in the package
    try:
        model = create_model_from_config(
            model_config, norm_params, master_seed=master_seed
        )
        print("Model built successfully.")
    except NameError:
        print("Error: `create_model_from_config` not found.")
        print(
            "Please ensure this factory function exists in the hybrid_models package."
        )
        return
    except Exception as e:
        print(f"Error building model: {e}")
        raise

    # --- 4. Setup Experiment ---
    print("Setting up experiment manager...")
    experiment_name = f"{EXPERIMENT_BASE_NAME}_seed{master_seed}"
    solver_config_obj = SolverConfig(
        **SOLVER_CONFIG_PARAMS
    )  # Create SolverConfig instance

    experiment = ExperimentManager(
        model=model,
        model_config=model_config,  # Pass the ModelConfig object
        norm_params=norm_params,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        output_dir=OUTPUT_DIR_BASE,  # Base directory
        experiment_name=experiment_name,  # Specific experiment subfolder name
    )
    print(f"Experiment results will be saved to: {experiment.output_dir}")

    # --- 5. Run Experiment Workflow ---
    # Generate documentation (optional, but good practice)
    print("Generating model documentation...")
    experiment.generate_model_documentation()
    experiment.save_normalization_parameters()

    # Train the model
    print("Training model...")
    trained_model = experiment.train(
        num_epochs=TRAINING_PARAMS["num_epochs"],
        learning_rate=TRAINING_PARAMS["learning_rate"],
        early_stopping_patience=TRAINING_PARAMS["early_stopping_patience"],
        early_stopping_min_delta=TRAINING_PARAMS["early_stopping_min_delta"],
        loss_metric=TRAINING_PARAMS["loss_metric"],
        component_weights=TRAINING_PARAMS["component_weights"],
        solver_config=solver_config_obj,  # Pass the solver config for loss calculation consistency
        save_checkpoints=TRAINING_PARAMS["save_checkpoints"],
        verbose=TRAINING_PARAMS["verbose"],
    )

    # Evaluate the trained model
    print("Evaluating model...")
    # Evaluation uses the solver config associated during training by default
    metrics = experiment.evaluate(
        state_names=["X", "P"], verbose=True  # Specify states to evaluate
    )

    # Visualize the results
    print("Generating visualizations...")
    experiment.visualize(
        state_names=["X", "P"],  # Specify states to visualize
        state_labels=VISUALIZATION_LABELS["state_labels"],
        component_names=VISUALIZATION_LABELS["component_names"],
        # validation_history is automatically included if available from train()
    )

    # Save all artifacts (model, history, configs, metrics)
    print("Saving all results...")
    result_paths = experiment.save_all_results()
    print(f"Results saved: {list(result_paths.keys())}")

    print(f"\n--- Experiment Run (Seed: {master_seed}) Completed Successfully! ---")
    print(f"All results saved to {experiment.output_dir}")

    # Optionally return results for further analysis in a notebook, etc.
    # return experiment


if __name__ == "__main__":
    # Run the main experiment function with the specified master seed
    main(master_seed=MASTER_SEED)
