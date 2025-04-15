"""
E. coli Fermentation Hybrid Model Example

Demonstrates building and training a hybrid model for an E. coli fermentation process
using the configuration-driven approach with the hybrid_models package.

This example uses real experimental data loaded from an Excel file.
"""
import os
import sys
import numpy as np # Often useful, though maybe not directly used here
import jax
import jax.numpy as jnp
import pandas as pd
from typing import List, Dict, Any, Tuple

# =============================================
# PATH SETUP (Relative to this script)
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================
# PACKAGE IMPORT (Handle running with/without installation)
# =============================================
try:
    from hybrid_models import (
        DatasetManager, VariableRegistry, VariableType,
        ModelConfig, NeuralNetworkConfig, create_model_from_config,
        SolverConfig, MSE, ExperimentManager
    )
    print("Imported 'hybrid_models' successfully (likely installed or in PYTHONPATH).")
except ImportError:
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
    print(f"Could not import 'hybrid_models' directly. Adding project root '{PROJECT_ROOT}' to sys.path.")
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    try:
        from hybrid_models import (
            DatasetManager, VariableRegistry, VariableType,
            ModelConfig, NeuralNetworkConfig, create_model_from_config,
            SolverConfig, MSE, ExperimentManager
        )
    except ImportError as e:
        print("\n--- ERROR ---")
        print("Failed to import 'hybrid_models'. Ensure:")
        print("1. 'hybrid_models' package directory exists at the project root.")
        print(f"2. You run this script from the project root OR install with: cd '{PROJECT_ROOT}' && pip install -e .")
        print("-------------\n")
        raise e

# =============================================
# CONFIGURATION SECTION
# =============================================

# --- Experiment Setup ---
MASTER_SEED = 45  # Can be changed for different initializations
EXPERIMENT_BASE_NAME = "ecoli_fermentation"
# Relative path to the data file within the DATA_DIR
DATA_FILE_NAME = "testtrain.xlsx"
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

# --- Data Schema/Loading Config ---
# Defines how to interpret columns in the Excel file
DATA_SCHEMA_CONFIG = {
    "time_column": "feedtimer(h)",
    "run_id_column": "RunID",
    "train_ratio": 0.8, # Let DatasetManager split by ratio if specific IDs aren't given
    "variables": [
        # (column_name, type, internal_name, is_output, calculate_rate)
        ("CDW(g/L)", VariableType.STATE, "X", True, False),
        ("Produktsol(g/L)", VariableType.STATE, "P", True, False),
        ("Temp(°C)", VariableType.CONTROL, "temp", False, False),
        ("InductorMASS(mg)", VariableType.CONTROL, "inductor_mass", False, False),
        ("Inductor(yesno)", VariableType.CONTROL, "inductor_switch", False, False),
        ("Reaktorvolumen(L)", VariableType.CONTROL, "reactor_volume", False, False),
        ("Feed(L)", VariableType.FEED, "feed", False, True), # calculate_rate=True
        ("Base(L)", VariableType.FEED, "base", False, True), # calculate_rate=True
    ]
}

# --- Neural Network Configuration ---
# Define the structure and inputs for each NN component
NEURAL_NETWORK_CONFIGS = [
     NeuralNetworkConfig(
        name="growth_rate",
        input_features=["X", "P", "temp", "feed", "inductor_mass"],
        hidden_dims=[32, 32, 32],
        output_activation="soft_sign", # Bounded activation might be suitable for rate
        seed=MASTER_SEED
    ),
     NeuralNetworkConfig(
        name="product_rate",
        input_features=["X", "P", "temp", "feed", "inductor_mass"],
        hidden_dims=[32, 32, 32],
        output_activation="softplus", # Ensure production rate > 0
        seed=MASTER_SEED + 1
    ),
]

# --- Trainable Parameter Configuration ---
# No explicit trainable parameters in this specific model structure (only NNs)
TRAINABLE_PARAMS_CONFIG = {}

# --- Solver Configuration ---
SOLVER_CONFIG_PARAMS = {
    "solver_type": "tsit5", # Default explicit RK solver
    "step_size_controller": "pid",
    "rtol": 1e-2, # Relaxed tolerances for potentially faster training
    "atol": 1e-4,
    "max_steps": 500000,
}

# --- Training Configuration ---
TRAINING_PARAMS = {
    "num_epochs": 10000,
    "learning_rate": 1e-3,
    "early_stopping_patience": 2500,
    "early_stopping_min_delta": 1e-6,
    "loss_metric": MSE,
    "component_weights": {"X": 1.0, "P": 10.0}, # Weight product loss higher
    "save_checkpoints": True,
    "verbose": True,
}

# --- Visualization Configuration ---
VISUALIZATION_LABELS = {
    "state_labels": {"X": "Biomass (CDW g/L)", "P": "Product (g/L)"},
    "component_names": ["Biomass Loss", "Product Loss"] # Match state order [X, P]
}


# =============================================
# MECHANISTIC MODEL COMPONENTS
# =============================================
# These functions define the mechanistic part of the E. coli fermentation model.

def calculate_dilution_rate(inputs: Dict[str, Any]) -> jax.Array:
    """Calculates the dilution rate D = F/V based on calculated flow rates."""
    volume = inputs.get("reactor_volume", 1.0)
    # Use rates calculated by DatasetManager (if calculate_rate=True was set)
    feed_rate = inputs.get("feed_rate", 0.0)
    base_rate = inputs.get("base_rate", 0.0)
    total_flow_rate = feed_rate + base_rate
    dilution_rate = jnp.where(volume > 1e-6, total_flow_rate / volume, 0.0)
    return dilution_rate

def ecoli_biomass_ode(inputs: Dict[str, Any]) -> jax.Array:
    """ODE component for biomass (X) dynamics: dX/dt = mu*X - D*X."""
    X = inputs.get("X", 1e-8) # Use .get with default for safety
    mu = inputs["growth_rate"]  # Provided by NN 'growth_rate'
    dilution_rate = calculate_dilution_rate(inputs)
    dXdt = mu * jnp.maximum(X, 1e-8) - dilution_rate * jnp.maximum(X, 1e-8) # Ensure X is positive
    return dXdt

def ecoli_product_ode(inputs: Dict[str, Any]) -> jax.Array:
    """ODE component for product (P) dynamics: dP/dt = vpx*X*switch - D*P."""
    X = inputs.get("X", 1e-8)
    P = inputs.get("P", 0.0)
    vpx = inputs["product_rate"]  # Provided by NN 'product_rate'
    inductor_switch = inputs.get("inductor_switch", 0.0)
    dilution_rate = calculate_dilution_rate(inputs)
    dPdt = vpx * jnp.maximum(X, 1e-8) * inductor_switch - dilution_rate * jnp.maximum(P, 0.0) # Ensure P >= 0
    return dPdt

# =============================================
# HELPER FUNCTION FOR LOADING DATA
# =============================================

def load_ecoli_data(file_path: str, schema_config: Dict) -> DatasetManager:
    """Loads E. coli fermentation data from an Excel file."""
    print(f"Loading E. coli data from: {file_path}")
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        # Use openpyxl engine if needed for .xlsx
        data_df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        print("Make sure 'openpyxl' is installed (`pip install openpyxl`)")
        raise

    manager = DatasetManager()
    # Load data, letting manager handle train/test split by ratio
    manager.load_from_dataframe(
        df=data_df,
        time_column=schema_config["time_column"],
        run_id_column=schema_config["run_id_column"],
        train_ratio=schema_config.get("train_ratio", 0.8) # Use ratio from config
        # train_run_ids and test_run_ids could be added to schema_config if needed
    )

    # Define variables based on the schema
    variables_list = []
    for var_def in schema_config["variables"]:
        col, vtype, internal = var_def[:3]
        output = var_def[3] if len(var_def) > 3 else False
        rate = var_def[4] if len(var_def) > 4 else False
        variables_list.append((col, vtype, internal, output, rate))

    # Add variables and calculate rates if requested
    manager.add_variables(variables_list, data=data_df)

    # Calculate normalization parameters based on the training split
    manager.calculate_norm_params()

    print(f"Loaded {len(manager.train_datasets)} training and {len(manager.test_datasets)} test datasets.")
    print(f"Normalization parameters calculated using training data: {list(manager.norm_params.keys())}")
    return manager

# =============================================
# MAIN EXECUTION SCRIPT
# =============================================

def main():
    """Main function to run the E. coli fermentation hybrid modeling experiment."""
    print(f"\n--- Starting E. coli Fermentation Hybrid Model Example ---")
    print(f"Script Location: {SCRIPT_DIR}")
    print(f"Using Master Seed: {MASTER_SEED}")
    print(f"Target Data Directory: {DATA_DIR}")
    print(f"Target Results Directory Base: {RESULTS_DIR}")

    # --- 1. Data Loading ---
    try:
        data_manager = load_ecoli_data(DATA_FILE_PATH, DATA_SCHEMA_CONFIG)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    norm_params = data_manager.norm_params
    train_datasets = data_manager.prepare_training_data()
    test_datasets = data_manager.prepare_test_data()

    if not train_datasets:
        print("Error: No training data loaded after processing. Exiting.")
        return
    # It's okay if test_datasets is empty, ExperimentManager handles it
    if not test_datasets:
        print("Warning: No test data loaded after processing. Evaluation will be only on training data.")


    # --- 2. Define Model Configuration ---
    print("Defining model configuration...")
    model_config = ModelConfig(
         state_names=["X", "P"], # Order matters
         mechanistic_components={
             "X": ecoli_biomass_ode,
             "P": ecoli_product_ode,
         },
         neural_networks=NEURAL_NETWORK_CONFIGS,
         trainable_parameters=TRAINABLE_PARAMS_CONFIG # Empty for this example
    )

    # --- 3. Build Model ---
    print("Building hybrid model from configuration...")
    try:
        model = create_model_from_config(
            model_config, norm_params, master_seed=MASTER_SEED
        )
        print("Model built successfully.")
    except Exception as e:
        print(f"Error building model: {e}")
        raise

    # --- 4. Setup Experiment ---
    print("Setting up experiment manager...")
    experiment_name = f"{EXPERIMENT_BASE_NAME}_seed{MASTER_SEED}"
    exp_manager_output_dir = RESULTS_DIR # Save into examples/ecoli_fermentation/results/

    try:
        experiment = ExperimentManager(
            model=model,
            model_config=model_config,
            norm_params=norm_params,
            train_datasets=train_datasets,
            test_datasets=test_datasets, # Pass even if empty
            output_dir=exp_manager_output_dir,
            experiment_name=experiment_name
        )
        print(f"Experiment results will be saved to: {experiment.output_dir}")
    except Exception as e:
        print(f"Error initializing ExperimentManager: {e}")
        raise

    # --- 5. Run Experiment Workflow ---
    print("Generating initial documentation...")
    experiment.generate_model_documentation()
    experiment.save_normalization_parameters()

    print("Starting experiment run...")
    solver_config_obj = SolverConfig(**SOLVER_CONFIG_PARAMS)
    eval_solver_config_obj = SolverConfig.for_evaluation() # Use default high-accuracy for eval

    try:
        # --- Using individual steps ---
        print("Training model...")
        experiment.train(
            num_epochs=TRAINING_PARAMS["num_epochs"],
            learning_rate=TRAINING_PARAMS["learning_rate"],
            early_stopping_patience=TRAINING_PARAMS["early_stopping_patience"],
            early_stopping_min_delta=TRAINING_PARAMS["early_stopping_min_delta"],
            loss_metric=TRAINING_PARAMS["loss_metric"],
            component_weights=TRAINING_PARAMS["component_weights"],
            validation_datasets=test_datasets if test_datasets else None,
            solver_config=solver_config_obj,
            save_checkpoints=TRAINING_PARAMS["save_checkpoints"],
            verbose=TRAINING_PARAMS["verbose"],
        )

        if experiment.trained_model:
            print("Evaluating model...")
            experiment.evaluate(
                state_names=["X", "P"],
                solver_config=eval_solver_config_obj,
                verbose=True
            )

            print("Generating visualizations...")
            experiment.visualize(
                state_names=["X", "P"],
                state_labels=VISUALIZATION_LABELS["state_labels"],
                component_names=VISUALIZATION_LABELS["component_names"],
                solver_config=eval_solver_config_obj,
            )

            print("Saving all results...")
            experiment.save_all_results()
            experiment.save_results_summary()
        else:
             print("Training did not complete successfully or was skipped. Skipping subsequent steps.")

    except Exception as e:
         print(f"\n--- ERROR during experiment execution ---")
         print(f"{type(e).__name__}: {e}")
         raise

    print(f"\n--- E. coli Fermentation Example (Seed: {MASTER_SEED}) Completed Successfully! ---")
    print(f"Data used from: {DATA_FILE_PATH}")
    if experiment.trained_model:
        print(f"Results saved to: {experiment.output_dir}")
    else:
        print("Results directory may be incomplete due to errors during execution.")


if __name__ == "__main__":
    try:
        main()
    except Exception as main_exception:
        print(f"\n--- UNHANDLED EXCEPTION IN MAIN ---")
        print(f"{type(main_exception).__name__}: {main_exception}")
        # import traceback
        # traceback.print_exc()