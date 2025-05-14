"""
Protein Refolding Hybrid Model with Denaturant-Dependent Kinetics

Demonstrates building and training a hybrid model for protein refolding kinetics
using a mechanistic model: k_fold = a_fold * (1 + urea)^(-b_fold)
where a_fold is predicted by a neural network and b_fold is a trainable parameter.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from typing import Dict, Any

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
    print("Imported 'hybrid_models' successfully.")
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
EXPERIMENT_BASE_NAME = "protein_refolding_denaturant_dependent"
# Relative path to the data file within the DATA_DIR
DATA_FILE_NAME = "combined_refolds.xlsx"
DATA_FILE_PATH = os.path.join(DATA_DIR, DATA_FILE_NAME)

# --- Data Schema/Loading Config ---
DATA_SCHEMA_CONFIG = {
    "time_column": "Refolding Time [min]",
    "run_id_column": "Experiment ID",
    "train_ratio": 0.6,  # Let DatasetManager split by ratio
    "variables": [
        # (column_name, type, internal_name, is_output, calculate_rate)
        ("Native Product Monomer [mg/L]", VariableType.STATE, "native_protein", True, False),
        ("I0 [mg/L]", VariableType.PARAMETER, "initial_protein", False, False),
        ("DTT [mM]", VariableType.PARAMETER, "dtt", False, False),
        ("GSSG [mM]", VariableType.PARAMETER, "gssg", False, False),
        ("Dilution Factor", VariableType.PARAMETER, "dilution", False, False),
        ("pH", VariableType.PARAMETER, "ph", False, False),
        ("Final Urea [M]", VariableType.PARAMETER, "urea", False, False),
    ]
}

# --- Neural Network Configuration ---
# Neural network for intrinsic rate factor (a_fold)
NEURAL_NETWORK_CONFIGS = [
    NeuralNetworkConfig(
        name="a_fold",  # Changed from k_fold to a_fold
        # Removed urea from inputs as it's used directly in the mechanistic formula
        input_features=["dtt", "gssg", "dilution", "ph"],
        hidden_dims=[32,32],
        output_activation="softplus",  # Ensure positive rate
        seed=MASTER_SEED
    )
]

# --- Trainable Parameter Configuration ---
# Added b_fold as a trainable parameter
TRAINABLE_PARAMS_CONFIG = {
    "b_fold": {
        "initial_value": 1.0,  # Initial guess - often around 1-2 in folding studies
        "transform": "softplus"  # Ensure positivity
    }
}

# --- Solver Configuration ---
SOLVER_CONFIG_PARAMS = {
    "solver_type": "dopri5",
    "step_size_controller": "pid",
    "rtol": 1e-2,  # Looser tolerance for speed
    "atol": 1e-3,  # Looser tolerance for speed
    "max_steps": 500000,
}

# --- Training Configuration ---
TRAINING_PARAMS = {
    "num_epochs": 20000,
    "learning_rate": 1e-4,
    "early_stopping_patience": 3000,
    "early_stopping_min_delta": 1e-6,
    "loss_metric": MSE,
    "component_weights": {"native_protein": 1.0},
    "save_checkpoints": True,
    "verbose": True,
}

# --- Visualization Configuration ---
VISUALIZATION_LABELS = {
    "state_labels": {"native_protein": "Native Protein [mg/L]"},
    "component_names": ["Native Protein Loss"]
}

# =============================================
# MECHANISTIC MODEL COMPONENTS
# =============================================

def native_protein_formation(inputs: Dict[str, Any]) -> jax.Array:
    """
    ODE component for native protein formation with denaturant-dependent kinetics.

    Formula: k_fold = a_fold * (1 + urea)^(-b_fold)
    where a_fold is from neural network and b_fold is a trainable parameter

    dNative/dt = k_fold * (Initial - Native)
    """
    i0 = inputs["initial_protein"]  # Initial total protein
    native = inputs["native_protein"]  # Current native protein

    # Get parameters for the kinetic formula
    urea = inputs["urea"]  # Denaturant concentration
    a_fold = inputs["a_fold"]  # Intrinsic rate from neural network
    b_fold = inputs["b_fold"]  # Denaturant sensitivity parameter

    # Calculate folding rate using the denaturant-dependent formula
    # Note: Using negative exponent as higher denaturant typically slows folding
    k_fold = a_fold * (1 + urea)**(-b_fold)

    # First-order kinetics: unfolded to native
    return k_fold * jnp.maximum(i0 - native, 0.0)  # Ensure remaining protein is non-negative

# =============================================
# HELPER FUNCTION FOR LOADING DATA
# =============================================

def load_refolding_data(file_path: str, schema_config: Dict) -> DatasetManager:
    """Loads protein refolding data from an Excel file."""
    print(f"Loading protein refolding data from: {file_path}")
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        # Use openpyxl engine for .xlsx
        data_df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        print("Make sure 'openpyxl' is installed (`pip install openpyxl`)")
        raise

    # Rename columns for easier working
    column_map = {
        "Refolding Time [min]": "time",
        "Native Product Monomer [mg/L]": "native_protein",
        "Refolding Yield [%]": "yield",  # We'll still load this for potential future use
        "I0 [mg/L]": "initial_protein",
        "DTT [mM]": "dtt",
        "GSSG [mM]": "gssg",
        "Dilution Factor": "dilution",
        "pH": "ph",
        "Final Urea [M]": "urea",
        "Experiment ID": "exp_id",
    }
    data_df = data_df.rename(columns=column_map)

    manager = DatasetManager()
    # Load data, letting manager handle train/test split by ratio
    manager.load_from_dataframe(
        df=data_df,
        time_column="time",
        run_id_column="exp_id",
        train_ratio=schema_config.get("train_ratio", 0.6)
    )

    # Define variables based on the schema - Yield is omitted
    variables_registry = VariableRegistry()
    variables_registry.add_state("native_protein", is_output=True)
    variables_registry.add_parameter("initial_protein")
    variables_registry.add_parameter("dtt")
    variables_registry.add_parameter("gssg")
    variables_registry.add_parameter("dilution")
    variables_registry.add_parameter("ph")
    variables_registry.add_parameter("urea")

    # Add variables to datasets
    manager.add_variables(variables_registry.to_list(), data=data_df)

    # Calculate normalization parameters
    manager.calculate_norm_params()

    print(f"Loaded {len(manager.train_datasets)} training and {len(manager.test_datasets)} test datasets.")
    print(f"Normalization parameters calculated using training data: {list(manager.norm_params.keys())}")
    return manager

# =============================================
# HELPER FUNCTION FOR CALCULATING YIELD
# =============================================

def calculate_yield(native_protein: float, initial_protein: float) -> float:
    """
    Calculate refolding yield from native protein concentration and initial protein.

    Args:
        native_protein: Native protein concentration in mg/L
        initial_protein: Initial protein concentration in mg/L

    Returns:
        Refolding yield in percentage
    """
    # Ensure we don't divide by zero
    if initial_protein <= 0:
        return 0.0

    return 100.0 * native_protein / initial_protein

# =============================================
# MAIN EXECUTION SCRIPT
# =============================================

def main():
    """Main function to run the protein refolding hybrid modeling experiment."""
    print(f"\n--- Starting Protein Refolding Model with Denaturant-Dependent Kinetics ---")
    print(f"Script Location: {SCRIPT_DIR}")
    print(f"Using Master Seed: {MASTER_SEED}")
    print(f"Target Data Directory: {DATA_DIR}")
    print(f"Target Results Directory: {RESULTS_DIR}")

    # --- 1. Data Loading ---
    try:
        data_manager = load_refolding_data(DATA_FILE_PATH, DATA_SCHEMA_CONFIG)
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

    if not test_datasets:
        print("Warning: No test data loaded after processing. Evaluation will be only on training data.")

    # --- 2. Define Model Configuration ---
    print("Defining model configuration...")
    model_config = ModelConfig(
         state_names=["native_protein"],  # Only native_protein as state
         mechanistic_components={
             "native_protein": native_protein_formation,
         },
         neural_networks=NEURAL_NETWORK_CONFIGS,
         trainable_parameters=TRAINABLE_PARAMS_CONFIG
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

    try:
        experiment = ExperimentManager(
            model=model,
            model_config=model_config,
            norm_params=norm_params,
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            output_dir=RESULTS_DIR,
            experiment_name=experiment_name
        )
        print(f"Experiment results will be saved to: {experiment.output_dir}")
    except Exception as e:
        print(f"Error initializing ExperimentManager: {e}")
        raise

    # --- 5. Run Experiment Workflow ---
    print("Generating initial documentation...")
    try:
        experiment.generate_model_documentation()
        experiment.save_normalization_parameters()
    except UnicodeEncodeError:
        print("Warning: Unicode encoding error when saving documentation.")
        print("Using UTF-8 encoding instead...")

        # Fix for Windows encoding issues
        import builtins
        original_open = builtins.open

        def patched_open(*args, **kwargs):
            if 'encoding' not in kwargs and len(args) < 3:
                kwargs['encoding'] = 'utf-8'
            return original_open(*args, **kwargs)

        try:
            builtins.open = patched_open
            experiment.generate_model_documentation()
            experiment.save_normalization_parameters()
        finally:
            builtins.open = original_open

    print("Starting experiment run...")
    solver_config_obj = SolverConfig(**SOLVER_CONFIG_PARAMS)
    eval_solver_config_obj = SolverConfig.for_evaluation()  # Higher accuracy for eval

    try:
        # Train the model
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
                state_names=["native_protein"],
                solver_config=eval_solver_config_obj,
                verbose=True
            )

            print("Generating visualizations...")
            experiment.visualize(
                state_names=["native_protein"],
                state_labels=VISUALIZATION_LABELS["state_labels"],
                component_names=VISUALIZATION_LABELS["component_names"],
                solver_config=eval_solver_config_obj,
            )

            print("Saving all results...")
            experiment.save_all_results()
            experiment.save_results_summary()

            # Examine the trained b_fold parameter value
            b_fold_value = experiment.trained_model.trainable_parameters.get("b_fold", None)
            if b_fold_value is not None:
                # Get the transformed value
                b_fold_transformed = jax.nn.softplus(b_fold_value)
                print(f"\nTrained denaturant sensitivity parameter (b_fold): {float(b_fold_transformed):.4f}")
                print("This represents the sensitivity of the protein folding rate to denaturant concentration.")
                print("Higher values indicate greater sensitivity to denaturant (faster decrease in rate with increasing urea).")

            # Inform user about the model approach
            print("\nModel Details:")
            print("- Uses a denaturant-dependent kinetic model: k_fold = a_fold * (1 + urea)^(-b_fold)")
            print("- a_fold (intrinsic rate) is predicted by a neural network based on solution conditions")
            print("- b_fold (denaturant sensitivity) is a trainable parameter")
            print("- The model predicts native protein concentration over time")
            print("- Yield can be calculated as: yield(%) = 100 * native_protein / initial_protein")
        else:
            print("Training did not complete successfully or was skipped. Skipping subsequent steps.")

    except Exception as e:
        print(f"\n--- ERROR during experiment execution ---")
        print(f"{type(e).__name__}: {e}")
        raise

    print(f"\n--- Protein Refolding Model with Denaturant Dependence (Seed: {MASTER_SEED}) Completed Successfully! ---")
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