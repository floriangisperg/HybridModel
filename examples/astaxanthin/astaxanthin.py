"""
Astaxanthin Production Hybrid Model Example (Refactored for Standard Layout)

Demonstrates building and training a hybrid model for astaxanthin production
using the configuration-driven approach with the hybrid_models package.
Assumes this script resides in examples/astaxanthin/ and the package
hybrid_models/ is at the project root. Best run with the package installed
via `pip install -e .` from the project root.

This example includes:
- Synthetic data generation based on a known ground truth model.
- Data loading from generated files.
- Definition of a hybrid model with:
    - Mechanistic ODE components.
    - Directly trainable parameters (k_c, y_sx).
    - Neural network replacements (mu_m, beta).
- Training, evaluation, and visualization using ExperimentManager.
"""
import os
import sys  # Needed for import fallback
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from scipy.integrate import solve_ivp
from typing import List, Dict, Any, Tuple

# =============================================
# PATH SETUP (Relative to this script)
# =============================================
# Get the directory where this script (run_astaxanthin.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define data and results directories relative to the script's location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results") # For ExperimentManager output

# Ensure data and results directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True) # Create base results dir for the example

# =============================================
# PACKAGE IMPORT (Handle running with/without installation)
# =============================================
try:
    # Try importing assuming the package is installed or PYTHONPATH is set
    from hybrid_models import (
        DatasetManager, VariableRegistry, VariableType,
        ModelConfig, NeuralNetworkConfig, create_model_from_config,
        SolverConfig, MSE, ExperimentManager
    )
    print("Imported 'hybrid_models' successfully (likely installed or in PYTHONPATH).")
except ImportError:
    # Fallback: If not installed, add the project root (two levels up) to sys.path
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
        print("Failed to import 'hybrid_models' even after modifying sys.path.")
        print("Please ensure:")
        print("1. The 'hybrid_models' package directory exists at the project root.")
        print("2. You run this script from the project root OR install the package using:")
        print(f"   cd '{PROJECT_ROOT}'") # Added quotes for paths with spaces
        print("   pip install -e .")
        print("-------------\n")
        raise e

# =============================================
# EXAMPLE-SPECIFIC: GROUND TRUTH & DATA GENERATION
# =============================================

class GroundTruthModel:
    """Defines the known ODE system for generating synthetic data."""
    def __init__(self):
        self.mu_m = 0.43      # maximum specific growth rate
        self.K_S = 65.0       # substrate saturation constant (will be k_c)
        self.mu_d = 2.10e-3   # specific decay rate (simplified out in hybrid)
        self.Y_S = 2.58       # substrate yield coefficient (will be y_sx)
        self.b = 0.236        # growth-independent yield coefficient (will be beta)
        self.k_d = 6.48e-2    # specific consumption rate for astaxanthin (simplified out)
        self.K_p = 2.50       # product saturation constant (simplified out)

    def __call__(self, t, y):
        X, S, P = y
        X = max(1e-10, X); S = max(1e-10, S); P = max(1e-10, P) # Prevent negatives

        # Using parameter names consistent with the hybrid model's target
        mu_m = self.mu_m
        k_c = self.K_S
        y_sx = self.Y_S
        beta = self.b # Simplification: ground truth uses beta directly for P rate

        # Simplified ODEs matching the structure we'll build in the hybrid model
        # We ignore mu_d, k_d, K_p for this hybrid example to focus on parameter replacement
        dX_dt = mu_m * S / (S + k_c + 1e-8) * X # Add epsilon for stability
        dS_dt = -y_sx * mu_m * S / (S + k_c + 1e-8) * X # Add epsilon
        dP_dt = beta * X

        return np.array([dX_dt, dS_dt, dP_dt])

def generate_and_save_data(initial_conditions: List[List[float]],
                           t_max: float,
                           noise_level: float,
                           file_prefix: str):
    """Generates synthetic data and saves it to CSV files in the data directory."""
    model = GroundTruthModel()
    all_rows = []
    run_ids = []

    print(f"Generating data for prefix '{file_prefix}'...")
    for exp_idx, y0 in enumerate(initial_conditions):
        run_id = f"{file_prefix}_Exp{exp_idx+1}"
        run_ids.append(run_id)

        # Generate irregular time points
        base_points = np.linspace(0, t_max, 15)
        time_noise = np.random.uniform(-t_max * 0.05, t_max * 0.05, size=len(base_points) - 2)
        t_points = np.sort(np.concatenate([
            [0], base_points[1:-1] + time_noise, [t_max]
        ]))
        t_points = np.maximum(0, t_points) # Ensure no negative times

        # Solve ODE
        solution = solve_ivp(model, [0, t_max], y0, t_eval=t_points, method='RK45')

        # Extract results and add noise
        times = solution.t
        X = solution.y[0] * (1 + noise_level * np.random.randn(len(times)))
        S = solution.y[1] * (1 + noise_level * np.random.randn(len(times)))
        P = solution.y[2] * (1 + noise_level * np.random.randn(len(times)))

        # Ensure positive values after adding noise
        X = np.maximum(X, 1e-8)
        S = np.maximum(S, 1e-8)
        P = np.maximum(P, 1e-8)

        # Create rows for DataFrame
        for i in range(len(times)):
            all_rows.append({
                'time': times[i],
                'biomass': X[i],
                'substrate': S[i],
                'product': P[i],
                'RunID': run_id
            })

    df = pd.DataFrame(all_rows)
    file_path = os.path.join(DATA_DIR, f"{file_prefix}_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved generated data to: {file_path}")
    return file_path, run_ids

# =============================================
# CONFIGURATION SECTION
# =============================================

# --- Experiment Setup ---
MASTER_SEED = 42
EXPERIMENT_BASE_NAME = "astaxanthin_hybrid"
# Output dir for ExperimentManager is relative to this script's location now
EXPERIMENT_OUTPUT_DIR = "results" # Results will be in examples/astaxanthin/results/

# --- Data Generation Config ---
T_MAX_SIMULATION = 168.0 # Simulation end time (hours)
NOISE_LEVEL = 0.075       # Relative noise level for synthetic data
# Initial conditions for different runs
Y0_LIST = {
    "train": [[0.1, 10, 0], [0.2, 5, 0], [0.05, 14, 0], [0.1, 12, 0]],
    "test":  [[0.15, 7.5, 0], [0.05, 5, 0], [0.2, 15, 0], [0.075, 7.5, 0]]
}

# --- Data Schema/Loading Config ---
DATA_SCHEMA_CONFIG = {
    "time_column": "time",
    "run_id_column": "RunID",
    "variables": [
        # (column_name, type, internal_name, is_output, calculate_rate)
        ("biomass", VariableType.STATE, "X", True, False),
        ("substrate", VariableType.STATE, "S", True, False),
        ("product", VariableType.STATE, "P", True, False),
    ]
}

# --- Neural Network Configuration ---
# Define the structure and inputs for each NN component being replaced
NEURAL_NETWORK_CONFIGS = [
     NeuralNetworkConfig(
        name="mu_m", # Replaces the mu_m parameter
        input_features=['X', 'S', 'P'], # Inputs to the NN
        hidden_dims=[16, 16, 16],
        output_activation="soft_sign",
        seed=MASTER_SEED # For documentation/reproducibility tracking
    ),
     NeuralNetworkConfig(
        name="beta", # Replaces the beta parameter
        input_features=['X', 'S', 'P'], # Inputs to the NN
        hidden_dims=[16, 16, 16],
        output_activation="soft_sign",
        seed=MASTER_SEED + 1 # Use different seed for different NNs if desired
    ),
]

# --- Trainable Parameter Configuration ---
# Define parameters that are directly part of the model and trained
TRAINABLE_PARAMS_CONFIG = {
    'k_c': { # Substrate saturation constant (replaces K_S)
        'initial_value': 50.0,
        'bounds': (1.0, 200.0), # Bounds primarily for reference or potential sigmoid transform
        'transform': 'softplus' # Use softplus for positivity
    },
    'y_sx': { # Substrate yield coefficient (replaces Y_S)
        'initial_value': 2.5,
        'bounds': (0.1, 10.0), # Bounds primarily for reference
        'transform': 'softplus' # Use softplus for positivity
    }
}

# --- Solver Configuration ---
SOLVER_CONFIG_PARAMS = {
    "solver_type": "dopri5", # A robust adaptive solver
    "step_size_controller": "pid",
    "rtol": 1e-4, # Tighter tolerance for potentially stiff system
    "atol": 1e-6,
    "max_steps": 200000, # Increased steps
}

# --- Training Configuration ---
TRAINING_PARAMS = {
    "num_epochs": 15000, # Consider reducing for quick testing if needed
    "learning_rate": 1e-3,
    "early_stopping_patience": 3000,
    "early_stopping_min_delta": 1e-6,
    "loss_metric": MSE,
    "component_weights": {'X': 5.0, 'S': 5.0, 'P': 5.0}, # Equal weight for this example
    "save_checkpoints": True,
    "verbose": True,
}

# --- Visualization Configuration ---
VISUALIZATION_LABELS = {
    "state_labels": {'X': 'Biomass (g/L)', 'S': 'Substrate (g/L)', 'P': 'Product (mg/L)'},
    # Ensure this matches the order of states in ModelConfig.state_names
    "component_names": ['Biomass Loss', 'Substrate Loss', 'Product Loss', 'Reg Loss (if any)'] # Adjusted for potential regularization loss in aux output
}


# =============================================
# MECHANISTIC MODEL COMPONENTS (for Hybrid Model)
# =============================================
# These functions define the structure used by the hybrid model.
# They use names corresponding to NN outputs and trainable parameters.

def hybrid_biomass_ode(inputs: Dict[str, Any]) -> jax.Array:
    """ODE component for biomass (X) dynamics in the hybrid model."""
    X = inputs['X']
    S = inputs['S']
    mu_m = inputs['mu_m']  # Provided by Neural Network 'mu_m'
    k_c = inputs['k_c']    # Provided by Trainable Parameter 'k_c'

    # Simplified biomass ODE (ignoring decay term from ground truth for this example)
    dX_dt = mu_m * S / (S + k_c + 1e-8) * X # Add epsilon for stability
    return dX_dt

def hybrid_substrate_ode(inputs: Dict[str, Any]) -> jax.Array:
    """ODE component for substrate (S) dynamics in the hybrid model."""
    X = inputs['X']
    S = inputs['S']
    mu_m = inputs['mu_m']  # Provided by Neural Network 'mu_m'
    y_sx = inputs['y_sx']  # Provided by Trainable Parameter 'y_sx'
    k_c = inputs['k_c']    # Provided by Trainable Parameter 'k_c'

    # Simplified substrate ODE
    dS_dt = -y_sx * mu_m * S / (S + k_c + 1e-8) * X # Add epsilon
    return dS_dt

def hybrid_product_ode(inputs: Dict[str, Any]) -> jax.Array:
    """ODE component for product (P) dynamics in the hybrid model."""
    X = inputs['X']
    beta = inputs['beta']  # Provided by Neural Network 'beta'

    # Simplified product ODE (matches ground truth structure here)
    dP_dt = beta * X
    return dP_dt

# =============================================
# HELPER FUNCTION FOR LOADING DATA
# =============================================

def load_data_from_files(train_file: str, test_file: str, schema_config: Dict) -> DatasetManager:
    """Loads training and test data from CSV files into a single DatasetManager."""
    print("Loading data from files...")
    if not os.path.exists(train_file):
         raise FileNotFoundError(f"Training data file not found: {train_file}")
    if not os.path.exists(test_file):
         raise FileNotFoundError(f"Test data file not found: {test_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Combine dataframes temporarily for DatasetManager loading, but keep track of RunIDs
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Get the unique RunIDs for train and test sets
    train_run_ids = train_df[schema_config["run_id_column"]].unique().tolist()
    test_run_ids = test_df[schema_config["run_id_column"]].unique().tolist()

    if not train_run_ids:
        raise ValueError("No RunIDs found in the training data file.")
    if not test_run_ids:
        print("Warning: No RunIDs found in the test data file. Test set will be empty.")


    manager = DatasetManager()
    # Load combined data but specify train/test split using RunIDs
    manager.load_from_dataframe(
        df=combined_df,
        time_column=schema_config["time_column"],
        run_id_column=schema_config["run_id_column"],
        train_run_ids=train_run_ids,
        test_run_ids=test_run_ids
    )

    # Define variables based on the schema
    variables_list = []
    for var_def in schema_config["variables"]:
        # Unpack tuple with defaults for output and rate calculation
        col, vtype, internal = var_def[:3]
        output = var_def[3] if len(var_def) > 3 else False
        rate = var_def[4] if len(var_def) > 4 else False
        variables_list.append((col, vtype, internal, output, rate))

    # Add variables to the datasets within the manager
    manager.add_variables(variables_list, data=combined_df) # Pass combined_df for lookup

    # Calculate normalization parameters based ONLY on the training data split
    manager.calculate_norm_params()

    print(f"Loaded {len(manager.train_datasets)} training and {len(manager.test_datasets)} test datasets from files.")
    print(f"Normalization parameters calculated using training data: {list(manager.norm_params.keys())}")
    return manager

# =============================================
# MAIN EXECUTION SCRIPT
# =============================================

def main():
    """Main function to run the astaxanthin hybrid modeling experiment."""
    print(f"\n--- Starting Astaxanthin Hybrid Model Example ---")
    print(f"Script Location: {SCRIPT_DIR}")
    print(f"Using Master Seed: {MASTER_SEED}")
    print(f"Target Data Directory: {DATA_DIR}")
    print(f"Target Results Directory Base: {RESULTS_DIR}")

    # --- 1. Data Generation & Saving ---
    # Generate data only if it doesn't exist to save time on reruns
    train_data_file = os.path.join(DATA_DIR, "train_data.csv")
    test_data_file = os.path.join(DATA_DIR, "test_data.csv")
    if not os.path.exists(train_data_file) or not os.path.exists(test_data_file):
        print("Generating and saving synthetic data...")
        generate_and_save_data(Y0_LIST["train"], T_MAX_SIMULATION, NOISE_LEVEL, "train")
        generate_and_save_data(Y0_LIST["test"], T_MAX_SIMULATION, NOISE_LEVEL, "test")
    else:
        print(f"Using existing data files: {train_data_file}, {test_data_file}")

    # --- 2. Data Loading ---
    try:
        data_manager = load_data_from_files(train_data_file, test_data_file, DATA_SCHEMA_CONFIG)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
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
        print("Warning: No test data loaded after processing. Evaluation will be limited.")


    # --- 3. Define Model Configuration ---
    print("Defining model configuration...")
    model_config = ModelConfig(
         state_names=["X", "S", "P"], # Order matters for loss component names if using defaults
         mechanistic_components={
             "X": hybrid_biomass_ode,
             "S": hybrid_substrate_ode,
             "P": hybrid_product_ode,
         },
         neural_networks=NEURAL_NETWORK_CONFIGS,
         trainable_parameters=TRAINABLE_PARAMS_CONFIG # Include trainable params config
    )


    # --- 4. Build Model ---
    print("Building hybrid model from configuration...")
    try:
        model = create_model_from_config(
            model_config, norm_params, master_seed=MASTER_SEED
        )
        print("Model built successfully.")
    except Exception as e:
        print(f"Error building model: {e}")
        raise

    # --- 5. Setup Experiment ---
    print("Setting up experiment manager...")
    experiment_name = f"{EXPERIMENT_BASE_NAME}_seed{MASTER_SEED}"
    # Set the output directory for ExperimentManager relative to the script
    exp_manager_output_dir = RESULTS_DIR

    try:
        experiment = ExperimentManager(
            model=model,
            model_config=model_config,
            norm_params=norm_params,
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            output_dir=exp_manager_output_dir, # Base directory for this example's results
            experiment_name=experiment_name   # Subfolder within RESULTS_DIR
        )
        print(f"Experiment results will be saved to: {experiment.output_dir}")
    except Exception as e:
        print(f"Error initializing ExperimentManager: {e}")
        raise

    # --- 6. Run Experiment Workflow ---
    print("Generating initial documentation...")
    experiment.generate_model_documentation()
    experiment.save_normalization_parameters()

    print("Starting experiment run...")
    solver_config_obj = SolverConfig(**SOLVER_CONFIG_PARAMS)
    eval_solver_config_obj = SolverConfig.for_evaluation(solver_type="dopri8")  # Example: override solver

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
            # validation_datasets=test_datasets if test_datasets else None, # Use test set for validation loss tracking
            solver_config=solver_config_obj, # Use training solver for loss calc
            save_checkpoints=TRAINING_PARAMS["save_checkpoints"],
            verbose=TRAINING_PARAMS["verbose"],
        )

        if experiment.trained_model: # Proceed only if training produced a model
            print("Evaluating model...")
            experiment.evaluate(
                state_names=["X", "S", "P"],
                # Use a potentially more accurate solver for final evaluation
                solver_config=eval_solver_config_obj,
                verbose=True
            )

            print("Generating visualizations...")
            experiment.visualize(
                state_names=["X", "S", "P"],
                state_labels=VISUALIZATION_LABELS["state_labels"],
                # Adjust component names if regularization was added to loss
                component_names=VISUALIZATION_LABELS["component_names"][:len(model_config.state_names)],
                solver_config=eval_solver_config_obj, # Use consistent solver for plots
            )

            print("Saving all results...")
            experiment.save_all_results()
            experiment.save_results_summary()
        else:
             print("Training did not complete successfully or was skipped. Skipping evaluation, visualization, and saving.")

    except Exception as e:
         print(f"\n--- ERROR during experiment execution ---")
         print(f"{type(e).__name__}: {e}")
         # Optionally re-raise e or handle specific errors
         raise # Re-raise to see the full traceback


    print(f"\n--- Astaxanthin Example (Seed: {MASTER_SEED}) Completed Successfully! ---")
    print(f"Data used from: {DATA_DIR}")
    if experiment.trained_model:
        print(f"Results saved to: {experiment.output_dir}")
    else:
        print("Results directory may be incomplete due to errors during execution.")


if __name__ == "__main__":
    # Add basic error handling for the main execution
    try:
        main()
    except Exception as main_exception:
        print(f"\n--- UNHANDLED EXCEPTION IN MAIN ---")
        print(f"{type(main_exception).__name__}: {main_exception}")
