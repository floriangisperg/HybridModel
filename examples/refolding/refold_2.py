import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

# Import your hybrid modeling framework
from hybrid_models import (
    DatasetManager,
    VariableType,
    VariableRegistry,
    HybridModelBuilder,
    train_hybrid_model,
    evaluate_model_performance,
    plot_all_results,
    create_hybrid_model_loss,
    MSE,
    SolverConfig,
)

# Set a random seed for reproducibility
seed = 45
key = jax.random.PRNGKey(seed)

# Path to the data file
data_path = os.path.join("data", "combined_refolds.xlsx")


# Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load Excel file
    data = pd.read_excel(file_path)

    # Rename columns if needed to make them easier to work with
    data = data.rename(
        columns={
            "Refolding Time [min]": "time",
            "Native Product Monomer [mg/L]": "native_protein",
            "Refolding Yield [%]": "yield",
            "I0 [mg/L]": "initial_protein",
            "DTT [mM]": "dtt",
            "GSSG [mM]": "gssg",
            "Dilution Factor": "dilution",
            "pH": "ph",
            "Final Urea [M]": "urea",
            "Experiment ID": "exp_id",
        }
    )

    return data


# Load the data
data = load_and_preprocess_data(data_path)

# Initialize dataset manager
manager = DatasetManager()
manager.load_from_dataframe(
    df=data,
    time_column="time",
    run_id_column="exp_id",
    train_ratio=0.6,  # 80% for training, 20% for testing
)

# Define variable registry
variables = VariableRegistry()
variables.add_state(
    "native_protein", is_output=True
)  # Native protein as state variable
variables.add_state("yield", is_output=True)  # Refolding yield as state variable
variables.add_parameter("initial_protein")  # Initial protein concentration as parameter
variables.add_parameter("dtt")  # DTT concentration as parameter
variables.add_parameter("gssg")  # GSSG concentration as parameter
variables.add_parameter("dilution")  # Dilution factor as parameter
variables.add_parameter("ph")  # pH as parameter
variables.add_parameter("urea")  # Urea concentration as parameter

# Add variables to the dataset manager
manager.add_variables(variables.to_list(), data)

# Calculate normalization parameters
manager.calculate_norm_params()
norm_params = manager.norm_params

# Prepare datasets for training and testing
train_datasets = manager.prepare_training_data()
test_datasets = manager.prepare_test_data()


# Define the model-building function
def build_refolding_model(norm_params, key):
    # Split key for different neural networks
    key1, key2 = jax.random.split(key)

    # Create model builder
    builder = HybridModelBuilder()
    builder.set_normalization_params(norm_params)

    # Add state variables
    builder.add_state("native_protein")
    builder.add_state("yield")

    # Define mechanistic components for native protein formation (first-order reaction)
    def native_protein_formation(inputs):
        # First-order kinetics with rate constant predicted by NN
        i0 = inputs["initial_protein"]  # Initial total protein
        native = inputs["native_protein"]  # Current native protein
        k_fold = inputs["k_fold"]  # Folding rate constant (predicted by NN)

        # Rate = k_fold * (Initial protein - Current native protein)
        return k_fold * (i0 - native)

    # Define mechanistic component for refolding yield (derived from native protein)
    def yield_formation(inputs):
        # Yield is derived from native protein / initial protein * 100
        # But for ODE dynamics, we can also model it with its own rate
        i0 = inputs["initial_protein"]
        current_yield = inputs["yield"]
        k_yield = inputs["k_yield"]  # Rate constant for yield change

        # Target yield based on current native protein
        target_yield = (inputs["native_protein"] / i0) * 100

        # Rate of change of yield approaching target
        return k_yield * (target_yield - current_yield)

    # Add mechanistic components
    builder.add_mechanistic_component("native_protein", native_protein_formation)
    builder.add_mechanistic_component("yield", yield_formation)

    # Replace kinetic parameters with neural networks

    # Neural network for folding rate constant
    builder.replace_with_nn(
        name="k_fold",
        input_features=["native_protein", "dtt", "gssg", "dilution", "ph", "urea"],
        hidden_dims=[16,32,16],
        output_activation=jax.nn.softplus,  # Ensure positive rate constant
        key=key1,
    )

    # Neural network for yield rate constant
    builder.replace_with_nn(
        name="k_yield",
        input_features=["yield", "dtt", "gssg", "dilution", "ph", "urea"],
        hidden_dims=[8],
        output_activation=jax.nn.softplus,  # Ensure positive rate constant
        key=key2,
    )

    # Build the model
    return builder.build()


# Build the model
model = build_refolding_model(norm_params, key)

# Create loss function
loss_fn = create_hybrid_model_loss(
    state_names=["native_protein", "yield"],
    loss_metric=MSE,
    component_weights={
        "native_protein": 1.0,
        "yield": 1.0,
    },  # Weight native protein higher if needed
)

# Train the model
print("Training the hybrid model...")
trained_model, history = train_hybrid_model(
    model=model,
    datasets=train_datasets,
    loss_fn=loss_fn,
    num_epochs=20000,
    learning_rate=7e-4,
    early_stopping_patience=5000,
    verbose=True,
)

# Evaluate the model
print("Evaluating the model...")


# Define a solver function that will be used for evaluation
def solve_for_dataset(model, dataset):
    solver_config = SolverConfig.for_evaluation()
    return model.solve(
        initial_state=dataset["initial_state"],
        t_span=(dataset["times"][0], dataset["times"][-1]),
        evaluation_times=dataset["times"],
        args={"static_inputs": dataset.get("static_inputs", {})},
        solver=solver_config.get_solver(),
        stepsize_controller=solver_config.get_step_size_controller(),
        rtol=solver_config.rtol,
        atol=solver_config.atol,
    )


# Evaluate on test datasets
evaluation_results = evaluate_model_performance(
    model=trained_model,
    datasets=test_datasets,
    solve_fn=solve_for_dataset,
    state_names=["native_protein", "yield"],
    verbose=True,
)

# Plot results
print("Generating plots...")
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Create visualization
plot_all_results(
    model=trained_model,
    train_datasets=train_datasets,
    test_datasets=test_datasets,
    history=history,
    solve_fn=solve_for_dataset,
    state_names=["native_protein", "yield"],
    output_dir=output_dir,
    state_labels={
        "native_protein": "Native Protein [mg/L]",
        "yield": "Refolding Yield [%]",
    },
)

print(f"Results saved to {output_dir}")
