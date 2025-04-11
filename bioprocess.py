"""
Bioprocess hybrid modeling example using the improved framework.

This script demonstrates the use of the enhanced hybrid modeling framework
with explicit control over model architecture, solver configuration, and
comprehensive model documentation and persistence. It includes robust
key generation for reproducible neural network initialization.
"""

import jax
import jax.numpy as jnp
import pandas as pd
import os

# Import core hybrid modeling framework
from hybrid_models import (
    HybridModelBuilder,
    VariableRegistry,
    MSE,
    RelativeMSE,
    SolverConfig,
    ExperimentManager,
    ModelConfig,
    NeuralNetworkConfig,
    create_initial_random_key,  # Import the key helper
)

from hybrid_models.data import DatasetManager, VariableType


# =============================================
# DATA LOADING AND PREPROCESSING (Unchanged)
# =============================================


def load_bioprocess_data(
    file_path, train_run_ids=None, test_run_ids=None, train_ratio=0.8
):
    """Load bioprocess experimental data."""
    # (Content remains the same)
    data = pd.read_excel(file_path)
    manager = DatasetManager()
    manager.load_from_dataframe(
        df=data,
        time_column="feedtimer(h)",
        run_id_column="RunID",
        train_run_ids=train_run_ids,
        test_run_ids=test_run_ids,
        train_ratio=train_ratio,
    )
    variables = VariableRegistry()
    variables.add_state("CDW(g/L)", internal_name="X")
    variables.add_state("Produktsol(g/L)", internal_name="P")
    variables.add_control("Temp(°C)", internal_name="temp")
    variables.add_control("InductorMASS(mg)", internal_name="inductor_mass")
    variables.add_control("Inductor(yesno)", internal_name="inductor_switch")
    variables.add_control("Reaktorvolumen(L)", internal_name="reactor_volume")
    variables.add_feed("Feed(L)", internal_name="feed")
    variables.add_feed("Base(L)", internal_name="base")
    manager.add_variables(variables.to_list(), data)
    manager.calculate_norm_params()
    return manager


# =============================================
# DEFINE BIOPROCESS MODEL (Updated with Explicit Key Handling)
# =============================================


def define_bioprocess_model(
    norm_params,
    ann_config,  # Require ann_config to be passed
    master_seed=42,  # Add default master seed, can be overridden
):
    """
    Define the bioprocess model with explicit, reproducible key generation.

    Args:
        norm_params: Normalization parameters.
        ann_config: Dictionary with neural network configurations (dims, activation, features).
        master_seed: The integer seed used to generate the master key for NN initialization.

    Returns:
        Tuple of (model, model_config)
    """
    print(
        f"--- Defining model using master_seed: {master_seed} ---"
    )  # Log the seed used

    # Create model builder
    builder = HybridModelBuilder()
    builder.set_normalization_params(norm_params)

    # Add state variables
    builder.add_state("X")
    builder.add_state("P")

    # Create model config for documentation
    model_config = ModelConfig(state_names=["X", "P"], mechanistic_components={})

    # Define mechanistic parts (remains the same)
    def calculate_dilution_rate(inputs):
        volume = inputs.get("reactor_volume", 1.0)
        feed_rate = inputs.get("feed_rate", 0.0)
        base_rate = inputs.get("base_rate", 0.0)
        total_flow_rate = feed_rate + base_rate
        dilution_rate = jnp.where(volume > 1e-6, total_flow_rate / volume, 0.0)
        return dilution_rate

    def biomass_ode(inputs):
        X = inputs["X"]
        mu = inputs["growth_rate"]
        dilution_rate = calculate_dilution_rate(inputs)
        dXdt = mu * X - dilution_rate * X
        return dXdt

    def product_ode(inputs):
        X = inputs["X"]
        P = inputs["P"]
        vpx = inputs["product_rate"]
        inductor_switch = inputs.get("inductor_switch", 0.0)
        dilution_rate = calculate_dilution_rate(inputs)
        dPdt = vpx * X * inductor_switch - dilution_rate * P
        return dPdt

    # Add mechanistic components
    builder.add_mechanistic_component("X", biomass_ode)
    builder.add_mechanistic_component("P", product_ode)
    model_config.mechanistic_components["X"] = biomass_ode
    model_config.mechanistic_components["P"] = product_ode

    # --- Explicit Key Generation ---
    master_key = create_initial_random_key(master_seed)
    key_growth, key_product = jax.random.split(master_key)
    # --- End Explicit Key Generation ---

    # Define and add NNs using explicit keys and config from ann_config
    # Growth Rate NN
    if "growth_rate" in ann_config:
        growth_settings = ann_config["growth_rate"]
        growth_nn_config_doc = NeuralNetworkConfig(
            name="growth_rate",
            input_features=growth_settings["input_features"],
            hidden_dims=growth_settings["hidden_dims"],
            output_activation=growth_settings["output_activation"],
            seed=master_seed,  # Store master seed for info
        )
        model_config.add_nn(growth_nn_config_doc)
        builder.replace_with_nn(
            name="growth_rate",
            input_features=growth_settings["input_features"],
            hidden_dims=growth_settings["hidden_dims"],
            output_activation=growth_nn_config_doc.get_activation_fn(),  # Use helper
            key=key_growth,  # Pass explicit key
        )
        print(f"  Added 'growth_rate' NN with key derived from seed {master_seed}")
    else:
        print("Warning: 'growth_rate' configuration missing in ann_config.")

    # Product Rate NN
    if "product_rate" in ann_config:
        product_settings = ann_config["product_rate"]
        product_nn_config_doc = NeuralNetworkConfig(
            name="product_rate",
            input_features=product_settings["input_features"],
            hidden_dims=product_settings["hidden_dims"],
            output_activation=product_settings["output_activation"],
            seed=master_seed,  # Store master seed for info (could also store subkey info if desired)
        )
        model_config.add_nn(product_nn_config_doc)
        builder.replace_with_nn(
            name="product_rate",
            input_features=product_settings["input_features"],
            hidden_dims=product_settings["hidden_dims"],
            output_activation=product_nn_config_doc.get_activation_fn(),  # Use helper
            key=key_product,  # Pass explicit key
        )
        print(f"  Added 'product_rate' NN with key derived from seed {master_seed}")
    else:
        print("Warning: 'product_rate' configuration missing in ann_config.")

    # Build the model
    model = builder.build()
    return model, model_config


# =============================================
# MAIN FUNCTION
# =============================================


def main(master_seed_for_init=42):  # Allow passing seed to main
    """Main function to run the bioprocess hybrid modeling experiment."""
    print(f"*** Starting Experiment Run with Master Seed: {master_seed_for_init} ***")

    # Configure model architecture (ensure dimensions match desired setup)
    ann_config = {
        "growth_rate": {
            "hidden_dims": [32, 32, 32],  # Match old script architecture
            "output_activation": "soft_sign",
            "input_features": ["X", "P", "temp", "feed", "inductor_mass"],
            # 'seed' field here is now just informational
        },
        "product_rate": {
            "hidden_dims": [32, 32, 32],  # Match old script architecture
            "output_activation": "softplus",
            "input_features": ["X", "P", "temp", "feed", "inductor_mass"],
            # 'seed' field here is now just informational
        },
    }

    # Configure solver
    solver_config = SolverConfig(
        solver_type="tsit5",
        step_size_controller="pid",
        rtol=1e-2,
        atol=1e-4,
        max_steps=500000,
    )

    # Load data
    print("Loading data...")
    data_manager = load_bioprocess_data("testtrain.xlsx", train_ratio=0.8)
    print(
        f"Loaded {len(data_manager.train_datasets)} training datasets and "
        f"{len(data_manager.test_datasets)} test datasets"
    )

    # Get normalization parameters
    norm_params = data_manager.norm_params

    # Prepare datasets
    train_datasets = data_manager.prepare_training_data()
    test_datasets = data_manager.prepare_test_data()

    # Build model with explicit master seed and correct architecture
    print("Building hybrid model...")
    model, model_config = define_bioprocess_model(
        norm_params,
        ann_config,
        master_seed=master_seed_for_init,  # Pass the chosen seed
    )

    # Create experiment manager
    # Experiment name can include the seed for clarity
    experiment_name = f"bioprocess_experiment_seed{master_seed_for_init}"
    output_dir_base = "results"  # Keep base directory consistent
    experiment = ExperimentManager(
        model=model,
        model_config=model_config,
        norm_params=norm_params,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        output_dir=output_dir_base,  # Manager will create subfolder
        experiment_name=experiment_name,
    )
    print(f"Experiment results will be saved to: {experiment.output_dir}")

    # Generate model documentation
    print("Generating model documentation...")
    experiment.generate_model_documentation()

    # Save normalization parameters
    experiment.save_normalization_parameters()

    # Train model
    print("Training model...")
    # The ExperimentManager uses the correctly initialized model
    trained_model = experiment.train(
        num_epochs=10000,
        learning_rate=1e-3,
        early_stopping_patience=2500,
        component_weights={"X": 1.0, "P": 10.0},
        loss_metric=MSE,
        solver_config=solver_config,  # Pass solver config for training loss
        save_checkpoints=True,  # Optional: save intermediate models
        verbose=True,  # Show training progress
    )

    # Evaluate model
    print("Evaluating model...")
    # ExperimentManager uses the same solver_config used for training by default
    # if it was passed to train(). Otherwise it defaults to eval settings.
    # Since we passed it to train, it should use rtol=1e-2, atol=1e-4 here too.
    metrics = experiment.evaluate(verbose=True)

    # Visualize results
    print("Generating visualizations...")
    # Again, uses the solver_config associated with the experiment
    experiment.visualize(
        state_labels={"X": "Biomass (CDW g/L)", "P": "Product (g/L)"},
        component_names=["Biomass Loss", "Product Loss"],  # Matches loss aux output
    )

    # Save all results
    print("Saving all results...")
    result_paths = experiment.save_all_results()

    print(f"Experiment Run {master_seed_for_init} completed successfully!")
    print(f"All results saved to {experiment.output_dir}")

    # Return results if needed for further analysis
    return trained_model, train_datasets, test_datasets, experiment


if __name__ == "__main__":
    # --- Run with the seed that matches the old script ---
    main(master_seed_for_init=42)
