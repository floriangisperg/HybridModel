"""
Model configuration and documentation utilities for hybrid models.

This module provides utilities for configuring neural networks and
generating human-readable documentation of hybrid models.
"""

import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
import jax
import jax.numpy as jnp
import os
import inspect

from .builder import HybridModelBuilder
from .utils import create_initial_random_key


@dataclass
class NeuralNetworkConfig:
    """
    Configuration for neural network component in hybrid models.
    """

    name: str
    input_features: List[str]
    hidden_dims: List[int] = field(default_factory=lambda: [32, 32])
    output_activation: Optional[Union[str, Callable]] = None
    dropout_rate: float = 0.0
    weight_init: str = "normal"
    seed: int = 0

    def get_activation_fn(self):
        """
        Get the activation function based on the string name or return the function.
        """
        if isinstance(self.output_activation, str):
            activations = {
                "relu": jax.nn.relu,
                "sigmoid": jax.nn.sigmoid,
                "tanh": jax.nn.tanh,
                "softplus": jax.nn.softplus,
                "soft_sign": jax.nn.soft_sign,
                "elu": jax.nn.elu,
                "leaky_relu": jax.nn.leaky_relu,
                "none": None,
            }
            return activations.get(self.output_activation.lower(), None)
        return self.output_activation

    def get_random_key(self):
        """
        Get a JAX random key based on the seed.
        """
        return jax.random.PRNGKey(self.seed)


@dataclass
class ModelConfig:
    """
    Configuration for hybrid model structure.
    """

    state_names: List[str]
    mechanistic_components: Dict[str, Callable] = field(default_factory=dict)
    neural_networks: List[NeuralNetworkConfig] = field(default_factory=list)
    trainable_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model configuration to a dictionary.
        """
        # --- UPDATE THIS METHOD ---
        mech_components_repr = {}
        for name, func in self.mechanistic_components.items():
            try:
                mech_components_repr[name] = f"Function: {func.__name__}"
            except AttributeError:
                mech_components_repr[name] = (
                    "Function: <lambda> or other non-named callable"
                )

        return {
            "state_names": self.state_names,
            "mechanistic_components": mech_components_repr,  # Use the representation
            "neural_networks": [
                {
                    "name": nn.name,
                    "input_features": nn.input_features,
                    "hidden_dims": nn.hidden_dims,
                    "output_activation": (
                        nn.output_activation
                        if isinstance(nn.output_activation, str)
                        else (
                            f"Function: {nn.output_activation.__name__ if hasattr(nn.output_activation, '__name__') else 'custom'}"
                            if nn.output_activation is not None
                            else "None"
                        )  # Handle None case
                    ),
                    "dropout_rate": nn.dropout_rate,
                    "seed": nn.seed,
                }
                for nn in self.neural_networks
            ],
            "trainable_parameters": (
                {
                    name: {
                        "initial_value": info.get("initial_value"),
                        "bounds": info.get("bounds"),
                        "transform": info.get("transform"),
                    }
                    for name, info in self.trainable_parameters.items()
                }
                if self.trainable_parameters
                else {}
            ),
        }

    def add_nn(self, nn_config: NeuralNetworkConfig):
        """
        Add a neural network configuration.
        """
        self.neural_networks.append(nn_config)
        return self


def describe_model(model, model_config: ModelConfig, norm_params: Dict) -> str:
    """
    Generate a detailed text description of the model structure.
    ... (Args remain the same) ...
    """
    doc = []
    doc.append("=" * 60)
    doc.append("HYBRID MODEL DOCUMENTATION")
    doc.append("=" * 60)
    doc.append(
        f"Generated for ModelConfig defined with states: {model_config.state_names}"
    )
    doc.append("")

    # State variables
    doc.append("STATE VARIABLES:")
    doc.append("-" * 30)
    for state in model_config.state_names:
        doc.append(f"- {state}")
    doc.append("")

    # Mechanistic components
    doc.append("MECHANISTIC COMPONENTS (State Equations):")
    doc.append("-" * 30)
    if model_config.mechanistic_components:
        for name, func in model_config.mechanistic_components.items():
            # Only describe components that define state derivatives
            if name in model_config.state_names:
                doc.append(f"Component for d{name}/dt:")
                try:
                    func_name = func.__name__
                    if func_name == "<lambda>":
                        func_name = "Lambda Function"
                except AttributeError:
                    func_name = "Callable Object"
                doc.append(f"  Function: {func_name}")

                # --- REVISED SOURCE CODE DISPLAY ---
                try:
                    source = inspect.getsource(func)
                    # Dedent the source code to remove leading whitespace common to all lines
                    dedented_source = textwrap.dedent(source)
                    # Optional: Limit the number of lines shown or strip decorators/docstrings
                    source_lines = dedented_source.strip().split("\n")
                    # Remove potential signature line if present
                    if source_lines and source_lines[0].strip().startswith("def "):
                        source_lines = source_lines[1:]
                    # Remove docstring if present (simple check for triple quotes)
                    if source_lines and source_lines[0].strip().startswith(
                        ('"""', "'''")
                    ):
                        end_docstring = -1
                        for idx, line in enumerate(source_lines):
                            if line.strip().endswith(('"""', "'''")):
                                end_docstring = idx
                                break
                        if end_docstring != -1:
                            source_lines = source_lines[end_docstring + 1 :]

                    # Indent the relevant source lines for display
                    formatted_source = "\n".join(
                        ["    " + line for line in source_lines if line.strip()]
                    )  # Ignore empty lines
                    if formatted_source:
                        doc.append("  Implementation:")
                        doc.append(formatted_source)
                    else:
                        doc.append("  <Could not extract source body>")

                except (TypeError, OSError, IOError):
                    doc.append("  <Source code not available>")
                doc.append("")
                # --- END REVISED SOURCE CODE DISPLAY ---
    else:
        doc.append("  None defined in ModelConfig.")
    doc.append("")

    # Neural network components
    doc.append("NEURAL NETWORK COMPONENTS:")
    doc.append("-" * 30)
    if model_config.neural_networks:
        for nn_config in model_config.neural_networks:
            doc.append(f"Neural Network replacing/defining: '{nn_config.name}'")
            doc.append(f"  Input Features: {', '.join(nn_config.input_features)}")
            doc.append(f"  Architecture (Hidden Dims): {nn_config.hidden_dims}")

            # Get activation name
            activation = nn_config.output_activation
            if isinstance(activation, str):
                activation_name = activation
            elif activation is None:
                activation_name = "None (linear)"
            else:  # If it's a callable function
                activation_name = (
                    activation.__name__
                    if hasattr(activation, "__name__")
                    else "Custom Callable"
                )

            doc.append(f"  Output Activation: {activation_name}")
            doc.append(f"  Initialization Seed Used: {nn_config.seed}")  # Add seed info
            doc.append("")
    else:
        doc.append("  None defined in ModelConfig.")
    doc.append("")

    # --- ADD THIS NEW SECTION ---
    doc.append("TRAINABLE PARAMETERS:")
    doc.append("-" * 30)
    if model_config.trainable_parameters:
        for name, param_info in model_config.trainable_parameters.items():
            doc.append(f"Parameter Name: '{name}'")
            doc.append(f"  Initial Value: {param_info.get('initial_value', 'N/A')}")
            bounds = param_info.get("bounds")
            if bounds:
                doc.append(f"  Bounds: {bounds}")
            transform = param_info.get("transform", "none")
            if transform != "none":
                doc.append(f"  Transformation: {transform}")
            doc.append("")
    else:
        doc.append("  None defined in ModelConfig.")
    doc.append("")
    # --- END NEW SECTION ---

    # Normalization parameters (Optional: Could be long)
    # doc.append("NORMALIZATION PARAMETERS:")
    # doc.append("-" * 30)
    # if norm_params:
    #     for key, value in norm_params.items():
    #         doc.append(f"  {key}: {value:.4f}") # Format float values
    # else:
    #     doc.append("  None provided.")
    # doc.append("")

    doc.append("=" * 60)

    return "\n".join(doc)


def save_model_description(
    model,
    model_config: ModelConfig,
    norm_params: Dict,
    filepath: str = "model_description.txt",
):
    """
    Save the model description to a text file.
    """
    description = describe_model(model, model_config, norm_params)

    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "w") as f:
        f.write(description)
    return filepath


def save_normalization_params(
    norm_params: Dict, filepath: str = "normalization_params.txt"
):
    """
    Save normalization parameters to a text file.

    Args:
        norm_params: Dictionary of normalization parameters
        filepath: Path to save the parameters
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "w") as f:
        f.write("NORMALIZATION PARAMETERS\n")
        f.write("=" * 30 + "\n\n")

        # Group parameters by variable (e.g., group X_mean and X_std together)
        param_groups = {}
        for key, value in norm_params.items():
            if "_mean" in key:
                var_name = key.replace("_mean", "")
                if var_name not in param_groups:
                    param_groups[var_name] = {}
                param_groups[var_name]["mean"] = value
            elif "_std" in key:
                var_name = key.replace("_std", "")
                if var_name not in param_groups:
                    param_groups[var_name] = {}
                param_groups[var_name]["std"] = value

        # Write grouped parameters
        for var_name, params in param_groups.items():
            f.write(f"Variable: {var_name}\n")
            for param_type, value in params.items():
                f.write(f"  {param_type.capitalize()}: {value}\n")
            f.write("\n")

    return filepath


def create_model_from_config(
    model_config: ModelConfig, norm_params: Dict, master_seed: int = 42
):
    """
    Builds a HybridODESystem model directly from a ModelConfig object.

    This function encapsulates the logic of using the HybridModelBuilder based
    on a configuration object, simplifying model creation in user scripts.

    Args:
        model_config: The ModelConfig object defining the model structure.
        norm_params: Dictionary of normalization parameters.
        master_seed: The master random seed for initializing neural networks.

    Returns:
        An instance of HybridODESystem.
    """
    print(f"Building model from config using master_seed: {master_seed}")
    builder = HybridModelBuilder()
    builder.set_normalization_params(norm_params)

    # 1. Add States
    for state in model_config.state_names:
        builder.add_state(state)
    print(f"  Added states: {model_config.state_names}")

    # 2. Add Mechanistic Components
    if model_config.mechanistic_components:
        for name, func in model_config.mechanistic_components.items():
            # Ensure component is intended for a state variable's derivative
            if name in model_config.state_names:
                builder.add_mechanistic_component(name, func)
                print(f"  Added mechanistic component for state: {name}")
            else:
                # If the name doesn't match a state, it might be intended
                # as an intermediate calculation used by NNs or other components.
                # The current builder design doesn't explicitly store these,
                # they are just functions used during ODE calculation.
                # If these intermediate calculations need to be explicitly managed
                # or replaced, the builder/config design might need extension.
                # For now, we only register components whose names match states.
                print(
                    f"  Skipping registration of mechanistic component '{name}' (does not match a state name)"
                )

    # 3. Add Neural Network Replacements/Definitions
    if model_config.neural_networks:
        master_key = create_initial_random_key(master_seed)
        # Create a unique key for each neural network
        nn_keys = jax.random.split(master_key, len(model_config.neural_networks))
        print(
            f"  Generating {len(model_config.neural_networks)} NN keys from master_seed {master_seed}"
        )

        for i, nn_conf in enumerate(model_config.neural_networks):
            activation_fn = nn_conf.get_activation_fn()  # Get the actual JAX function
            builder.replace_with_nn(
                name=nn_conf.name,
                input_features=nn_conf.input_features,
                hidden_dims=nn_conf.hidden_dims,
                output_activation=activation_fn,
                key=nn_keys[i],  # Pass the unique key
            )
            print(f"  Added NN component: {nn_conf.name}")

    # 4. Add Trainable Parameters (if they were part of ModelConfig)
    if model_config.trainable_parameters:  # Check if the dict is not empty
        print(
            f"  Adding {len(model_config.trainable_parameters)} trainable parameters..."
        )
        for name, param_info in model_config.trainable_parameters.items():
            if "initial_value" not in param_info:
                print(
                    f"Warning: Trainable parameter '{name}' missing 'initial_value'. Skipping."
                )
                continue

            builder.add_trainable_parameter(
                name=name,
                initial_value=param_info["initial_value"],  # Required key
                bounds=param_info.get("bounds"),  # Optional
                transform=param_info.get(
                    "transform", "none"
                ),  # Optional, default 'none'
            )
            print(
                f"    Added trainable parameter: {name} (initial={param_info['initial_value']})"
            )
    else:
        print("  No trainable parameters defined in config.")

    # 5. Build the model
    model = builder.build()
    print("Model building complete.")
    return model
