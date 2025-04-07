"""
Model configuration and documentation utilities for hybrid models.

This module provides utilities for configuring neural networks and
generating human-readable documentation of hybrid models.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
import jax
import jax.numpy as jnp
import os
import inspect


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
                "none": None
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model configuration to a dictionary.
        """
        return {
            "state_names": self.state_names,
            "mechanistic_components": {name: "Function" for name in self.mechanistic_components},
            "neural_networks": [
                {
                    "name": nn.name,
                    "input_features": nn.input_features,
                    "hidden_dims": nn.hidden_dims,
                    "output_activation": (
                        nn.output_activation if isinstance(nn.output_activation, str)
                        else f"Function: {nn.output_activation.__name__ if hasattr(nn.output_activation, '__name__') else 'custom'}"
                    ),
                    "dropout_rate": nn.dropout_rate
                }
                for nn in self.neural_networks
            ]
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

    Args:
        model: The hybrid model instance
        model_config: The model configuration
        norm_params: Normalization parameters

    Returns:
        Formatted string with model description
    """
    doc = []
    doc.append("=" * 60)
    doc.append("HYBRID MODEL DOCUMENTATION")
    doc.append("=" * 60)
    doc.append("")

    # State variables
    doc.append("STATE VARIABLES:")
    doc.append("-" * 30)
    for state in model_config.state_names:
        doc.append(f"- {state}")
    doc.append("")

    # Mechanistic components
    doc.append("MECHANISTIC COMPONENTS:")
    doc.append("-" * 30)
    for name, func in model_config.mechanistic_components.items():
        doc.append(f"Component: {name}")
        # Try to get the function source code
        try:
            source = inspect.getsource(func)
            # Clean up the source code a bit
            source = "\n".join(["    " + line for line in source.split("\n")])
            doc.append("Source:")
            doc.append(source)
        except:
            doc.append("    <Source code not available>")
        doc.append("")

    # Neural network components
    doc.append("NEURAL NETWORK COMPONENTS:")
    doc.append("-" * 30)
    for nn_config in model_config.neural_networks:
        doc.append(f"Neural Network: {nn_config.name}")
        doc.append(f"  Input Features: {', '.join(nn_config.input_features)}")
        doc.append(f"  Architecture: {nn_config.hidden_dims}")

        # Get activation name
        activation = nn_config.output_activation
        if isinstance(activation, str):
            activation_name = activation
        elif activation is None:
            activation_name = "None (linear)"
        else:
            activation_name = activation.__name__ if hasattr(activation, "__name__") else "custom"

        doc.append(f"  Output Activation: {activation_name}")
        doc.append("")

    # Normalization parameters
    doc.append("NORMALIZATION PARAMETERS:")
    doc.append("-" * 30)
    for key, value in norm_params.items():
        doc.append(f"{key}: {value}")

    return "\n".join(doc)


def save_model_description(model, model_config: ModelConfig, norm_params: Dict,
                           filepath: str = "model_description.txt"):
    """
    Save the model description to a text file.
    """
    description = describe_model(model, model_config, norm_params)

    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(description)
    return filepath


def save_normalization_params(norm_params: Dict, filepath: str = "normalization_params.txt"):
    """
    Save normalization parameters to a text file.

    Args:
        norm_params: Dictionary of normalization parameters
        filepath: Path to save the parameters
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("NORMALIZATION PARAMETERS\n")
        f.write("=" * 30 + "\n\n")

        # Group parameters by variable (e.g., group X_mean and X_std together)
        param_groups = {}
        for key, value in norm_params.items():
            if '_mean' in key:
                var_name = key.replace('_mean', '')
                if var_name not in param_groups:
                    param_groups[var_name] = {}
                param_groups[var_name]['mean'] = value
            elif '_std' in key:
                var_name = key.replace('_std', '')
                if var_name not in param_groups:
                    param_groups[var_name] = {}
                param_groups[var_name]['std'] = value

        # Write grouped parameters
        for var_name, params in param_groups.items():
            f.write(f"Variable: {var_name}\n")
            for param_type, value in params.items():
                f.write(f"  {param_type.capitalize()}: {value}\n")
            f.write("\n")

    return filepath