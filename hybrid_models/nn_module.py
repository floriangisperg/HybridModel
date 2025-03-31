import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Dict, Callable, Optional
from jaxtyping import Array, Float, PyTree


class ConfigurableNN(eqx.Module):
    """
    A configurable neural network that can replace parts of a mechanistic model.
    """
    layers: List  # Neural network layers
    norm_params: Dict[str, float]  # Normalization parameters
    input_features: List[str]  # List of input feature names

    def __init__(
            self,
            norm_params: Dict,
            input_features: List[str],
            hidden_dims: List[int] = [16, 16],
            output_activation: Optional[Callable] = None,
            key=None
    ):
        self.norm_params = norm_params
        self.input_features = input_features

        # Build the layers
        if key is None:
            key = jax.random.PRNGKey(0)

        # Input dimension is number of features
        input_dim = len(input_features)

        layers = []

        # First layer: input -> first hidden
        layer_key, key = jax.random.split(key)
        layers.append(eqx.nn.Linear(input_dim, hidden_dims[0], key=layer_key))
        layers.append(jax.nn.relu)

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layer_key, key = jax.random.split(key)
            layers.append(eqx.nn.Linear(hidden_dims[i], hidden_dims[i + 1], key=layer_key))
            layers.append(jax.nn.relu)

        # Output layer
        layer_key, key = jax.random.split(key)
        layers.append(eqx.nn.Linear(hidden_dims[-1], 1, key=layer_key))

        # Optional output activation
        if output_activation is not None:
            layers.append(output_activation)

        self.layers = layers

    def __call__(self, inputs: Dict[str, Float[Array, "..."]]) -> Float[Array, ""]:
        """
        Forward pass of the neural network.

        Args:
            inputs: Dictionary of input values for each feature

        Returns:
            Scalar prediction
        """
        # Get normalized inputs for each requested feature
        normalized_inputs = []

        for feature in self.input_features:
            # Check if normalization is needed
            mean_key = f"{feature}_mean"
            std_key = f"{feature}_std"

            if mean_key in self.norm_params and std_key in self.norm_params:
                # Standard scaling for continuous variables
                normalized_value = (inputs[feature] - self.norm_params[mean_key]) / (
                        self.norm_params[std_key] + 1e-8)
                normalized_inputs.append(normalized_value)
            else:
                # Use raw value if normalization params not available
                normalized_inputs.append(inputs[feature])

        # Combine normalized inputs
        x = jnp.array(normalized_inputs)

        # Forward pass through layers
        for layer in self.layers:
            x = layer(x)

        # Return scalar output
        return x[0]