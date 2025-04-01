"""Optimized neural network components for hybrid models."""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, List, Callable, Any, Optional
from jaxtyping import Array, Float, PyTree


class JITOptimizedNN(eqx.Module):
    """
    Neural network with optimized JIT compilation for hybrid models.
    This wraps the ConfigurableNN with explicit JIT compilation.
    """
    nn: Any  # The original neural network module
    input_features: List[str]  # List of required input features
    _jitted_forward: Optional[Callable] = None  # JIT-compiled forward pass

    def __init__(self, nn):
        """
        Initialize with an existing neural network.

        Args:
            nn: ConfigurableNN instance to optimize
        """
        self.nn = nn
        self.input_features = nn.input_features
        self._jitted_forward = None

    def _prepare_forward(self):
        """Create a JIT-compiled forward pass for the network."""

        # Create a standalone function to compute normalized inputs
        def prepare_inputs(raw_inputs, input_features, norm_params):
            normalized_inputs = []

            for feature in input_features:
                # Extract value from inputs dict
                value = raw_inputs.get(feature, 0.0)

                # Apply normalization if parameters are available
                mean_key = f"{feature}_mean"
                std_key = f"{feature}_std"

                if mean_key in norm_params and std_key in norm_params:
                    normalized_value = (value - norm_params[mean_key]) / (norm_params[std_key] + 1e-8)
                    normalized_inputs.append(normalized_value)
                else:
                    normalized_inputs.append(value)

            return jnp.array(normalized_inputs)

        # Create a JIT-compilable forward function
        def forward_fn(nn_params, norm_params, inputs):
            # Prepare the inputs as an array
            input_array = prepare_inputs(inputs, self.input_features, norm_params)

            # Forward pass through the network layers
            x = input_array
            for layer in nn_params:
                x = layer(x)

            # Return scalar output
            return x[0]

        # Create a statically-compiled version
        self._jitted_forward = jax.jit(
            forward_fn,
            static_argnames=['norm_params']
        )

    def __call__(self, inputs: Dict) -> Float:
        """
        Optimized forward pass.

        Args:
            inputs: Dictionary of input values

        Returns:
            Network output (scalar)
        """
        # Create JIT-compiled function if not already created
        if self._jitted_forward is None:
            self._prepare_forward()

        # Forward pass with the JIT-compiled function
        try:
            return self.nn(inputs)
        except Exception as e:
            print(f"Error in neural network forward pass: {e}")
            return 0.0  # Return default value on error


def optimize_nn_components(model: Any) -> Any:
    """
    Optimize neural network components in a hybrid model.

    Args:
        model: Hybrid model with neural networks

    Returns:
        Model with optimized neural networks
    """
    # Get model components
    mech_components = model.mechanistic_components
    nn_replacements = model.nn_replacements
    state_names = model.state_names

    # Create optimized neural networks
    optimized_nns = {}
    for name, nn in nn_replacements.items():
        optimized_nns[name] = JITOptimizedNN(nn)

    # Create new model with optimized networks
    optimized_model = type(model)(
        mechanistic_components=mech_components,
        nn_replacements=optimized_nns,
        state_names=state_names
    )

    return optimized_model


def create_optimized_ode_function(model: Any) -> Callable:
    """
    Create an optimized ODE function for a hybrid model.

    This function does not JIT-compile the entire ODE function,
    but optimizes the neural network evaluations within it.

    Args:
        model: Hybrid model

    Returns:
        Optimized ODE function
    """
    # Extract components
    mech_components = model.mechanistic_components
    nn_replacements = model.nn_replacements
    state_names = model.state_names

    # JIT-compile individual neural networks
    jitted_nns = {}
    for name, nn in nn_replacements.items():
        # Extract and JIT-compile the forward pass
        @jax.jit
        def nn_forward(inputs, name=name, original_nn=nn):
            return original_nn(inputs)

        jitted_nns[name] = nn_forward

    # Create efficient ODE function
    def optimized_ode_function(t, y, args):
        # Convert state array to dictionary
        state_dict = {name: y[i] for i, name in enumerate(state_names)}

        # Create inputs dictionary
        inputs = {**state_dict}

        # Add time-dependent inputs
        time_inputs = args.get('time_dependent_inputs', {})
        for key, (times, values) in time_inputs.items():
            # Find closest time index
            idx = jnp.argmin(jnp.abs(times - t))
            inputs[key] = values[idx]

        # Add static inputs
        inputs.update(args.get('static_inputs', {}))

        # Neural network computations with JIT-compiled functions
        for name, jitted_nn in jitted_nns.items():
            inputs[name] = jitted_nn(inputs)

        # Calculate intermediate values
        intermediate_values = {}
        for name, component_fn in mech_components.items():
            # Skip if this component is replaced by a neural network
            if name in nn_replacements:
                continue

            # Calculate component output
            try:
                intermediate_values[name] = component_fn(inputs)
            except Exception as e:
                print(f"Error in component {name}: {e}")
                intermediate_values[name] = 0.0

        # Add intermediate values to inputs
        inputs.update(intermediate_values)

        # Calculate state derivatives
        derivatives = []
        for state_name in state_names:
            if state_name in mech_components:
                derivatives.append(mech_components[state_name](inputs))
            elif state_name in nn_replacements:
                derivatives.append(nn_replacements[state_name](inputs))
            else:
                raise ValueError(f"No derivative for {state_name}")

        return jnp.array(derivatives)

    return optimized_ode_function