import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

# **** ADD Tuple ****
from typing import Dict, List, Tuple, Any, Callable, Optional
from jaxtyping import Array, Float, PyTree

# **** REMOVE get_value_at_time or keep for other uses, but ODE uses interp ****
# from .utils import get_value_at_time # No longer needed for ODE function

# **** IMPORT the new interpolation function ****
from .utils import interp_linear  # Use the JIT-compatible function


class HybridODESystem(eqx.Module):
    """
    A general framework for hybrid ODE systems that combine mechanistic models
    with neural networks and trainable parameters. Accesses time-dependent inputs
    via linear interpolation.
    """

    mechanistic_components: Dict[str, Callable]
    nn_replacements: Dict[str, eqx.Module]
    trainable_parameters: Dict[str, jnp.ndarray]
    parameter_transforms: Dict[str, Dict]
    state_names: List[str]

    # __init__ remains the same
    def __init__(
        self,
        mechanistic_components: Dict[str, Callable],
        nn_replacements: Dict[str, eqx.Module],
        trainable_parameters: Dict[str, jnp.ndarray],
        parameter_transforms: Dict[str, Dict],
        state_names: List[str],
    ):
        self.mechanistic_components = mechanistic_components
        self.nn_replacements = nn_replacements
        self.trainable_parameters = trainable_parameters
        self.parameter_transforms = parameter_transforms
        self.state_names = state_names

    # _transform_parameter remains the same
    def _transform_parameter(self, name: str, value: jnp.ndarray) -> jnp.ndarray:
        # ... (implementation as before) ...
        transform_info = self.parameter_transforms.get(name, {})
        transform_type = transform_info.get("transform", "none")
        bounds = transform_info.get("bounds", None)

        if transform_type == "sigmoid" and bounds:
            lower, upper = bounds
            return lower + (upper - lower) * jax.nn.sigmoid(value)
        elif transform_type == "softplus":
            return jax.nn.softplus(value)
        elif transform_type == "exp":
            return jnp.exp(value)
        else:
            return value

    # **** MODIFIED: ode_function uses linear interpolation ****
    def ode_function(
        self, t: float, y: Float[Array, "D"], args: Dict
    ) -> Float[Array, "D"]:
        """
        The ODE function that combines mechanistic, neural network, and trainable
        parameter components. Uses linear interpolation for time-dependent inputs.
        """
        # Convert state array to dictionary
        state_dict = {name: y[i] for i, name in enumerate(self.state_names)}

        # Create inputs dictionary for components, starting with current state
        inputs = {**state_dict}

        # Add static inputs (constants for the run)
        inputs.update(args.get("static_inputs", {}))

        # Add transformed trainable parameters to inputs
        for name, param in self.trainable_parameters.items():
            inputs[name] = self._transform_parameter(name, param)

        # --- Interpolate Time-Dependent Inputs ---
        time_inputs = args.get("time_dependent_inputs", {})
        for key, times_values_tuple in time_inputs.items():
            if isinstance(times_values_tuple, tuple) and len(times_values_tuple) == 2:
                dense_times, dense_values = times_values_tuple
                # Use the JIT-compatible linear interpolation function from utils
                inputs[key] = interp_linear(t, dense_times, dense_values)
            else:
                # This path should ideally not be hit if prepare_for_training is correct
                # Raise an error to make incorrect formatting obvious during development/use
                raise TypeError(
                    f"Invalid format for time_dependent_input '{key}'. "
                    f"Expected a (times, values) tuple, but got {type(times_values_tuple)}. "
                    "Check dataset preparation."
                )

        # Compute neural network outputs (NNs now receive interpolated inputs)
        for name, nn in self.nn_replacements.items():
            # NNs need all required inputs available in the 'inputs' dict
            try:
                inputs[name] = nn(inputs)
            except KeyError as e:
                raise KeyError(
                    f"Input key {e} missing for NN '{name}'. Available inputs: {list(inputs.keys())}"
                ) from e

        # Process each component to calculate derivatives
        derivatives = []
        for state_name in self.state_names:
            # Check if derivative is defined by mechanistic or NN component
            component_func = self.mechanistic_components.get(state_name)
            nn_replacement = self.nn_replacements.get(state_name)

            if component_func is not None:
                # State has a mechanistic function
                try:
                    deriv = component_func(inputs)
                    derivatives.append(deriv)
                except KeyError as e:
                    raise KeyError(
                        f"Input key {e} missing for mechanistic component '{state_name}'. Available inputs: {list(inputs.keys())}"
                    ) from e
            elif nn_replacement is not None:
                # State derivative is *directly* calculated by a neural network
                # (Less common for state derivatives, more common for intermediate rates)
                try:
                    deriv = nn_replacement(inputs)
                    derivatives.append(deriv)
                except KeyError as e:
                    raise KeyError(
                        f"Input key {e} missing for NN derivative replacement '{state_name}'. Available inputs: {list(inputs.keys())}"
                    ) from e
            else:
                # State doesn't have a defined derivative function
                raise ValueError(
                    f"No mechanistic component or direct NN replacement found for the derivative of state '{state_name}'. "
                    f"Defined mechanistic components: {list(self.mechanistic_components.keys())}. "
                    f"Defined NN replacements: {list(self.nn_replacements.keys())}."
                )

        # Ensure output is a JAX array
        return jnp.asarray(derivatives)

    # solve method remains the same
    def solve(
        self,
        initial_state: Dict[str, float],
        t_span: Tuple[float, float],
        evaluation_times: Float[Array, "N"],  # These are the sparse times
        args: Dict,  # Contains time_dependent_inputs as (dense_t, dense_v) tuples
        solver=diffrax.Tsit5(),
        rtol=1e-3,
        atol=1e-6,
        max_steps=100000,
        dt0=0.1,
        stepsize_controller=None,
        **kwargs,
    ) -> Dict[str, Float[Array, "..."]]:
        # ... (implementation as before) ...
        # Convert initial state dictionary to array
        y0 = jnp.array([initial_state[name] for name in self.state_names])

        # Define ODE term using the (now interpolation-aware) ode_function
        term = diffrax.ODETerm(self.ode_function)  # Pass the method directly

        # Save solution only at the requested sparse evaluation_times
        saveat = diffrax.SaveAt(ts=evaluation_times)

        # Use provided stepsize_controller or create default PIDController
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        # Solve ODE
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=dt0,
            y0=y0,
            args=args,  # Pass args containing time_dependent_inputs etc.
            saveat=saveat,
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
            **kwargs,
        )

        # Extract solution and return as dictionary aligned with evaluation_times
        solution = {"times": sol.ts}  # Should match evaluation_times
        for i, name in enumerate(self.state_names):
            solution[name] = sol.ys[:, i]

        return solution
