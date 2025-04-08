import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from typing import Dict, List, Tuple, Any, Callable, Optional
from jaxtyping import Array, Float, PyTree


def get_value_at_time(t: float, times: Float[Array, "N"], values: Float[Array, "N"]) -> float:
    """Get value at specific time t using nearest-time interpolation."""
    idx = jnp.argmin(jnp.abs(times - t))
    return values[idx]


class HybridODESystem(eqx.Module):
    """
    A general framework for hybrid ODE systems that combine mechanistic models with neural networks
    and trainable parameters.
    """
    mechanistic_components: Dict[str, Callable]
    nn_replacements: Dict[str, eqx.Module]
    trainable_parameters: Dict[str, jnp.ndarray]
    parameter_transforms: Dict[str, Dict]
    state_names: List[str]

    def __init__(
            self,
            mechanistic_components: Dict[str, Callable],
            nn_replacements: Dict[str, eqx.Module],
            trainable_parameters: Dict[str, jnp.ndarray],
            parameter_transforms: Dict[str, Dict],
            state_names: List[str]
    ):
        """
        Initialize the hybrid ODE system.

        Args:
            mechanistic_components: Dictionary of mechanistic model components
            nn_replacements: Dictionary of neural network replacements
            trainable_parameters: Dictionary of trainable parameters
            parameter_transforms: Dictionary of parameter transformation settings
            state_names: Names of the state variables in the correct order
        """
        self.mechanistic_components = mechanistic_components
        self.nn_replacements = nn_replacements
        self.trainable_parameters = trainable_parameters
        self.parameter_transforms = parameter_transforms
        self.state_names = state_names

    def _transform_parameter(self, name: str, value: jnp.ndarray) -> jnp.ndarray:
        """Apply the appropriate transformation to a parameter."""
        transform_info = self.parameter_transforms.get(name, {})
        transform_type = transform_info.get('transform', 'none')
        bounds = transform_info.get('bounds', None)

        if transform_type == 'sigmoid' and bounds:
            # Apply sigmoid transform to bound the parameter
            lower, upper = bounds
            return lower + (upper - lower) * jax.nn.sigmoid(value)
        elif transform_type == 'softplus':
            # Ensure parameter is positive
            return jax.nn.softplus(value)
        elif transform_type == 'exp':
            # Exponential transform
            return jnp.exp(value)
        else:
            # No transformation
            return value

    def ode_function(self, t: float, y: Float[Array, "D"], args: Dict) -> Float[Array, "D"]:
        """
        The ODE function that combines mechanistic, neural network, and trainable parameter components.
        """
        # Convert state array to dictionary
        state_dict = {name: y[i] for i, name in enumerate(self.state_names)}

        # Create inputs dictionary for components
        inputs = {**state_dict}

        # Add time-dependent inputs
        time_inputs = args.get('time_dependent_inputs', {})
        for key, (times, values) in time_inputs.items():
            inputs[key] = get_value_at_time(t, times, values)

        # Add static inputs
        inputs.update(args.get('static_inputs', {}))

        # Add transformed trainable parameters to inputs
        for name, param in self.trainable_parameters.items():
            inputs[name] = self._transform_parameter(name, param)

        # Compute neural network outputs
        for name, nn in self.nn_replacements.items():
            inputs[name] = nn(inputs)

        # Process each component to calculate derivatives
        derivatives = []
        for state_name in self.state_names:
            if state_name in self.mechanistic_components:
                # State has a mechanistic function
                derivatives.append(self.mechanistic_components[state_name](inputs))
            elif state_name in self.nn_replacements:
                # State is directly calculated by a neural network
                derivatives.append(self.nn_replacements[state_name](inputs))
            else:
                # State doesn't have a defined derivative
                raise ValueError(f"No derivative defined for state {state_name}")

        return jnp.array(derivatives)

    def solve(
            self,
            initial_state: Dict[str, float],
            t_span: Tuple[float, float],
            evaluation_times: Float[Array, "N"],
            args: Dict,
            solver=diffrax.Tsit5(),
            rtol=1e-3,
            atol=1e-6,
            max_steps=100000,
            dt0=0.1,
            stepsize_controller=None,  # Optional explicit stepsize controller
            **kwargs
    ) -> Dict[str, Float[Array, "..."]]:
        """
        Solve the ODE system.

        Args:
            initial_state: Dictionary of initial states
            t_span: (t0, t1) time span to solve over
            evaluation_times: Times at which to evaluate the solution
            args: Additional arguments for the ODE function
            solver: Diffrax solver
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_steps: Maximum number of steps
            dt0: Initial step size
            stepsize_controller: Optional explicit stepsize controller (if None, creates PIDController)
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing solution arrays
        """
        # Convert initial state dictionary to array
        y0 = jnp.array([initial_state[name] for name in self.state_names])

        # Define ODE term
        term = diffrax.ODETerm(lambda t, y, args: self.ode_function(t, y, args))

        # Set up saveat
        saveat = diffrax.SaveAt(ts=evaluation_times)

        # Use provided stepsize_controller or create default one based on rtol/atol
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        # Solve ODE with robust settings
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=saveat,
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
            **kwargs
        )

        # Extract solution and return as dictionary
        solution = {
            'times': sol.ts,
        }

        # Add each state's solution
        for i, name in enumerate(self.state_names):
            solution[name] = sol.ys[:, i]

        return solution