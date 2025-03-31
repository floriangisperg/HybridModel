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
    A general framework for hybrid ODE systems that combine mechanistic models with neural networks.
    """
    mechanistic_components: Dict[str, Callable[[Dict[str, Float[Array, "..."]]],
    Float[Array, ""]]]  # Dict of mechanistic model components
    nn_replacements: Dict[str, eqx.Module]  # Dict of neural network replacements
    state_names: List[str]  # Names of state variables

    def __init__(
            self,
            mechanistic_components: Dict[str, Callable],
            nn_replacements: Dict[str, eqx.Module],
            state_names: List[str]
    ):
        """
        Initialize the hybrid ODE system.

        Args:
            mechanistic_components: Dictionary of mechanistic model components as callables
            nn_replacements: Dictionary of neural network replacements for specific components
            state_names: Names of the state variables in the correct order
        """
        self.mechanistic_components = mechanistic_components
        self.nn_replacements = nn_replacements
        self.state_names = state_names

    def ode_function(self, t: float, y: Float[Array, "D"], args: Dict) -> Float[Array, "D"]:
        """
        The ODE function that combines mechanistic and neural network components.

        Args:
            t: Current time
            y: Current state (as an array)
            args: Additional arguments

        Returns:
            Derivatives of states
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

        # IMPORTANT: First compute neural network outputs
        # This ensures they're available for mechanistic components that need them
        for name, nn in self.nn_replacements.items():
            inputs[name] = nn(inputs)

        # Process each component to calculate intermediate values
        intermediate_values = {}
        for name, component_fn in self.mechanistic_components.items():
            # Skip if this component is replaced by a neural network
            if name in self.nn_replacements:
                continue

            # Calculate component output using inputs that now include neural network outputs
            intermediate_values[name] = component_fn(inputs)

        # Add intermediate values to inputs
        inputs.update(intermediate_values)

        # Calculate state derivatives
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
            max_steps=100000,  # Increased from 50000
            dt0=0.01  # Made explicit to allow customization
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

        Returns:
            Dictionary containing solution arrays
        """
        # Convert initial state dictionary to array
        y0 = jnp.array([initial_state[name] for name in self.state_names])

        # Define ODE term
        term = diffrax.ODETerm(lambda t, y, args: self.ode_function(t, y, args))

        # Set up saveat
        saveat = diffrax.SaveAt(ts=evaluation_times)

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
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol)
        )

        # Extract solution and return as dictionary
        solution = {
            'times': sol.ts,
        }

        # Add each state's solution
        for i, name in enumerate(self.state_names):
            solution[name] = sol.ys[:, i]

        return solution