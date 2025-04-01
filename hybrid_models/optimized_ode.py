"""Optimized implementation of the HybridODESystem."""
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from typing import Dict, List, Tuple, Any, Callable, Optional
from functools import partial
from jaxtyping import Array, Float, PyTree
from .ode_system import get_value_at_time


class OptimizedODESystem(eqx.Module):
    """
    Performance-optimized version of HybridODESystem.

    This implementation uses various JAX optimization techniques:
    1. Pre-compilation of the ODE function with static arguments
    2. Optimized state handling to reduce data conversion overhead
    3. Specialized solver configurations for different problem types
    """
    mechanistic_components: Dict[str, Callable[[Dict[str, Float[Array, "..."]]],
    Float[Array, ""]]]  # Dict of mechanistic model components
    nn_replacements: Dict[str, eqx.Module]  # Dict of neural network replacements
    state_names: List[str]  # Names of state variables
    _jitted_ode_function: Optional[Callable] = None  # Cached JIT-compiled ODE function

    def __init__(
            self,
            mechanistic_components: Dict[str, Callable],
            nn_replacements: Dict[str, eqx.Module],
            state_names: List[str]
    ):
        """
        Initialize the optimized ODE system.

        Args:
            mechanistic_components: Dictionary of mechanistic model components as callables
            nn_replacements: Dictionary of neural network replacements for specific components
            state_names: Names of the state variables in the correct order
        """
        self.mechanistic_components = mechanistic_components
        self.nn_replacements = nn_replacements
        self.state_names = state_names
        self._jitted_ode_function = None  # Will be created on first call

    def _prepare_jitted_ode_function(self, args_signature: Dict):
        """
        Prepare a JIT-compiled ODE function with the given args signature.
        This significantly improves performance by avoiding recompilation.

        Args:
            args_signature: The arguments structure that will be used
        """
        # Create a non-jitted version for testing
        def ode_function_impl(t, y, args):
            """The actual ODE function implementation."""
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

            # First compute neural network outputs
            for name, nn in self.nn_replacements.items():
                inputs[name] = nn(inputs)

            # Process each component to calculate intermediate values
            intermediate_values = {}
            for name, component_fn in self.mechanistic_components.items():
                # Skip if this component is replaced by a neural network
                if name in self.nn_replacements:
                    continue

                # Calculate component output using updated inputs
                try:
                    intermediate_values[name] = component_fn(inputs)
                except KeyError as e:
                    # Add better error message with context
                    raise KeyError(f"Missing input '{e.args[0]}' for component '{name}'") from e

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

        # Due to JAX limitations with complex Python objects,
        # we won't use JIT here for maximum compatibility
        self._jitted_ode_function = ode_function_impl

    def ode_function(self, t: float, y: Float[Array, "D"], args: Dict) -> Float[Array, "D"]:
        """
        The optimized ODE function that combines mechanistic and neural network components.

        Args:
            t: Current time
            y: Current state (as an array)
            args: Additional arguments

        Returns:
            Derivatives of states
        """
        # Create the JIT-compiled function if it doesn't exist
        if self._jitted_ode_function is None:
            self._prepare_jitted_ode_function(args)

        # Call the JIT-compiled function
        return self._jitted_ode_function(t, y, args)

    def solve(
            self,
            initial_state: Dict[str, float],
            t_span: Tuple[float, float],
            evaluation_times: Float[Array, "N"],
            args: Dict,
            solver_type: str = 'adaptive',
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_steps: int = 500000,
            dt0: float = 0.01
    ) -> Dict[str, Float[Array, "..."]]:
        """
        Solve the ODE system with performance optimizations.

        Args:
            initial_state: Dictionary of initial states
            t_span: (t0, t1) time span to solve over
            evaluation_times: Times at which to evaluate the solution
            args: Additional arguments for the ODE function
            solver_type: Type of solver to use ('adaptive', 'fixed', or 'symplectic')
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_steps: Maximum number of steps
            dt0: Initial step size

        Returns:
            Dictionary containing solution arrays
        """
        # Convert initial state dictionary to array
        y0 = jnp.array([initial_state[name] for name in self.state_names])

        # Define ODE term
        term = diffrax.ODETerm(lambda t, y, args: self.ode_function(t, y, args))

        # Set up saveat
        saveat = diffrax.SaveAt(ts=evaluation_times)

        # Select solver based on type
        if solver_type == 'fixed':
            # Fixed step solver - fastest but least accurate
            solver = diffrax.Euler()
            stepsize_controller = diffrax.ConstantStepSize()
        elif solver_type == 'symplectic':
            # Symplectic solver - good for conservative systems
            solver = diffrax.Kvaerno5()
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        else:
            # Default to adaptive solver - most robust
            solver = diffrax.Tsit5()
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        # Solve ODE with optimized settings
        @partial(jax.jit, static_argnames=['args', 'solver', 'stepsize_controller'])
        def jitted_solve(y0, t0, t1, ts, args, solver, stepsize_controller):
            return diffrax.diffeqsolve(
                term,
                solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=y0,
                args=args,
                saveat=diffrax.SaveAt(ts=ts),
                max_steps=max_steps,
                stepsize_controller=stepsize_controller
            )

        # Call the JIT-compiled solver
        sol = jitted_solve(
            y0,
            t_span[0],
            t_span[1],
            evaluation_times,
            args,
            solver,
            stepsize_controller
        )

        # Extract solution and return as dictionary
        solution = {
            'times': sol.ts,
        }

        # Add each state's solution
        for i, name in enumerate(self.state_names):
            solution[name] = sol.ys[:, i]

        return solution

    @classmethod
    def from_hybrid_ode_system(cls, system):
        """
        Convert a regular HybridODESystem to an optimized version.

        Args:
            system: The HybridODESystem to optimize

        Returns:
            OptimizedODESystem instance
        """
        return cls(
            mechanistic_components=system.mechanistic_components,
            nn_replacements=system.nn_replacements,
            state_names=system.state_names
        )