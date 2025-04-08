import jax.numpy as jnp

from typing import Dict, List, Callable, Optional, Tuple
from .nn_module import ConfigurableNN
from .ode_system import HybridODESystem, get_value_at_time
# Import updated training module with validation support
from .training import train_hybrid_model
from .evaluation import calculate_metrics, evaluate_hybrid_model
from .utils import (normalize_data, combine_normalization_params,
                    calculate_rate, create_initial_random_key)
# Visualization
from .visualization import (plot_training_history, plot_state_predictions,
                           plot_all_results)
# Evaluation
from .evaluation_utils import (evaluate_model_performance, aggregate_evaluation_results,
                              create_metrics_summary, compare_models)
# Loss function
from .loss import (LossMetric, MSE, RelativeMSE, MAE, WeightedMSE, NRMSE,
                  create_hybrid_model_loss, mse_loss, mae_loss)
# Data utilities
from .data_utils import VariableRegistry
# Solver utilities with advanced configuration
from .solver import SolverConfig, solve_for_dataset
# Experiment management
from .experiment import ExperimentManager
# Model configuration and documentation
from .model_utils import ModelConfig, NeuralNetworkConfig, save_model_description, save_normalization_params
# Model persistence
from .persistence import save_model, load_model, save_training_results, load_training_results

class HybridModelBuilder:
    """
    Builder for creating hybrid models that combine mechanistic components with neural networks.
    """

    def __init__(self):
        self.state_names = []
        self.mechanistic_components = {}
        self.nn_replacements = {}
        self.norm_params = {}

    def add_state(self, name: str):
        """
        Add a state variable to the model.

        Args:
            name: Name of the state variable
        """
        if name not in self.state_names:
            self.state_names.append(name)

    def add_mechanistic_component(self, name: str, component_fn: callable):
        """
        Add a mechanistic component to the model.

        Args:
            name: Name of the component
            component_fn: The mechanistic function for this component
        """
        self.mechanistic_components[name] = component_fn

    def add_trainable_parameter(self, name: str, initial_value: float,
                                bounds: Optional[Tuple[float, float]] = None,
                                transform: str = "none"):
        """
        Add a directly trainable parameter to the model.

        Args:
            name: Name of the parameter
            initial_value: Initial value for the parameter
            bounds: Optional tuple of (min_value, max_value) to constrain the parameter
            transform: Transformation to apply ("none", "sigmoid", "softplus", "exp")
        """
        # Initialize trainable parameters dict if not already present
        if not hasattr(self, 'trainable_parameters'):
            self.trainable_parameters = {}
            self.parameter_transforms = {}

        # Store as JAX Array to make it trainable
        param_value = jnp.array(initial_value, dtype=float)

        # Store parameter info
        self.trainable_parameters[name] = param_value
        self.parameter_transforms[name] = {
            'transform': transform,
            'bounds': bounds
        }

        return self  # Allow method chaining

    def replace_with_nn(self, name: str, input_features: List[str],
                        hidden_dims: List[int] = [16, 16],
                        output_activation: callable = None,
                        key=None):
        """
        Replace a component with a neural network.

        Args:
            name: Name of the component to replace
            input_features: List of input features for the neural network
            hidden_dims: Hidden layer dimensions
            output_activation: Activation function for the output
            key: Random key for initialization
        """
        # Create neural network for this component
        nn = ConfigurableNN(
            norm_params=self.norm_params,
            input_features=input_features,
            hidden_dims=hidden_dims,
            output_activation=output_activation,
            key=key
        )

        self.nn_replacements[name] = nn

    def set_normalization_params(self, norm_params: Dict):
        """
        Set normalization parameters for the model.

        Args:
            norm_params: Dictionary of normalization parameters
        """
        self.norm_params = norm_params

    def build(self) -> HybridODESystem:
        """
        Build and return the hybrid ODE system.

        Returns:
            HybridODESystem instance
        """
        # Initialize trainable parameters if not defined
        if not hasattr(self, 'trainable_parameters'):
            self.trainable_parameters = {}
            self.parameter_transforms = {}

        return HybridODESystem(
            mechanistic_components=self.mechanistic_components,
            nn_replacements=self.nn_replacements,
            trainable_parameters=self.trainable_parameters,
            parameter_transforms=self.parameter_transforms,
            state_names=self.state_names
        )