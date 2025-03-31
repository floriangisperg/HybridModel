from typing import Dict, List, Callable, Optional
from .nn_module import ConfigurableNN
from .ode_system import HybridODESystem, get_value_at_time
from .training import train_hybrid_model
from .evaluation import calculate_metrics, evaluate_hybrid_model
from .utils import (normalize_data, combine_normalization_params,
                    calculate_rate, create_initial_random_key)


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
        return HybridODESystem(
            mechanistic_components=self.mechanistic_components,
            nn_replacements=self.nn_replacements,
            state_names=self.state_names
        )