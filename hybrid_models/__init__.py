import jax.numpy as jnp

from typing import Dict, List, Callable, Optional, Tuple
from .nn_module import ConfigurableNN
from .ode_system import HybridODESystem, get_value_at_time

# Import updated training module with validation support
from .training import train_hybrid_model
from .evaluation import calculate_metrics, evaluate_hybrid_model
from .utils import (
    normalize_data,
    combine_normalization_params,
    calculate_rate,
    create_initial_random_key,
)

# Visualization
from .visualization import (
    plot_training_history,
    plot_state_predictions,
    plot_all_results,
)

# Evaluation
from .evaluation_utils import (
    evaluate_model_performance,
    aggregate_evaluation_results,
    create_metrics_summary,
    compare_models,
)

# Loss function
from .loss import (
    LossMetric,
    MSE,
    RelativeMSE,
    MAE,
    WeightedMSE,
    NRMSE,
    create_hybrid_model_loss,
    mse_loss,
    mae_loss,
)

from .data import DatasetManager, VariableType, TimeSeriesData

# Data utilities
from .data_utils import VariableRegistry

# Solver utilities with advanced configuration
from .solver import SolverConfig, solve_for_dataset

# Experiment management
from .experiment import ExperimentManager

# Model configuration and documentation
from .model_utils import (
    ModelConfig,
    NeuralNetworkConfig,
    save_model_description,
    save_normalization_params,
    create_model_from_config,
)

# Model persistence
from .persistence import (
    save_model,
    load_model,
    save_training_results,
    load_training_results,
)

from .builder import HybridModelBuilder
