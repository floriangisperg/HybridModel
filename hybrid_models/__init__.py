# In hybrid_models/__init__.py

import jax.numpy as jnp
from typing import Dict, List, Callable, Optional, Tuple

# Existing imports
from .nn_module import ConfigurableNN
from .ode_system import HybridODESystem  # ,get_value_at_time
from .training import train_hybrid_model
from .evaluation import calculate_metrics, evaluate_hybrid_model

# **** ADD IMPORTS FROM UTILS ****
from .utils import (
    normalize_data,
    combine_normalization_params,
    calculate_rate,
    create_initial_random_key,
    interp_linear,  # <-- Add this import
    # get_interpolated_value_at_time # <-- Add this too if you created the wrapper
)

# **** END ADDED IMPORTS ****

# Existing imports
from .visualization import (
    plot_training_history,
    plot_state_predictions,
    plot_all_results,
)
from .evaluation_utils import (
    evaluate_model_performance,
    create_metrics_summary,
    compare_models,
)
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
from .data_utils import VariableRegistry
from .solver import SolverConfig, solve_for_dataset
from .experiment import ExperimentManager
from .model_utils import (
    ModelConfig,
    NeuralNetworkConfig,
    save_model_description,
    save_normalization_params,
    create_model_from_config,
)
from .persistence import (
    save_model,
    load_model,
    save_training_results,
    load_training_results,
)
from .builder import HybridModelBuilder
