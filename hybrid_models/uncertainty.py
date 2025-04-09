"""
Uncertainty quantification for hybrid models.

This module provides optional extensions for quantifying uncertainty
in hybrid model predictions without modifying the core framework.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import matplotlib.pyplot as plt
import os
import time
import copy

from hybrid_models.training import train_hybrid_model
from hybrid_models.nn_module import ConfigurableNN
from hybrid_models.utils import create_initial_random_key
from jaxtyping import Array, Float, PyTree


class EnsembleModel:
    """
    Ensemble of hybrid models for uncertainty quantification.

    This class manages multiple instances of a hybrid model to provide
    uncertainty estimates in predictions without modifying the core model.
    """

    def __init__(self, models: List[Any], state_names: Optional[List[str]] = None):
        """
        Initialize with a list of pre-built models.

        Args:
            models: List of hybrid models, typically built with different random seeds
            state_names: Optional list of state names for reference
        """
        self.models = models
        self.n_models = len(models)
        self.state_names = state_names
        self.trained = False

    @classmethod
    def from_builder(cls, builder, n_models: int = 10, state_names: Optional[List[str]] = None, seed: int = 42):
        """
        Create an ensemble from a model builder.

        Args:
            builder: HybridModelBuilder instance
            n_models: Number of models in the ensemble
            state_names: Optional list of state names
            seed: Random seed for initialization

        Returns:
            An EnsembleModel instance with multiple models
        """
        # Create master random key
        master_key = create_initial_random_key(seed)
        model_keys = jax.random.split(master_key, n_models)

        models = []

        for i, key in enumerate(model_keys):
            # Deep copy the builder to avoid modifying the original
            builder_copy = copy.deepcopy(builder)

            # Generate keys for neural networks
            nn_keys = jax.random.split(key, 10)  # Generate enough keys for multiple NNs
            key_idx = 0

            # For each neural network replacement in the builder, use a different key
            for name, nn_config in getattr(builder_copy, 'nn_replacements', {}).items():
                if key_idx < len(nn_keys):
                    # Create new NN with different initialization
                    builder_copy.replace_with_nn(
                        name=name,
                        input_features=nn_config.input_features,
                        hidden_dims=[layer.out_features for layer in nn_config.layers
                                     if hasattr(layer, 'out_features')][:-1],
                        output_activation=nn_config.layers[-1] if not hasattr(nn_config.layers[-1],
                                                                              'out_features') else None,
                        key=nn_keys[key_idx]
                    )
                    key_idx += 1

            # Build the model with potentially different initializations
            model = builder_copy.build()
            models.append(model)

        if state_names is None and hasattr(builder, 'state_names'):
            state_names = builder.state_names

        return cls(models, state_names)

    def train(self,
              datasets: List[Dict],
              loss_fn: Optional[Callable] = None,
              num_epochs: int = 1000,
              learning_rate: float = 1e-3,
              early_stopping_patience: Optional[int] = None,
              verbose: bool = True,
              **train_kwargs) -> Tuple['EnsembleModel', List[Dict]]:
        """
        Train all models in the ensemble.

        Args:
            datasets: List of datasets for training
            loss_fn: Loss function (if None, uses mse_loss from framework)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Optional patience for early stopping
            verbose: Whether to print progress
            **train_kwargs: Additional kwargs for train_hybrid_model

        Returns:
            Tuple of (self, list of training histories)
        """
        if loss_fn is None:
            from hybrid_models.loss import mse_loss
            loss_fn = mse_loss

        trained_models = []
        history_list = []

        for i, model in enumerate(self.models):
            if verbose:
                print(f"Training model {i + 1}/{self.n_models}")
                start_time = time.time()

            trained_model, history = train_hybrid_model(
                model=model,
                datasets=datasets,
                loss_fn=loss_fn,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                verbose=verbose,
                **train_kwargs
            )

            if verbose:
                elapsed_time = time.time() - start_time
                print(f"Model {i + 1} training completed in {elapsed_time:.2f} seconds.")

            trained_models.append(trained_model)
            history_list.append(history)

        # Update models with trained versions
        self.models = trained_models
        self.trained = True

        # Return self for method chaining
        return self, history_list

    def predict(self,
                dataset: Dict,
                solver_kwargs: Optional[Dict] = None) -> Dict:
        """
        Make predictions with uncertainty estimates for a single dataset.

        Args:
            dataset: Dataset containing initial conditions and times
            solver_kwargs: Optional solver configuration

        Returns:
            Dictionary with mean predictions, confidence intervals, etc.
        """
        if not self.trained:
            print("Warning: Models may not be trained. Running prediction with possibly untrained models.")

        if solver_kwargs is None:
            solver_kwargs = {}

        # Collect predictions from all models
        predictions = []

        for model in self.models:
            solution = model.solve(
                initial_state=dataset['initial_state'],
                t_span=(dataset['times'][0], dataset['times'][-1]),
                evaluation_times=dataset['times'],
                args={
                    'time_dependent_inputs': dataset.get('time_dependent_inputs', {}),
                    'static_inputs': dataset.get('static_inputs', {})
                },
                **solver_kwargs
            )
            predictions.append(solution)

        # Compute statistics across ensemble
        result = {'times': dataset['times']}
        result['mean'] = {}
        result['std'] = {}
        result['lower'] = {}  # 2.5 percentile
        result['upper'] = {}  # 97.5 percentile

        # Get all keys except 'times'
        all_keys = set()
        for p in predictions:
            all_keys.update(p.keys())
        all_keys.discard('times')

        # Get state names if not provided
        if self.state_names is None:
            self.state_names = list(all_keys)

        for key in all_keys:
            if key != 'times':  # Skip time points
                # Stack predictions across models
                try:
                    stacked = jnp.stack([p[key] for p in predictions])

                    # Calculate statistics
                    result['mean'][key] = jnp.mean(stacked, axis=0)
                    result['std'][key] = jnp.std(stacked, axis=0)

                    # Calculate percentiles (approximate for JAX compatibility)
                    sorted_vals = jnp.sort(stacked, axis=0)
                    lower_idx = max(0, int(0.025 * self.n_models))
                    upper_idx = min(self.n_models - 1, int(0.975 * self.n_models))

                    result['lower'][key] = sorted_vals[lower_idx]
                    result['upper'][key] = sorted_vals[upper_idx]
                except:
                    print(f"Warning: Could not calculate statistics for {key}")

        return result

    def predict_all(self,
                    datasets: List[Dict],
                    solver_kwargs: Optional[Dict] = None) -> List[Dict]:
        """
        Make predictions with uncertainty for multiple datasets.

        Args:
            datasets: List of datasets
            solver_kwargs: Optional solver configuration

        Returns:
            List of prediction dictionaries with uncertainty estimates
        """
        return [self.predict(dataset, solver_kwargs) for dataset in datasets]

    def evaluate(self,
                 datasets: List[Dict],
                 solver_kwargs: Optional[Dict] = None) -> Dict:
        """
        Evaluate the ensemble model on datasets.

        Args:
            datasets: List of datasets for evaluation
            solver_kwargs: Optional solver configuration

        Returns:
            Dictionary of evaluation metrics with uncertainty
        """
        results = {}

        # Make predictions for all datasets
        predictions = self.predict_all(datasets, solver_kwargs)

        # Evaluate each dataset
        for i, (dataset, prediction) in enumerate(zip(datasets, predictions)):
            dataset_metrics = {}

            for state in self.state_names:
                true_key = f"{state}_true"
                if true_key in dataset and state in prediction['mean']:
                    # Calculate metrics for mean prediction
                    from hybrid_models.evaluation import calculate_metrics
                    metrics = calculate_metrics(dataset[true_key], prediction['mean'][state])
                    dataset_metrics[state] = metrics

            results[f"dataset_{i}"] = dataset_metrics

        # Calculate overall metrics (simplified)
        overall_metrics = {}

        for state in self.state_names:
            state_metrics = [dataset_metrics.get(state, {})
                             for dataset_metrics in results.values()
                             if state in dataset_metrics]

            if state_metrics:
                # Average metrics across datasets
                avg_metrics = {}
                for metric in ['mse', 'rmse', 'r2', 'mae']:
                    values = [m.get(metric, 0) for m in state_metrics if metric in m]
                    if values:
                        avg_metrics[metric] = sum(values) / len(values)

                overall_metrics[state] = avg_metrics

        results['overall'] = overall_metrics

        return results


def plot_prediction_with_uncertainty(prediction: Dict,
                                     dataset: Dict,
                                     state_name: str,
                                     output_dir: str = "results",
                                     filename: Optional[str] = None,
                                     title: Optional[str] = None,
                                     figsize: Tuple[int, int] = (10, 6),
                                     show_error_bars: bool = False):
    """
    Plot predictions with uncertainty for a single state variable.

    Args:
        prediction: Prediction dictionary from EnsembleModel.predict
        dataset: Dataset containing true values
        state_name: Name of state variable to plot
        output_dir: Directory to save the plot
        filename: Optional custom filename
        title: Optional custom title
        figsize: Figure size as (width, height)
        show_error_bars: Whether to show error bars instead of bands
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    plt.figure(figsize=figsize)

    # Get time points
    times = prediction['times']

    # Plot true values if available
    true_key = f"{state_name}_true"
    if true_key in dataset:
        plt.plot(times, dataset[true_key], 'ko-',
                 label='Measured', alpha=0.7, markersize=4)

    # Get predictions and uncertainties
    mean = prediction['mean'][state_name]
    std = prediction['std'][state_name]
    lower = prediction['lower'][state_name]
    upper = prediction['upper'][state_name]

    # Plot mean prediction
    plt.plot(times, mean, 'r-', label='Prediction (mean)', linewidth=2)

    if show_error_bars:
        # Plot error bars at intervals
        n_points = len(times)
        interval = max(1, n_points // 10)  # Show approximately 10 error bars
        plt.errorbar(times[::interval], mean[::interval],
                     yerr=std[::interval], fmt='ro', alpha=0.6,
                     label='68% Confidence')
    else:
        # Plot confidence intervals as bands
        plt.fill_between(times, mean - std, mean + std,
                         color='red', alpha=0.3, label='68% CI')
        plt.fill_between(times, lower, upper,
                         color='red', alpha=0.1, label='95% CI')

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel(state_name)

    if title:
        plt.title(title)
    else:
        plt.title(f'Prediction with Uncertainty: {state_name}')

    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Save figure
    if filename is None:
        filename = f"uncertainty_{state_name}.png"

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_predictions_with_uncertainty(prediction: Dict,
                                      dataset: Dict,
                                      state_names: Optional[List[str]] = None,
                                      output_dir: str = "results",
                                      prefix: str = "",
                                      figsize: Tuple[int, int] = (10, 6)):
    """
    Plot predictions with uncertainty for multiple state variables.

    Args:
        prediction: Prediction dictionary from EnsembleModel.predict
        dataset: Dataset containing true values
        state_names: List of state variables to plot (if None, uses all)
        output_dir: Directory to save plots
        prefix: Prefix for filenames
        figsize: Figure size as (width, height)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # If state_names not provided, use all states in prediction
    if state_names is None:
        state_names = list(prediction['mean'].keys())

    # Plot each state individually
    for state_name in state_names:
        if state_name in prediction['mean']:
            plot_prediction_with_uncertainty(
                prediction=prediction,
                dataset=dataset,
                state_name=state_name,
                output_dir=output_dir,
                filename=f"{prefix}uncertainty_{state_name}.png",
                figsize=figsize
            )

    # Create a combined plot if there are multiple states
    if len(state_names) > 1:
        fig, axes = plt.subplots(len(state_names), 1,
                                 figsize=(figsize[0], figsize[1] * len(state_names) / 2))

        for i, state_name in enumerate(state_names):
            if state_name not in prediction['mean']:
                continue

            ax = axes[i] if len(state_names) > 1 else axes

            # Get time points
            times = prediction['times']

            # Plot true values if available
            true_key = f"{state_name}_true"
            if true_key in dataset:
                ax.plot(times, dataset[true_key], 'ko-',
                        label='Measured', alpha=0.7, markersize=4)

            # Get predictions and uncertainties
            mean = prediction['mean'][state_name]
            std = prediction['std'][state_name]

            # Plot mean and confidence interval
            ax.plot(times, mean, 'r-', label='Prediction', linewidth=2)
            ax.fill_between(times, mean - std, mean + std,
                            color='red', alpha=0.2, label='68% CI')

            # Labels
            ax.set_ylabel(state_name)
            ax.set_title(state_name)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.legend(loc='best')

        # Add common x-label
        fig.text(0.5, 0.04, 'Time', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}combined_uncertainty.png"))
        plt.close()


def compare_uncertainties(ensemble_predictions: Dict,
                          true_values: Dict,
                          state_name: str,
                          output_dir: str = "results",
                          filename: Optional[str] = None,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6)):
    """
    Create a plot comparing predictions with true values and highlighting uncertainty.

    Args:
        ensemble_predictions: Dictionary of ensemble predictions with uncertainties
        true_values: Dictionary with true values
        state_name: Name of the state variable to plot
        output_dir: Directory to save the plot
        filename: Optional custom filename
        title: Optional custom title
        figsize: Figure size as (width, height)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    plt.figure(figsize=figsize)

    # Get time points
    times = ensemble_predictions['times']

    # Get predictions and uncertainties
    mean = ensemble_predictions['mean'][state_name]
    std = ensemble_predictions['std'][state_name]

    # Plot true values
    true_key = f"{state_name}_true"
    if true_key in true_values:
        plt.plot(times, true_values[true_key], 'ko-',
                 label='Measured', alpha=0.7, markersize=4)

    # Plot mean prediction
    plt.plot(times, mean, 'r-', label='Ensemble Mean', linewidth=2)

    # Plot confidence intervals
    plt.fill_between(times, mean - std, mean + std,
                     color='red', alpha=0.3, label='68% CI')
    plt.fill_between(times, mean - 2 * std, mean + 2 * std,
                     color='red', alpha=0.1, label='95% CI')

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel(state_name)

    if title:
        plt.title(title)
    else:
        plt.title(f'Prediction vs. Measured with Uncertainty: {state_name}')

    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Save figure
    if filename is None:
        filename = f"uncertainty_comparison_{state_name}.png"

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()