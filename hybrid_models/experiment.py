"""
Experiment management for hybrid model training and evaluation.

This module provides classes and functions to manage the end-to-end
workflow for hybrid model experiments, from data loading to evaluation.
"""
import os
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import jax
import pandas as pd

from .loss import create_hybrid_model_loss, MSE, LossMetric
from .training import train_hybrid_model
from .evaluation_utils import evaluate_model_performance
from .visualization import plot_all_results
from .solver import solve_for_dataset, SolverConfig
from .model_utils import ModelConfig, save_model_description, save_normalization_params
from .persistence import save_model, load_model, save_training_results


class ExperimentManager:
    """
    Manager for hybrid model experiments.

    This class orchestrates the end-to-end workflow for training and evaluating
    hybrid models, including data management, training, evaluation, and visualization.
    """

    def __init__(
        self,
        model: Any,
        model_config: ModelConfig,
        norm_params: Dict[str, float],
        train_datasets: List[Dict],
        test_datasets: Optional[List[Dict]] = None,
        output_dir: str = "results",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the experiment manager.

        Args:
            model: The hybrid model to train
            model_config: Configuration of the model
            norm_params: Normalization parameters
            train_datasets: List of training datasets
            test_datasets: Optional list of test datasets
            output_dir: Directory to save results
            experiment_name: Optional name for the experiment (used for file naming)
        """
        self.model = model
        self.model_config = model_config
        self.norm_params = norm_params
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

        # Set up output directory
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.output_dir = os.path.join(output_dir, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Placeholder for trained model and results
        self.trained_model = None
        self.training_history = None
        self.validation_history = None
        self.training_metrics = None
        self.test_metrics = None
        self.solver_config = None

    def create_loss_function(
        self,
        state_names: List[str],
        loss_metric: LossMetric = MSE,
        component_weights: Optional[Dict[str, float]] = None,
        solver_config: Optional[SolverConfig] = None
    ) -> Callable:
        """
        Create a loss function for model training.

        Args:
            state_names: Names of state variables to include in loss
            loss_metric: Loss metric to use
            component_weights: Optional weights for each state variable
            solver_config: Optional solver configuration

        Returns:
            Loss function that takes (model, datasets) and returns (loss, aux)
        """
        # Use training solver config by default
        solver_config = solver_config or SolverConfig.for_training()
        self.solver_config = solver_config

        # Create a custom solve function with this solver config
        def custom_solve_fn(model, dataset):
            return solve_for_dataset(model, dataset, solver_config)

        # Create the loss function
        loss_fn = create_hybrid_model_loss(
            solve_fn=custom_solve_fn,
            state_names=state_names,
            loss_metric=loss_metric,
            component_weights=component_weights
        )

        return loss_fn

    def generate_model_documentation(self, filepath: Optional[str] = None) -> str:
        """
        Generate detailed documentation of the model structure.

        Args:
            filepath: Optional path to save the documentation

        Returns:
            Path to the saved documentation file or the documentation string
        """
        # Generate model description
        if filepath is None:
            filepath = os.path.join(self.output_dir, "model_documentation.txt")

        # Create and save the description
        return save_model_description(
            model=self.model,
            model_config=self.model_config,
            norm_params=self.norm_params,
            filepath=filepath
        )

    def save_normalization_parameters(self, filepath: Optional[str] = None) -> str:
        """
        Save normalization parameters to a text file.

        Args:
            filepath: Optional path to save the parameters

        Returns:
            Path to the saved file
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, "normalization_params.txt")

        return save_normalization_params(self.norm_params, filepath)

    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model.

        Args:
            filepath: Optional path to save the model

        Returns:
            Path to the saved model file
        """
        if self.trained_model is None:
            raise ValueError("Model must be trained before saving. Call train() first.")

        if filepath is None:
            filepath = os.path.join(self.output_dir, "trained_model.eqx")

        # Create metadata
        metadata = {
            "experiment_name": self.experiment_name,
            "training_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_config": self.model_config.to_dict(),
            "norm_params": self.norm_params
        }

        # Add solver config if available
        if self.solver_config:
            metadata["solver_config"] = {
                "solver_type": self.solver_config.solver_type,
                "step_size_controller": self.solver_config.step_size_controller,
                "rtol": self.solver_config.rtol,
                "atol": self.solver_config.atol,
                "max_steps": self.solver_config.max_steps
            }

        return save_model(self.trained_model, filepath, metadata)

    def save_all_results(self) -> Dict[str, str]:
        """
        Save all experiment results including model, configurations, and metrics.

        Returns:
            Dictionary with paths to all saved files
        """
        if self.trained_model is None:
            raise ValueError("Model must be trained before saving results. Call train() first.")

        # Convert model config and solver config to dictionaries
        model_config_dict = self.model_config.to_dict()

        solver_config_dict = None
        if self.solver_config:
            solver_config_dict = {
                "solver_type": self.solver_config.solver_type,
                "step_size_controller": self.solver_config.step_size_controller,
                "rtol": self.solver_config.rtol,
                "atol": self.solver_config.atol,
                "max_steps": self.solver_config.max_steps
            }

        # Get evaluation metrics
        metrics_dict = None
        if self.training_metrics or self.test_metrics:
            metrics_dict = {
                "training": self.training_metrics,
                "test": self.test_metrics
            }

        # Save everything
        paths = save_training_results(
            output_dir=self.output_dir,
            model=self.trained_model,
            training_history=self.training_history,
            model_config=model_config_dict,
            norm_params=self.norm_params,
            solver_config=solver_config_dict,
            metrics=metrics_dict
        )

        # Generate additional documentation
        paths["model_documentation"] = self.generate_model_documentation()
        paths["norm_params_txt"] = self.save_normalization_parameters()

        return paths

    def train(
        self,
        state_names: Optional[List[str]] = None,
        num_epochs: int = 1000,
        learning_rate: float = 1e-3,
        early_stopping_patience: Optional[int] = None,
        component_weights: Optional[Dict[str, float]] = None,
        loss_metric: LossMetric = MSE,
        validation_datasets: Optional[List[Dict]] = None,
        solver_config: Optional[SolverConfig] = None,
        save_checkpoints: bool = False,
        checkpoint_interval: int = 100,
        verbose: bool = True
    ) -> Any:
        """
        Train the model with enhanced options.

        Args:
            state_names: Names of state variables for loss calculation (defaults to model state_names)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Optional patience for early stopping
            component_weights: Optional weights for state variables in loss
            loss_metric: Loss metric to use
            validation_datasets: Optional datasets for validation
            solver_config: Optional solver configuration
            save_checkpoints: Whether to save intermediate checkpoints
            checkpoint_interval: Interval (in epochs) for checkpoint saving
            verbose: Whether to print progress

        Returns:
            Trained model
        """
        # Use model state names if none provided
        if state_names is None:
            state_names = self.model_config.state_names

        # Create loss function
        loss_fn = self.create_loss_function(
            state_names=state_names,
            loss_metric=loss_metric,
            component_weights=component_weights,
            solver_config=solver_config
        )

        # Train the model
        if verbose:
            print(f"Training model with {num_epochs} epochs...")
            start_time = time.time()

        if validation_datasets:
            self.trained_model, self.training_history, self.validation_history = train_hybrid_model(
                model=self.model,
                datasets=self.train_datasets,
                loss_fn=loss_fn,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                validation_datasets=validation_datasets,
                verbose=verbose
            )
        else:
            self.trained_model, self.training_history = train_hybrid_model(
                model=self.model,
                datasets=self.train_datasets,
                loss_fn=loss_fn,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                verbose=verbose
            )

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Training completed in {elapsed_time:.1f} seconds.")

        # Save documentation immediately after training
        self.generate_model_documentation()
        self.save_normalization_parameters()

        # Save final model if requested
        if save_checkpoints:
            self.save_model()

        return self.trained_model

    def evaluate(
            self,
            state_names: Optional[List[str]] = None,
            solver_config: Optional[SolverConfig] = None,
            verbose: bool = True
    ) -> Dict:
        """
        Evaluate the trained model.

        Args:
            state_names: State variable names to evaluate (defaults to model state_names)
            solver_config: Optional solver configuration (defaults to the same config used in training)
            verbose: Whether to print results

        Returns:
            Dictionary of evaluation metrics
        """
        if self.trained_model is None:
            raise ValueError("Model must be trained before evaluation. Call train() first.")

        # Use model state names if none provided
        if state_names is None:
            state_names = self.model_config.state_names

        # Use the same solver config as training by default
        if solver_config is None:
            if self.solver_config is not None:
                solver_config = self.solver_config
            else:
                # Fall back to evaluation config if no training config is available
                solver_config = SolverConfig.for_evaluation()
                if verbose:
                    print("Warning: No training solver configuration found. Using default evaluation solver.")

        # Create a custom solve function with this solver config
        def custom_solve_fn(model, dataset):
            return solve_for_dataset(model, dataset, solver_config)

        # Evaluate on training data
        if verbose:
            print(f"Evaluating on training data using solver: {solver_config.solver_type}...")

        self.training_metrics = evaluate_model_performance(
            model=self.trained_model,
            datasets=self.train_datasets,
            solve_fn=custom_solve_fn,
            state_names=state_names,
            dataset_type="Training",
            save_metrics=True,
            output_dir=self.output_dir,
            metrics_filename="training_metrics.txt",
            verbose=verbose
        )

        # Evaluate on test data if available
        if self.test_datasets:
            if verbose:
                print(f"Evaluating on test data using solver: {solver_config.solver_type}...")

            self.test_metrics = evaluate_model_performance(
                model=self.trained_model,
                datasets=self.test_datasets,
                solve_fn=custom_solve_fn,
                state_names=state_names,
                dataset_type="Test",
                save_metrics=True,
                output_dir=self.output_dir,
                metrics_filename="test_metrics.txt",
                verbose=verbose
            )

        return {
            'training': self.training_metrics,
            'test': self.test_metrics
        }

    def visualize(
            self,
            state_names: Optional[List[str]] = None,
            state_labels: Optional[Dict[str, str]] = None,
            component_names: Optional[List[str]] = None,
            solver_config: Optional[SolverConfig] = None
    ):
        """
        Visualize training results and model predictions.

        Args:
            state_names: State variable names to visualize (defaults to model state_names)
            state_labels: Optional display labels for state variables
            component_names: Optional names for loss components
            solver_config: Optional solver configuration (defaults to the same config used in training)
        """
        if self.trained_model is None:
            raise ValueError("Model must be trained before visualization. Call train() first.")

        # Use model state names if none provided
        if state_names is None:
            state_names = self.model_config.state_names

        # Use the same solver config as training by default
        if solver_config is None:
            if self.solver_config is not None:
                solver_config = self.solver_config
            else:
                # Fall back to evaluation config if no training config is available
                solver_config = SolverConfig.for_evaluation()
                print("Warning: No training solver configuration found. Using default evaluation solver.")

        # Create a custom solve function with this solver config
        def custom_solve_fn(model, dataset):
            return solve_for_dataset(model, dataset, solver_config)

        # Create visualizations
        plot_all_results(
            model=self.trained_model,
            train_datasets=self.train_datasets,
            test_datasets=self.test_datasets,
            history=self.training_history,
            solve_fn=custom_solve_fn,
            state_names=state_names,
            output_dir=self.output_dir,
            state_labels=state_labels,
            component_names=component_names,
            validation_history=self.validation_history
        )

    def save_results_summary(self, filename: str = "experiment_summary.txt"):
        """
        Save a summary of the experiment results.

        Args:
            filename: Name of the summary file
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write(f"=== Experiment: {self.experiment_name} ===\n\n")

            # Training parameters
            f.write("Training Parameters:\n")
            if self.training_history:
                f.write(f"  Epochs: {len(self.training_history['loss'])}\n")
                f.write(f"  Final loss: {self.training_history['loss'][-1]:.6f}\n")
                f.write(f"  Best loss: {min(self.training_history['loss']):.6f}\n")

            # Performance metrics
            f.write("\nPerformance Metrics:\n")

            # Add training metrics
            if self.training_metrics and 'aggregate' in self.training_metrics:
                f.write("  Training Data:\n")
                for state, metrics in self.training_metrics['aggregate'].items():
                    f.write(f"    {state}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}\n")

            # Add test metrics
            if self.test_metrics and 'aggregate' in self.test_metrics:
                f.write("  Test Data:\n")
                for state, metrics in self.test_metrics['aggregate'].items():
                    f.write(f"    {state}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}\n")

            # Add timestamp
            import datetime
            f.write(f"\nSummary generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")