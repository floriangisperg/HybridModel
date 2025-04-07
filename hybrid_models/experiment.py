"""
Experiment management for hybrid model training and evaluation.

This module provides classes and functions to manage the end-to-end
workflow for hybrid model experiments, from data loading to evaluation.
"""
import os
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
import jax
import pandas as pd

from .loss import create_hybrid_model_loss, MSE, LossMetric
from .training import train_hybrid_model
from .evaluation_utils import evaluate_model_performance
from .visualization import plot_all_results
from .solver import solve_for_dataset, SolverConfig


class ExperimentManager:
    """
    Manager for hybrid model experiments.

    This class orchestrates the end-to-end workflow for training and evaluating
    hybrid models, including data management, training, evaluation, and visualization.
    """

    def __init__(
            self,
            model: Any,
            train_datasets: List[Dict],
            test_datasets: Optional[List[Dict]] = None,
            output_dir: str = "results",
            experiment_name: Optional[str] = None
    ):
        """
        Initialize the experiment manager.

        Args:
            model: The hybrid model to train
            train_datasets: List of training datasets
            test_datasets: Optional list of test datasets
            output_dir: Directory to save results
            experiment_name: Optional name for the experiment (used for file naming)
        """
        self.model = model
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

    def train(
            self,
            state_names: List[str],
            num_epochs: int = 1000,
            learning_rate: float = 1e-3,
            early_stopping_patience: Optional[int] = None,
            component_weights: Optional[Dict[str, float]] = None,
            loss_metric: LossMetric = MSE,
            validation_datasets: Optional[List[Dict]] = None,
            solver_config: Optional[SolverConfig] = None,
            verbose: bool = True
    ) -> Any:
        """
        Train the model.

        Args:
            state_names: Names of state variables for loss calculation
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Optional patience for early stopping
            component_weights: Optional weights for state variables in loss
            loss_metric: Loss metric to use
            validation_datasets: Optional datasets for validation
            solver_config: Optional solver configuration
            verbose: Whether to print progress

        Returns:
            Trained model
        """
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

        return self.trained_model

    def evaluate(
            self,
            state_names: List[str],
            solver_config: Optional[SolverConfig] = None,
            verbose: bool = True
    ) -> Dict:
        """
        Evaluate the trained model.

        Args:
            state_names: State variable names to evaluate
            solver_config: Optional solver configuration
            verbose: Whether to print results

        Returns:
            Dictionary of evaluation metrics
        """
        if self.trained_model is None:
            raise ValueError("Model must be trained before evaluation. Call train() first.")

        # Use evaluation solver config by default
        solver_config = solver_config or SolverConfig.for_evaluation()

        # Create a custom solve function with this solver config
        def custom_solve_fn(model, dataset):
            return solve_for_dataset(model, dataset, solver_config)

        # Evaluate on training data
        if verbose:
            print("Evaluating on training data...")

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
                print("Evaluating on test data...")

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
            state_names: List[str],
            state_labels: Optional[Dict[str, str]] = None,
            component_names: Optional[List[str]] = None,
            solver_config: Optional[SolverConfig] = None
    ):
        """
        Visualize training results and model predictions.

        Args:
            state_names: State variable names to visualize
            state_labels: Optional display labels for state variables
            component_names: Optional names for loss components
            solver_config: Optional solver configuration
        """
        if self.trained_model is None:
            raise ValueError("Model must be trained before visualization. Call train() first.")

        # Use evaluation solver config by default
        solver_config = solver_config or SolverConfig.for_evaluation()

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