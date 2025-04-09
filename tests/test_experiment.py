"""Tests for the experiment management module."""
import pytest
import jax
import jax.numpy as jnp
import os
import shutil
import tempfile
import equinox as eqx
from hybrid_models.experiment import ExperimentManager
from hybrid_models.model_utils import ModelConfig
from hybrid_models.solver import SolverConfig
from hybrid_models.loss import MSE


class SimpleModel(eqx.Module):
    """A simple model for testing the ExperimentManager."""
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self):
        self.weight = jnp.array(1.0)
        self.bias = jnp.array(0.0)

    def solve(self, initial_state, t_span, evaluation_times, args={}, **kwargs):
        """Simple solve function that mimics an ODE solver."""
        times = evaluation_times
        X = self.weight * times + self.bias
        P = self.weight * 0.5 * times + self.bias

        return {
            'times': times,
            'X': X,
            'P': P
        }


@pytest.fixture
def experiment_setup():
    """Set up temporary directory and simple model for testing."""
    # Create a temporary directory for test output
    temp_dir = tempfile.mkdtemp()

    # Create a simple model
    model = SimpleModel()

    # Create a model configuration
    model_config = ModelConfig(
        state_names=['X', 'P'],
        mechanistic_components={'X': lambda x: x, 'P': lambda x: x},
        neural_networks=[]
    )

    # Create normalization parameters
    norm_params = {
        'X_mean': 0.0,
        'X_std': 1.0,
        'P_mean': 0.0,
        'P_std': 1.0
    }

    # Create simple training datasets
    train_datasets = [
        {
            'initial_state': {'X': 0.0, 'P': 0.0},
            'times': jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            'X_true': jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            'P_true': jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
        }
    ]

    # Create simple test datasets
    test_datasets = [
        {
            'initial_state': {'X': 0.0, 'P': 0.0},
            'times': jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            'X_true': jnp.array([0.0, 1.1, 2.2, 3.3, 4.4]),
            'P_true': jnp.array([0.0, 0.55, 1.1, 1.65, 2.2])
        }
    ]

    # Create the experiment manager
    experiment = ExperimentManager(
        model=model,
        model_config=model_config,
        norm_params=norm_params,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        output_dir=temp_dir,
        experiment_name="test_experiment"
    )

    yield experiment

    # Clean up temporary directory after tests
    shutil.rmtree(temp_dir)


def test_experiment_initialization(experiment_setup):
    """Test that the experiment manager initializes correctly."""
    experiment = experiment_setup

    # Check attributes are set correctly
    assert experiment.model is not None
    assert experiment.model_config is not None
    assert experiment.norm_params is not None
    assert experiment.train_datasets is not None
    assert experiment.test_datasets is not None
    assert "test_experiment" in experiment.output_dir
    assert os.path.exists(experiment.output_dir)


def test_create_loss_function(experiment_setup):
    """Test creating a loss function."""
    experiment = experiment_setup

    # Create a loss function
    loss_fn = experiment.create_loss_function(
        state_names=['X', 'P'],
        loss_metric=MSE,
        component_weights={'X': 1.0, 'P': 2.0}
    )

    # The loss function should be callable
    assert callable(loss_fn)

    # Test the loss function with the model and datasets
    loss_value, aux = loss_fn(experiment.model, experiment.train_datasets)

    # Loss value should be a scalar
    assert jnp.isscalar(loss_value)
    # aux should contain component losses
    assert len(aux) == 2  # One for X and one for P


def test_generate_model_documentation(experiment_setup):
    """Test generating model documentation."""
    experiment = experiment_setup

    # Generate documentation
    doc_path = experiment.generate_model_documentation()

    # Check that file was created
    assert os.path.exists(doc_path)

    # Check content (basic validation)
    with open(doc_path, 'r') as f:
        content = f.read()
        assert "HYBRID MODEL DOCUMENTATION" in content
        assert "STATE VARIABLES" in content
        assert "X" in content
        assert "P" in content


def test_save_normalization_parameters(experiment_setup):
    """Test saving normalization parameters."""
    experiment = experiment_setup

    # Save parameters
    params_path = experiment.save_normalization_parameters()

    # Check that file was created
    assert os.path.exists(params_path)

    # Check content (basic validation)
    with open(params_path, 'r') as f:
        content = f.read()
        assert "NORMALIZATION PARAMETERS" in content
        assert "X" in content
        assert "P" in content


def test_train_and_evaluate(experiment_setup):
    """Test training and evaluating the model."""
    experiment = experiment_setup

    # Train the model (minimal training for test speed)
    trained_model = experiment.train(
        state_names=['X', 'P'],
        num_epochs=5,  # Small number for testing
        learning_rate=0.1,
        verbose=False
    )

    # Check that trained model was created
    assert trained_model is not None
    assert experiment.trained_model is not None
    assert experiment.training_history is not None
    assert 'loss' in experiment.training_history

    # Evaluate the model
    metrics = experiment.evaluate(verbose=False)

    # Check that metrics were calculated
    assert 'training' in metrics
    assert 'test' in metrics
    assert experiment.training_metrics is not None
    assert experiment.test_metrics is not None


def test_visualize(experiment_setup):
    """Test visualization functionality."""
    experiment = experiment_setup

    # Train the model first (required)
    experiment.train(num_epochs=2, verbose=False)

    # Generate visualizations
    experiment.visualize(
        state_names=['X', 'P'],
        state_labels={'X': 'X State', 'P': 'P State'}
    )

    # Check that visualization files were created
    assert os.path.exists(os.path.join(experiment.output_dir, "training_loss.png"))
    assert os.path.exists(os.path.join(experiment.output_dir, "training_dataset_1_predictions.png"))
    assert os.path.exists(os.path.join(experiment.output_dir, "test_dataset_1_predictions.png"))


def test_save_all_results(experiment_setup):
    """Test saving all results."""
    experiment = experiment_setup

    # Train the model first (required)
    experiment.train(num_epochs=2, verbose=False)
    experiment.evaluate(verbose=False)

    # Save all results
    result_paths = experiment.save_all_results()

    # Check that result paths were returned
    assert isinstance(result_paths, dict)
    assert "model" in result_paths
    assert "history" in result_paths
    assert "norm_params" in result_paths
    assert "model_documentation" in result_paths

    # Check that files exist
    for path in result_paths.values():
        assert os.path.exists(path)


def test_save_results_summary(experiment_setup):
    """Test saving a results summary."""
    experiment = experiment_setup

    # Train and evaluate the model first
    experiment.train(num_epochs=2, verbose=False)
    experiment.evaluate(verbose=False)

    # Save summary
    experiment.save_results_summary("test_summary.txt")

    # Check that summary file was created
    summary_path = os.path.join(experiment.output_dir, "test_summary.txt")
    assert os.path.exists(summary_path)

    # Check content
    with open(summary_path, 'r') as f:
        content = f.read()
        assert "Experiment: test_experiment" in content
        assert "Training Parameters" in content
        assert "Performance Metrics" in content