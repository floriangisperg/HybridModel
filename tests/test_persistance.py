"""Tests for the model persistence module."""
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import os
import tempfile
import shutil
import json
import pickle
from hybrid_models.persistence import (
    save_model,
    load_model,
    save_training_results,
    load_training_results
)


class SimpleModel(eqx.Module):
    """A simple model for testing persistence functions."""
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, weight=1.0, bias=0.0):
        self.weight = jnp.array(weight)
        self.bias = jnp.array(bias)

    def __call__(self, x):
        return self.weight * x + self.bias


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


def test_save_and_load_model(temp_dir):
    """Test saving and loading a model."""
    # Create a simple model
    model = SimpleModel(weight=2.5, bias=1.5)

    # Save the model
    model_path = os.path.join(temp_dir, "test_model.eqx")
    metadata = {"test_key": "test_value"}

    save_model(model, model_path, metadata)

    # Check that model file was created
    assert os.path.exists(model_path)

    # Check that metadata file was created
    metadata_path = f"{model_path}.meta.json"
    assert os.path.exists(metadata_path)

    # Load the model back
    loaded_model, loaded_metadata = load_model(model_path, SimpleModel())

    # Check that model parameters were preserved
    assert jnp.allclose(loaded_model.weight, model.weight)
    assert jnp.allclose(loaded_model.bias, model.bias)

    # Check that metadata was preserved
    assert loaded_metadata["test_key"] == "test_value"


def test_save_and_load_model_without_metadata(temp_dir):
    """Test saving and loading a model without metadata."""
    # Create a simple model
    model = SimpleModel(weight=3.0, bias=2.0)

    # Save the model without metadata
    model_path = os.path.join(temp_dir, "test_model_no_meta.eqx")
    save_model(model, model_path)

    # Check that model file was created
    assert os.path.exists(model_path)

    # Load the model back
    loaded_model, loaded_metadata = load_model(model_path, SimpleModel())

    # Check that model parameters were preserved
    assert jnp.allclose(loaded_model.weight, model.weight)
    assert jnp.allclose(loaded_model.bias, model.bias)

    # Metadata should be None
    assert loaded_metadata is None


def test_save_and_load_training_results(temp_dir):
    """Test saving and loading complete training results."""
    # Create a simple model
    model = SimpleModel(weight=2.0, bias=1.0)

    # Create training history
    training_history = {
        "loss": [10.0, 5.0, 2.0, 1.0, 0.5],
        "aux": [(10.0, 5.0), (5.0, 2.5), (2.0, 1.0), (1.0, 0.5), (0.5, 0.25)]
    }

    # Create model config and normalization parameters
    model_config = {
        "architecture": "simple",
        "state_names": ["X", "P"]
    }

    norm_params = {
        "X_mean": 0.0,
        "X_std": 1.0,
        "P_mean": 0.0,
        "P_std": 1.0
    }

    # Create solver config and metrics
    solver_config = {
        "solver_type": "tsit5",
        "rtol": 1e-3,
        "atol": 1e-6
    }

    metrics = {
        "training": {"X": {"r2": 0.99, "rmse": 0.01}},
        "test": {"X": {"r2": 0.95, "rmse": 0.05}}
    }

    # Save training results
    result_paths = save_training_results(
        output_dir=temp_dir,
        model=model,
        training_history=training_history,
        model_config=model_config,
        norm_params=norm_params,
        solver_config=solver_config,
        metrics=metrics
    )

    # Check that all files were created
    for path in result_paths.values():
        assert os.path.exists(path)

    # Check that expected keys are in result_paths
    assert "model" in result_paths
    assert "history" in result_paths
    assert "norm_params" in result_paths
    assert "solver_config" in result_paths
    assert "metrics" in result_paths

    # Load training results back
    loaded_results = load_training_results(temp_dir, SimpleModel())

    # Check that all components were loaded correctly
    assert "model" in loaded_results
    assert "training_history" in loaded_results
    assert "norm_params" in loaded_results
    assert "solver_config" in loaded_results
    assert "metrics" in loaded_results

    # Check specifics of loaded data
    assert jnp.allclose(loaded_results["model"].weight, model.weight)
    assert jnp.allclose(loaded_results["model"].bias, model.bias)
    assert loaded_results["training_history"]["loss"] == training_history["loss"]
    assert loaded_results["norm_params"]["X_mean"] == norm_params["X_mean"]
    assert loaded_results["solver_config"]["solver_type"] == solver_config["solver_type"]
    assert loaded_results["metrics"]["training"]["X"]["r2"] == metrics["training"]["X"]["r2"]


def test_load_partial_training_results(temp_dir):
    """Test loading training results when not all files are present."""
    # Create a simple model
    model = SimpleModel(weight=2.0, bias=1.0)

    # Create only model config and norm params
    model_config = {"architecture": "simple"}
    norm_params = {"X_mean": 0.0, "X_std": 1.0}

    # Save only model and norm_params
    model_path = os.path.join(temp_dir, "model.eqx")
    norm_path = os.path.join(temp_dir, "normalization_params.json")

    # Save model with metadata
    metadata = {"model_config": model_config}
    save_model(model, model_path, metadata)

    # Save norm params directly
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f)

    # Load training results
    loaded_results = load_training_results(temp_dir, SimpleModel())

    # Check what was loaded
    assert "model" in loaded_results
    assert "metadata" in loaded_results
    assert "norm_params" in loaded_results

    # Check contents
    assert jnp.allclose(loaded_results["model"].weight, model.weight)
    assert loaded_results["metadata"]["model_config"] == model_config
    assert loaded_results["norm_params"]["X_mean"] == norm_params["X_mean"]

    # Optional components should not be present
    assert "training_history" not in loaded_results
    assert "solver_config" not in loaded_results
    assert "metrics" not in loaded_results