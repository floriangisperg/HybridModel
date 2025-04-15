"""Tests for the model utilities module (model_utils.py)."""

import pytest
import jax
import jax.numpy as jnp
import os
from hybrid_models.model_utils import (
    NeuralNetworkConfig,
    ModelConfig,
    describe_model,
    save_model_description,
    save_normalization_params,
    create_model_from_config,  # Import the factory function
)
from hybrid_models.builder import (
    HybridModelBuilder,
)  # Needed for ModelConfig tests indirectly
from hybrid_models.ode_system import HybridODESystem  # To check factory output type


# --- Fixtures ---
@pytest.fixture
def sample_norm_params():
    return {"X_mean": 10.0, "X_std": 2.0, "P_mean": 5.0, "P_std": 1.0}


@pytest.fixture
def sample_nn_config():
    return NeuralNetworkConfig(
        name="growth_rate",
        input_features=["X", "P"],
        hidden_dims=[16, 16],
        output_activation="relu",
        seed=123,
    )


@pytest.fixture
def sample_trainable_params_config():
    return {
        "k1": {"initial_value": 0.5, "transform": "softplus"},
        "k2": {"initial_value": 10.0, "bounds": (0, 20), "transform": "sigmoid"},
    }


@pytest.fixture
def sample_mechanistic_funcs():
    def biomass_ode(inputs):
        return inputs["growth_rate"] * inputs["X"] * inputs["k1"]

    def product_ode(inputs):
        return inputs["k2"] * inputs["X"]

    return {"X": biomass_ode, "P": product_ode}


@pytest.fixture
def sample_model_config(
    sample_mechanistic_funcs, sample_nn_config, sample_trainable_params_config
):
    return ModelConfig(
        state_names=["X", "P"],
        mechanistic_components=sample_mechanistic_funcs,
        neural_networks=[sample_nn_config],
        trainable_parameters=sample_trainable_params_config,
    )


# --- Tests for NeuralNetworkConfig ---


def test_nn_config_get_activation():
    conf_relu = NeuralNetworkConfig("nn1", ["f1"], output_activation="relu")
    conf_sig = NeuralNetworkConfig("nn2", ["f1"], output_activation="sigmoid")
    conf_callable = NeuralNetworkConfig("nn3", ["f1"], output_activation=jnp.tanh)
    conf_none = NeuralNetworkConfig("nn4", ["f1"], output_activation=None)
    conf_invalid = NeuralNetworkConfig("nn5", ["f1"], output_activation="invalid_act")

    assert conf_relu.get_activation_fn() is jax.nn.relu
    assert conf_sig.get_activation_fn() is jax.nn.sigmoid
    assert conf_callable.get_activation_fn() is jnp.tanh
    assert conf_none.get_activation_fn() is None
    assert conf_invalid.get_activation_fn() is None  # Default for invalid string


def test_nn_config_get_random_key():
    conf1 = NeuralNetworkConfig("nn1", ["f1"], seed=42)
    conf2 = NeuralNetworkConfig("nn2", ["f1"], seed=42)
    conf3 = NeuralNetworkConfig("nn3", ["f1"], seed=43)
    key1 = conf1.get_random_key()
    key2 = conf2.get_random_key()
    key3 = conf3.get_random_key()
    assert isinstance(key1, jax.Array)
    assert jnp.array_equal(key1, key2)
    assert not jnp.array_equal(key1, key3)


# --- Tests for ModelConfig ---


def test_model_config_to_dict(sample_model_config):
    config_dict = sample_model_config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["state_names"] == ["X", "P"]
    assert "X" in config_dict["mechanistic_components"]
    assert len(config_dict["neural_networks"]) == 1
    assert config_dict["neural_networks"][0]["name"] == "growth_rate"
    assert "k1" in config_dict["trainable_parameters"]
    assert "k2" in config_dict["trainable_parameters"]
    assert config_dict["trainable_parameters"]["k1"]["transform"] == "softplus"


# --- Tests for Documentation ---


def test_describe_model(sample_model_config, sample_norm_params):
    # We don't need a real model instance if describe_model only uses config
    description = describe_model(
        model=None, model_config=sample_model_config, norm_params=sample_norm_params
    )

    assert isinstance(description, str)
    assert "HYBRID MODEL DOCUMENTATION" in description
    assert "STATE VARIABLES" in description
    assert "- X" in description
    assert "MECHANISTIC COMPONENTS" in description
    # Check for mechanistic func name
    assert "biomass_ode" in description
    assert "NEURAL NETWORK COMPONENTS" in description
    # Check for NN name
    assert "'growth_rate'" in description
    assert "Input Features: X, P" in description
    assert "TRAINABLE PARAMETERS" in description
    # Check for trainable param name
    assert "Parameter Name: 'k1'" in description
    assert "Initial Value: 0.5" in description
    assert "Transformation: softplus" in description
    assert "Parameter Name: 'k2'" in description
    assert "Bounds: (0, 20)" in description
    assert "Transformation: sigmoid" in description
    # Check that norm params are NOT included by default now
    # assert "NORMALIZATION PARAMETERS" not in description # Or check presence if uncommented


def test_save_model_description(sample_model_config, sample_norm_params, tmp_path):
    f_path = tmp_path / "model_desc.txt"
    returned_path = save_model_description(
        None, sample_model_config, sample_norm_params, filepath=str(f_path)
    )
    assert f_path.exists()
    assert str(returned_path) == str(f_path)
    content = f_path.read_text()
    assert "HYBRID MODEL DOCUMENTATION" in content
    assert "'k1'" in content  # Check trainable param is mentioned


def test_save_normalization_params(sample_norm_params, tmp_path):
    f_path = tmp_path / "norm_params.txt"
    returned_path = save_normalization_params(sample_norm_params, filepath=str(f_path))
    assert f_path.exists()
    assert str(returned_path) == str(f_path)
    content = f_path.read_text()
    assert "NORMALIZATION PARAMETERS" in content
    assert "Variable: X" in content
    assert "Mean: 10.0" in content
    assert "Std: 2.0" in content


# --- Tests for create_model_from_config ---


def test_create_model_from_config_basic(sample_model_config, sample_norm_params):
    model = create_model_from_config(
        sample_model_config, sample_norm_params, master_seed=99
    )

    assert isinstance(model, HybridODESystem)
    assert model.state_names == ["X", "P"]
    assert "X" in model.mechanistic_components
    assert callable(model.mechanistic_components["X"])
    assert "growth_rate" in model.nn_replacements
    assert "k1" in model.trainable_parameters
    assert "k2" in model.trainable_parameters
    assert jnp.isclose(
        model.trainable_parameters["k1"], jnp.array(0.5)
    )  # Check initial value
    assert model.parameter_transforms["k1"]["transform"] == "softplus"
    assert model.parameter_transforms["k2"]["bounds"] == (0, 20)


def test_create_model_from_config_no_optional(sample_norm_params):
    # Config with only mandatory fields
    config = ModelConfig(
        state_names=["Z"],
        # No mechanistic, NNs, or trainable params
    )
    model = create_model_from_config(config, sample_norm_params, master_seed=1)
    assert isinstance(model, HybridODESystem)
    assert model.state_names == ["Z"]
    assert not model.mechanistic_components
    assert not model.nn_replacements
    assert not model.trainable_parameters
