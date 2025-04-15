"""Tests for the advanced evaluation utilities (evaluation_utils.py)."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from hybrid_models.evaluation_utils import (
    evaluate_model_performance,
    create_metrics_summary,
    compare_models,
)
from hybrid_models.evaluation import calculate_metrics
from typing import Any, Dict, List  # Added imports

# --- Fixtures ---


# Mock solve function (MODIFIED)
def mock_solve_fn(model: Any, dataset: Dict) -> Dict:
    """Simple mock: returns values based on dataset name or a fixed offset. Handles missing keys."""
    t = dataset["times"]
    # Use dataset ID to create different mock behavior per dataset if needed
    offset = 0.1 if "1" in dataset.get("id", "0") else 0.2

    solution = {"times": t}

    # Safely generate mock prediction for X
    if "X_true" in dataset:
        solution["X"] = dataset["X_true"] + offset
    else:
        # Fallback if X_true is missing (e.g., return zeros + offset)
        solution["X"] = jnp.zeros_like(t) + offset

    # Safely generate mock prediction for P
    if "P_true" in dataset:
        solution["P"] = dataset["P_true"] - offset / 2.0
    else:
        # Fallback if P_true is missing (e.g., return zeros - offset/2)
        # Important: Still include 'P' key if evaluate_model_performance expects it based on state_names
        solution["P"] = jnp.zeros_like(t) - offset / 2.0

    # Example: Include state 'Y' in solution, even if 'Y_true' isn't in data
    solution["Y"] = jnp.ones_like(t)

    return solution


# Mock datasets fixture remains the same
@pytest.fixture
def mock_datasets() -> List[Dict]:
    times1 = jnp.array([0.0, 1.0, 2.0])
    times2 = jnp.array([0.0, 1.0, 2.0, 3.0])
    return [
        {
            "id": "Dataset_1",
            "times": times1,
            "X_true": jnp.array([1.0, 2.0, 3.0]),
            "P_true": jnp.array([5.0, 5.0, 5.0]),
        },
        {
            "id": "Dataset_2",
            "times": times2,
            "X_true": jnp.array([2.0, 3.0, 4.0, 5.0]),
            "P_true": jnp.array([1.0, 2.0, 1.0, 2.0]),
        },
        # Dataset with missing P_true value
        {"id": "Dataset_3", "times": times1, "X_true": jnp.array([1.0, 1.0, 1.0])},
    ]


# --- Tests for evaluate_model_performance ---


def test_evaluate_model_performance_structure(mock_datasets):
    results = evaluate_model_performance(
        model=None,  # Mock model (not used by mock_solve_fn)
        datasets=mock_datasets,
        solve_fn=mock_solve_fn,
        state_names=["X", "P"],  # Explicitly ask for X and P
        dataset_type="Test",
        verbose=False,
    )

    assert isinstance(results, dict)
    assert "Test_1" in results
    assert "Test_2" in results
    assert "Test_3" in results
    assert "overall" in results

    assert "X" in results["Test_1"]
    assert "P" in results["Test_1"]
    assert "r2" in results["Test_1"]["X"]
    assert "rmse" in results["Test_1"]["X"]

    assert "X" in results["Test_3"]
    assert "P" not in results["Test_3"]  # Metrics for P cannot be calculated for Test_3

    # Check structure of overall result (P should be calculated overall from datasets 1 & 2)
    assert "X" in results["overall"]
    assert "P" in results["overall"]
    assert "r2" in results["overall"]["X"]
    assert "rmse" in results["overall"]["X"]
    assert "P" in results["overall"]


def test_evaluate_model_performance_overall_calc(mock_datasets):
    # Use only first two datasets which have both X and P
    datasets_for_test = mock_datasets[:2]
    results = evaluate_model_performance(
        model=None,
        datasets=datasets_for_test,
        solve_fn=mock_solve_fn,
        state_names=["X", "P"],
        dataset_type="Test",
        verbose=False,
    )

    # Calculate metrics manually on concatenated data
    all_true_X = jnp.concatenate([ds["X_true"] for ds in datasets_for_test])
    all_true_P = jnp.concatenate([ds["P_true"] for ds in datasets_for_test])
    sol1 = mock_solve_fn(None, datasets_for_test[0])
    sol2 = mock_solve_fn(None, datasets_for_test[1])
    all_pred_X = jnp.concatenate([sol1["X"], sol2["X"]])
    all_pred_P = jnp.concatenate([sol1["P"], sol2["P"]])

    expected_overall_X = calculate_metrics(all_true_X, all_pred_X)
    expected_overall_P = calculate_metrics(all_true_P, all_pred_P)

    # Compare calculated overall metrics with expected
    assert jnp.allclose(results["overall"]["X"]["r2"], expected_overall_X["r2"])
    assert jnp.allclose(results["overall"]["X"]["rmse"], expected_overall_X["rmse"])
    assert jnp.allclose(results["overall"]["P"]["r2"], expected_overall_P["r2"])
    assert jnp.allclose(results["overall"]["P"]["rmse"], expected_overall_P["rmse"])


def test_evaluate_model_performance_state_discovery(mock_datasets):
    results = evaluate_model_performance(
        model=None,
        datasets=mock_datasets,
        solve_fn=mock_solve_fn,
        state_names=None,  # Trigger auto-discovery
        verbose=False,
    )
    # Should discover X and P from *_true keys
    assert "X" in results["overall"]
    assert "P" in results["overall"]
    # Y should not be evaluated as Y_true is missing
    assert "Y" not in results.get("Test_1", {})
    assert "Y" not in results.get("overall", {})


def test_evaluate_model_performance_saving(mock_datasets, tmp_path):
    output_dir = tmp_path / "eval_results"
    metrics_filename = "test_run_metrics.txt"
    results = evaluate_model_performance(
        model=None,
        datasets=mock_datasets,
        solve_fn=mock_solve_fn,
        state_names=["X", "P"],
        dataset_type="Val",
        verbose=False,
        save_metrics=True,
        output_dir=str(output_dir),
        metrics_filename=metrics_filename,
    )

    f_path = output_dir / metrics_filename
    assert f_path.exists()
    content = f_path.read_text()
    assert "=== Val Evaluation Metrics ===" in content
    assert "Val_1" in content
    assert "Val_2" in content
    assert "Overall Metrics" in content
    assert "r2:" in content  # Check if metrics are printed


# --- Tests for create_metrics_summary ---


@pytest.fixture
def sample_evaluation_results():
    # Sample structure returned by evaluate_model_performance
    return {
        "Train_1": {"X": {"r2": 0.9, "rmse": 0.1}, "P": {"r2": 0.8, "rmse": 0.2}},
        "Train_2": {"X": {"r2": 0.95, "rmse": 0.05}, "P": {"r2": 0.85, "rmse": 0.15}},
        "overall": {"X": {"r2": 0.92, "rmse": 0.08}, "P": {"r2": 0.82, "rmse": 0.18}},
    }


def test_create_metrics_summary_dataframe(sample_evaluation_results):
    df = create_metrics_summary(sample_evaluation_results, format_type="dataframe")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6  # 2 datasets * 2 states + 1 overall * 2 states
    assert "Dataset" in df.columns
    assert "State" in df.columns
    assert "r2" in df.columns
    assert "rmse" in df.columns
    # Check one value
    assert df[(df["Dataset"] == "overall") & (df["State"] == "X")]["r2"].iloc[0] == 0.92


def test_create_metrics_summary_flat_dict(sample_evaluation_results):
    flat_dict = create_metrics_summary(
        sample_evaluation_results, format_type="flat_dict"
    )
    assert isinstance(flat_dict, dict)
    assert "Train_1.X.r2" in flat_dict
    assert "overall.P.rmse" in flat_dict
    assert flat_dict["Train_1.X.r2"] == 0.9
    assert flat_dict["overall.P.rmse"] == 0.18


def test_create_metrics_summary_dict(sample_evaluation_results):
    dict_summary = create_metrics_summary(sample_evaluation_results, format_type="dict")
    # Should just return the original dictionary
    assert dict_summary == sample_evaluation_results


# --- Tests for compare_models ---


def test_compare_models(mock_datasets):
    # Create two mock models (just need distinct objects)
    model1 = "ModelA"
    model2 = "ModelB"

    # Mock solve function that gives slightly different results per model
    def compare_solve_fn(model, dataset):
        offset = 0.1 if model == "ModelA" else 0.15
        return {
            "times": dataset["times"],
            "X": dataset["X_true"] + offset,
            "P": dataset["P_true"] - offset,
        }

    comparison_df = compare_models(
        models=[model1, model2],
        model_names=["ModelA", "ModelB"],
        datasets=mock_datasets[:2],  # Use first two valid datasets
        solve_fn=compare_solve_fn,
        state_names=["X", "P"],
        dataset_type="Compare",
    )

    assert isinstance(comparison_df, pd.DataFrame)
    assert len(comparison_df) == 4  # 2 models * 2 states
    assert "Model" in comparison_df.columns
    assert "State" in comparison_df.columns
    assert "r2" in comparison_df.columns
    assert comparison_df["Model"].tolist() == ["ModelA", "ModelA", "ModelB", "ModelB"]
    assert comparison_df["State"].tolist() == ["X", "P", "X", "P"]
    # Check that metrics differ slightly between models
    r2_a = comparison_df[comparison_df["Model"] == "ModelA"]["r2"].values
    r2_b = comparison_df[comparison_df["Model"] == "ModelB"]["r2"].values
    assert not np.allclose(r2_a, r2_b)
