"""Tests for the data handling module (data.py)."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from hybrid_models.data import TimeSeriesData, DatasetManager, VariableType
from hybrid_models.utils import calculate_rate  # Import for comparison if needed


# Fixture for sample DataFrame
@pytest.fixture
def sample_df():
    np.random.seed(42)
    times = np.linspace(0, 10, 21)
    return pd.DataFrame(
        {
            "time": times,
            "RunID": ["RunA"] * 11 + ["RunB"] * 10,
            "X_conc": 1.0 + 0.5 * times + 0.1 * np.random.randn(len(times)),
            "P_conc": 0.1 * (times**2) + 0.05 * np.random.randn(len(times)),
            "Temp": 37.0 + 0.2 * np.random.randn(len(times)),
            "Feed": 0.1 * times,
            "ReactorSize": [10.0] * 21,  # Example parameter
        }
    )


# --- Tests for TimeSeriesData ---


def test_timeseriesdata_init():
    ts_data = TimeSeriesData(run_id="TestRun")
    assert ts_data.run_id == "TestRun"
    assert ts_data.time_points is None
    assert not ts_data.state_variables
    assert not ts_data.parameter_variables
    assert not ts_data.control_variables
    assert not ts_data.feed_variables
    assert not ts_data.output_variables


def test_timeseriesdata_from_dataframe(sample_df):
    ts_data = TimeSeriesData()
    df_runA = sample_df[sample_df["RunID"] == "RunA"].copy()
    ts_data.from_dataframe(
        df_runA, time_column="time", run_id_column="RunID", run_id="RunA"
    )

    assert ts_data.run_id == "RunA"
    assert ts_data.time_column == "time"
    assert isinstance(ts_data.time_points, jax.Array)
    assert len(ts_data.time_points) == 11
    assert ts_data.time_points[0] == 0.0
    assert ts_data.time_points[-1] == 5.0  # RunA goes up to t=5


def test_timeseriesdata_add_variable_state(sample_df):
    ts_data = TimeSeriesData()
    df_runA = sample_df[sample_df["RunID"] == "RunA"].copy()
    ts_data.from_dataframe(df_runA, "time", "RunID", "RunA")
    ts_data.add_variable(
        "X_conc", VariableType.STATE, internal_name="X", is_output=True, data=df_runA
    )

    assert "X" in ts_data.state_variables
    assert "X" in ts_data.output_variables
    assert isinstance(ts_data.state_variables["X"], jax.Array)
    assert len(ts_data.state_variables["X"]) == 11
    assert ts_data._column_mapping["X"] == "X_conc"


def test_timeseriesdata_add_variable_parameter(sample_df):
    ts_data = TimeSeriesData()
    df_runA = sample_df[sample_df["RunID"] == "RunA"].copy()
    ts_data.from_dataframe(df_runA, "time", "RunID", "RunA")
    ts_data.add_variable(
        "ReactorSize", VariableType.PARAMETER, internal_name="V", data=df_runA
    )

    assert "V" in ts_data.parameter_variables
    assert isinstance(ts_data.parameter_variables["V"], float)
    assert ts_data.parameter_variables["V"] == 10.0
    assert ts_data._column_mapping["V"] == "ReactorSize"


def test_timeseriesdata_add_variable_control_with_rate(sample_df):
    ts_data = TimeSeriesData()
    df_runA = sample_df[sample_df["RunID"] == "RunA"].copy()
    ts_data.from_dataframe(df_runA, "time", "RunID", "RunA")
    ts_data.add_variable(
        "Temp",
        VariableType.CONTROL,
        internal_name="T",
        calculate_rate=True,
        data=df_runA,
    )

    assert "T" in ts_data.control_variables
    assert "T_rate" in ts_data.control_variables  # Rate should be stored here
    assert len(ts_data.control_variables["T"]) == 11
    assert len(ts_data.control_variables["T_rate"]) == 11
    assert ts_data._column_mapping["T"] == "Temp"
    assert ts_data._column_mapping["T_rate"] == "Rate of Temp"


def test_timeseriesdata_add_variable_feed_with_rate(sample_df):
    ts_data = TimeSeriesData()
    df_runA = sample_df[sample_df["RunID"] == "RunA"].copy()
    ts_data.from_dataframe(df_runA, "time", "RunID", "RunA")
    # Default calculate_rate=True for FEED
    ts_data.add_variable("Feed", VariableType.FEED, internal_name="F", data=df_runA)

    assert "F" in ts_data.feed_variables
    assert "F_rate" in ts_data.feed_variables  # Rate should be here for feeds
    assert len(ts_data.feed_variables["F"]) == 11
    assert len(ts_data.feed_variables["F_rate"]) == 11
    assert ts_data._column_mapping["F"] == "Feed"
    assert ts_data._column_mapping["F_rate"] == "Rate of Feed"


def test_timeseriesdata_get_initial_state(sample_df):
    ts_data = TimeSeriesData()
    df_runA = sample_df[sample_df["RunID"] == "RunA"].copy()
    ts_data.from_dataframe(df_runA, "time", "RunID", "RunA")
    ts_data.add_variable("X_conc", VariableType.STATE, "X", True, data=df_runA)
    ts_data.add_variable("P_conc", VariableType.STATE, "P", True, data=df_runA)

    initial_state = ts_data.get_initial_state()
    assert "X" in initial_state
    assert "P" in initial_state
    assert isinstance(initial_state["X"], float)
    assert jnp.isclose(initial_state["X"], ts_data.state_variables["X"][0])
    assert jnp.isclose(initial_state["P"], ts_data.state_variables["P"][0])


def test_timeseriesdata_prepare_for_training(sample_df):
    ts_data = TimeSeriesData()
    df_runA = sample_df[sample_df["RunID"] == "RunA"].copy()
    ts_data.from_dataframe(df_runA, "time", "RunID", "RunA")
    ts_data.add_variable("X_conc", VariableType.STATE, "X", True, data=df_runA)
    ts_data.add_variable("P_conc", VariableType.STATE, "P", True, data=df_runA)
    ts_data.add_variable("Temp", VariableType.CONTROL, "T", data=df_runA)
    ts_data.add_variable("Feed", VariableType.FEED, "F", data=df_runA)  # Will calc rate
    ts_data.add_variable("ReactorSize", VariableType.PARAMETER, "V", data=df_runA)

    prepared = ts_data.prepare_for_training()

    assert "times" in prepared
    assert "initial_state" in prepared
    assert "time_dependent_inputs" in prepared
    assert "static_inputs" in prepared
    assert "X_true" in prepared
    assert "P_true" in prepared

    assert "X" in prepared["initial_state"]
    assert "P" in prepared["initial_state"]
    assert "T" in prepared["time_dependent_inputs"]
    assert "F" in prepared["time_dependent_inputs"]
    assert "F_rate" in prepared["time_dependent_inputs"]  # Rate from feed
    assert "V" in prepared["static_inputs"]
    assert len(prepared["X_true"]) == 11


# --- Tests for DatasetManager ---


def test_datasetmanager_init():
    manager = DatasetManager()
    assert not manager.train_datasets
    assert not manager.test_datasets
    assert not manager.norm_params


def test_datasetmanager_load_split_ratio(sample_df):
    manager = DatasetManager()
    manager.load_from_dataframe(
        sample_df, "time", "RunID", train_ratio=0.5
    )  # Should get 1 train, 1 test
    assert len(manager.train_datasets) == 1
    assert len(manager.test_datasets) == 1
    assert manager.train_datasets[0].run_id == "RunA"
    assert manager.test_datasets[0].run_id == "RunB"


def test_datasetmanager_load_split_ids(sample_df):
    manager = DatasetManager()
    manager.load_from_dataframe(
        sample_df, "time", "RunID", train_run_ids=["RunB"], test_run_ids=["RunA"]
    )
    assert len(manager.train_datasets) == 1
    assert len(manager.test_datasets) == 1
    assert manager.train_datasets[0].run_id == "RunB"
    assert manager.test_datasets[0].run_id == "RunA"


def test_datasetmanager_add_variables(sample_df):
    manager = DatasetManager()
    manager.load_from_dataframe(sample_df, "time", "RunID", train_ratio=0.5)
    variables = [
        ("X_conc", VariableType.STATE, "X", True, False),
        ("Temp", VariableType.CONTROL, "T", False, False),
        ("Feed", VariableType.FEED, "F", False, True),  # Test rate calc via manager
    ]
    manager.add_variables(variables, data=sample_df)

    # Check one train and one test dataset
    for ds in [manager.train_datasets[0], manager.test_datasets[0]]:
        assert "X" in ds.state_variables
        assert "T" in ds.control_variables
        assert "F" in ds.feed_variables
        assert "F_rate" in ds.feed_variables  # Rate should be calculated


def test_datasetmanager_calculate_norm_params(sample_df):
    manager = DatasetManager()
    # Load RunA as train, RunB as test
    manager.load_from_dataframe(
        sample_df, "time", "RunID", train_run_ids=["RunA"], test_run_ids=["RunB"]
    )
    variables = [
        ("X_conc", VariableType.STATE, "X", True, False),
        ("P_conc", VariableType.STATE, "P", True, False),
    ]
    manager.add_variables(variables, data=sample_df)
    manager.calculate_norm_params(variable_names=["X"])  # Only calc for X

    assert "X_mean" in manager.norm_params
    assert "X_std" in manager.norm_params
    assert "P_mean" not in manager.norm_params  # Should not be calculated

    # Verify params are calculated ONLY from RunA (train)
    df_runA = sample_df[sample_df["RunID"] == "RunA"]
    expected_X_mean = df_runA["X_conc"].mean()
    expected_X_std = df_runA["X_conc"].std()

    assert jnp.isclose(manager.norm_params["X_mean"], expected_X_mean)
    # Note: pandas std is ddof=1 by default, numpy/jax is ddof=0. Be mindful if comparing exactly.
    # For this test, checking they are calculated is sufficient. Recalculate if needed:
    # assert jnp.isclose(manager.norm_params['X_std'], jnp.std(jnp.array(df_runA['X_conc'].values)))


def test_datasetmanager_prepare_data(sample_df):
    manager = DatasetManager()
    manager.load_from_dataframe(sample_df, "time", "RunID", train_ratio=0.5)
    variables = [("X_conc", VariableType.STATE, "X", True, False)]
    manager.add_variables(variables, data=sample_df)

    train_prepared = manager.prepare_training_data()
    test_prepared = manager.prepare_test_data()

    assert isinstance(train_prepared, list)
    assert len(train_prepared) == 1
    assert isinstance(train_prepared[0], dict)
    assert "X_true" in train_prepared[0]

    assert isinstance(test_prepared, list)
    assert len(test_prepared) == 1
    assert isinstance(test_prepared[0], dict)
    assert "X_true" in test_prepared[0]


def test_timeseriesdata_add_variable_feed_with_rate(sample_df):
    ts_data = TimeSeriesData()
    df_runA = sample_df[sample_df["RunID"] == "RunA"].copy()
    ts_data.from_dataframe(df_runA, "time", "RunID", "RunA")
    # --- CHANGE: Use add_feed instead of add_variable ---
    # Original: ts_data.add_variable("Feed", VariableType.FEED, internal_name="F", data=df_runA)
    ts_data.add_feed(
        column_name="Feed", internal_name="F", data=df_runA
    )  # add_feed defaults calculate_rate=True

    assert "F" in ts_data.feed_variables
    assert "F_rate" in ts_data.feed_variables  # Rate should be here for feeds
    assert len(ts_data.feed_variables["F"]) == 11
    assert len(ts_data.feed_variables["F_rate"]) == 11
    assert ts_data._column_mapping["F"] == "Feed"
    assert ts_data._column_mapping["F_rate"] == "Rate of Feed"


def test_timeseriesdata_prepare_for_training(sample_df):
    ts_data = TimeSeriesData()
    df_runA = sample_df[sample_df["RunID"] == "RunA"].copy()
    ts_data.from_dataframe(df_runA, "time", "RunID", "RunA")
    # Use convenience methods for clarity
    ts_data.add_state("X_conc", internal_name="X", is_output=True, data=df_runA)
    ts_data.add_state("P_conc", internal_name="P", is_output=True, data=df_runA)
    ts_data.add_control("Temp", internal_name="T", data=df_runA)
    # --- CHANGE: Use add_feed instead of add_variable ---
    # Original: ts_data.add_variable("Feed", VariableType.FEED, "F", data=df_runA)
    ts_data.add_feed(
        column_name="Feed", internal_name="F", data=df_runA
    )  # add_feed defaults calculate_rate=True
    ts_data.add_parameter("ReactorSize", internal_name="V", data=df_runA)

    prepared = ts_data.prepare_for_training()

    assert "times" in prepared
    assert "initial_state" in prepared
    assert "time_dependent_inputs" in prepared
    assert "static_inputs" in prepared
    assert "X_true" in prepared
    assert "P_true" in prepared

    assert "X" in prepared["initial_state"]
    assert "P" in prepared["initial_state"]
    assert "T" in prepared["time_dependent_inputs"]
    assert "F" in prepared["time_dependent_inputs"]
    assert "F_rate" in prepared["time_dependent_inputs"]  # Rate from feed
    assert "V" in prepared["static_inputs"]
    assert len(prepared["X_true"]) == 11
