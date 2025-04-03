"""Tests for the data management module."""
import pytest
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from hybrid_models.data import TimeSeriesData, DatasetManager, VariableType


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'time': [0.0, 1.0, 2.0, 3.0, 4.0],
        'RunID': [1, 1, 1, 1, 1],
        'X': [1.0, 1.5, 2.2, 3.0, 3.8],
        'P': [0.0, 0.3, 0.8, 1.5, 2.1],
        'temp': [30.0, 32.0, 35.0, 37.0, 37.0],
        'feed': [0.0, 0.1, 0.2, 0.3, 0.4],
        'param': [5.0, 5.0, 5.0, 5.0, 5.0]
    })


@pytest.fixture
def multi_run_dataframe():
    """Create a DataFrame with multiple runs for testing."""
    runs = []
    for run_id in range(1, 4):  # 3 runs
        times = np.linspace(0, 10, 11)  # 0 to 10 in steps of 1.0
        X = 1.0 + 0.5 * times * (1 + 0.1 * run_id)  # Slight variation between runs
        P = 0.1 * times ** 2 * (1 + 0.05 * run_id)
        temp = 37.0 * np.ones_like(times)
        feed = 0.1 * times
        param = 5.0 * run_id * np.ones_like(times)

        run_data = pd.DataFrame({
            'time': times,
            'RunID': run_id,
            'X': X,
            'P': P,
            'temp': temp,
            'feed': feed,
            'param': param
        })
        runs.append(run_data)

    return pd.concat(runs, ignore_index=True)


def test_timeseries_data_initialization():
    """Test TimeSeriesData initialization."""
    ts = TimeSeriesData(run_id="test_run")
    assert ts.run_id == "test_run"
    assert ts.time_points is None
    assert len(ts.state_variables) == 0
    assert len(ts.parameter_variables) == 0
    assert len(ts.control_variables) == 0
    assert len(ts.feed_variables) == 0
    assert len(ts.output_variables) == 0


def test_timeseries_set_time_column(sample_dataframe):
    """Test setting the time column."""
    ts = TimeSeriesData()
    ts.set_time_column('time', sample_dataframe)

    assert ts.time_column == 'time'
    assert ts.time_points is not None
    assert len(ts.time_points) == len(sample_dataframe)
    assert jnp.allclose(ts.time_points, jnp.array(sample_dataframe['time']))


def test_timeseries_from_dataframe(sample_dataframe):
    """Test loading data from DataFrame."""
    ts = TimeSeriesData().from_dataframe(
        df=sample_dataframe,
        time_column='time',
        run_id_column='RunID',
        run_id=1
    )

    assert ts.run_id == "1"
    assert ts.time_column == 'time'
    assert len(ts.time_points) == len(sample_dataframe)

    # Test filtering by run_id
    ts2 = TimeSeriesData().from_dataframe(
        df=sample_dataframe,
        time_column='time',
        run_id_column='RunID',
        run_id=2  # This run_id doesn't exist
    )

    # This should raise an error since run_id=2 doesn't exist
    with pytest.raises(ValueError):
        ts2 = TimeSeriesData().from_dataframe(
            df=sample_dataframe,
            time_column='time',
            run_id_column='RunID',
            run_id=2
        )


def test_timeseries_add_variable(sample_dataframe):
    """Test adding variables of different types."""
    ts = TimeSeriesData().from_dataframe(
        df=sample_dataframe,
        time_column='time'
    )

    # Add state variable
    ts.add_variable('X', VariableType.STATE, 'X_state', True, False, sample_dataframe)
    assert 'X_state' in ts.state_variables
    assert 'X_state' in ts.output_variables
    assert len(ts.state_variables['X_state']) == len(sample_dataframe)

    # Add parameter
    ts.add_variable('param', VariableType.PARAMETER, 'p', False, False, sample_dataframe)
    assert 'p' in ts.parameter_variables
    assert ts.parameter_variables['p'] == 5.0

    # Add control variable
    ts.add_variable('temp', VariableType.CONTROL, 'temperature', False, False, sample_dataframe)
    assert 'temperature' in ts.control_variables

    # Add feed variable with rate calculation
    ts.add_variable('feed', VariableType.FEED, 'feed_rate', False, True, sample_dataframe)
    assert 'feed_rate' in ts.feed_variables
    assert 'feed_rate_rate' in ts.feed_variables  # Rate should be calculated

    # Test shorthand methods
    ts.add_state('P', 'product', True, False, sample_dataframe)
    assert 'product' in ts.state_variables
    assert 'product' in ts.output_variables

    ts.add_control('temp', 'temp2', True, sample_dataframe)
    assert 'temp2' in ts.control_variables
    assert 'temp2_rate' in ts.control_variables  # Rate calculated


def test_timeseries_get_initial_state(sample_dataframe):
    """Test getting initial state."""
    ts = TimeSeriesData().from_dataframe(
        df=sample_dataframe,
        time_column='time'
    )

    ts.add_state('X', 'X', True, False, sample_dataframe)
    ts.add_state('P', 'P', True, False, sample_dataframe)

    initial_state = ts.get_initial_state()
    assert 'X' in initial_state
    assert 'P' in initial_state
    assert initial_state['X'] == 1.0  # First value in X
    assert initial_state['P'] == 0.0  # First value in P


def test_timeseries_prepare_for_training(sample_dataframe):
    """Test preparing dataset for training."""
    ts = TimeSeriesData().from_dataframe(
        df=sample_dataframe,
        time_column='time'
    )

    ts.add_state('X', 'X', True, False, sample_dataframe)
    ts.add_state('P', 'P', True, False, sample_dataframe)
    ts.add_control('temp', 'temp', False, False, sample_dataframe)
    ts.add_feed('feed', 'feed', True, sample_dataframe)
    ts.add_parameter('param', 'param', sample_dataframe)

    dataset = ts.prepare_for_training()

    assert 'times' in dataset
    assert 'initial_state' in dataset
    assert 'time_dependent_inputs' in dataset
    assert 'static_inputs' in dataset
    assert 'X_true' in dataset
    assert 'P_true' in dataset

    assert 'temp' in dataset['time_dependent_inputs']
    assert 'feed' in dataset['time_dependent_inputs']
    assert 'feed_rate' in dataset['time_dependent_inputs']
    assert 'param' in dataset['static_inputs']


def test_dataset_manager_load_from_dataframe(multi_run_dataframe):
    """Test loading data with DatasetManager."""
    manager = DatasetManager()
    manager.load_from_dataframe(
        df=multi_run_dataframe,
        time_column='time',
        run_id_column='RunID',
        train_ratio=0.67  # Should give 2 training, 1 testing
    )

    assert len(manager.train_datasets) == 2
    assert len(manager.test_datasets) == 1

    # Test with specific run IDs
    manager2 = DatasetManager()
    manager2.load_from_dataframe(
        df=multi_run_dataframe,
        time_column='time',
        run_id_column='RunID',
        train_run_ids=[1, 3],
        test_run_ids=[2]
    )

    assert len(manager2.train_datasets) == 2
    assert len(manager2.test_datasets) == 1
    assert manager2.train_datasets[0].run_id in ["1", "3"]
    assert manager2.train_datasets[1].run_id in ["1", "3"]
    assert manager2.test_datasets[0].run_id == "2"


def test_dataset_manager_add_variables(multi_run_dataframe):
    """Test adding variables to all datasets."""
    manager = DatasetManager()
    manager.load_from_dataframe(
        df=multi_run_dataframe,
        time_column='time',
        run_id_column='RunID',
        train_run_ids=[1, 2],
        test_run_ids=[3]
    )

    variable_definitions = [
        ('X', VariableType.STATE, 'X', True, False),
        ('P', VariableType.STATE, 'P', True, False),
        ('temp', VariableType.CONTROL, 'temp', False, False),
        ('feed', VariableType.FEED, 'feed', False, True),
        ('param', VariableType.PARAMETER, 'param', False, False)
    ]

    manager.add_variables(variable_definitions, multi_run_dataframe)

    # Check that variables were added to all datasets
    for dataset in manager.train_datasets + manager.test_datasets:
        assert 'X' in dataset.state_variables
        assert 'P' in dataset.state_variables
        assert 'temp' in dataset.control_variables
        assert 'feed' in dataset.feed_variables
        assert 'feed_rate' in dataset.feed_variables
        assert 'param' in dataset.parameter_variables


def test_dataset_manager_calculate_norm_params(multi_run_dataframe):
    """Test calculating normalization parameters."""
    manager = DatasetManager()
    manager.load_from_dataframe(
        df=multi_run_dataframe,
        time_column='time',
        run_id_column='RunID',
        train_run_ids=[1, 2],
        test_run_ids=[3]
    )

    variable_definitions = [
        ('X', VariableType.STATE, 'X', True, False),
        ('P', VariableType.STATE, 'P', True, False),
        ('temp', VariableType.CONTROL, 'temp', False, False)
    ]

    manager.add_variables(variable_definitions, multi_run_dataframe)
    manager.calculate_norm_params()

    # Check that normalization parameters were calculated
    assert 'X_mean' in manager.norm_params
    assert 'X_std' in manager.norm_params
    assert 'P_mean' in manager.norm_params
    assert 'P_std' in manager.norm_params
    assert 'temp_mean' in manager.norm_params
    assert 'temp_std' in manager.norm_params


def test_dataset_manager_prepare_data(multi_run_dataframe):
    """Test preparing training and test data."""
    manager = DatasetManager()
    manager.load_from_dataframe(
        df=multi_run_dataframe,
        time_column='time',
        run_id_column='RunID',
        train_run_ids=[1, 2],
        test_run_ids=[3]
    )

    variable_definitions = [
        ('X', VariableType.STATE, 'X', True, False),
        ('P', VariableType.STATE, 'P', True, False),
        ('temp', VariableType.CONTROL, 'temp', False, False),
        ('feed', VariableType.FEED, 'feed', False, True)
    ]

    manager.add_variables(variable_definitions, multi_run_dataframe)
    manager.calculate_norm_params()

    train_data = manager.prepare_training_data()
    test_data = manager.prepare_test_data()

    assert len(train_data) == 2
    assert len(test_data) == 1

    for dataset in train_data + test_data:
        assert 'times' in dataset
        assert 'initial_state' in dataset
        assert 'time_dependent_inputs' in dataset
        assert 'X_true' in dataset
        assert 'P_true' in dataset