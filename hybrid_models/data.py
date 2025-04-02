import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
from jaxtyping import Array, Float, PyTree
from enum import Enum


class VariableType(Enum):
    """Enumeration of variable types for clear classification."""
    STATE = 'state'  # State variables (like cell density, metabolites)
    PARAMETER = 'parameter'  # Process parameters, constant throughout a run
    CONTROL = 'control'  # Controlled variables (like temperature, pH)
    FEED = 'feed'  # Feed variables that affect mass balance


@dataclass
class TimeSeriesData:
    """
    A simple dataclass for handling time-series data in hybrid modeling.
    This class is domain-agnostic and can be used for any application.
    """
    run_id: Optional[str] = None

    # Time related data
    time_points: Optional[Array] = None
    time_column: Optional[str] = None

    # Variables by category
    state_variables: Dict[str, Array] = field(default_factory=dict)
    parameter_variables: Dict[str, Any] = field(default_factory=dict)
    control_variables: Dict[str, Array] = field(default_factory=dict)
    feed_variables: Dict[str, Array] = field(default_factory=dict)

    # Target variables for training
    output_variables: Dict[str, Array] = field(default_factory=dict)

    # Original column mappings
    _column_mapping: Dict[str, str] = field(default_factory=dict)

    def set_time_column(self, time_column: str, data: Optional[pd.DataFrame] = None):
        """
        Set the time column and extract time points if data is provided.
        """
        self.time_column = time_column

        if data is not None and time_column in data.columns:
            # Convert to JAX array
            self.time_points = jnp.array(data[time_column].values)

        return self

    def from_dataframe(self, df: pd.DataFrame, time_column: str, run_id_column: Optional[str] = None,
                       run_id: Optional[str] = None):
        """
        Load data from a pandas DataFrame.
        """
        # Handle run ID filtering
        if run_id_column is not None and run_id is not None:
            if run_id_column in df.columns:
                df = df[df[run_id_column] == run_id]
                if len(df) == 0:
                    raise ValueError(f"No data found for run_id={run_id}")
                self.run_id = str(run_id)

        # Sort by time column
        if time_column in df.columns:
            df = df.sort_values(time_column)

        # Set time column
        self.set_time_column(time_column, df)

        return self

    def add_variable(self, column_name: str, variable_type: VariableType,
                     internal_name: Optional[str] = None, is_output: bool = False,
                     calculate_rate: bool = False, data: Optional[pd.DataFrame] = None):
        """
        Add a variable from a DataFrame.
        """
        # Use internal name or column name
        internal_name = internal_name or column_name

        # Store mapping
        self._column_mapping[internal_name] = column_name

        if data is None:
            # If no data provided, assume this is called after loading a variable
            # and we're just setting metadata
            return self

        if column_name not in data.columns:
            print(f"Warning: Column '{column_name}' not found in data. Skipping.")
            return self

        # Extract data based on variable type
        if variable_type == VariableType.PARAMETER:
            # For parameters, take the first non-NaN value
            values = data[column_name].dropna().values
            if len(values) > 0:
                self.parameter_variables[internal_name] = (
                    float(values[0]) if np.issubdtype(values.dtype, np.number) else values[0]
                )
        else:
            # For time series variables, convert to JAX array
            values = data[column_name].values
            variable_array = jnp.array(values)

            # Add to appropriate dictionary based on type
            if variable_type == VariableType.STATE:
                self.state_variables[internal_name] = variable_array
                if is_output:
                    self.output_variables[internal_name] = variable_array
            elif variable_type == VariableType.CONTROL:
                self.control_variables[internal_name] = variable_array
            elif variable_type == VariableType.FEED:
                self.feed_variables[internal_name] = variable_array

            # Calculate rate if requested
            if calculate_rate and self.time_points is not None:
                rate_name = f"{internal_name}_rate"
                from hybrid_models.utils import calculate_rate as calc_rate_func
                rate_values = calc_rate_func(self.time_points, variable_array)

                # Determine where to store the rate based on the original variable type
                if variable_type == VariableType.FEED:
                    self.feed_variables[rate_name] = rate_values
                else:
                    self.control_variables[rate_name] = rate_values

                self._column_mapping[rate_name] = f"Rate of {column_name}"

        return self

    def add_state(self, column_name: str, internal_name: Optional[str] = None,
                  is_output: bool = True, calculate_rate: bool = False,
                  data: Optional[pd.DataFrame] = None):
        """Add a state variable."""
        return self.add_variable(column_name, VariableType.STATE, internal_name,
                                 is_output, calculate_rate, data)

    def add_parameter(self, column_name: str, internal_name: Optional[str] = None,
                      data: Optional[pd.DataFrame] = None):
        """Add a parameter variable."""
        return self.add_variable(column_name, VariableType.PARAMETER, internal_name,
                                 False, False, data)

    def add_control(self, column_name: str, internal_name: Optional[str] = None,
                    calculate_rate: bool = False, data: Optional[pd.DataFrame] = None):
        """Add a control variable."""
        return self.add_variable(column_name, VariableType.CONTROL, internal_name,
                                 False, calculate_rate, data)

    def add_feed(self, column_name: str, internal_name: Optional[str] = None,
                 calculate_rate: bool = True, data: Optional[pd.DataFrame] = None):
        """Add a feed variable."""
        return self.add_variable(column_name, VariableType.FEED, internal_name,
                                 False, calculate_rate, data)

    def get_initial_state(self) -> Dict[str, float]:
        """Get the initial values of all state variables."""
        initial_state = {}
        for name, values in self.state_variables.items():
            if len(values) > 0:
                # Use the first value, handling NaN values carefully
                if jnp.isnan(values[0]):
                    non_nan_indices = jnp.where(~jnp.isnan(values))[0]
                    if len(non_nan_indices) > 0:
                        initial_state[name] = float(values[non_nan_indices[0]])
                    else:
                        print(f"Warning: All values for {name} are NaN. Using 0.0 as initial state.")
                        initial_state[name] = 0.0
                else:
                    initial_state[name] = float(values[0])
        return initial_state

    def prepare_for_training(self) -> Dict:
        """
        Prepare the dataset for training by converting to the format
        expected by the training functions.
        """
        # Get initial state
        initial_state = self.get_initial_state()

        # Prepare time-dependent inputs
        time_dependent_inputs = {}

        # Add control and feed variables
        for name, values in {**self.control_variables, **self.feed_variables}.items():
            time_dependent_inputs[name] = (self.time_points, values)

        # Prepare static inputs (parameters)
        static_inputs = {**self.parameter_variables}

        # Prepare dataset
        dataset = {
            'times': self.time_points,
            'initial_state': initial_state,
            'time_dependent_inputs': time_dependent_inputs,
            'static_inputs': static_inputs
        }

        # Add true outputs for loss calculation
        for name, values in self.output_variables.items():
            dataset[f'{name}_true'] = values

        return dataset


class DatasetManager:
    """
    Manager for handling multiple datasets with train/test split functionality.
    """

    def __init__(self):
        self.train_datasets: List[TimeSeriesData] = []
        self.test_datasets: List[TimeSeriesData] = []
        self.norm_params: Dict[str, float] = {}

    def load_from_dataframe(self, df: pd.DataFrame, time_column: str,
                            run_id_column: Optional[str] = None,
                            train_run_ids: Optional[List] = None,
                            test_run_ids: Optional[List] = None,
                            train_ratio: float = 0.8):
        """
        Load data from a DataFrame, splitting into train and test datasets.

        Args:
            df: DataFrame containing the data
            time_column: Name of the time column
            run_id_column: Column that contains run IDs
            train_run_ids: Specific run IDs to use for training
            test_run_ids: Specific run IDs to use for testing
            train_ratio: Ratio of runs to use for training if not specified
        """
        if run_id_column and run_id_column in df.columns:
            all_run_ids = df[run_id_column].unique()

            # Determine train and test run IDs
            if train_run_ids is None and test_run_ids is None:
                # Split based on train_ratio
                n_train = max(1, int(len(all_run_ids) * train_ratio))
                train_run_ids = all_run_ids[:n_train]
                test_run_ids = all_run_ids[n_train:]
            elif train_run_ids is None:
                # Use all runs not in test_run_ids for training
                train_run_ids = [rid for rid in all_run_ids if rid not in test_run_ids]
            elif test_run_ids is None:
                # Use all runs not in train_run_ids for testing
                test_run_ids = [rid for rid in all_run_ids if rid not in train_run_ids]

            # Load training datasets
            for run_id in train_run_ids:
                dataset = TimeSeriesData(run_id=str(run_id))
                dataset.from_dataframe(df, time_column, run_id_column, run_id)
                self.train_datasets.append(dataset)

            # Load test datasets
            for run_id in test_run_ids:
                dataset = TimeSeriesData(run_id=str(run_id))
                dataset.from_dataframe(df, time_column, run_id_column, run_id)
                self.test_datasets.append(dataset)
        else:
            # If no run_id_column, use the entire dataset for training
            dataset = TimeSeriesData()
            dataset.from_dataframe(df, time_column)
            self.train_datasets.append(dataset)

        return self

    def add_variables(self, variable_definitions: List[Tuple], data: pd.DataFrame = None):
        """
        Add variables to all datasets based on definitions.

        Args:
            variable_definitions: List of tuples with
                (column_name, variable_type, internal_name, is_output, calculate_rate)
            data: Optional DataFrame to use (if not already loaded)
        """
        # Process each dataset
        for dataset in self.train_datasets + self.test_datasets:
            run_data = None

            # If data is provided and we have a run_id, filter the data
            if data is not None and dataset.run_id is not None:
                run_id_value = dataset.run_id

                # Find the run_id column by checking for a column with the matching value
                for col in data.columns:
                    if data[col].astype(str).eq(run_id_value).any():
                        run_data = data[data[col].astype(str) == run_id_value]
                        break

                if run_data is None:
                    run_data = data  # Use all data if we can't filter by run_id

            # Add each variable
            for var_def in variable_definitions:
                if len(var_def) == 2:
                    column_name, var_type = var_def
                    internal_name, is_output, calculate_rate = None, False, False
                elif len(var_def) == 3:
                    column_name, var_type, internal_name = var_def
                    is_output, calculate_rate = False, False
                elif len(var_def) == 4:
                    column_name, var_type, internal_name, is_output = var_def
                    calculate_rate = False
                elif len(var_def) >= 5:
                    column_name, var_type, internal_name, is_output, calculate_rate = var_def

                dataset.add_variable(
                    column_name=column_name,
                    variable_type=var_type,
                    internal_name=internal_name,
                    is_output=is_output,
                    calculate_rate=calculate_rate,
                    data=run_data
                )

        return self

    def calculate_norm_params(self, variable_names: Optional[List[str]] = None):
        """
        Calculate normalization parameters from training datasets only.

        Args:
            variable_names: Optional list of specific variable names to normalize
                            If None, uses all state, control, and feed variables
        """
        # Initialize container for all values
        all_values = {}

        # Find all relevant variables if not specified
        if variable_names is None:
            variable_names = set()
            for dataset in self.train_datasets:
                variable_names.update(dataset.state_variables.keys())
                variable_names.update(dataset.control_variables.keys())
                variable_names.update(dataset.feed_variables.keys())
            variable_names = list(variable_names)

        # Collect all values for each variable from training datasets only
        for dataset in self.train_datasets:
            for var_name in variable_names:
                if var_name not in all_values:
                    all_values[var_name] = []

                # Check in each variable category
                for var_dict in [dataset.state_variables, dataset.control_variables,
                                 dataset.feed_variables]:
                    if var_name in var_dict:
                        values = var_dict[var_name]
                        # Filter out NaN values
                        valid_values = values[~jnp.isnan(values)] if jnp.isnan(values).any() else values
                        if len(valid_values) > 0:
                            all_values[var_name].append(valid_values)

        # Calculate normalization parameters
        for var_name, value_list in all_values.items():
            if value_list and len(value_list) > 0:
                # Concatenate all values
                all_data = jnp.concatenate(value_list)

                # Standard scaling
                mean_val = float(jnp.mean(all_data))
                std_val = float(jnp.std(all_data))

                # Store normalization parameters
                self.norm_params[f"{var_name}_mean"] = mean_val
                self.norm_params[f"{var_name}_std"] = max(std_val, 1e-8)  # Avoid division by zero

        return self

    def prepare_training_data(self) -> List[Dict]:
        """Prepare training datasets for model training."""
        return [dataset.prepare_for_training() for dataset in self.train_datasets]

    def prepare_test_data(self) -> List[Dict]:
        """Prepare test datasets for model evaluation."""
        return [dataset.prepare_for_training() for dataset in self.test_datasets]