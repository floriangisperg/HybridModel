import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from jaxtyping import Array, Float, PyTree
from enum import Enum


class VariableType(Enum):
    """Enumeration of variable types for clear classification."""
    STATE = 'state'  # X variables - state/response variables (like cell density, metabolites)
    PARAMETER = 'parameter'  # Z variables - process parameters, constant throughout a run
    CONTROL = 'control'  # W variables - controlled variables (like temperature, pH)
    FEED = 'feed'  # F variables - feed variables that affect mass balance
    QUALITY = 'quality'  # Y variables - typically measured at the end (quality attributes)


class TimeSeriesData:
    """
    A general class for handling time-series data in hybrid modeling.
    This class is domain-agnostic and can be used for any application.
    """

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize a time-series dataset.

        Args:
            run_id: Optional identifier for this dataset
        """
        self.run_id = run_id
        self.time_points = None
        self.time_column = None

        # Initialize dictionaries for different variable types (based on paper nomenclature)
        self.state_variables = {}  # X variables - process state/response
        self.parameter_variables = {}  # Z variables - process parameters (constant)
        self.control_variables = {}  # W variables - controlled variables
        self.feed_variables = {}  # F variables - feeds
        self.quality_variables = {}  # Y variables - quality attributes

        # Collection of all output variables (for training)
        self.output_variables = {}

        # Track original column names from data source
        self._column_mapping = {}  # Maps internal names to original column names

        # Store normalization parameters and settings
        self.norm_params = {}
        self.normalized_variables = set()

        # Original data source (for reference)
        self._data_source = None
        self._variable_metadata = {}  # Store metadata about variables (units, descriptions, etc.)

    def set_time_column(self, time_column: str, data: Optional[pd.DataFrame] = None):
        """
        Set the time column and extract time points if data is provided.

        Args:
            time_column: Name of the time column
            data: Optional DataFrame containing the time data

        Returns:
            Self for method chaining
        """
        self.time_column = time_column

        if data is not None:
            if time_column in data.columns:
                # Convert to JAX array and handle potential NaN values
                self.time_points = jnp.array(data[time_column].fillna(method='ffill').values)
            else:
                raise ValueError(f"Time column '{time_column}' not found in data")

        return self

    def from_dataframe(self, df: pd.DataFrame, time_column: str, run_id_column: Optional[str] = None,
                       run_id: Optional[str] = None):
        """
        Load data from a pandas DataFrame.

        Args:
            df: DataFrame containing the data
            time_column: Name of the time column
            run_id_column: Optional column that contains run IDs
            run_id: Specific run ID to filter if run_id_column is provided

        Returns:
            Self for method chaining
        """
        # Store original data source
        self._data_source = df.copy()

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
        Add a variable from the data source.

        Args:
            column_name: Original column name in the data source
            variable_type: Type of variable (STATE, PARAMETER, CONTROL, FEED, QUALITY)
            internal_name: Optional name to use internally (defaults to column_name)
            is_output: Whether this variable is an output (target) variable
            calculate_rate: Whether to calculate rate of change for this variable
            data: Optional DataFrame to extract data from (uses stored data source if None)

        Returns:
            Self for method chaining
        """
        # Use internal name or column name
        internal_name = internal_name or column_name

        # Store mapping
        self._column_mapping[internal_name] = column_name

        # Get data source
        df = data if data is not None else self._data_source

        if df is None:
            raise ValueError("No data source available. Either provide data or use from_dataframe first.")

        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found in data. Skipping.")
            return self

        # Extract and convert to JAX array based on variable type
        if variable_type == VariableType.PARAMETER:
            # For parameters (Z variables), take the first non-NaN value
            values = df[column_name].dropna().values
            if len(values) > 0:
                self.parameter_variables[internal_name] = float(values[0]) if np.issubdtype(values.dtype,
                                                                                            np.number) else values[0]
        else:
            # For time series variables, convert to JAX array
            values = df[column_name].values
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
            elif variable_type == VariableType.QUALITY:
                self.quality_variables[internal_name] = variable_array
                if is_output:
                    self.output_variables[internal_name] = variable_array

            # Calculate rate if requested
            if calculate_rate and self.time_points is not None:
                rate_name = f"{internal_name}_rate"
                rate_values = self.calculate_derivative(variable_array)

                # Determine where to store the rate based on the original variable type
                if variable_type == VariableType.FEED:
                    self.feed_variables[rate_name] = rate_values
                else:
                    self.control_variables[rate_name] = rate_values

                self._column_mapping[rate_name] = f"Rate of {column_name}"

        return self

    def add_state(self, column_name: str, internal_name: Optional[str] = None,
                  is_output: bool = True, calculate_rate: bool = False):
        """
        Add a state variable (X variable in nomenclature).

        Args:
            column_name: Original column name in the data source
            internal_name: Optional name to use internally
            is_output: Whether this variable is an output variable
            calculate_rate: Whether to calculate rate of change

        Returns:
            Self for method chaining
        """
        return self.add_variable(column_name, VariableType.STATE, internal_name, is_output, calculate_rate)

    def add_parameter(self, column_name: str, internal_name: Optional[str] = None):
        """
        Add a parameter variable (Z variable in nomenclature) - constant throughout a run.

        Args:
            column_name: Original column name in the data source
            internal_name: Optional name to use internally

        Returns:
            Self for method chaining
        """
        return self.add_variable(column_name, VariableType.PARAMETER, internal_name)

    def add_control(self, column_name: str, internal_name: Optional[str] = None,
                    calculate_rate: bool = False):
        """
        Add a control variable (W variable in nomenclature).

        Args:
            column_name: Original column name in the data source
            internal_name: Optional name to use internally
            calculate_rate: Whether to calculate rate of change

        Returns:
            Self for method chaining
        """
        return self.add_variable(column_name, VariableType.CONTROL, internal_name, False, calculate_rate)

    def add_feed(self, column_name: str, internal_name: Optional[str] = None,
                 calculate_rate: bool = True):
        """
        Add a feed variable (F variable in nomenclature).
        Feed rates are often important, so calculate_rate defaults to True.

        Args:
            column_name: Original column name in the data source
            internal_name: Optional name to use internally
            calculate_rate: Whether to calculate rate of change (defaults to True for feeds)

        Returns:
            Self for method chaining
        """
        return self.add_variable(column_name, VariableType.FEED, internal_name, False, calculate_rate)

    def add_quality(self, column_name: str, internal_name: Optional[str] = None,
                    is_output: bool = True):
        """
        Add a quality attribute (Y variable in nomenclature).

        Args:
            column_name: Original column name in the data source
            internal_name: Optional name to use internally
            is_output: Whether this variable is an output variable (defaults to True)

        Returns:
            Self for method chaining
        """
        return self.add_variable(column_name, VariableType.QUALITY, internal_name, is_output)

    def calculate_derivative(self, values: Array) -> Array:
        """
        Calculate rate of change (derivative) of values.

        Args:
            values: Values at time points

        Returns:
            Array of rates of change
        """
        if self.time_points is None:
            raise ValueError("Time points not set. Cannot calculate derivative.")

        # Handle the case of empty arrays
        if len(values) <= 1:
            return jnp.array([])

        # Initialize rates array
        rates = jnp.zeros_like(values)

        # Calculate rates using forward differences
        for i in range(len(self.time_points) - 1):
            dt = self.time_points[i + 1] - self.time_points[i]
            if dt > 0:
                rates = rates.at[i].set((values[i + 1] - values[i]) / dt)

        # For the last point, use the previous rate
        rates = rates.at[-1].set(rates[-2] if len(rates) > 1 else 0.0)

        return rates

    def interpolate(self, selection: Union[str, List[str]] = "all", method: str = "linear"):
        """
        Interpolate variables to handle missing values.

        Args:
            selection: 'all', 'state', 'control', 'feed', 'quality', or a list of specific variable names
            method: Interpolation method ('linear', 'nearest', 'zero', 'slinear', etc.)

        Returns:
            Self for method chaining
        """
        if self.time_points is None:
            raise ValueError("Time points not set. Cannot interpolate.")

        # Determine which variables to interpolate
        variables_to_interpolate = {}

        if selection == "all":
            variables_to_interpolate.update(self.state_variables)
            variables_to_interpolate.update(self.control_variables)
            variables_to_interpolate.update(self.feed_variables)
            variables_to_interpolate.update(self.quality_variables)
        elif selection == "state":
            variables_to_interpolate.update(self.state_variables)
        elif selection == "control":
            variables_to_interpolate.update(self.control_variables)
        elif selection == "feed":
            variables_to_interpolate.update(self.feed_variables)
        elif selection == "quality":
            variables_to_interpolate.update(self.quality_variables)
        elif isinstance(selection, list):
            for var_name in selection:
                if var_name in self.state_variables:
                    variables_to_interpolate[var_name] = self.state_variables[var_name]
                elif var_name in self.control_variables:
                    variables_to_interpolate[var_name] = self.control_variables[var_name]
                elif var_name in self.feed_variables:
                    variables_to_interpolate[var_name] = self.feed_variables[var_name]
                elif var_name in self.quality_variables:
                    variables_to_interpolate[var_name] = self.quality_variables[var_name]

        # Perform interpolation
        for var_name, values in variables_to_interpolate.items():
            # Check if there are NaN values
            if jnp.isnan(values).any():
                # Create a mask for valid (non-NaN) values
                valid_mask = ~jnp.isnan(values)
                valid_indices = jnp.where(valid_mask)[0]

                if len(valid_indices) < 2:
                    print(f"Warning: Not enough valid points to interpolate {var_name}. Skipping.")
                    continue

                # Get valid time points and values
                valid_times = self.time_points[valid_indices]
                valid_values = values[valid_indices]

                # Create an interpolation function
                from scipy.interpolate import interp1d
                interp_func = interp1d(valid_times, valid_values, kind=method,
                                       bounds_error=False, fill_value="extrapolate")

                # Apply interpolation to all time points
                interpolated_values = interp_func(self.time_points)
                interpolated_values = jnp.array(interpolated_values)

                # Update the variable with interpolated values
                if var_name in self.state_variables:
                    self.state_variables[var_name] = interpolated_values
                    # Also update output variables if needed
                    if var_name in self.output_variables:
                        self.output_variables[var_name] = interpolated_values
                elif var_name in self.control_variables:
                    self.control_variables[var_name] = interpolated_values
                elif var_name in self.feed_variables:
                    self.feed_variables[var_name] = interpolated_values
                elif var_name in self.quality_variables:
                    self.quality_variables[var_name] = interpolated_values
                    # Also update output variables if needed
                    if var_name in self.output_variables:
                        self.output_variables[var_name] = interpolated_values

        return self

    def normalize(self, selection: Union[str, List[str]] = "all", norm_params: Optional[Dict] = None,
                  method: str = "standard"):
        """
        Normalize variables.

        Args:
            selection: 'all', 'state', 'control', 'feed', 'quality', or a list of specific variable names
            norm_params: Optional normalization parameters to use (calculates from data if None)
            method: Normalization method ('standard', 'minmax')

        Returns:
            Self for method chaining
        """
        # Determine which variables to normalize
        variables_to_normalize = {}

        if selection == "all":
            variables_to_normalize.update(self.state_variables)
            variables_to_normalize.update(self.control_variables)
            variables_to_normalize.update(self.feed_variables)
            variables_to_normalize.update(self.quality_variables)
        elif selection == "state":
            variables_to_normalize.update(self.state_variables)
        elif selection == "control":
            variables_to_normalize.update(self.control_variables)
        elif selection == "feed":
            variables_to_normalize.update(self.feed_variables)
        elif selection == "quality":
            variables_to_normalize.update(self.quality_variables)
        elif isinstance(selection, list):
            for var_name in selection:
                if var_name in self.state_variables:
                    variables_to_normalize[var_name] = self.state_variables[var_name]
                elif var_name in self.control_variables:
                    variables_to_normalize[var_name] = self.control_variables[var_name]
                elif var_name in self.feed_variables:
                    variables_to_normalize[var_name] = self.feed_variables[var_name]
                elif var_name in self.quality_variables:
                    variables_to_normalize[var_name] = self.quality_variables[var_name]

        # Calculate normalization parameters if not provided
        if norm_params is None:
            norm_params = {}
            for var_name, values in variables_to_normalize.items():
                # Filter out NaN values
                valid_values = values[~jnp.isnan(values)] if jnp.isnan(values).any() else values

                if len(valid_values) == 0:
                    print(f"Warning: No valid values to normalize {var_name}. Skipping.")
                    continue

                if method == "standard":
                    # Standard scaling
                    mean_val = float(jnp.mean(valid_values))
                    std_val = float(jnp.std(valid_values))

                    # Store normalization parameters
                    norm_params[f"{var_name}_mean"] = mean_val
                    norm_params[f"{var_name}_std"] = max(std_val, 1e-8)  # Avoid division by zero

                elif method == "minmax":
                    # Min-max scaling
                    min_val = float(jnp.min(valid_values))
                    max_val = float(jnp.max(valid_values))

                    # Store normalization parameters
                    norm_params[f"{var_name}_min"] = min_val
                    norm_params[
                        f"{var_name}_max"] = max_val if max_val > min_val else min_val + 1.0  # Avoid division by zero

        # Store normalization parameters
        self.norm_params.update(norm_params)

        # Add normalized variables to the tracking set
        self.normalized_variables.update(variables_to_normalize.keys())

        return self

    def get_initial_state(self) -> Dict[str, float]:
        """Get the initial values of all state variables."""
        initial_state = {}
        for name, values in self.state_variables.items():
            if len(values) > 0:
                # Use the first non-NaN value, or 0.0 if all are NaN
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

    def get_all_input_variables(self) -> Dict[str, Array]:
        """Get all input variables (controls and feeds) combined."""
        all_inputs = {}
        all_inputs.update(self.control_variables)
        all_inputs.update(self.feed_variables)
        return all_inputs

    def prepare_for_training(self) -> Dict:
        """
        Prepare the dataset for training by converting to the format
        expected by the training functions.
        """
        # Get initial state
        initial_state = self.get_initial_state()

        # Prepare time-dependent inputs
        time_dependent_inputs = {}

        # Add control variables
        for name, values in self.control_variables.items():
            time_dependent_inputs[name] = (self.time_points, values)

        # Add feed variables
        for name, values in self.feed_variables.items():
            time_dependent_inputs[name] = (self.time_points, values)

        # Prepare static inputs (parameters)
        static_inputs = {**self.parameter_variables}

        # Prepare dataset
        dataset = {
            'times': self.time_points,
            'initial_state': initial_state,
            'time_dependent_inputs': time_dependent_inputs,
            'static_inputs': static_inputs
            # Remove run_id to avoid JAX jit compilation issues
        }

        # Add true outputs for loss calculation
        for name, values in self.output_variables.items():
            dataset[f'{name}_true'] = values

        return dataset

    @classmethod
    def load_datasets_from_dataframe(cls, df: pd.DataFrame, time_column: str,
                                     run_id_column: Optional[str] = None,
                                     max_runs: Optional[int] = None) -> List['TimeSeriesData']:
        """
        Load multiple datasets from a DataFrame, splitting by run ID.

        Args:
            df: DataFrame containing the data
            time_column: Name of the time column
            run_id_column: Column that contains run IDs
            max_runs: Maximum number of runs to load

        Returns:
            List of TimeSeriesData objects
        """
        datasets = []

        if run_id_column is not None and run_id_column in df.columns:
            # Get unique run IDs
            run_ids = df[run_id_column].unique()

            # Limit to max_runs if specified
            if max_runs is not None:
                run_ids = run_ids[:max_runs]

            # Create a dataset for each run
            for run_id in run_ids:
                dataset = cls(run_id=str(run_id))
                dataset.from_dataframe(df, time_column, run_id_column, run_id)
                datasets.append(dataset)
        else:
            # If no run ID column, treat the entire dataset as a single run
            dataset = cls()
            dataset.from_dataframe(df, time_column)
            datasets.append(dataset)

        return datasets

    @classmethod
    def calculate_normalization_parameters(cls, datasets: List['TimeSeriesData'],
                                           selection: Union[str, List[str]] = "all",
                                           method: str = "standard") -> Dict[str, float]:
        """
        Calculate normalization parameters for variables across datasets.

        Args:
            datasets: List of TimeSeriesData objects
            selection: 'all', 'state', 'control', 'feed', 'quality', or list of variable names
            method: Normalization method ('standard', 'minmax')

        Returns:
            Dictionary of normalization parameters
        """
        # Initialize containers for variables
        all_values = {}

        for dataset in datasets:
            # Determine which variables to include based on selection
            if selection == "all":
                variables_dicts = [
                    dataset.state_variables,
                    dataset.control_variables,
                    dataset.feed_variables,
                    dataset.quality_variables
                ]
            elif selection == "state":
                variables_dicts = [dataset.state_variables]
            elif selection == "control":
                variables_dicts = [dataset.control_variables]
            elif selection == "feed":
                variables_dicts = [dataset.feed_variables]
            elif selection == "quality":
                variables_dicts = [dataset.quality_variables]
            elif isinstance(selection, list):
                # For specific variable names, we need to check each dictionary
                variables_dicts = []
                selected_vars = {}
                for var_name in selection:
                    for var_dict in [dataset.state_variables, dataset.control_variables,
                                     dataset.feed_variables, dataset.quality_variables]:
                        if var_name in var_dict:
                            selected_vars[var_name] = var_dict[var_name]
                variables_dicts = [selected_vars]
            else:
                raise ValueError(f"Invalid selection: {selection}")

            # Collect values from each dictionary
            for variables_dict in variables_dicts:
                for var_name, values in variables_dict.items():
                    if var_name not in all_values:
                        all_values[var_name] = []

                    # Filter out NaN values
                    valid_values = values[~jnp.isnan(values)] if jnp.isnan(values).any() else values
                    if len(valid_values) > 0:
                        all_values[var_name].append(valid_values)

        # Calculate normalization parameters
        norm_params = {}
        for var_name, value_list in all_values.items():
            if value_list and len(value_list) > 0:
                # Concatenate all values
                all_data = jnp.concatenate(value_list)

                if method == "standard":
                    # Standard scaling
                    mean_val = float(jnp.mean(all_data))
                    std_val = float(jnp.std(all_data))

                    # Store normalization parameters
                    norm_params[f"{var_name}_mean"] = mean_val
                    norm_params[f"{var_name}_std"] = max(std_val, 1e-8)  # Avoid division by zero

                elif method == "minmax":
                    # Min-max scaling
                    min_val = float(jnp.min(all_data))
                    max_val = float(jnp.max(all_data))

                    # Store normalization parameters
                    norm_params[f"{var_name}_min"] = min_val
                    norm_params[
                        f"{var_name}_max"] = max_val if max_val > min_val else min_val + 1.0  # Avoid division by zero

        return norm_params

    @classmethod
    def apply_normalization_parameters(cls, datasets: List['TimeSeriesData'],
                                       norm_params: Dict[str, float],
                                       selection: Union[str, List[str]] = "all"):
        """
        Apply normalization parameters to datasets.

        Args:
            datasets: List of TimeSeriesData objects
            norm_params: Dictionary of normalization parameters
            selection: 'all', 'state', 'control', 'feed', 'quality', or list of variable names
        """
        for dataset in datasets:
            # Store normalization parameters
            dataset.norm_params.update(norm_params)

            # Track which variables are normalized based on selection
            if selection == "all":
                dataset.normalized_variables.update(dataset.state_variables.keys())
                dataset.normalized_variables.update(dataset.control_variables.keys())
                dataset.normalized_variables.update(dataset.feed_variables.keys())
                dataset.normalized_variables.update(dataset.quality_variables.keys())
            elif selection == "state":
                dataset.normalized_variables.update(dataset.state_variables.keys())
            elif selection == "control":
                dataset.normalized_variables.update(dataset.control_variables.keys())
            elif selection == "feed":
                dataset.normalized_variables.update(dataset.feed_variables.keys())
            elif selection == "quality":
                dataset.normalized_variables.update(dataset.quality_variables.keys())
            elif isinstance(selection, list):
                dataset.normalized_variables.update(selection)


def prepare_datasets_for_training(datasets: List[TimeSeriesData]) -> List[Dict]:
    """
    Prepare multiple datasets for training.

    Args:
        datasets: List of TimeSeriesData objects

    Returns:
        List of dictionaries in the format expected by the training functions
    """
    return [dataset.prepare_for_training() for dataset in datasets]