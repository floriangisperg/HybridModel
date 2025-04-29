import jax
import jax.numpy as jnp
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# **** ADD Tuple and Union to imports ****
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from jaxtyping import Array, Float, PyTree
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)  # Define the logger object


# Keep VariableType Enum as is
class VariableType(Enum):
    STATE = "state"
    PARAMETER = "parameter"
    CONTROL = "control"
    FEED = "feed"


@dataclass
class TimeSeriesData:
    """
    Handles time-series data for hybrid modeling, supporting variables
    with potentially different time frequencies.
    """

    run_id: Optional[str] = None

    # Sparse time points primarily used for state variables and evaluation
    time_points: Optional[Array] = None
    time_column: Optional[str] = None  # Name of the original time column

    # State variables are assumed to be aligned with sparse time_points
    state_variables: Dict[str, Array] = field(default_factory=dict)
    # Parameters are static within a run
    parameter_variables: Dict[str, Any] = field(default_factory=dict)

    # **** MODIFIED: Store (times, values) tuples for potentially dense variables ****
    control_variables: Dict[str, Tuple[Array, Array]] = field(default_factory=dict)
    feed_variables: Dict[str, Tuple[Array, Array]] = field(default_factory=dict)

    # Output variables for training loss (aligned with sparse time_points)
    output_variables: Dict[str, Array] = field(default_factory=dict)

    # Original column mappings
    _column_mapping: Dict[str, str] = field(default_factory=dict)

    def set_time_column(self, time_column: str, data: Optional[pd.DataFrame] = None):
        """
        Set the main (potentially sparse) time column and extract time points
        if data is provided. This is typically aligned with state measurements.
        """
        self.time_column = time_column

        if data is not None and time_column in data.columns:
            # Extract UNIQUE, SORTED time points for the main sparse grid
            # Drop NaNs in time column before processing
            unique_times = data[time_column].dropna().unique()
            # Convert to JAX array and ensure it's sorted
            self.time_points = jnp.sort(jnp.array(unique_times))
            if len(self.time_points) < len(data[time_column].dropna()):
                print(
                    f"Warning: Duplicate time points found in main time column '{time_column}'. Using unique sorted times."
                )

        return self

    def from_dataframe(
        self,
        df: pd.DataFrame,
        time_column: str,
        run_id_column: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """
        Load base data (run_id, main time points) from a pandas DataFrame.
        Actual variable loading happens in add_variable.
        """
        filtered_df = df.copy()  # Work on a copy
        # Handle run ID filtering
        if run_id_column is not None and run_id is not None:
            if run_id_column in filtered_df.columns:
                # Ensure consistent type for comparison if run_id is numeric
                try:
                    run_id_val = type(filtered_df[run_id_column].iloc[0])(run_id)
                except (ValueError, TypeError):
                    run_id_val = str(run_id)  # Fallback to string comparison

                filtered_df = filtered_df[filtered_df[run_id_column] == run_id_val]
                if len(filtered_df) == 0:
                    raise ValueError(
                        f"No data found for run_id={run_id} in column {run_id_column}"
                    )
                self.run_id = str(run_id)
            else:
                print(
                    f"Warning: run_id_column '{run_id_column}' not found in DataFrame."
                )

        # Set the main sparse time points from the (potentially filtered) data
        self.set_time_column(time_column, filtered_df)

        return self  # Return self to allow chaining if needed

    # **** MODIFIED: add_variable handles dense/sparse storage ****
    def add_variable(
        self,
        column_name: str,
        variable_type: VariableType,
        internal_name: Optional[str] = None,
        is_output: bool = False,
        calculate_rate: bool = False,
        data: Optional[pd.DataFrame] = None,
        main_time_column: Optional[str] = None,
    ):  # Need main time column name
        """
        Add a variable, handling dense storage for controls/feeds
        and sparse storage for states/parameters.
        Requires the original DataFrame (`data`) and the name of the time column
        (`main_time_column`) associated with that DataFrame.
        """
        if data is None:
            raise ValueError(
                "`data` (original DataFrame) must be provided to add_variable."
            )
        if main_time_column is None:
            # Try to use the one set previously, otherwise raise error
            main_time_column = self.time_column
            if main_time_column is None:
                raise ValueError(
                    "`main_time_column` must be provided or set previously via `set_time_column`."
                )

        # Use internal name or column name
        internal_name = internal_name or column_name

        # Store mapping
        self._column_mapping[internal_name] = column_name

        if column_name not in data.columns:
            print(
                f"Warning: Column '{column_name}' not found in data. Skipping variable '{internal_name}'."
            )
            return self
        if main_time_column not in data.columns:
            print(
                f"Warning: Main time column '{main_time_column}' not found in data. Skipping variable '{internal_name}'."
            )
            return self

        # Prepare data subset for this variable (handle potential NaNs)
        # Select relevant columns and drop rows where BOTH time AND value are NaN
        var_df = data[[main_time_column, column_name]].dropna(
            subset=[main_time_column, column_name], how="all"
        )
        # Drop rows where the specific variable value is NaN, but keep time
        var_df = var_df.dropna(subset=[column_name])
        # Sort by time for interpolation and rate calculation
        var_df = var_df.sort_values(by=main_time_column)

        if len(var_df) == 0:
            print(
                f"Warning: No valid data points found for column '{column_name}' after dropping NaNs. Skipping variable '{internal_name}'."
            )
            return self

        # Extract time and value arrays for this specific variable
        var_times = jnp.array(var_df[main_time_column].values)
        var_values = jnp.array(var_df[column_name].values)

        # --- Logic based on VariableType ---
        if variable_type == VariableType.PARAMETER:
            # Parameters are constant, take the first value found
            self.parameter_variables[internal_name] = (
                float(var_values[0])
                if jnp.issubdtype(var_values.dtype, jnp.number)
                else var_values[0]
            )
        elif variable_type == VariableType.STATE:
            # States align with the main sparse time grid (self.time_points)
            if self.time_points is None:
                raise ValueError(
                    "Cannot add STATE variable before `time_points` are set (e.g., via `from_dataframe`)."
                )
            # Interpolate state values onto the main sparse time grid
            # Use linear interpolation; nearest might miss trends between sparse points
            from hybrid_models.utils import (
                interp_linear,
            )  # Import here or ensure available

            aligned_values = interp_linear(self.time_points, var_times, var_values)
            self.state_variables[internal_name] = aligned_values
            if is_output:
                self.output_variables[internal_name] = aligned_values
            # Optionally calculate rate on the *sparse* grid if needed for loss/analysis later
            if calculate_rate:
                rate_name = f"{internal_name}_rate"
                from hybrid_models.utils import calculate_rate as calc_rate_func

                rate_values = calc_rate_func(self.time_points, aligned_values)
                # Store rate sparsely as well (e.g., if needed as an output target)
                self.state_variables[rate_name] = rate_values
                self._column_mapping[rate_name] = f"Rate of {column_name} (Sparse)"

        elif variable_type in [VariableType.CONTROL, VariableType.FEED]:
            # Controls and Feeds store the dense (time, value) tuple
            target_dict = (
                self.control_variables
                if variable_type == VariableType.CONTROL
                else self.feed_variables
            )
            target_dict[internal_name] = (var_times, var_values)

            # Calculate rate on the DENSE grid if requested
            if calculate_rate:
                rate_name = f"{internal_name}_rate"
                from hybrid_models.utils import calculate_rate as calc_rate_func

                rate_values = calc_rate_func(
                    var_times, var_values
                )  # Calculate on dense data
                # Store rate as a dense (time, value) tuple as well
                target_dict[rate_name] = (var_times, rate_values)
                self._column_mapping[rate_name] = f"Rate of {column_name} (Dense)"
        else:
            print(
                f"Warning: Unknown VariableType '{variable_type}'. Skipping variable '{internal_name}'."
            )

        return self

    # --- Update Helper methods to call the modified add_variable ---
    # Need to pass 'data' and 'main_time_column' down
    def add_state(
        self,
        column_name: str,
        internal_name: Optional[str] = None,
        is_output: bool = True,
        calculate_rate: bool = False,
        data: Optional[pd.DataFrame] = None,
        main_time_column: Optional[str] = None,
    ):
        return self.add_variable(
            column_name,
            VariableType.STATE,
            internal_name,
            is_output,
            calculate_rate,
            data,
            main_time_column,
        )

    def add_parameter(
        self,
        column_name: str,
        internal_name: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        main_time_column: Optional[str] = None,
    ):
        return self.add_variable(
            column_name,
            VariableType.PARAMETER,
            internal_name,
            False,
            False,
            data,
            main_time_column,
        )

    def add_control(
        self,
        column_name: str,
        internal_name: Optional[str] = None,
        calculate_rate: bool = False,
        data: Optional[pd.DataFrame] = None,
        main_time_column: Optional[str] = None,
    ):
        return self.add_variable(
            column_name,
            VariableType.CONTROL,
            internal_name,
            False,
            calculate_rate,
            data,
            main_time_column,
        )

    def add_feed(
        self,
        column_name: str,
        internal_name: Optional[str] = None,
        calculate_rate: bool = True,
        data: Optional[pd.DataFrame] = None,
        main_time_column: Optional[str] = None,
    ):
        # Note: Default calculate_rate=True for feeds might often make sense
        return self.add_variable(
            column_name,
            VariableType.FEED,
            internal_name,
            False,
            calculate_rate,
            data,
            main_time_column,
        )

    def get_initial_state(self) -> Dict[str, float]:
        """Get the initial values of all state variables (from sparse grid)."""
        if self.time_points is None or len(self.time_points) == 0:
            raise ValueError(
                "Cannot get initial state: `time_points` is not set or is empty."
            )
        initial_state = {}
        # Initial state corresponds to the first point in the sparse time_points grid
        t0 = self.time_points[0]
        for name, values in self.state_variables.items():
            if name.endswith("_rate"):
                continue  # Skip rate variables

            if values is not None and len(values) > 0:
                # The values should already be aligned with self.time_points
                initial_val = values[0]
                # Handle potential NaN at the first point (though interpolation should minimize this)
                if jnp.isnan(initial_val):
                    # Find first non-nan value
                    non_nan_indices = jnp.where(~jnp.isnan(values))[0]
                    if len(non_nan_indices) > 0:
                        initial_state[name] = float(values[non_nan_indices[0]])
                        print(
                            f"Warning: First value for state '{name}' was NaN. Using first non-NaN value: {initial_state[name]}"
                        )
                    else:
                        print(
                            f"Warning: All values for state {name} are NaN. Using 0.0 as initial state."
                        )
                        initial_state[name] = 0.0
                else:
                    initial_state[name] = float(initial_val)
            else:
                print(
                    f"Warning: State variable '{name}' has no values. Using 0.0 as initial state."
                )
                initial_state[name] = 0.0
        return initial_state

    # **** MODIFIED: prepare_for_training handles dense tuples ****
    def prepare_for_training(self) -> Dict:
        """
        Prepare the dataset into the dictionary format expected by the
        training and solving functions.
        """
        if self.time_points is None:
            raise ValueError("Cannot prepare for training: `time_points` not set.")

        initial_state = self.get_initial_state()

        # Prepare time-dependent inputs (controls, feeds)
        # These now contain the (dense_times, dense_values) tuples
        time_dependent_inputs = {}
        for name, times_values_tuple in {
            **self.control_variables,
            **self.feed_variables,
        }.items():
            time_dependent_inputs[name] = times_values_tuple  # Pass the tuple directly

        # Prepare static inputs (parameters)
        static_inputs = {**self.parameter_variables}

        # Prepare dataset dictionary
        dataset = {
            # 'times': the sparse evaluation times for loss calculation
            "times": self.time_points,
            "initial_state": initial_state,
            "time_dependent_inputs": time_dependent_inputs,
            "static_inputs": static_inputs,
        }

        # Add true outputs for loss calculation (aligned with sparse 'times')
        for name, values in self.output_variables.items():
            # Ensure output variables have same length as sparse time points
            if len(values) != len(self.time_points):
                print(
                    f"Warning: Length mismatch for output variable '{name}' ({len(values)}) and sparse time points ({len(self.time_points)}). This might indicate an issue in data alignment."
                )
                # Attempt to truncate/pad? Or raise error? For now, just add.
                # Depending on use case, might need interpolation here too, but typically outputs match sparse states.
            dataset[f"{name}_true"] = values

        return dataset


# --- DatasetManager Modifications ---


class DatasetManager:
    """
    Manager for handling multiple datasets with train/test split functionality.
    Now supports dense time series for controls/feeds.
    """

    def __init__(self):
        self.train_datasets: List[TimeSeriesData] = []
        self.test_datasets: List[TimeSeriesData] = []
        self.norm_params: Dict[str, float] = {}
        self._main_time_column: Optional[str] = None  # Store the main time column name

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        time_column: str,
        run_id_column: Optional[str] = None,
        train_run_ids: Optional[List] = None,
        test_run_ids: Optional[List] = None,
        train_ratio: float = 0.8,
    ):
        """Loads data, creating TimeSeriesData objects for train/test splits."""
        self._main_time_column = time_column  # Store for later use in add_variables

        if run_id_column and run_id_column in df.columns:
            all_run_ids = df[run_id_column].unique()
            train_ids, test_ids = self._determine_split_ids(
                all_run_ids, train_run_ids, test_run_ids, train_ratio
            )

            log.info(
                f"Splitting runs. Training: {len(train_ids)}, Testing: {len(test_ids)}"
            )

            # Load training datasets
            for run_id in train_ids:
                dataset = TimeSeriesData()
                # Pass the full df, from_dataframe will filter by run_id
                dataset.from_dataframe(df, time_column, run_id_column, run_id)
                self.train_datasets.append(dataset)

            # Load test datasets
            for run_id in test_ids:
                dataset = TimeSeriesData()
                dataset.from_dataframe(df, time_column, run_id_column, run_id)
                self.test_datasets.append(dataset)
        else:
            log.warning(
                "No run_id_column provided or found. Treating entire DataFrame as a single training dataset."
            )
            dataset = TimeSeriesData()
            dataset.from_dataframe(df, time_column)
            self.train_datasets.append(dataset)

        return self

    def _determine_split_ids(
        self, all_run_ids, train_run_ids, test_run_ids, train_ratio
    ):
        """Helper to determine train/test run IDs based on input."""
        if train_run_ids is None and test_run_ids is None:
            # Split based on train_ratio
            np.random.shuffle(all_run_ids)  # Shuffle for random split
            n_total = len(all_run_ids)
            n_train = max(1, int(n_total * train_ratio)) if n_total > 1 else n_total
            train_ids = all_run_ids[:n_train]
            test_ids = all_run_ids[n_train:]
        elif train_run_ids is None:
            # Use all runs not in test_run_ids for training
            test_ids = list(test_run_ids)
            train_ids = [rid for rid in all_run_ids if rid not in test_ids]
        elif test_run_ids is None:
            # Use all runs not in train_run_ids for testing
            train_ids = list(train_run_ids)
            test_ids = [rid for rid in all_run_ids if rid not in train_ids]
        else:
            # Both provided explicitly
            train_ids = list(train_run_ids)
            test_ids = list(test_run_ids)
            # Optional: Check for overlap or missing runs
            overlap = set(train_ids) & set(test_ids)
            if overlap:
                print(
                    f"Warning: Overlap detected between train_run_ids and test_run_ids: {overlap}"
                )
            missing = set(all_run_ids) - set(train_ids) - set(test_ids)
            if missing:
                print(
                    f"Warning: Some runs are in neither train nor test sets: {missing}"
                )

        return train_ids, test_ids

    # **** MODIFIED: add_variables passes necessary info to TimeSeriesData.add_variable ****
    def add_variables(self, variable_definitions: List[Tuple], data: pd.DataFrame):
        """
        Add variables to all loaded datasets based on definitions.

        Args:
            variable_definitions: List of tuples compatible with VariableRegistry.to_list()
                                   e.g., (column_name, variable_type, internal_name, is_output, calculate_rate)
            data: The *original, complete* pandas DataFrame used for loading.
                  This is needed to access the full dense time series for controls/feeds.
        """
        if self._main_time_column is None:
            raise RuntimeError(
                "DatasetManager's main time column not set. Call load_from_dataframe first."
            )
        if data is None:
            raise ValueError(
                "The original complete DataFrame `data` must be provided to `add_variables`."
            )

        # Process each dataset (train and test)
        for dataset in self.train_datasets + self.test_datasets:
            # Filter the original DataFrame for the specific run_id of the current dataset
            # This provides the necessary context (full data for the run) to add_variable
            run_data = data
            if dataset.run_id is not None:
                run_id_column = self._find_run_id_column(data, dataset.run_id)
                if run_id_column:
                    # Ensure type consistency for filtering
                    try:
                        run_id_val = type(data[run_id_column].iloc[0])(dataset.run_id)
                    except (ValueError, TypeError):
                        run_id_val = str(dataset.run_id)
                    run_data = data[data[run_id_column] == run_id_val].copy()
                else:
                    print(
                        f"Warning: Could not definitively find run_id_column for run_id '{dataset.run_id}' when adding variables. Using full DataFrame."
                    )

            # Add each variable defined in the list
            for var_def in variable_definitions:
                # Unpack definition tuple carefully
                col, vtype = var_def[0], var_def[1]
                internal = var_def[2] if len(var_def) > 2 else None
                output = (
                    var_def[3] if len(var_def) > 3 else (vtype == VariableType.STATE)
                )  # Default output=True for STATE
                rate = (
                    var_def[4] if len(var_def) > 4 else (vtype == VariableType.FEED)
                )  # Default rate=True for FEED

                # Call the dataset's add_variable method, passing the filtered run_data
                dataset.add_variable(
                    column_name=col,
                    variable_type=vtype,
                    internal_name=internal,
                    is_output=output,
                    calculate_rate=rate,
                    data=run_data,  # Pass the DataFrame for this run
                    main_time_column=self._main_time_column,  # Pass time column name
                )
        return self

    def _find_run_id_column(self, df, run_id_value):
        """Helper to find the column matching the run_id (used if not explicitly stored)."""
        for col in df.columns:
            # Check if the run_id_value (as string) exists in the column (as string)
            if df[col].astype(str).eq(str(run_id_value)).any():
                return col
        return None

    def calculate_norm_params(self, variable_names: Optional[List[str]] = None):
        """
        Calculate normalization parameters from training datasets only.
        Handles both sparse (Array) and dense (Tuple[Array, Array]) data.
        """
        all_values_dict = {}

        # Gather all *values* (not times) from training datasets
        for dataset in self.train_datasets:
            # Collect sparse state variables
            for name, values_arr in dataset.state_variables.items():
                if name.endswith("_rate"):
                    continue  # Skip rates for now unless explicitly requested
                if variable_names is None or name in variable_names:
                    if name not in all_values_dict:
                        all_values_dict[name] = []
                    valid_values = values_arr[~jnp.isnan(values_arr)]
                    if len(valid_values) > 0:
                        all_values_dict[name].append(valid_values)

            # Collect dense control/feed variables (use the value array from tuple)
            for dense_dict in [dataset.control_variables, dataset.feed_variables]:
                for name, (times_arr, values_arr) in dense_dict.items():
                    if name.endswith("_rate"):
                        continue  # Skip rates for now unless explicitly requested
                    if variable_names is None or name in variable_names:
                        if name not in all_values_dict:
                            all_values_dict[name] = []
                        valid_values = values_arr[~jnp.isnan(values_arr)]
                        if len(valid_values) > 0:
                            all_values_dict[name].append(valid_values)

        # Calculate normalization parameters (mean, std)
        self.norm_params = {}
        for name, list_of_arrays in all_values_dict.items():
            if list_of_arrays:
                all_data = jnp.concatenate(list_of_arrays)
                if len(all_data) > 0:
                    mean_val = float(jnp.mean(all_data))
                    std_val = float(jnp.std(all_data))
                    self.norm_params[f"{name}_mean"] = mean_val
                    self.norm_params[f"{name}_std"] = max(
                        std_val, 1e-8
                    )  # Avoid zero std

        return self

    # prepare_training_data and prepare_test_data should work correctly
    # as they rely on TimeSeriesData.prepare_for_training() which was updated.
    def prepare_training_data(self) -> List[Dict]:
        """Prepare training datasets for model training."""
        prepared = []
        for i, dataset in enumerate(self.train_datasets):
            try:
                prepared.append(dataset.prepare_for_training())
            except Exception as e:
                print(
                    f"Error preparing training dataset {i} (run_id: {dataset.run_id}): {e}"
                )
                # Decide whether to skip or raise
                # raise e
        return prepared

    def prepare_test_data(self) -> List[Dict]:
        """Prepare test datasets for model evaluation."""
        prepared = []
        for i, dataset in enumerate(self.test_datasets):
            try:
                prepared.append(dataset.prepare_for_training())
            except Exception as e:
                print(
                    f"Error preparing test dataset {i} (run_id: {dataset.run_id}): {e}"
                )
                # Decide whether to skip or raise
                # raise e
        return prepared
