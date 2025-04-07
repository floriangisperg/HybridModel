"""
Utilities for data management in hybrid modeling.

This module provides helper classes and functions for working with data
in the hybrid modeling framework.
"""
from typing import List, Optional, Tuple, Dict, Any
from .data import VariableType


class VariableRegistry:
    """
    A registry for declaring data variables with a fluent, self-documenting API.
    This class simplifies the process of creating variable definitions for the DatasetManager.
    """

    def __init__(self):
        """Initialize an empty variable registry."""
        self._variables = []

    def add_state(self, column_name: str, internal_name: Optional[str] = None,
                  is_output: bool = True, calculate_rate: bool = False) -> 'VariableRegistry':
        """
        Add a state variable (like cell density, product concentration).

        Args:
            column_name: Name of the column in the dataset
            internal_name: Optional internal name for the variable (if None, uses column_name)
            is_output: Whether this variable is an output/target for training
            calculate_rate: Whether to calculate rate of change for this variable

        Returns:
            Self for method chaining
        """
        self._variables.append((column_name, VariableType.STATE, internal_name, is_output, calculate_rate))
        return self

    def add_parameter(self, column_name: str, internal_name: Optional[str] = None) -> 'VariableRegistry':
        """
        Add a parameter variable (constant within a run).

        Args:
            column_name: Name of the column in the dataset
            internal_name: Optional internal name for the variable

        Returns:
            Self for method chaining
        """
        self._variables.append((column_name, VariableType.PARAMETER, internal_name, False, False))
        return self

    def add_control(self, column_name: str, internal_name: Optional[str] = None,
                    calculate_rate: bool = False) -> 'VariableRegistry':
        """
        Add a control variable (like temperature, pH).

        Args:
            column_name: Name of the column in the dataset
            internal_name: Optional internal name for the variable
            calculate_rate: Whether to calculate rate of change for this variable

        Returns:
            Self for method chaining
        """
        self._variables.append((column_name, VariableType.CONTROL, internal_name, False, calculate_rate))
        return self

    def add_feed(self, column_name: str, internal_name: Optional[str] = None,
                 calculate_rate: bool = True) -> 'VariableRegistry':
        """
        Add a feed variable (like nutrient feed, base addition).

        Args:
            column_name: Name of the column in the dataset
            internal_name: Optional internal name for the variable
            calculate_rate: Whether to calculate rate of change (defaults to True for feeds)

        Returns:
            Self for method chaining
        """
        self._variables.append((column_name, VariableType.FEED, internal_name, False, calculate_rate))
        return self

    def to_list(self) -> List[Tuple]:
        """
        Convert the registry to a list of variable definition tuples.

        Returns:
            List of variable definition tuples compatible with DatasetManager.add_variables()
        """
        return self._variables


def create_time_series_dataset(data: Dict[str, Any],
                               time_column: str,
                               variable_registry: VariableRegistry,
                               run_id_column: Optional[str] = None,
                               run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a time series dataset in the format required by the hybrid modeling framework.

    Args:
        data: DataFrame or Dictionary containing time series data
        time_column: Name of the column containing time values
        variable_registry: VariableRegistry with variable definitions
        run_id_column: Optional column name containing run IDs
        run_id: Optional specific run ID to filter by

    Returns:
        Dictionary with the dataset in the format required for training
    """
    # This is a placeholder for future implementation
    # The function would convert raw data to the required format without using DatasetManager
    pass