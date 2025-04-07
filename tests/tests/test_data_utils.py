"""Tests for the data utilities module."""
import pytest
from hybrid_models import VariableRegistry
from hybrid_models.data import VariableType


def test_variable_registry_creation():
    """Test creating a VariableRegistry and adding variables."""
    registry = VariableRegistry()

    # Add different types of variables
    registry.add_state('Biomass', 'X')
    registry.add_parameter('Reactor_Size', 'V')
    registry.add_control('Temperature', 'T', calculate_rate=True)
    registry.add_feed('Glucose', 'G')

    # Convert to list and check
    variables = registry.to_list()

    # Check number of variables
    assert len(variables) == 4

    # Check each variable
    assert variables[0] == ('Biomass', VariableType.STATE, 'X', True, False)
    assert variables[1] == ('Reactor_Size', VariableType.PARAMETER, 'V', False, False)
    assert variables[2] == ('Temperature', VariableType.CONTROL, 'T', False, True)
    assert variables[3] == ('Glucose', VariableType.FEED, 'G', False, True)


def test_variable_registry_method_chaining():
    """Test the fluent interface with method chaining."""
    registry = VariableRegistry()

    # Use method chaining
    registry.add_state('Biomass', 'X') \
        .add_state('Product', 'P') \
        .add_control('Temperature', 'T')

    variables = registry.to_list()

    # Check that all variables were added
    assert len(variables) == 3
    assert variables[0][0] == 'Biomass'
    assert variables[1][0] == 'Product'
    assert variables[2][0] == 'Temperature'


def test_variable_registry_defaults():
    """Test the default values for variable options."""
    registry = VariableRegistry()

    # Add variables with default options
    registry.add_state('Biomass')  # No internal name specified
    registry.add_feed('Glucose', calculate_rate=False)  # Override default for calculate_rate

    variables = registry.to_list()

    # Check defaults
    assert variables[0] == ('Biomass', VariableType.STATE, None, True, False)
    assert variables[1] == ('Glucose', VariableType.FEED, None, False, False)