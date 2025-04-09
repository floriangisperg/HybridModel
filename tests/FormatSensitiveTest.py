"""Examples of format-sensitive testing approaches."""
import re
import jax.numpy as jnp
import pytest


def test_flexible_string_representation():
    """Demonstrate flexible testing of string representations."""
    # Example string representation
    config_str = "Solver: DOPRI5, Controller: pid, rtol: 0.0001, atol: 1e-08, max_steps: 100000"

    # BAD: Fragile exact matches - prone to breaking with minor formatting changes
    # assert "rtol: 1e-4" in config_str  # This will fail if formatted as 0.0001!

    # GOOD: Check essential content with less format dependency
    assert "DOPRI5" in config_str
    assert "pid" in config_str

    # BETTER: Use regular expressions for flexible format matching
    assert re.search(r"rtol:\s+0?\.0001|1e-4", config_str) is not None
    assert re.search(r"atol:\s+1e-0?8", config_str) is not None

    # ALTERNATIVE: Extract values and check numerically
    rtol_match = re.search(r"rtol:\s+([\d\.e+-]+)", config_str)
    if rtol_match:
        rtol_value = float(rtol_match.group(1))
        assert jnp.isclose(rtol_value, 1e-4)


def test_numeric_output_comparison():
    """Demonstrate proper testing of numeric outputs."""
    # Example numeric output
    result = 0.10000000000000001  # From floating point calculation
    expected = 0.1  # What we actually expect

    # BAD: Direct equality check - often fails due to floating point imprecision
    # assert result == expected

    # GOOD: Use approximate equality with appropriate tolerance
    assert jnp.isclose(result, expected)

    # BETTER: Specify appropriate tolerances for the context
    assert jnp.isclose(result, expected, rtol=1e-10, atol=1e-10)


def test_array_comparison():
    """Demonstrate testing array outputs appropriately."""
    # Example arrays
    result_array = jnp.array([0.1, 0.2, 0.30000000000000004])
    expected_array = jnp.array([0.1, 0.2, 0.3])

    # BAD: Element-wise equality - prone to floating point issues
    # assert (result_array == expected_array).all()

    # GOOD: Use allclose for array comparison
    assert jnp.allclose(result_array, expected_array)

    # BETTER: Check properties when exact values aren't critical
    assert result_array.shape == expected_array.shape
    assert result_array.dtype == expected_array.dtype
    assert jnp.all(result_array >= 0)  # Domain-specific constraints


def test_dictionary_output():
    """Demonstrate testing dictionary outputs appropriately."""
    # Example dictionary output
    result_dict = {
        'X': jnp.array([1.0, 2.0, 3.0]),
        'P': jnp.array([0.1, 0.2, 0.3]),
        'times': jnp.array([0.0, 1.0, 2.0]),
        'metadata': {'solver': 'tsit5', 'steps': 120}
    }

    # BAD: Full equality check when only some parts matter
    # assert result_dict == expected_dict

    # GOOD: Check critical keys and properties
    assert 'X' in result_dict
    assert 'P' in result_dict
    assert 'times' in result_dict
    assert result_dict['X'].shape == (3,)

    # BETTER: Test domain-specific properties
    assert result_dict['X'][0] <= result_dict['X'][-1]  # Increasing values
    assert result_dict['times'][0] < result_dict['times'][-1]
    assert result_dict['X'].shape == result_dict['times'].shape


def test_conditional_format_checking():
    """Demonstrate handling variable formats intelligently."""

    # Example function that might return different format strings
    def get_solver_info(use_scientific=False):
        if use_scientific:
            return "rtol: 1.00e-04, atol: 1.00e-08"
        else:
            return "rtol: 0.0001, atol: 0.00000001"

    # Test both formats
    sci_format = get_solver_info(use_scientific=True)
    decimal_format = get_solver_info(use_scientific=False)

    # Use a flexible approach that handles both
    def check_tolerances(text, rtol_expected, atol_expected):
        # Extract values using regex that handles both formats
        rtol_match = re.search(r"rtol:\s+([\d\.e+-]+)", text)
        atol_match = re.search(r"atol:\s+([\d\.e+-]+)", text)

        # Convert to float and compare
        if rtol_match and atol_match:
            rtol_value = float(rtol_match.group(1))
            atol_value = float(atol_match.group(1))
            assert jnp.isclose(rtol_value, rtol_expected)
            assert jnp.isclose(atol_value, atol_expected)
        else:
            pytest.fail(f"Could not extract rtol/atol values from: {text}")

    # Check both formats with the same validation
    check_tolerances(sci_format, 1e-4, 1e-8)
    check_tolerances(decimal_format, 1e-4, 1e-8)