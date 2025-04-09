"""Example of maintaining test parameter synchronization."""


# APPROACH 1: Create factory functions for test objects
def create_test_ode_system():
    """
    Factory function that creates a test ODE system with default parameters.

    When implementation changes, update only this function.
    """

    def sample_component(inputs):
        return inputs['X'] * 0.5

    return HybridODESystem(
        mechanistic_components={'X': sample_component},
        nn_replacements={},
        trainable_parameters={},  # Recently added parameter
        parameter_transforms={},  # Recently added parameter
        state_names=['X']
    )


# In tests:
def test_using_factory():
    # Get a test system with correct parameters
    ode_system = create_test_ode_system()
    # Continue with test...


# APPROACH 2: Create test fixtures with all parameters
@pytest.fixture
def ode_system_fixture():
    """Pytest fixture that provides a test ODE system."""

    def sample_component(inputs):
        return inputs['X'] * 0.5

    return HybridODESystem(
        mechanistic_components={'X': sample_component},
        nn_replacements={},
        trainable_parameters={},
        parameter_transforms={},
        state_names=['X']
    )


# In tests:
def test_with_fixture(ode_system_fixture):
    # Use the fixture directly
    result = ode_system_fixture.solve(...)
    # Continue with test...


# APPROACH 3: Create a test configuration class
class TestConfig:
    """
    Central configuration for test parameters.

    Update this when implementation changes.
    """

    @staticmethod
    def get_default_ode_params():
        """Get default parameters for ODE system tests."""
        return {
            'mechanistic_components': {'X': lambda inputs: inputs['X'] * 0.5},
            'nn_replacements': {},
            'trainable_parameters': {},
            'parameter_transforms': {},
            'state_names': ['X']
        }


# In tests:
def test_with_config():
    params = TestConfig.get_default_ode_params()
    ode_system = HybridODESystem(**params)
    # Continue with test...