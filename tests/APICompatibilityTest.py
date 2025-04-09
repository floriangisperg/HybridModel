"""Strategies for API compatibility testing."""
import pytest
import diffrax
import jax
import optax


def test_diffrax_api_compatibility():
    """Test that we're using actual diffrax APIs correctly."""
    # 1. Check that expected classes exist
    assert hasattr(diffrax, 'Tsit5')
    assert hasattr(diffrax, 'PIDController')
    assert hasattr(diffrax, 'ConstantStepSize')
    assert hasattr(diffrax, 'ODETerm')

    # 2. Check that we can instantiate classes as expected
    pid_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    assert pid_controller.rtol == 1e-3
    assert pid_controller.atol == 1e-6

    # 3. Check method signatures/parameters
    # Create a minimal example to confirm API usage
    def simple_ode(t, y, args):
        return -y

    term = diffrax.ODETerm(simple_ode)
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=0.1,
        y0=jnp.array([1.0]),
        saveat=diffrax.SaveAt(ts=jnp.array([0.0, 0.5, 1.0]))
    )

    # Check result has expected attributes
    assert hasattr(sol, 'ts')
    assert hasattr(sol, 'ys')


def test_jax_api_compatibility():
    """Test that we're using actual JAX APIs correctly."""
    # 1. Check that expected functions/modules exist
    assert hasattr(jax, 'jit')
    assert hasattr(jax, 'grad')
    assert hasattr(jax, 'value_and_grad')
    assert hasattr(jax, 'vmap')

    # 2. Check that JAX functions work as expected
    @jax.jit
    def simple_function(x):
        return x * 2

    result = simple_function(jnp.array(3.0))
    assert result == 6.0


def test_optax_api_compatibility():
    """Test that we're using actual Optax APIs correctly."""
    # 1. Check that expected optimizers exist
    assert hasattr(optax, 'adam')
    assert hasattr(optax, 'sgd')
    assert hasattr(optax, 'apply_updates')

    # 2. Check that optimizer initialization works
    optimizer = optax.adam(learning_rate=1e-3)

    # 3. Check optimizer interface
    params = {'w': jnp.array(1.0), 'b': jnp.array(0.0)}
    opt_state = optimizer.init(params)

    # 4. Check update function
    grads = {'w': jnp.array(0.1), 'b': jnp.array(0.05)}
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    # 5. Check that parameters were updated
    assert new_params['w'] != params['w']
    assert new_params['b'] != params['b']


# Add validation at import time:
def validate_crucial_apis():
    """
    Validate crucial APIs at import time.

    This can be called during module initialization to fail fast if
    a critical API is missing or incompatible.
    """
    try:
        # Check diffrax
        assert hasattr(diffrax, 'PIDController')
        assert hasattr(diffrax, 'Tsit5')

        # Check JAX
        assert hasattr(jax, 'jit')
        assert hasattr(jax, 'grad')

        # Check other dependencies as needed
    except AssertionError as e:
        raise ImportError(f"Critical API validation failed: {e}")
    except Exception as e:
        raise ImportError(f"Unexpected error validating APIs: {e}")

    return True

# Call this in __init__.py
# validate_crucial_apis()