"""Integration tests for the hybrid modeling framework."""
import pytest
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from hybrid_models import (
    HybridModelBuilder,
    train_hybrid_model,
    evaluate_hybrid_model,
    calculate_metrics,
    create_initial_random_key
)
from hybrid_models.data import DatasetManager, VariableType


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for integration testing."""
    # Create DataFrame with multiple runs
    runs = []
    for run_id in range(1, 4):  # 3 runs
        times = np.linspace(0, 10, 11)  # 0 to 10 in steps of 1.0
        X = 1.0 + 0.5 * times * (1 + 0.1 * run_id)  # Slight variation between runs
        P = 0.1 * times ** 2 * (1 + 0.05 * run_id)
        temp = 37.0 * np.ones_like(times)
        feed = 0.1 * times
        inductor = np.ones_like(times)

        run_data = pd.DataFrame({
            'time': times,
            'RunID': run_id,
            'X': X,
            'P': P,
            'temp': temp,
            'feed': feed,
            'inductor': inductor
        })
        runs.append(run_data)

    return pd.concat(runs, ignore_index=True)


def test_end_to_end_workflow(synthetic_data):
    """Test the complete workflow from data loading to model evaluation."""
    # Step 1: Load and prepare data
    manager = DatasetManager()
    manager.load_from_dataframe(
        df=synthetic_data,
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
        ('inductor', VariableType.CONTROL, 'inductor', False, False)
    ]

    manager.add_variables(variable_definitions, synthetic_data)
    manager.calculate_norm_params()

    train_datasets = manager.prepare_training_data()
    test_datasets = manager.prepare_test_data()

    # Step 2: Build model
    builder = HybridModelBuilder()
    builder.set_normalization_params(manager.norm_params)
    builder.add_state('X')
    builder.add_state('P')

    def biomass_ode(inputs):
        X = inputs['X']
        mu = inputs['growth_rate']
        return mu * X

    def product_ode(inputs):
        X = inputs['X']
        qp = inputs['product_rate']
        inductor = inputs.get('inductor', 1.0)
        return qp * X * inductor

    builder.add_mechanistic_component('X', biomass_ode)
    builder.add_mechanistic_component('P', product_ode)

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    builder.replace_with_nn(
        name='growth_rate',
        input_features=['X', 'temp'],
        hidden_dims=[4, 4],
        key=key1
    )

    builder.replace_with_nn(
        name='product_rate',
        input_features=['X', 'P', 'temp'],
        hidden_dims=[4, 4],
        output_activation=jax.nn.softplus,
        key=key2
    )

    model = builder.build()

    # Step 3: Define loss function
    def loss_function(model, datasets):
        total_loss = 0.0

        for dataset in datasets:
            solution = model.solve(
                initial_state=dataset['initial_state'],
                t_span=(dataset['times'][0], dataset['times'][-1]),
                evaluation_times=dataset['times'],
                args={
                    'time_dependent_inputs': dataset['time_dependent_inputs'],
                    'static_inputs': dataset.get('static_inputs', {})
                },
                rtol=1e-2,
                atol=1e-4
            )

            X_pred = solution['X']
            P_pred = solution['P']
            X_true = dataset['X_true']
            P_true = dataset['P_true']

            loss = jnp.mean(jnp.square(X_pred - X_true)) + jnp.mean(jnp.square(P_pred - P_true))
            total_loss += loss

        return total_loss / len(datasets), None

    # Step 4: Train model (minimal epochs for testing)
    trained_model, history = train_hybrid_model(
        model=model,
        datasets=train_datasets,
        loss_fn=loss_function,
        num_epochs=5,  # Just a few epochs for testing
        learning_rate=1e-3
    )

    # Check that training actually happened
    assert len(history['loss']) == 5

    # Step 5: Evaluate model
    def solve_for_dataset(model, dataset):
        return model.solve(
            initial_state=dataset['initial_state'],
            t_span=(dataset['times'][0], dataset['times'][-1]),
            evaluation_times=dataset['times'],
            args={
                'time_dependent_inputs': dataset['time_dependent_inputs'],
                'static_inputs': dataset.get('static_inputs', {})
            },
            rtol=1e-2,
            atol=1e-4
        )

    results = evaluate_hybrid_model(trained_model, test_datasets, solve_for_dataset)

    # Assertions to verify workflow
    assert 'dataset_0' in results
    assert 'overall' in results
    assert 'X' in results['dataset_0']
    assert 'P' in results['dataset_0']
    assert 'r2' in results['dataset_0']['X']
    assert 'rmse' in results['dataset_0']['X']


def test_bioprocess_model_structure(synthetic_data):
    """Test the structure of a bioprocess hybrid model similar to the example script."""
    # Simplified test of just the model structure, not training

    # Step 1: Prepare data and norm params
    manager = DatasetManager()
    manager.load_from_dataframe(
        df=synthetic_data,
        time_column='time',
        run_id_column='RunID',
        train_ratio=0.7
    )

    variable_definitions = [
        ('X', VariableType.STATE, 'X', True, False),
        ('P', VariableType.STATE, 'P', True, False),
        ('temp', VariableType.CONTROL, 'temp', False, False),
        ('feed', VariableType.FEED, 'feed', False, True),
        ('inductor', VariableType.CONTROL, 'inductor', False, False)
    ]

    manager.add_variables(variable_definitions, synthetic_data)
    manager.calculate_norm_params()

    # Step 2: Define a bioprocess model structure similar to example
    builder = HybridModelBuilder()
    builder.set_normalization_params(manager.norm_params)

    # Add states
    builder.add_state('X')
    builder.add_state('P')

    # Define dilution rate calculation
    def calculate_dilution_rate(inputs):
        volume = inputs.get('reactor_volume', 1.0)
        feed_rate = inputs.get('feed_rate', 0.0)
        return jnp.where(volume > 1e-6, feed_rate / volume, 0.0)

    # Define biomass ODE (mechanistic part)
    def biomass_ode(inputs):
        X = inputs['X']
        mu = inputs['growth_rate']  # Will be replaced by neural network
        dilution_rate = calculate_dilution_rate(inputs)
        return mu * X - dilution_rate * X

    # Define product ODE (mechanistic part)
    def product_ode(inputs):
        X = inputs['X']
        P = inputs['P']
        vpx = inputs['product_rate']  # Will be replaced by neural network
        inductor = inputs.get('inductor', 0.0)
        dilution_rate = calculate_dilution_rate(inputs)
        return vpx * X * inductor - dilution_rate * P

    # Add mechanistic components
    builder.add_mechanistic_component('X', biomass_ode)
    builder.add_mechanistic_component('P', product_ode)

    # Create random key for neural network initialization
    key = create_initial_random_key(42)
    key1, key2 = jax.random.split(key)

    # Replace growth rate with neural network
    builder.replace_with_nn(
        name='growth_rate',
        input_features=['X', 'P', 'temp', 'feed', 'inductor'],
        hidden_dims=[8, 8],
        key=key1
    )

    # Replace product formation rate with neural network
    builder.replace_with_nn(
        name='product_rate',
        input_features=['X', 'P', 'temp', 'feed', 'inductor'],
        hidden_dims=[8, 8],
        output_activation=jax.nn.softplus,  # Ensure non-negative rate
        key=key2
    )

    # Build model
    model = builder.build()

    # Check model structure
    assert model.state_names == ['X', 'P']
    assert 'X' in model.mechanistic_components
    assert 'P' in model.mechanistic_components
    assert 'growth_rate' in model.nn_replacements
    assert 'product_rate' in model.nn_replacements

    # Test if ODE functions can be called
    inputs_dict = {
        'X': jnp.array(1.0),
        'P': jnp.array(0.5),
        'temp': jnp.array(37.0),
        'feed': jnp.array(0.1),
        'feed_rate': jnp.array(0.01),
        'inductor': jnp.array(1.0),
        'reactor_volume': jnp.array(10.0)
    }

    # Create test inputs for the neural networks
    nn_inputs = model.nn_replacements['growth_rate'](inputs_dict)
    assert isinstance(nn_inputs, jnp.ndarray)

    # Add neural network outputs to inputs
    inputs_dict['growth_rate'] = model.nn_replacements['growth_rate'](inputs_dict)
    inputs_dict['product_rate'] = model.nn_replacements['product_rate'](inputs_dict)

    # Test mechanistic components
    x_derivative = model.mechanistic_components['X'](inputs_dict)
    p_derivative = model.mechanistic_components['P'](inputs_dict)

    assert isinstance(x_derivative, jnp.ndarray)
    assert isinstance(p_derivative, jnp.ndarray)