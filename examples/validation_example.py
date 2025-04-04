"""
Example demonstrating how to use validation datasets during training.
"""

import jax
import jax.numpy as jnp
from hybrid_models import (
    HybridModelBuilder,
    train_hybrid_model,
    create_initial_random_key
)
from hybrid_models.visualization import plot_all_results
from hybrid_models.evaluation_utils import evaluate_model_performance


def create_example_datasets():
    """Create synthetic datasets for demonstration."""
    # Create a basic time series
    times = jnp.linspace(0, 10, 21)  # 0 to 10 in steps of 0.5

    # Training datasets (3 datasets with different growth rates)
    train_datasets = []
    for growth_rate in [0.2, 0.25, 0.3]:
        # Simple growth curve for X (biomass) with some noise
        X = 1.0 + growth_rate * times * (
                    1 + 0.1 * jax.random.normal(jax.random.PRNGKey(int(growth_rate * 100)), shape=times.shape))

        # Simple product formation for P
        P = 0.1 * times ** 1.5 * (
                    1 + 0.1 * jax.random.normal(jax.random.PRNGKey(int(growth_rate * 200)), shape=times.shape))

        # Ensure values are positive
        X = jnp.maximum(X, 0.1)
        P = jnp.maximum(P, 0.01)

        dataset = {
            'X_true': X,
            'P_true': P,
            'times': times,
            'initial_state': {
                'X': float(X[0]),
                'P': float(P[0])
            },
            'time_dependent_inputs': {
                'temp': (times, 37.0 * jnp.ones_like(times)),  # Constant temperature
                'feed': (times, 0.1 * times),  # Linear feed increase
                'inductor_switch': (times, jnp.ones_like(times)),  # Inductor always on
            }
        }
        train_datasets.append(dataset)

    # Validation dataset (slightly different growth rate)
    validation_datasets = []
    for growth_rate in [0.27]:
        X = 1.0 + growth_rate * times * (
                    1 + 0.1 * jax.random.normal(jax.random.PRNGKey(int(growth_rate * 300)), shape=times.shape))
        P = 0.1 * times ** 1.5 * (
                    1 + 0.1 * jax.random.normal(jax.random.PRNGKey(int(growth_rate * 400)), shape=times.shape))

        X = jnp.maximum(X, 0.1)
        P = jnp.maximum(P, 0.01)

        dataset = {
            'X_true': X,
            'P_true': P,
            'times': times,
            'initial_state': {
                'X': float(X[0]),
                'P': float(P[0])
            },
            'time_dependent_inputs': {
                'temp': (times, 37.0 * jnp.ones_like(times)),
                'feed': (times, 0.1 * times),
                'inductor_switch': (times, jnp.ones_like(times)),
            }
        }
        validation_datasets.append(dataset)

    # Test dataset (completely different growth rate)
    test_datasets = []
    for growth_rate in [0.35]:
        X = 1.0 + growth_rate * times * (
                    1 + 0.1 * jax.random.normal(jax.random.PRNGKey(int(growth_rate * 500)), shape=times.shape))
        P = 0.1 * times ** 1.5 * (
                    1 + 0.1 * jax.random.normal(jax.random.PRNGKey(int(growth_rate * 600)), shape=times.shape))

        X = jnp.maximum(X, 0.1)
        P = jnp.maximum(P, 0.01)

        dataset = {
            'X_true': X,
            'P_true': P,
            'times': times,
            'initial_state': {
                'X': float(X[0]),
                'P': float(P[0])
            },
            'time_dependent_inputs': {
                'temp': (times, 37.0 * jnp.ones_like(times)),
                'feed': (times, 0.1 * times),
                'inductor_switch': (times, jnp.ones_like(times)),
            }
        }
        test_datasets.append(dataset)

    return train_datasets, validation_datasets, test_datasets


def create_simple_model():
    """Create a simple hybrid model for demonstration."""
    # Create model builder
    builder = HybridModelBuilder()

    # Add state variables
    builder.add_state('X')  # Biomass
    builder.add_state('P')  # Product

    # Define simple growth model
    def biomass_ode(inputs):
        X = inputs['X']
        mu = inputs['growth_rate']  # Will be replaced by neural network
        return mu * X

    # Define simple product formation
    def product_ode(inputs):
        X = inputs['X']
        qp = inputs['product_rate']  # Will be replaced by neural network
        inductor_switch = inputs.get('inductor_switch', 1.0)
        return qp * X * inductor_switch

    # Add mechanistic components
    builder.add_mechanistic_component('X', biomass_ode)
    builder.add_mechanistic_component('P', product_ode)

    # Create random key for neural network initialization
    key = create_initial_random_key(42)
    key1, key2 = jax.random.split(key)

    # Replace growth rate with neural network
    builder.replace_with_nn(
        name='growth_rate',
        input_features=['X', 'temp', 'feed'],
        hidden_dims=[4, 4],  # Small network for demonstration
        key=key1
    )

    # Replace product formation rate with neural network
    builder.replace_with_nn(
        name='product_rate',
        input_features=['X', 'P', 'feed'],
        hidden_dims=[4, 4],  # Small network for demonstration
        output_activation=jax.nn.softplus,  # Ensure non-negative rate
        key=key2
    )

    # Build and return the model
    return builder.build()


def simple_loss_function(model, datasets):
    """Loss function for model training."""
    total_loss = 0.0
    total_x_loss = 0.0
    total_p_loss = 0.0

    for dataset in datasets:
        # Get predictions
        solution = model.solve(
            initial_state=dataset['initial_state'],
            t_span=(dataset['times'][0], dataset['times'][-1]),
            evaluation_times=dataset['times'],
            args={
                'time_dependent_inputs': dataset['time_dependent_inputs']
            },
            rtol=1e-2,
            atol=1e-4
        )

        # Calculate loss
        X_pred = solution['X']
        P_pred = solution['P']
        X_true = dataset['X_true']
        P_true = dataset['P_true']

        X_loss = jnp.mean(jnp.square(X_pred - X_true))
        P_loss = jnp.mean(jnp.square(P_pred - P_true))

        # Add to total loss
        run_loss = X_loss + P_loss
        total_loss += run_loss
        total_x_loss += X_loss
        total_p_loss += P_loss

    # Return average loss
    n_datasets = len(datasets)
    return total_loss / n_datasets, (total_x_loss / n_datasets, total_p_loss / n_datasets)


def solve_for_dataset(model, dataset):
    """Solve the model for a given dataset."""
    solution = model.solve(
        initial_state=dataset['initial_state'],
        t_span=(dataset['times'][0], dataset['times'][-1]),
        evaluation_times=dataset['times'],
        args={
            'time_dependent_inputs': dataset['time_dependent_inputs']
        },
        rtol=1e-2,
        atol=1e-4
    )

    return solution


def main():
    """Main function demonstrating validation datasets during training."""
    # Create datasets
    print("Creating datasets...")
    train_datasets, validation_datasets, test_datasets = create_example_datasets()

    # Create model
    print("Creating model...")
    model = create_simple_model()

    # Train model with validation
    print("Training model with validation...")
    try:
        trained_model, history, validation_history = train_hybrid_model(
            model=model,
            datasets=train_datasets,
            loss_fn=simple_loss_function,
            num_epochs=200,
            learning_rate=1e-3,
            early_stopping_patience=30,
            validation_datasets=validation_datasets,
            verbose=True
        )
        print("Training complete")
    except Exception as e:
        print(f"Error during training: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nFalling back to returning the untrained model")
        history = {"loss": [], "aux": []}
        validation_history = {"loss": [], "aux": []}
        trained_model = model

    # Plot results including validation loss
    print("Plotting results...")
    plot_all_results(
        model=trained_model,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        history=history,
        solve_fn=solve_for_dataset,
        state_names=['X', 'P'],
        output_dir="validation_example_results",
        state_labels={'X': 'Biomass', 'P': 'Product'},
        component_names=['Biomass Loss', 'Product Loss'],
        validation_history=validation_history
    )

    # Evaluate on training data
    print("\nEvaluating on training data...")
    train_evaluation = evaluate_model_performance(
        model=trained_model,
        datasets=train_datasets,
        solve_fn=solve_for_dataset,
        state_names=['X', 'P'],
        dataset_type="Training",
        save_metrics=True,
        output_dir="validation_example_results"
    )

    # Evaluate on validation data
    print("\nEvaluating on validation data...")
    validation_evaluation = evaluate_model_performance(
        model=trained_model,
        datasets=validation_datasets,
        solve_fn=solve_for_dataset,
        state_names=['X', 'P'],
        dataset_type="Validation",
        save_metrics=True,
        output_dir="validation_example_results",
        metrics_filename="validation_metrics.txt"
    )

    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_evaluation = evaluate_model_performance(
        model=trained_model,
        datasets=test_datasets,
        solve_fn=solve_for_dataset,
        state_names=['X', 'P'],
        dataset_type="Test",
        save_metrics=True,
        output_dir="validation_example_results",
        metrics_filename="test_metrics.txt"
    )

    return trained_model, train_datasets, validation_datasets, test_datasets, history, validation_history


if __name__ == "__main__":
    main()