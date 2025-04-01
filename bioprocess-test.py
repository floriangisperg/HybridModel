# test_optimizations.py
from bioprocess import (
    load_bioprocess_data,
    define_bioprocess_model,
    prepare_bioprocess_dataset,
    bioprocess_loss_function
)
from hybrid_models import train_hybrid_optimized


def main():
    # Load your data and create model (using your existing code)
    print("Loading data...")
    runs = load_bioprocess_data('Train_data_masked.xlsx', max_runs=2)
    norm_params = runs[0]['norm_params']

    # Build model
    print("Building model...")
    model = define_bioprocess_model(norm_params)

    # Prepare datasets
    print("Preparing datasets...")
    datasets = prepare_bioprocess_dataset(runs)

    # Train model with optimizations
    print("Training optimized model...")
    trained_model, history = train_hybrid_optimized(
        model=model,
        datasets=datasets,
        loss_fn=bioprocess_loss_function,
        num_epochs=10,  # Start with a small number for testing
        learning_rate=1e-3,
        use_parallel=True,
        verbose=True
    )

    print("Training complete!")
    print(f"Initial loss: {history['loss'][0]}")
    print(f"Final loss: {history['loss'][-1]}")

    return trained_model, history


if __name__ == "__main__":
    main()