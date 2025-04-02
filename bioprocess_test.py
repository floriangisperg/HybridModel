import pandas as pd
# Import our framework (assuming it's installed or in the same directory)
from hybrid_models import (
    HybridModelBuilder,
    train_hybrid_model,
    evaluate_hybrid_model,
    calculate_metrics,
    normalize_data,
    calculate_rate,
    create_initial_random_key,
    # Import new data handling classes and functions
    TimeSeriesData,
    prepare_datasets_for_training,
    VariableType
)

# Load data from Excel
data = pd.read_excel("Train_data_masked.xlsx")

# Use our new data loading framework
datasets = TimeSeriesData.load_datasets_from_dataframe(
    df=data,
    time_column='feedtimer(h)',
    run_id_column='RunID',
    max_runs=2
)

# Inspect the raw data from the first dataset
print("Raw data (first few rows) for dataset 0:")
print(datasets[0]._data_source.head())

# Configure each dataset and inspect key steps
for i, dataset in enumerate(datasets):
    print(f"\n--- Configuring dataset {i} ---")

    # Add state variables (X variables)
    dataset.add_state('CDW(g/L)', 'X')  # Biomass concentration
    dataset.add_state('Produktsol(g/L)', 'P')  # Product concentration

    # Add control variables (W variables)
    dataset.add_control('Temp(°C)', 'temp')  # Temperature

    # Add feed variables (F variables) - calculate rates automatically
    dataset.add_feed('Feed(L)', 'feed')  # Glucose feed
    dataset.add_feed('Base(L)', 'base')  # Base addition

    # Add more control variables
    dataset.add_control('InductorMASS(mg)', 'inductor_mass')
    dataset.add_control('Inductor(yesno)', 'inductor_switch')
    dataset.add_control('Reaktorvolumen(L)', 'reactor_volume')

    # Interpolate to handle missing values
    dataset.interpolate()

    # Inspect time points (first 10 points)
    print("Time points (first 10):", dataset.time_points[:10])

    # Inspect state variables (first 10 values each)
    print("State Variables:")
    for name, arr in dataset.state_variables.items():
        print(f"  {name}: {arr[:10]}")

    # Inspect control variables (first 10 values each)
    print("Control Variables:")
    for name, arr in dataset.control_variables.items():
        print(f"  {name}: {arr[:10]}")

    # Inspect feed variables (first 10 values each)
    print("Feed Variables:")
    for name, arr in dataset.feed_variables.items():
        print(f"  {name}: {arr[:10]}")

    # Optionally, prepare the dataset for training and inspect the structure
    training_dataset = dataset.prepare_for_training()
    print("Prepared Training Dataset Keys:", training_dataset.keys())
    # For example, inspect the initial state and a sample of time_dependent_inputs
    print("Initial state:", training_dataset['initial_state'])
    for key, (times, values) in training_dataset['time_dependent_inputs'].items():
        print(f"Input '{key}' - first 10 time points: {times[:10]}, first 10 values: {values[:10]}")

dummy = True
