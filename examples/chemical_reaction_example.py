"""
Chemical reaction network example for hybrid modeling framework.

This example simulates a complex chemical reaction network to generate synthetic data,
then applies hybrid modeling to fit a simplified kinetic model to the data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import os

# Import hybrid modeling framework
from hybrid_models import (
    HybridModelBuilder,
    VariableRegistry,
    MSE,
    SolverConfig,
    ExperimentManager,
    ModelConfig,
    NeuralNetworkConfig
)
from hybrid_models.data import DatasetManager, VariableType


# =============================================
# GROUND TRUTH MODEL FOR DATA GENERATION
# =============================================

def generate_chemical_kinetics_data(
        n_experiments=8,
        time_points=50,
        t_max=10.0,
        add_noise=True,
        noise_level=0.02,
        random_seed=42
):
    """
    Generate synthetic data from a complex chemical reaction network.

    Args:
        n_experiments: Number of experimental conditions
        time_points: Number of time points per experiment
        t_max: Maximum simulation time
        add_noise: Whether to add measurement noise
        noise_level: Standard deviation of noise relative to signal
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with time-series data from all experiments
    """
    np.random.seed(random_seed)

    # Define Arrhenius parameters for each reaction
    # A + B → C
    A1 = 1e10  # pre-exponential factor
    E1 = 50000  # activation energy J/mol

    # C → D
    A2 = 1e12
    E2 = 60000

    # D → E (catalyst-dependent)
    A3 = 1e8
    E3 = 40000

    # A + E → F
    A4 = 1e9
    E4 = 45000

    # F + C → G
    A5 = 1e11
    E5 = 55000

    # Universal gas constant
    R = 8.314  # J/(mol·K)

    def reaction_rates(t, y, params):
        """Calculate reaction rates for the complex model."""
        A, B, C, D, E, F, G = y
        T, catalyst = params['T'], params['catalyst']

        # Temperature-dependent rate constants (Arrhenius equation)
        k1 = A1 * np.exp(-E1 / (R * T))
        k2 = A2 * np.exp(-E2 / (R * T))
        k3 = A3 * np.exp(-E3 / (R * T))
        k4 = A4 * np.exp(-E4 / (R * T))
        k5 = A5 * np.exp(-E5 / (R * T))

        # Reaction rates with complex effects
        r1 = k1 * A * B  # A + B → C

        # Inhibition effect: D inhibits its own formation
        inhibition_factor = 1.0 / (1.0 + D / 0.2)
        r2 = k2 * C * inhibition_factor  # C → D with inhibition

        # Catalyst effect
        r3 = k3 * D * catalyst  # D → E (catalyst-dependent)

        # Simple reaction
        r4 = k4 * A * E  # A + E → F

        # F + C → G with substrate inhibition at high F
        substrate_inhibition = 1.0 / (1.0 + F / 0.5)
        r5 = k5 * F * C * substrate_inhibition  # F + C → G

        return r1, r2, r3, r4, r5

    def ode_system(t, y, params):
        """ODE system for the complex chemical reaction network."""
        A, B, C, D, E, F, G = y

        r1, r2, r3, r4, r5 = reaction_rates(t, y, params)

        # Rate equations
        dA_dt = -r1 - r4  # A consumed in two reactions
        dB_dt = -r1  # B consumed in one reaction
        dC_dt = r1 - r2 - r5  # C produced by r1, consumed by r2 and r5
        dD_dt = r2 - r3  # D produced by r2, consumed by r3
        dE_dt = r3 - r4  # E produced by r3, consumed by r4
        dF_dt = r4 - r5  # F produced by r4, consumed by r5
        dG_dt = r5  # G produced by r5

        return [dA_dt, dB_dt, dC_dt, dD_dt, dE_dt, dF_dt, dG_dt]

    # Generate experimental conditions
    experiments = []
    for exp_id in range(n_experiments):
        # Randomize parameters for each experiment
        T = np.random.uniform(300, 400)  # Temperature between 300-400 K
        catalyst = np.random.uniform(0.1, 1.0)  # Catalyst concentration

        # Randomize initial concentrations
        A0 = np.random.uniform(0.8, 1.2)  # Initial concentration of A
        B0 = np.random.uniform(0.8, 1.2)  # Initial concentration of B

        # Initial state (A, B, C, D, E, F, G)
        y0 = [A0, B0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Parameters for this experiment
        params = {
            'T': T,
            'catalyst': catalyst
        }

        # Time points for this experiment
        t_eval = np.linspace(0, t_max, time_points)

        # Solve ODE system
        solution = solve_ivp(
            ode_system,
            [0, t_max],
            y0,
            t_eval=t_eval,
            args=(params,),
            method='LSODA'
        )

        # Extract results
        times = solution.t
        concentrations = solution.y

        # For each time point, create a row in the dataset
        for i, t in enumerate(times):
            # Add noise if requested
            if add_noise:
                noise = np.random.normal(0, noise_level, size=len(concentrations))
                measured = [max(0, conc[i] * (1 + noise[j])) for j, conc in enumerate(concentrations)]
            else:
                measured = [conc[i] for conc in concentrations]

            # Store all data for this time point
            row = {
                'ExpID': exp_id,
                'Time': t,
                'Temp': T,
                'Catalyst': catalyst,
                'A': measured[0],
                'B': measured[1],
                'C': measured[2],
                'D': measured[3],
                'E': measured[4],
                'F': measured[5],
                'G': measured[6]
            }
            experiments.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(experiments)

    return df


# =============================================
# PREPARE DATA FOR HYBRID MODELING
# =============================================

def prepare_chemical_data(df, train_ratio=0.7):
    """
    Prepare chemical kinetics data for hybrid modeling.

    Args:
        df: DataFrame with chemical kinetics data
        train_ratio: Fraction of experiments to use for training

    Returns:
        DatasetManager with prepared datasets
    """
    # Create dataset manager
    manager = DatasetManager()

    # Get unique experiment IDs
    exp_ids = df['ExpID'].unique()
    n_experiments = len(exp_ids)

    # Split into training and test sets
    n_train = int(n_experiments * train_ratio)
    train_ids = exp_ids[:n_train]
    test_ids = exp_ids[n_train:]

    # Load data with train/test split
    manager.load_from_dataframe(
        df=df,
        time_column='Time',
        run_id_column='ExpID',
        train_run_ids=train_ids,
        test_run_ids=test_ids
    )

    # Define variables using VariableRegistry
    variables = VariableRegistry()

    # State variables (chemical concentrations - outputs)
    variables.add_state('A', is_output=True)
    variables.add_state('B', is_output=True)
    variables.add_state('C', is_output=True)
    variables.add_state('D', is_output=True)
    variables.add_state('E', is_output=True)
    variables.add_state('F', is_output=True)
    variables.add_state('G', is_output=True)

    # Control variables (temperature, catalyst)
    variables.add_control('Temp')
    variables.add_control('Catalyst')

    # Add variables to datasets
    manager.add_variables(variables.to_list(), df)

    # Calculate normalization parameters
    manager.calculate_norm_params()

    return manager


# =============================================
# DEFINE SIMPLIFIED KINETIC MODEL
# =============================================

def define_simplified_model(norm_params, ann_config=None):
    """
    Define a simplified kinetic model with hybrid components.

    This model excludes the F+C→G reaction and simplifies catalyst effects.

    Args:
        norm_params: Normalization parameters
        ann_config: Optional dictionary with neural network configurations

    Returns:
        Tuple of (model, model_config)
    """
    # Set default ANN configuration if none provided
    if ann_config is None:
        ann_config = {
            'r3_enhancement': {  # Capture catalyst effects on r3
                'hidden_dims': [32, 32, 32],
                'output_activation': 'softplus',
                'input_features': ['D', 'Catalyst', 'Temp'],
                'seed': 42
            },
            'r2_inhibition': {  # Capture inhibition of r2
                'hidden_dims': [32, 32, 32],
                'output_activation': 'sigmoid',
                'input_features': ['C', 'D', 'Temp'],
                'seed': 43
            },
            'missing_reaction': {  # Capture the missing F+C→G reaction
                'hidden_dims': [32, 32, 32],
                'output_activation': 'softplus',
                'input_features': ['F', 'C', 'Temp'],
                'seed': 44
            }
        }

    # Create model builder
    builder = HybridModelBuilder()

    # Set normalization parameters
    builder.set_normalization_params(norm_params)

    # Add state variables (all chemical species)
    builder.add_state('A')
    builder.add_state('B')
    builder.add_state('C')
    builder.add_state('D')
    builder.add_state('E')
    builder.add_state('F')
    builder.add_state('G')  # We'll use an NN to predict this since the reaction is "unknown"

    # Create model config for documentation
    model_config = ModelConfig(
        state_names=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        mechanistic_components={}  # Will be filled below
    )

    # Define simplified kinetic model functions

    # Simplified Arrhenius equation helper
    def simplified_arrhenius(A_factor, E_activation, T):
        """Simplified Arrhenius equation for rate constants."""
        R = 8.314  # J/(mol·K)
        return A_factor * jnp.exp(-E_activation / (R * T))

    # A + B → C
    def dA_dt(inputs):
        A = inputs['A']
        B = inputs['B']
        T = inputs['Temp']

        # Simplified r1 rate
        k1 = simplified_arrhenius(1e10, 50000, T)
        r1 = k1 * A * B

        # A consumed in reaction 1 and 4
        r4 = inputs.get('r4', 0.0)

        return -r1 - r4

    def dB_dt(inputs):
        A = inputs['A']
        B = inputs['B']
        T = inputs['Temp']

        # Simplified r1 rate
        k1 = simplified_arrhenius(1e10, 50000, T)
        r1 = k1 * A * B

        return -r1

    def dC_dt(inputs):
        A = inputs['A']
        B = inputs['B']
        C = inputs['C']
        T = inputs['Temp']

        # Reaction rates
        k1 = simplified_arrhenius(1e10, 50000, T)
        r1 = k1 * A * B

        # Use neural network to capture complex inhibition
        inhibition_factor = inputs['r2_inhibition']  # From neural network

        k2 = simplified_arrhenius(1e12, 60000, T)
        r2 = k2 * C * inhibition_factor

        # Include effect of missing reaction (to be learned)
        missing_reaction_effect = inputs.get('missing_reaction', 0.0)

        return r1 - r2 - missing_reaction_effect

    def dD_dt(inputs):
        C = inputs['C']
        D = inputs['D']
        T = inputs['Temp']

        # Use neural network to capture complex inhibition
        inhibition_factor = inputs['r2_inhibition']  # From neural network

        k2 = simplified_arrhenius(1e12, 60000, T)
        r2 = k2 * C * inhibition_factor

        # Apply catalyst enhancement learned by neural network
        catalyst_enhancement = inputs['r3_enhancement']  # From neural network

        k3 = simplified_arrhenius(1e8, 40000, T)
        r3 = k3 * D * catalyst_enhancement

        return r2 - r3

    def dE_dt(inputs):
        D = inputs['D']
        E = inputs['E']
        T = inputs['Temp']

        # Apply catalyst enhancement learned by neural network
        catalyst_enhancement = inputs['r3_enhancement']  # From neural network

        k3 = simplified_arrhenius(1e8, 40000, T)
        r3 = k3 * D * catalyst_enhancement

        # E is also consumed in r4
        k4 = simplified_arrhenius(1e9, 45000, T)
        A = inputs['A']
        r4 = k4 * A * E

        # Calculate r4 for use in dA_dt
        inputs['r4'] = r4

        return r3 - r4

    def dF_dt(inputs):
        A = inputs['A']
        E = inputs['E']
        T = inputs['Temp']

        k4 = simplified_arrhenius(1e9, 45000, T)
        r4 = k4 * A * E

        # Include effect of missing reaction (to be learned)
        missing_reaction_effect = inputs.get('missing_reaction', 0.0)

        return r4 - missing_reaction_effect

    def dG_dt(inputs):
        # G is only produced in the "missing" reaction that the model
        # doesn't explicitly know about - neural network will learn this
        missing_reaction_effect = inputs.get('missing_reaction', 0.0)

        return missing_reaction_effect

    # Add mechanistic components
    builder.add_mechanistic_component('A', dA_dt)
    builder.add_mechanistic_component('B', dB_dt)
    builder.add_mechanistic_component('C', dC_dt)
    builder.add_mechanistic_component('D', dD_dt)
    builder.add_mechanistic_component('E', dE_dt)
    builder.add_mechanistic_component('F', dF_dt)
    builder.add_mechanistic_component('G', dG_dt)

    # Add mechanistic components to model config
    model_config.mechanistic_components['A'] = dA_dt
    model_config.mechanistic_components['B'] = dB_dt
    model_config.mechanistic_components['C'] = dC_dt
    model_config.mechanistic_components['D'] = dD_dt
    model_config.mechanistic_components['E'] = dE_dt
    model_config.mechanistic_components['F'] = dF_dt
    model_config.mechanistic_components['G'] = dG_dt

    # Create neural network configurations
    for nn_name, nn_settings in ann_config.items():
        # Create neural network configuration
        nn_config = NeuralNetworkConfig(
            name=nn_name,
            input_features=nn_settings['input_features'],
            hidden_dims=nn_settings['hidden_dims'],
            output_activation=nn_settings['output_activation'],
            seed=nn_settings.get('seed', 0)
        )

        # Add to model config for documentation
        model_config.add_nn(nn_config)

        # Create the neural network in the builder
        builder.replace_with_nn(
            name=nn_name,
            input_features=nn_config.input_features,
            hidden_dims=nn_config.hidden_dims,
            output_activation=nn_config.get_activation_fn(),
            key=nn_config.get_random_key()
        )

    # Build the model
    model = builder.build()

    return model, model_config


# =============================================
# MAIN FUNCTION
# =============================================

def main():
    """Main function to run the chemical kinetics hybrid modeling example."""
    print("Chemical Kinetics Hybrid Modeling Example")
    print("=========================================")

    # Parameters
    n_experiments = 10
    output_dir = "examples/results/chemical_kinetics"
    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    kinetics_data = generate_chemical_kinetics_data(
        n_experiments=n_experiments,
        time_points=10,
        t_max=10.0,
        add_noise=True,
        noise_level=0.05
    )

    # Save raw data
    kinetics_data.to_csv(f"{output_dir}/raw_kinetics_data.csv", index=False)
    print(f"Generated data for {n_experiments} experiments with {len(kinetics_data)} data points")

    # Prepare data for hybrid modeling
    print("\nPreparing data for hybrid modeling...")
    data_manager = prepare_chemical_data(kinetics_data, train_ratio=0.7)

    print(f"Prepared {len(data_manager.train_datasets)} training datasets and "
          f"{len(data_manager.test_datasets)} test datasets")

    # Get normalization parameters
    norm_params = data_manager.norm_params

    # Prepare datasets for training and testing
    train_datasets = data_manager.prepare_training_data()
    test_datasets = data_manager.prepare_test_data()

    # Configure model architecture with custom neural networks
    ann_config = {
        'r3_enhancement': {  # Capture catalyst effects on r3
            'hidden_dims': [32, 32, 32],
            'output_activation': 'softplus',
            'input_features': ['D', 'Catalyst', 'Temp'],
            'seed': 42
        },
        'r2_inhibition': {  # Capture inhibition of r2
            'hidden_dims': [32, 32, 32],
            'output_activation': 'sigmoid',
            'input_features': ['C', 'D', 'Temp'],
            'seed': 43
        },
        'missing_reaction': {  # Capture the missing F+C→G reaction
            'hidden_dims': [32, 32, 32],
            'output_activation': 'softplus',
            'input_features': ['F', 'C', 'Temp'],
            'seed': 44
        }
    }

    # Configure solver
    solver_config = SolverConfig(
        solver_type="tsit5",  # Higher accuracy solver for this stiff system
        step_size_controller="pid",
        rtol=1e-3,
        atol=1e-6,
        max_steps=100000
    )

    # Build model with customized architecture
    print("\nBuilding hybrid model...")
    model, model_config = define_simplified_model(norm_params, ann_config)

    # Create experiment manager
    experiment = ExperimentManager(
        model=model,
        model_config=model_config,
        norm_params=norm_params,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        output_dir=output_dir,
        experiment_name="chemical_kinetics_example"
    )

    # Generate model documentation
    print("\nGenerating model documentation...")
    experiment.generate_model_documentation()

    # Save normalization parameters
    experiment.save_normalization_parameters()

    # Train model with customized settings
    print("\nTraining model...")
    trained_model = experiment.train(
        state_names=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        num_epochs=1000,  # Increase for better results
        learning_rate=1e-3,
        early_stopping_patience=100,
        component_weights={
            'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0,
            'E': 1.0, 'F': 1.0, 'G': 1.5  # Weight G higher since it depends on the missing reaction
        },
        loss_metric=MSE,
        solver_config=solver_config,
        save_checkpoints=True
    )

    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = experiment.evaluate(
        state_names=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        solver_config=solver_config,
        verbose=True
    )

    # Visualize results
    print("\nGenerating visualizations...")
    experiment.visualize(
        state_names=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        state_labels={
            'A': 'Reactant A', 'B': 'Reactant B',
            'C': 'Intermediate C', 'D': 'Intermediate D',
            'E': 'Intermediate E', 'F': 'Product F',
            'G': 'Product G (unknown reaction)'
        },
        component_names=[
            'A Loss', 'B Loss', 'C Loss', 'D Loss',
            'E Loss', 'F Loss', 'G Loss'
        ],
        solver_config=solver_config
    )

    # Save all results
    print("\nSaving all results...")
    result_paths = experiment.save_all_results()

    # Generate summary
    experiment.save_results_summary()

    print("\nChemical kinetics experiment completed successfully!")
    print(f"All results saved to {experiment.output_dir}")

    return trained_model, train_datasets, test_datasets, experiment


if __name__ == "__main__":
    main()