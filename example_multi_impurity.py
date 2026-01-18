#!/usr/bin/env python3
"""
Example script showing how to create custom multi-impurity tight-binding configurations.
"""

from qpi_G_OOP import SystemParameters, QPISimulation, QPIvisualiser
import numpy as np


def create_custom_multi_impurity_config():
    """Create a custom configuration for multi-impurity tight-binding simulation."""
    
    # Define custom parameters
    params = SystemParameters(
        gridsize=512,
        L=50.0,
        t=1.0,           # Hopping parameter
        mu=0.3,          # Chemical potential (doped)
        eta=0.08,        # Energy broadening
        V_s=0.25,        # Weak impurity strength for multiple impurities
        E_min=-1.5,      # Energy range appropriate for tight-binding
        E_max=1.5,
        n_frames=12,     # Fewer frames for quicker simulation
        model_type="square_lattice",
        t_prime=0.15,    # Next-nearest neighbor hopping
        use_all_bands=False,
        band_index=0
    )
    
    # Create custom impurity positions
    # Example: Create impurities in a specific pattern
    center = params.gridsize // 2
    impurity_positions = []
    
    # Central cluster of 3 impurities
    impurity_positions.extend([
        (center, center),
        (center - 30, center + 25),
        (center + 20, center - 35)
    ])
    
    # Two isolated impurities
    impurity_positions.extend([
        (center - 80, center - 80),
        (center + 90, center + 70)
    ])
    
    # Additional random impurities
    np.random.seed(42)  # For reproducibility
    for _ in range(5):  # Add 5 more random ones
        attempts = 0
        while attempts < 100:
            row = np.random.randint(50, params.gridsize - 50)
            col = np.random.randint(50, params.gridsize - 50)
            
            # Check minimum distance from existing impurities
            min_dist = min([np.sqrt((row - r)**2 + (col - c)**2) 
                           for r, c in impurity_positions])
            if min_dist > 40:  # Minimum separation
                impurity_positions.append((row, col))
                break
            attempts += 1
    
    print(f"Created {len(impurity_positions)} impurities at positions:")
    for i, (r, c) in enumerate(impurity_positions):
        print(f"  Impurity {i+1}: (row={r}, col={c})")
    
    return params, impurity_positions


def run_custom_simulation():
    """Run the custom multi-impurity simulation."""
    print("Creating custom multi-impurity tight-binding configuration...")
    params, impurity_positions = create_custom_multi_impurity_config()
    
    # Create the tight-binding model
    model = create_model(params.model_type, 
                        t=params.t, 
                        t_prime=params.t_prime, 
                        a=params.a)
    print(f"Created {params.model_type} model with t={params.t}, t'={params.t_prime}")
    
    # Create and run simulation
    print("Running QPI simulation...")
    simulation = QPISimulation(params, impurity_positions, model)
    visualiser = QPIvisualiser(simulation)
    
    # Output directory
    output_dir = "outputs/custom_multi_impurity"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create animations
    print("Generating visualizations...")
    animation_file = f"{output_dir}/custom_multi_impurity.mp4"
    fourier_file = f"{output_dir}/custom_multi_impurity_fourier.mp4"
    snapshot_file = f"{output_dir}/custom_multi_impurity_snapshot.png"
    
    # Save frames
    frames_dir = f"{output_dir}/frames"
    ani = visualiser.create_animation(animation_file, frames_dir=frames_dir)
    visualiser.create_fourier_animation(fourier_file, frames_dir=frames_dir)
    visualiser.save_mid_energy_snapshot(snapshot_file)
    
    print(f"\nSimulation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"- Main animation: {animation_file}")
    print(f"- Fourier analysis: {fourier_file}")
    print(f"- Snapshot: {snapshot_file}")
    print(f"- Individual frames: {frames_dir}")
    
    return simulation


def compare_models():
    """Compare QPI patterns for different tight-binding models with same impurity configuration."""
    
    print("\nRunning comparison of different tight-binding models...")
    
    # Common parameters
    base_params = {
        'gridsize': 512,
        'L': 50.0,
        'mu': 0.0,
        'eta': 0.1,
        'V_s': 0.3,
        'E_min': -1.0,
        'E_max': 1.0,
        'n_frames': 8  # Quick comparison
    }
    
    # Same impurity positions for fair comparison
    center = base_params['gridsize'] // 2
    impurity_positions = [
        (center, center),
        (center - 40, center + 30),
        (center + 35, center - 25)
    ]
    
    models_to_test = [
        {'name': 'square_lattice', 't': 1.0, 't_prime': 0.0},
        {'name': 'square_lattice_nnp', 't': 1.0, 't_prime': 0.2},  # With next-nearest neighbors
        {'name': 'graphene', 't': 2.7}
    ]
    
    for model_info in models_to_test:
        model_name = model_info['name']
        print(f"\nRunning {model_name} model...")
        
        # Create parameters
        if 'square_lattice' in model_name:
            params = SystemParameters(
                **base_params,
                model_type='square_lattice',
                t=model_info['t'],
                t_prime=model_info.get('t_prime', 0.0)
            )
            model = create_model('square_lattice', 
                               t=model_info['t'], 
                               t_prime=model_info.get('t_prime', 0.0),
                               a=params.a)
        else:  # graphene
            params = SystemParameters(
                **base_params,
                model_type='graphene',
                t=model_info['t']
            )
            model = create_model('graphene', t=model_info['t'], a=params.a)
        
        # Run simulation
        simulation = QPISimulation(params, impurity_positions, model)
        visualiser = QPIvisualiser(simulation)
        
        # Save results
        output_dir = f"outputs/comparison_{model_name}"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        snapshot_file = f"{output_dir}/comparison_{model_name}_snapshot.png"
        visualiser.save_mid_energy_snapshot(snapshot_file)
        
        print(f"  Saved snapshot: {snapshot_file}")
        print(f"  Extracted {len(simulation.extracted_k)} dispersion points")


if __name__ == "__main__":
    print("Custom Multi-Impurity Tight-Binding QPI Simulations")
    print("=" * 60)
    
    # Run custom simulation
    simulation = run_custom_simulation()
    
    # Run model comparison
    compare_models()
    
    print("\nAll simulations completed!")
    print("\nTo run more simulations, try:")
    print("  python run_qpi.py square_lattice_5_impurities")
    print("  python run_qpi.py graphene_10_impurities") 
    print("  python run_qpi.py random_15_impurities")