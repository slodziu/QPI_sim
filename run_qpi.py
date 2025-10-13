"""
Simple runner script for QPI simulations using predefined configurations.

Usage examples:
    python run_qpi.py fast_preview
    python run_qpi.py double_impurity
    python run_qpi.py high_quality_single
"""

import sys
from qpi_G_OOP import SystemParameters, QPISimulation, QPIVisualizer
from config import get_config, list_available_configs


def run_simulation(config_name: str):
    """Run a QPI simulation with the specified configuration."""
    try:
        # Get configuration
        config = get_config(config_name)
        print(f"Running simulation: {config.name}")
        print(f"Description: {config.description}")
        print("-" * 50)
        
        # Convert config to SystemParameters
        params = SystemParameters(
            gridsize=config.gridsize,
            L=config.L,
            t=config.t,
            mu=config.mu,
            eta=config.eta,
            V_s=config.V_s,
            E_min=config.E_min,
            E_max=config.E_max,
            n_frames=config.n_frames,
            rotation_angle=config.rotation_angle,
            disorder_strength=config.disorder_strength,
            zoom_factor=config.zoom_factor
        )
        
        # Get impurity positions
        impurity_positions = config.get_impurity_positions()
        print(f"Impurity positions: {len(impurity_positions)} impurities")
        
        # Create and run simulation
        simulation = QPISimulation(params, impurity_positions)
        visualizer = QPIVisualizer(simulation)
        
        # Generate filename with outputs folder
        import os
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        anim_filename = os.path.join(outputs_dir, f"qpi_{config.name}.mp4")
        snapshot_filename = os.path.join(outputs_dir, f"qpi_{config.name}_snapshot.png")
        
        # Create animation as MP4
        ani = visualizer.create_animation(anim_filename)
        
        # Save snapshot at mid-energy
        visualizer.save_mid_energy_snapshot(snapshot_filename)
        
        # Print results
        print(f"\nSimulation completed!")
        print(f"Extracted {len(simulation.extracted_k)} dispersion points")
        if len(simulation.extracted_E) > 0:
            print(f"Energy range: {min(simulation.extracted_E):.2f} to {max(simulation.extracted_E):.2f}")
        print(f"Animation saved as: {anim_filename}")
        print(f"Snapshot saved as: {snapshot_filename}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable configurations:")
        list_available_configs()
        return False
    except Exception as e:
        print(f"Simulation failed: {e}")
        return False
    
    return True


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python run_qpi.py <config_name>")
        print("\nAvailable configurations:")
        list_available_configs()
        sys.exit(1)
    
    config_name = sys.argv[1]
    
    if config_name == "list":
        list_available_configs()
        sys.exit(0)
    
    success = run_simulation(config_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()