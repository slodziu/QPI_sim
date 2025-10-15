"""
Simple runner script for QPI simulations using predefined configurations.

Usage examples:
    python run_qpi.py fast_preview
    python run_qpi.py double_impurity
    python run_qpi.py high_quality_single
"""

import sys
from qpi_G_OOP import SystemParameters, QPISimulation, QPIvisualiser
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
        visualiser = QPIvisualiser(simulation)
        
        # Generate folder structure: outputs/{config_name}/
        import os
        outputs_base = "outputs"
        config_output_dir = os.path.join(outputs_base, config.name)
        frames_dir = os.path.join(config_output_dir, "frames")
        
        # Create directories
        os.makedirs(config_output_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Generate filenames
        anim_filename = os.path.join(config_output_dir, f"qpi_{config.name}.mp4")
        snapshot_filename = os.path.join(config_output_dir, f"qpi_{config.name}_snapshot.png")
        
        # Create animation with individual frames
        ani = visualiser.create_animation(anim_filename, frames_dir=frames_dir)
        
        # Save snapshot at mid-energy
        visualiser.save_mid_energy_snapshot(snapshot_filename)
        
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