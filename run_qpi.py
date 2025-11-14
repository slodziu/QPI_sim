"""
Simple runner script for QPI simulations using predefined configurations.

Usage examples:
    python run_qpi.py fast_preview
    python run_qpi.py fast_preview --save-frames
    python run_qpi.py double_impurity
    python run_qpi.py high_quality_single --save-frames
"""

import sys
import argparse
from qpi_G_OOP import SystemParameters, QPISimulation, QPIvisualiser
from config import get_config, list_available_configs
from model_factory import create_model
from fermi_surface_3d import visualize_tight_binding_fermi_surface
from true_3d_fermi_surface import visualize_true_3d_fermi_surface


def run_simulation(config_name: str, save_frames: bool = False):
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
            zoom_factor=config.zoom_factor,
            model_type=getattr(config, 'model_type', 'parabolic'),
            t_prime=getattr(config, 't_prime', 0.0),
            use_all_bands=getattr(config, 'use_all_bands', False),
            band_index=getattr(config, 'band_index', 0),
            # Advanced tight-binding parameters
            ty=getattr(config, 'ty', 0.6),
            t_z=getattr(config, 't_z', 0.5),
            kz_slice=getattr(config, 'kz_slice', 0.0)
        )
        
        # Get impurity positions
        impurity_positions = config.get_impurity_positions()
        print(f"Impurity positions: {len(impurity_positions)} impurities")
        print(f"Model type: {params.model_type}")
        
        # Create tight-binding model
        model = None
        if params.model_type != "parabolic":
            if params.model_type == "square_lattice":
                model_params = {
                    't': params.t,
                    't_prime': params.t_prime,
                    'a': params.a
                }
            elif params.model_type == "graphene":
                model_params = {
                    't': params.t,
                    'a': params.a
                }
            elif params.model_type == "cubic_3d":
                model_params = {
                    't': params.t,
                    't_z': getattr(params, 't_z', 0.5),
                    'a': params.a,
                    'kz_slice': getattr(params, 'kz_slice', 0.0)
                }
            elif params.model_type == "anisotropic":
                model_params = {
                    'tx': params.t,
                    'ty': getattr(params, 'ty', 0.6),
                    't_prime': params.t_prime,
                    'a': params.a
                }
            elif params.model_type == "honeycomb":
                model_params = {
                    't': params.t,
                    'a': params.a
                }
            elif params.model_type == "true_3d_cubic":
                model_params = {
                    't': params.t,
                    't_prime': params.t_prime,
                    'a': params.a
                }
            elif params.model_type == "true_3d_aniso":
                model_params = {
                    'tx': params.t,
                    'ty': getattr(params, 'ty', 0.6),
                    'tz': getattr(params, 't_z', 0.4),
                    'a': params.a
                }
            elif params.model_type == "true_3d_complex":
                model_params = {
                    't1': params.t,
                    't2': getattr(params, 't_z', 0.3),
                    't3': getattr(params, 't_prime', 0.1),
                    'a': params.a
                }
            elif params.model_type == "multiband_2band":
                model_params = {
                    't1': params.t,
                    't2': params.t_z, 
                    'hybridization': params.t_prime,
                    'offset': 1.0,  # Fixed offset
                    'a': params.a
                }
            elif params.model_type == "multiband_3band":
                model_params = {
                    't': params.t,
                    't_z': params.t_z,
                    't_inter': params.t_prime,
                    'a': params.a
                }
            elif params.model_type == "multiband_4band":
                model_params = {
                    't_d': params.t,
                    't_p': params.t_z,
                    't_pd': params.t_prime,
                    'Delta': 2.0,  # Fixed d-p separation
                    'a': params.a
                }
            elif params.model_type == "ute2_full":
                model_params = {
                    't_ff': params.t,
                    't_fp': params.t_z,
                    't_pp': params.t_prime,
                    't_soc': 0.8,  # Fixed SOC strength
                    'U_onsite': 2.0,
                    'Te_onsite': 1.5,
                    'anisotropy_a': 1.0,
                    'anisotropy_b': 0.8,
                    'anisotropy_c': 0.6,
                    'a': params.a
                }
            elif params.model_type == "ute2_simplified":
                model_params = {
                    't_U': params.t,
                    't_UTe': params.t_z,
                    't_aniso_ratio': params.t_prime,
                    'soc_strength': 0.3,  # Fixed SOC
                    'a': params.a
                }
            else:
                # Default parameters for unknown models
                model_params = {
                    't': params.t,
                    'a': params.a
                }
            
            model = create_model(params.model_type, **model_params)
            print(f"Created {params.model_type} model with appropriate parameters")
            
            # For tight-binding models, choose appropriate visualization
            if params.model_type.startswith("true_3d") or params.model_type.startswith("multiband") or params.model_type.startswith("ute2"):
                print("\n🌟 TRUE 3D MODE: Generating Full 3D Fermi Surface")
                print("-" * 50)
                visualize_true_3d_fermi_surface(model, config.name, mu=params.mu)
            else:
                print("\n🔬 TIGHT-BINDING MODE: Generating 3D Fermi Surface Only")
                print("-" * 50)
                visualize_tight_binding_fermi_surface(model, config.name, mu=params.mu)
            return True
        
        # Create and run simulation
        simulation = QPISimulation(params, impurity_positions, model)
        visualiser = QPIvisualiser(simulation)
        
        # Generate folder structure: outputs/{config_name}/
        import os
        outputs_base = "outputs"
        config_output_dir = os.path.join(outputs_base, config.name)
        
        # Create directories
        os.makedirs(config_output_dir, exist_ok=True)
        
        # Only create frames directory if save_frames is True
        frames_dir = None
        if save_frames:
            frames_dir = os.path.join(config_output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
        
        # Generate filenames
        anim_filename = os.path.join(config_output_dir, f"qpi_{config.name}.mp4")
        fourier_anim_filename = os.path.join(config_output_dir, f"qpi_{config.name}_fourier.mp4")
        snapshot_filename = os.path.join(config_output_dir, f"qpi_{config.name}_snapshot.png")
        
        # Create main QPI animation with or without individual frames
        ani = visualiser.create_animation(anim_filename, frames_dir=frames_dir)
        
        # Create separate fourier analysis animation 
        visualiser.create_fourier_animation(fourier_anim_filename, frames_dir=frames_dir)
        
        # Save snapshot at mid-energy
        visualiser.save_mid_energy_snapshot(snapshot_filename)
        
        # Print results
        print(f"\nSimulation completed!")
        print(f"Extracted {len(simulation.extracted_k)} dispersion points")
        if len(simulation.extracted_E) > 0:
            print(f"Energy range: {min(simulation.extracted_E):.2f} to {max(simulation.extracted_E):.2f}")
        print(f"Main QPI animation saved as: {anim_filename}")
        print(f"Fourier analysis animation saved as: {fourier_anim_filename}")
        print(f"Snapshot saved as: {snapshot_filename}")
        if save_frames and frames_dir:
            print(f"QPI frames saved to: {os.path.join(frames_dir, 'qpi')}")
            print(f"Fourier frames saved to: {os.path.join(frames_dir, 'fourier')}")
        else:
            print("Individual frames not saved (use --save-frames to save individual frames)")
        
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
    parser = argparse.ArgumentParser(description='Run QPI simulations with predefined configurations.')
    parser.add_argument('config_name', nargs='?', help='Configuration name to run')
    parser.add_argument('--save-frames', action='store_true', 
                        help='Save individual frames to frames folder')
    parser.add_argument('--list', action='store_true', 
                        help='List available configurations')
    
    args = parser.parse_args()
    
    if args.list or args.config_name == "list":
        list_available_configs()
        sys.exit(0)
    
    if not args.config_name:
        parser.print_help()
        print("\nAvailable configurations:")
        list_available_configs()
        sys.exit(1)
    
    success = run_simulation(args.config_name, save_frames=args.save_frames)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()