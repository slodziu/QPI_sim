"""
QPI simulation runner with predefined configurations.

Usage:
    python run_qpi.py config_name [--save-frames]
    
Examples:
    python run_qpi.py high_quality_single
    python run_qpi.py random_30_impurities --save-frames
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from qpi_G_OOP import SystemParameters, QPISimulation, QPIvisualiser, DEFAULT_LDOS_COLORMAP
from config import get_config, list_available_configs


class CustomLayoutQPIVisualiser(QPIvisualiser):
    """QPI Visualiser with custom layout: top real space (full width), bottom two plots (half width each)."""
    
    def _setup_figure(self):
        """Initialize the figure with custom gridspec layout."""
        # Create figure with balanced layout:
        # Left: Real space LDOS plot (spans full height)  
        # Right: Two stacked plots (momentum space top, dispersion bottom)
        self.fig = plt.figure(figsize=(8, 5), dpi=300)
        
        gs = gridspec.GridSpec(2, 2, figure=self.fig, 
                              height_ratios=[1, 1],
                              width_ratios=[1.3, 1],
                              hspace=0.45, wspace=0.5,
                              top=0.95, bottom=0.08, 
                              left=0.08, right=0.95)
        
        # Left: Real space LDOS plot (spans both rows)
        self.ax1 = self.fig.add_subplot(gs[:, 0])  # All rows, first column
        
        # Right top: Momentum space plot
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Top right
        
        # Right bottom: Dispersion plot
        self.ax4 = self.fig.add_subplot(gs[1, 1])  # Bottom right
        
        # Set up real space plot (LDOS)
        self.im1 = self.ax1.imshow(
            np.zeros((self.params.gridsize, self.params.gridsize)), 
            origin='lower', cmap=DEFAULT_LDOS_COLORMAP, extent=[0, self.params.L, 0, self.params.L]
        )
        self.ax1.set_title("LDOS around impurities", fontsize=12)
        self.ax1.set_xlabel('x (physical units)', fontsize=10)
        self.ax1.set_ylabel('y (physical units)', fontsize=10)
        self.ax1.tick_params(axis='both', which='major', labelsize=8)
        self.ax1.tick_params(axis='both', which='major', labelsize=12)
        plt.colorbar(self.im1, ax=self.ax1, label='LDOS').ax.tick_params(labelsize=12)
        self.ax1.figure.axes[-1].set_ylabel('LDOS', fontsize=10)
        
        # Add panel label (a)
        self.ax1.text(0.02, 0.98, '(a)', transform=self.ax1.transAxes, fontsize=12, fontweight='bold',
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add energy text to real space plot
        self.energy_text = self.ax1.text(0.02, 0.88, '', transform=self.ax1.transAxes,
                                        verticalalignment='top', fontsize=9,
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set up momentum space plot
        dk = 2 * np.pi / self.params.L
        k_actual_max = dk * self.params.gridsize / 2
        
        # Set momentum space bounds based on energy range for better focus
        max_kF = np.sqrt(self.params.E_max)
        k_zoom = min(k_actual_max, max_kF * 5)  # Show up to 5*kF_max for QPI features
        
        self.im2 = self.ax2.imshow(
            np.zeros((self.params.gridsize, self.params.gridsize)), 
            origin='lower', cmap='plasma',
            extent=[-k_zoom, k_zoom, -k_zoom, k_zoom]  # Use zoomed range
        )
        self.ax2.set_xlim(-k_zoom, k_zoom)
        self.ax2.set_ylim(-k_zoom, k_zoom)
        self.ax2.set_title('Momentum Space: QPI Pattern', fontsize=12)
        self.ax2.set_xlabel('$k_x$ (1/a)', fontsize=10)
        self.ax2.set_ylabel('$k_y$ (1/a)', fontsize=10)
        self.ax2.tick_params(axis='both', which='major', labelsize=8)
        self.ax2.grid(False)
        self.ax2.tick_params(axis='both', which='major', labelsize=11)
        plt.colorbar(self.im2, ax=self.ax2, label='log|FFT(LDOS)|').ax.tick_params(labelsize=11)
        self.ax2.figure.axes[-1].set_ylabel('log|FFT(LDOS)|', fontsize=10)
        
        # Add panel label (b)
        self.ax2.text(0.05, 0.95, '(b)', transform=self.ax2.transAxes, fontsize=12, fontweight='bold',
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set up dispersion plot
        self.ax4.set_xlabel('$k_F$ (1/length units)', fontsize=10)
        self.ax4.set_ylabel('Energy E', fontsize=10)
        self.ax4.set_title('Dispersion: Theory vs Extracted', fontsize=12)
        self.ax4.tick_params(axis='both', which='major', labelsize=8)
        self.ax4.grid(True, alpha=0.3)
        self.ax4.tick_params(axis='both', which='major', labelsize=11)
        
        # Add panel label (c)
        self.ax4.text(0.05, 0.95, '(c)', transform=self.ax4.transAxes, fontsize=12, fontweight='bold',
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set plot bounds based on energy range
        k_disp_max = np.sqrt(self.params.E_max) + 1
        
        # Plot theoretical dispersion based on model type
        k_theory = np.linspace(-k_disp_max, k_disp_max, 400)
        self.theory_lines = self._plot_theoretical_dispersion(k_theory)
        
        self.extracted_scatter = self.ax4.scatter(
            [], [], c='red', s=50, alpha=0.7, label='From q=2k_F peaks'
        )
        
        self.ax4.legend(fontsize=9)
        self.ax4.set_xlim(-k_disp_max, k_disp_max)
        self.ax4.set_ylim(self.params.E_min - 2, self.params.E_max + 2)


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
            model_type="parabolic"
        )
        
        # Get impurity positions
        impurity_positions = config.get_impurity_positions()
        print(f"Impurity positions: {len(impurity_positions)} impurities")
        print(f"Using parabolic dispersion model")
        
        # Create and run simulation
        simulation = QPISimulation(params, impurity_positions, model)
        
        # Use custom layout visualizer
        visualiser = CustomLayoutQPIVisualiser(simulation)
        
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