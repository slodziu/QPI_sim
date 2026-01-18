"""
Custom Layout QPI Visualization

Modified QPI visualization with custom layout optimized for parabolic dispersion:
- Top: Large real space LDOS plot for detailed spatial features
- Bottom: Two equal-width plots (momentum space QPI + energy dispersion) 
  providing complementary momentum-space information

Uses RdBu_r colormap for enhanced contrast in LDOS visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from qpi_G_OOP import QPIvisualiser, QPISimulation, SystemParameters, GreensFunction, ImpuritySystem, QPIAnalyzer, DEFAULT_LDOS_COLORMAP
from config import FAST_PREVIEW, RANDOM_30_IMPURITIES


class CustomLayoutQPIVisualiser(QPIvisualiser):
    """QPI Visualiser with custom 2x2 layout where top spans full width."""
    
    def _setup_figure(self):
        """Initialize the figure with custom gridspec layout."""
        # Create figure with custom gridspec
        # Top row: Real space plot spans full width
        # Bottom row: Two equal-width plots (momentum space + dispersion)
        self.fig = plt.figure(figsize=(16, 12), dpi=100)
        
        # Create gridspec with height ratios 2:1 (top:bottom)
        gs = gridspec.GridSpec(2, 2, figure=self.fig, 
                              height_ratios=[2, 1], 
                              width_ratios=[1, 1],
                              hspace=0.3, wspace=0.3,
                              top=0.95, bottom=0.08, 
                              left=0.08, right=0.95)
        
        # Top row: Real space LDOS plot (spans both columns)
        self.ax1 = self.fig.add_subplot(gs[0, :])  # Top row, all columns
        
        # Bottom row: Momentum space and dispersion plots
        self.ax2 = self.fig.add_subplot(gs[1, 0])  # Bottom left
        self.ax4 = self.fig.add_subplot(gs[1, 1])  # Bottom right
        
        # Set up real space plot (LDOS)
        self.im1 = self.ax1.imshow(
            np.zeros((self.params.gridsize, self.params.gridsize)), 
            origin='lower', cmap=DEFAULT_LDOS_COLORMAP, extent=[0, self.params.L, 0, self.params.L]
        )
        self.ax1.set_title("LDOS around impurities", fontsize=12)
        self.ax1.set_xlabel('x (physical units)', fontsize=10)
        self.ax1.set_ylabel('y (physical units)', fontsize=10)
        plt.colorbar(self.im1, ax=self.ax1, label='LDOS')
        
        # Add energy text to real space plot
        self.energy_text = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes,
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
        self.ax2.set_xlabel('kx (1/a)', fontsize=10)
        self.ax2.set_ylabel('ky (1/a)', fontsize=10)
        self.ax2.grid(False)
        plt.colorbar(self.im2, ax=self.ax2, label='log|FFT(LDOS)|')
        
        # Set up dispersion plot
        self.ax4.set_xlabel('k_F (1/length units)', fontsize=10)
        self.ax4.set_ylabel('Energy E', fontsize=10)
        self.ax4.set_title('Dispersion: Theory vs Extracted', fontsize=12)
        self.ax4.grid(True, alpha=0.3)
        
        # Set plot bounds based on energy range
        k_disp_max = np.sqrt(self.params.E_max) + 1
        
        # Plot theoretical parabolic dispersion: E = k²
        k_theory = np.linspace(-k_disp_max, k_disp_max, 400)
        E_theory = k_theory**2
        self.theory_lines = [self.ax4.plot(k_theory, E_theory, 'b-', linewidth=2, label='Theory: E = k²')[0]]
        
        self.extracted_scatter = self.ax4.scatter(
            [], [], c='red', s=50, alpha=0.7, label='From q=2k_F peaks'
        )
        
        self.ax4.legend()
        self.ax4.set_xlim(-k_disp_max, k_disp_max)
        self.ax4.set_ylim(self.params.E_min - 2, self.params.E_max + 2)


def run_custom_layout_simulation(config, config_name: str):
    """Run parabolic dispersion QPI simulation with custom layout visualization."""
    print(f"Running {config_name} simulation with custom layout...")
    print(f"Description: {config.description}")
    print("-" * 60)
    
    # Convert config to SystemParameters - keep it simple for parabolic dispersion
    params = SystemParameters(
        gridsize=config.gridsize,
        L=config.L,
        t=config.t,
        mu=config.mu,
        eta=config.eta,
        V_s=config.V_s,
        E_min=config.E_min,
        E_max=config.E_max,
        n_frames=config.n_frames
    )
    
    # Set up impurity positions
    impurity_positions = config.get_impurity_positions()
    print(f"Impurity positions: {impurity_positions}")
    
    # Create simulation components for parabolic dispersion (no model needed)
    greens = GreensFunction(params)
    impurities = ImpuritySystem(impurity_positions)
    analyzer = QPIAnalyzer(params)
    
    # Create simulation (parabolic dispersion - no model)
    sim = QPISimulation(params, impurity_positions, model=None)
    
    # Use custom layout visualizer
    vis = CustomLayoutQPIVisualiser(sim)
    
    # Calculate and save snapshot at mid-energy
    mid_energy = (params.E_min + params.E_max) / 2
    print(f"Generating snapshot at E = {mid_energy:.2f}")
    
    # Run single energy calculation
    LDOS, fft_display, fft_complex, peak_q = sim.run_single_energy(mid_energy)
    k_F = sim.energy_to_kF(mid_energy)
    
    # Update all plots
    vis._update_real_space_plot(LDOS, mid_energy, k_F)
    vis._update_momentum_plot(fft_display)
    vis._update_dispersion_plot(peak_q)
    
    # Save the figure
    output_filename = f"outputs/{config_name}_custom_layout.png"
    vis.fig.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved custom layout plot to {output_filename}")
    
    # Don't show immediately - return for batch display
    return vis


if __name__ == "__main__":
    print("Creating custom layout QPI visualizations for parabolic dispersion...")
    print("=" * 70)
    
    # Run fast preview
    print("1. FAST PREVIEW SIMULATION:")
    vis1 = run_custom_layout_simulation(FAST_PREVIEW, "fast_preview")
    
    print("\n" + "=" * 70)
    
    # Run 30 random impurities  
    print("2. RANDOM 30 IMPURITIES SIMULATION:")
    vis2 = run_custom_layout_simulation(RANDOM_30_IMPURITIES, "random_30_impurities")
    
    print("\n✓ Both parabolic dispersion QPI simulations completed!")
    print("✓ Two separate plots generated with custom layout")
    
    # Show both plots
    plt.show()
    
    input("Press Enter to close plots...")