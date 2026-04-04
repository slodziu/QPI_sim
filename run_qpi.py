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
from typing import Optional
from qpi_G_OOP import SystemParameters, QPISimulation, QPIvisualiser, DEFAULT_LDOS_COLORMAP
from config import get_config, list_available_configs


THESIS_LDOS_PERCENTILE = 85  # Percentile for LDOS clipping (0-100); lower = more contrast boost and oscillation visibility
THESIS_LDOS_VMIN = 0 # Asymmetric LDOS colorbar: minimum (set to None to use percentile-based vmin)
THESIS_LDOS_VMAX = 2e-2   # Asymmetric LDOS colorbar: maximum (set to None to use percentile-based vmax)
THESIS_MOMENTUM_EXTENT_MULTIPLIER = 2.5  # k-space half-width in units of k_F; shows 2k_F feature comfortably
THESIS_SHOW_2KF_ARROW = True  # Draw red 2k_F arrow in momentum panel (b)
THESIS_SHOW_DISPERSION_GUIDE_ARROW = True  # Draw guidance arrow in dispersion panel (c)
THESIS_POSTER_SIZE = (3600, 2400)  # Output poster figure pixel size (12x8 inches at 300 DPI) - fits A4 nicely with margins
THESIS_POSTER_DPI = 300  # DPI for poster export matches savefig DPI

# Snapshot energies for specific configurations
THESIS_SNAPSHOT_ENERGIES = {
    'high_quality_single': 5.0,      # E=5 for high_quality_single
    'fast_preview': 10.0,            # E=10 for fast_preview (within E_min=10, E_max=20)
    'random_30_impurities': 10.0,    # E=10 for random_30_impurities
}


class CustomLayoutQPIVisualiser(QPIvisualiser):
    """QPI Visualiser with custom layout: top real space (full width), bottom two plots (half width each)."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with thesis plotting parameters."""
        # Store thesis configuration BEFORE calling super().__init__()
        # because parent init calls _setup_figure which needs these attributes
        self.thesis_ldos_percentile = THESIS_LDOS_PERCENTILE
        self.thesis_ldos_vmin = THESIS_LDOS_VMIN
        self.thesis_ldos_vmax = THESIS_LDOS_VMAX
        self.thesis_momentum_extent_multiplier = THESIS_MOMENTUM_EXTENT_MULTIPLIER
        self.thesis_show_2kf_arrow = THESIS_SHOW_2KF_ARROW
        self.thesis_show_dispersion_guide_arrow = THESIS_SHOW_DISPERSION_GUIDE_ARROW
        self.thesis_poster_size = THESIS_POSTER_SIZE
        self.thesis_poster_dpi = THESIS_POSTER_DPI
        self.thesis_snapshot_energies = THESIS_SNAPSHOT_ENERGIES
        
        # Storage for guidance arrow artist (panel c)
        self.dispersion_guide_arrow = None
        
        # Now call parent init
        super().__init__(*args, **kwargs)
    
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
        
        # For thesis: use dynamic per-frame extent, but initialize with max energy
        # At max energy, we'll display ±(2.5 * kF_max) in k-space
        max_kF = self.sim.energy_to_kF(self.params.E_max)
        k_initial_extent = self.thesis_momentum_extent_multiplier * max_kF  # ±2.5*kF_max
        k_zoom = min(k_actual_max, k_initial_extent)  # Cap at FFT extent
        
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
        
        # Store k_zoom for later frame updates
        self.k_zoom_initial = k_zoom
        
        # Set up dispersion plot
        self.ax4.set_xlabel(r'$k_\mathrm{F}$ (1/length units)', fontsize=10)
        self.ax4.set_ylabel('Energy E', fontsize=10)
        self.ax4.set_title('Dispersion: Theory vs Extracted', fontsize=12)
        self.ax4.tick_params(axis='both', which='major', labelsize=8)
        self.ax4.grid(True, alpha=0.3)
        self.ax4.tick_params(axis='both', which='major', labelsize=11)
        
        # Add panel label (c)
        self.ax4.text(0.05, 0.95, '(c)', transform=self.ax4.transAxes, fontsize=12, fontweight='bold',
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Store parameters for guidance arrow (will be computed per-frame)
        self.e_mid = (self.params.E_min + self.params.E_max) / 2  # Mid-energy for guidance arrow
        
        # Set plot bounds based on energy range
        k_disp_max = np.sqrt(self.params.E_max) + 1
        
        # Plot theoretical dispersion based on model type
        k_theory = np.linspace(-k_disp_max, k_disp_max, 400)
        self.theory_lines = self._plot_theoretical_dispersion(k_theory)
        
        self.extracted_scatter = self.ax4.scatter(
            [], [], c='red', s=50, alpha=0.7, label=r'From q=2$k_\mathrm{F}$ peaks'
        )
        
        self.ax4.legend(fontsize=9)
        self.ax4.set_xlim(-k_disp_max, k_disp_max)
        self.ax4.set_ylim(self.params.E_min - 2, self.params.E_max + 2)

    def animate_frame(self, frame_idx: int):
        """
        Animate a single frame with thesis enhancements (dynamic extents, arrows, etc.).
        
        Args:
            frame_idx: Frame index in the animation sequence
            
        Returns:
            List of artists that were modified
        """
        energy = self.params.E_min + (self.params.E_max - self.params.E_min) * frame_idx / (self.params.n_frames - 1)
        k_F = self.sim.energy_to_kF(energy)
        
        LDOS, fft_display, fft_complex, peak_q = self.sim.run_single_energy(energy)
        
        self._update_real_space_plot(LDOS, energy, k_F)
        self._update_momentum_plot(fft_display, energy=energy)  # Pass energy for dynamic extent
        self._update_dispersion_plot(peak_q, energy=energy)  # Pass energy for guidance arrow
        
        # Return all artists that need to be redrawn (including theory lines for persistence)
        artists = [self.im1, self.im2, self.energy_text, self.extracted_scatter]
        if hasattr(self, 'theory_lines'):
            artists.extend(self.theory_lines)
        if self.dispersion_guide_arrow is not None:
            artists.append(self.dispersion_guide_arrow)
        return artists

    def _update_real_space_plot(self, LDOS: np.ndarray, energy: float, k_F: float):
        """
        Update the real space LDOS plot with asymmetric or percentile-based contrast (thesis version).
        
        Args:
            LDOS: 2D LDOS array
            energy: Current energy value
            k_F: Fermi wavevector at current energy
        """
        self.im1.set_data(LDOS)
        
        # Use asymmetric colorbar if vmin/vmax are specified, otherwise use percentile-based
        if self.thesis_ldos_vmin is not None and self.thesis_ldos_vmax is not None:
            vmin = self.thesis_ldos_vmin
            vmax = self.thesis_ldos_vmax
        else:
            # Fall back to percentile-based contrast clipping for thesis quality
            abs_ldos = np.abs(LDOS)
            try:
                # Calculate percentile-based limits for robust contrast
                vmax_raw = np.percentile(abs_ldos, self.thesis_ldos_percentile)
                vmax_raw = max(vmax_raw, np.max(abs_ldos) * 0.01)  # Avoid collapse to zero
                vmin = -vmax_raw
                vmax = vmax_raw
            except:
                # Fallback if percentile calculation fails
                vmax = np.max(abs_ldos)
                vmin = -vmax * 0.1
        
        self.im1.set_clim(vmin=vmin, vmax=vmax)
        self.ax1.set_title(f"LDOS (E = {energy:.3f}, $k_\\mathrm{{F}}$ = {k_F:.2f})")
        self.energy_text.set_text(f'E = {energy:.2f}\n$k_\\mathrm{{F}}$ = {k_F:.2f}')
        
        for artist in self.ax1.lines:
            artist.remove()
        
        if len(self.sim.impurities.positions) > 1 and len(self.sim.impurities.positions) <= 5:
            self.ax1.legend(loc='upper right')
    
    def _update_momentum_plot(self, fft_display: np.ndarray, energy: float = None):
        """Update momentum plot with ±2.5*k_F extent"""
        fft_log = np.log10(fft_display + 1)
        
        # Calculate proper k-space bounds with thesis multiplier (dynamic per frame)
        dk = 2 * np.pi / self.params.L
        k_actual_max = dk * self.params.gridsize / 2
        
        # If energy is provided, compute current k_F; otherwise use max
        if energy is not None:
            current_kF = self.sim.energy_to_kF(energy)
        else:
            current_kF = self.sim.energy_to_kF(self.params.E_max)
        
        # Dynamic extent: ±2.5 * k_F for proper framing of the 2k_F feature
        k_zoom = self.thesis_momentum_extent_multiplier * current_kF
        
        # Crop the FFT data to match the zoom range
        center = fft_log.shape[0] // 2
        pixels_to_show = int(k_zoom / dk) if dk > 0 else center
        pixels_to_show = min(pixels_to_show, center)
        
        fft_cropped = fft_log[center-pixels_to_show:center+pixels_to_show, 
                             center-pixels_to_show:center+pixels_to_show]
        
        self.im2.set_data(fft_cropped)
        self.im2.set_extent([-k_zoom, k_zoom, -k_zoom, k_zoom])
        self.ax2.set_xlim(-k_zoom, k_zoom)
        self.ax2.set_ylim(-k_zoom, k_zoom)
        
        vmin_fft = np.min(fft_cropped)
        vmax_fft = np.max(fft_cropped)
        self.im2.set_clim(vmin=vmin_fft, vmax=vmax_fft)
        
        # Clean up old 2k_F arrow and annotations before redrawing
        if self.thesis_show_2kf_arrow and current_kF > 0:
            # Remove all patches (arrows) and texts (labels) except panel label
            for artist in list(self.ax2.patches):
                artist.remove()
            for txt in list(self.ax2.texts):
                if '(b)' not in txt.get_text():  # Keep panel label
                    txt.remove()
            
            # Remove legend if present
            legend = self.ax2.get_legend()
            if legend is not None:
                legend.remove()
            
            expected_2kF = 2 * current_kF
            if expected_2kF <= k_zoom:
                # Draw fresh arrow from origin to 2k_F
                self.ax2.annotate('', xy=(expected_2kF, 0), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
                # Add text label directly instead of legend to avoid accumulation
                self.ax2.text(expected_2kF * 0.5, k_zoom * 0.15, r'$2k_\mathrm{F}$',
                            fontsize=8, color='red', weight='bold',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    def _update_dispersion_plot(self, peak_q: Optional[float], energy: float = None):
        """
        Update the dispersion plot with guidance arrow (thesis version).
        
        Args:
            peak_q: Detected peak position in momentum space
            energy: Current energy value (for mid-energy guidance arrow placement)
        """
        self.sim.update_dispersion_data(peak_q)
        
        if len(self.sim.extracted_k) > 0:
            self.extracted_scatter.set_offsets(
                np.column_stack([self.sim.extracted_k, self.sim.extracted_E])
            )
        
        # Draw guidance arrow from (0, E_mid) to theory curve at E_mid
        if self.thesis_show_dispersion_guide_arrow:
            # Remove old guidance arrow and label if they exist
            if self.dispersion_guide_arrow is not None:
                self.dispersion_guide_arrow.remove()
                self.dispersion_guide_arrow = None
            # Remove old label texts from dispersion plot
            for txt in list(self.ax4.texts):
                if '2k' in txt.get_text() or 'mathrm{F}' in txt.get_text():
                    txt.remove()
            
            # Compute target point on E=k^2 at mid-energy
            # For parabolic: E = k^2, so k = sqrt(E)
            k_at_mid = np.sqrt(max(0, self.e_mid))
            
            # Draw bidirectional arrow from -k_F to +k_F at E_mid (shows 2k_F separation)
            if k_at_mid <= self.ax4.get_xlim()[1]:
                self.dispersion_guide_arrow = self.ax4.annotate(
                    '', xy=(k_at_mid, self.e_mid), xytext=(-k_at_mid, self.e_mid),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5, alpha=0.7)
                )
                # Add label to the arrow at center
                self.ax4.text(0, self.e_mid + 1.0, r'$2k_\mathrm{F}$',
                            fontsize=9, color='red', weight='bold',
                            horizontalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    def save_snapshot_at_energy(self, energy: float, filename: str):
        """
        Save a snapshot at a specific energy value.
        
        Args:
            energy: The energy at which to save the snapshot
            filename: Output filename
        """
        # Clamp energy to valid range
        energy = max(self.params.E_min, min(self.params.E_max, energy))
        
        # Run simulation at this energy and update plots
        k_F = self.sim.energy_to_kF(energy)
        LDOS, fft_display, fft_complex, peak_q = self.sim.run_single_energy(energy)
        
        self._update_real_space_plot(LDOS, energy, k_F)
        self._update_momentum_plot(fft_display, energy=energy)
        self._update_dispersion_plot(peak_q, energy=energy)
        
        # Save the figure
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"✓ Saved snapshot at E={energy:.2f} to {filename}")


def run_simulation(config_name: str, save_frames: bool = False, poster_frame: bool = False):
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
        simulation = QPISimulation(params, impurity_positions, model=None)
        
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
        
        # Save snapshot at config-specific energy or mid-energy
        snapshot_energy = THESIS_SNAPSHOT_ENERGIES.get(config.name, None)
        if snapshot_energy is not None:
            visualiser.save_snapshot_at_energy(snapshot_energy, snapshot_filename)
        else:
            visualiser.save_mid_energy_snapshot(snapshot_filename)

        # Optionally save a single zoomed poster frame
        if poster_frame:
            poster_dir = os.path.join(config_output_dir, "poster")
            visualiser.save_poster_frame(poster_dir, frame=0)
        
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
    parser.add_argument('--poster-frame', action='store_true',
                        help='Save a single zoomed azimuthal poster figure (first energy frame)')
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
    
    success = run_simulation(args.config_name, save_frames=args.save_frames, poster_frame=args.poster_frame)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()