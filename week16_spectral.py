#!/usr/bin/env python3
"""
Fermi Surface with 5f Orbital Spectral Weighting
=================================================
Plot UTe2 Fermi surface as thick contour lines weighted by U 5f orbital character.
"""

import numpy as np
import matplotlib.pyplot as plt
from UTe2_fixed import H_full, a, b, c, set_parameters
import os

def calculate_5f_weighted_fermi_surface(kz=0.0, resolution=512):
    """
    Calculate Fermi surface with 5f orbital spectral weights (vectorized).
    
    Parameters:
    - kz: out-of-plane momentum (typically 0 for 2D slice)
    - resolution: k-space grid resolution
    
    Returns:
    - energies: band energies [kx, ky, band]
    - weights_5f: 5f orbital weights [kx, ky, band]
    - kx_vals, ky_vals: momentum grids
    """
    print(f"Computing 5f-weighted Fermi surface at kz={kz:.3f}...")
    print(f"Resolution: {resolution}×{resolution}")
    
    # Create k-space grid
    kx_vals = np.linspace(-2*np.pi/a, 2*np.pi/a, resolution)
    ky_vals = np.linspace(-6*np.pi/b, 6*np.pi/b, resolution)
    
    # Create meshgrid and flatten for vectorized computation
    KX, KY = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    kx_flat = KX.flatten()
    ky_flat = KY.flatten()
    n_points = len(kx_flat)
    
    print("Diagonalizing Hamiltonian on k-space grid (vectorized)...")
    
    # Initialize arrays for all k-points
    n_bands = 4  # UTe2 has 4 bands
    H_array = np.zeros((n_points, n_bands, n_bands), dtype=complex)
    
    # Build Hamiltonians for all k-points at once
    for i in range(n_points):
        H_array[i] = H_full(kx_flat[i], ky_flat[i], kz)
    
    # Vectorized diagonalization
    eigenvals, eigenvecs = np.linalg.eigh(H_array)  # Shape: (n_points, 4), (n_points, 4, 4)
    
    # Reshape energies back to grid
    energies = eigenvals.reshape(resolution, resolution, n_bands)
    
    # Calculate 5f orbital weights (vectorized)
    # Hamiltonian basis: [U1, U2, Te1, Te2]
    # U 5f orbitals are indices 0 and 1
    u1_weights = np.abs(eigenvecs[:, 0, :])**2  # Shape: (n_points, 4)
    u2_weights = np.abs(eigenvecs[:, 1, :])**2  # Shape: (n_points, 4)
    weights_5f_flat = u1_weights + u2_weights
    
    # Reshape back to grid
    weights_5f = weights_5f_flat.reshape(resolution, resolution, n_bands)
    
    print("Calculation complete!")
    print(f"Energy range: {energies.min():.3f} to {energies.max():.3f} eV")
    print(f"5f weight range: {weights_5f.min():.3f} to {weights_5f.max():.3f}")
    
    return energies, weights_5f, kx_vals, ky_vals


def plot_5f_weighted_fermi_contours(energies, weights_5f, kx_vals, ky_vals, 
                                     kz=0.0, param_set='odd_parity_paper'):
    """
    Plot Fermi surface as contour lines with intensity based on 5f character.
    
    Parameters:
    - energies: band energies [kx, ky, band]
    - weights_5f: 5f orbital weights [kx, ky, band]
    - kx_vals, ky_vals: momentum grids
    - kz: out-of-plane momentum
    - param_set: parameter set name
    """
    print("\nCreating 5f-weighted Fermi surface contour plot...")
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Convert k-space to units of π/a and π/b
    kx_plot = kx_vals / (np.pi/a)
    ky_plot = ky_vals / (np.pi/b)
    
    # Create meshgrid for plotting - ky on x-axis, kx on y-axis
    # Use 'ij' indexing: KY[i,j] = ky_plot[i], KX[i,j] = kx_plot[j]
    KY, KX = np.meshgrid(ky_plot, kx_plot, indexing='ij')
    
    # Plot each band's Fermi surface with constant thickness but varying intensity
    n_bands = energies.shape[2]
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    # Constant linewidth for all contours
    linewidth = 8
    
    for band in range(n_bands):
        print(f"  Plotting {band_labels[band]}...")
        
        # Extract energies and weights for this band
        # Transpose: energies[i_kx, i_ky] -> E_band[i_ky, i_kx] to match meshgrid indexing
        E_band = energies[:, :, band].T  # Now [ky, kx] for correct orientation
        W_band = weights_5f[:, :, band].T
        
        # Find Fermi surface contours (E = 0)
        contour_levels = [0.0]
        
        # Create contour plot with ky on x-axis, kx on y-axis
        contours = ax.contour(KY, KX, E_band, levels=contour_levels, 
                              linewidths=0, alpha=0)
        
        # Extract contour segments and plot with intensity based on 5f weight
        from scipy.interpolate import RegularGridInterpolator
        from matplotlib.collections import LineCollection
        
        # Interpolator still uses original [kx, ky] indexing
        weight_interp = RegularGridInterpolator((kx_plot, ky_plot), weights_5f[:, :, band],
                                               method='linear', bounds_error=False,
                                               fill_value=0)
        
        # Iterate through all contour segments using allsegs
        for i_level, segments in enumerate(contours.allsegs):
            for seg_idx, segment in enumerate(segments):
                if len(segment) < 2:
                    continue
                
                # segment is a (N, 2) array - now [ky, kx] due to swap
                ky_contour = segment[:, 0]
                kx_contour = segment[:, 1]
                
                # Interpolate 5f weights along the contour - need [kx, ky] order
                weights_along_contour = weight_interp(np.column_stack([kx_contour, ky_contour]))
                
                # Create line segments for gradient coloring
                # Break contour into small segments, each with its own color
                points = np.array([ky_contour, kx_contour]).T.reshape(-1, 1, 2)
                segments_for_collection = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Map weights to grayscale: black (0) = max weight, white (1) = no weight
                # Use the average weight of the two endpoints for each segment
                segment_weights = (weights_along_contour[:-1] + weights_along_contour[1:]) / 2
                gray_values = 1.0 - segment_weights
                
                # Create colors array (N, 4) for RGBA
                colors = np.zeros((len(gray_values), 4))
                colors[:, 0] = gray_values  # R
                colors[:, 1] = gray_values  # G
                colors[:, 2] = gray_values  # B
                colors[:, 3] = 1.0          # Alpha
                
                # Create LineCollection with varying colors
                lc = LineCollection(segments_for_collection, colors=colors, 
                                   linewidths=linewidth, capstyle='round', joinstyle='round')
                ax.add_collection(lc)
                
                # Add label only for first segment of first contour of each band
                if seg_idx == 0 and i_level == 0:
                    # Add a dummy line for the legend
                    ax.plot([], [], color='gray', linewidth=linewidth, 
                           label=band_labels[band])
    
    # Style the plot
    ax.set_xlabel(r'$k_y$ (π/b)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$k_x$ (π/a)', fontsize=16, fontweight='bold')
    ax.set_title(f'UTe₂ Fermi Surface with U 5f Orbital Weighting\n' +
                f'(Black = max 5f character, White = no 5f character, kz={kz:.3f})', 
                fontsize=18, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.2, linewidth=1)
    ax.axvline(0, color='k', linestyle='-', alpha=0.2, linewidth=1)
    
    # Add Brillouin zone boundaries - swap for new axis orientation
    ax.axvline(-3, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(3, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(-1, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    # Set axis limits - swap for new axis orientation
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal', adjustable='box')

    # Add colorbar to show intensity scale
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    

    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Save plot
    os.makedirs('outputs/week16', exist_ok=True)
    save_path = f'outputs/week16/fermi_surface_5f_weighted_contours_kz_{kz:.3f}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {save_path}")
    
    plt.show()
    
    return fig, ax


def find_fermi_surface_point(energies, kx_plot, ky_plot, target_ky, below_axis=True):
    """
    Find a point on the Fermi surface at a specific ky value.
    
    Parameters:
    - energies: band energies [kx, ky, band]
    - kx_plot: kx values in π/a units
    - ky_plot: ky values in π/b units
    - target_ky: target ky value in π/b units
    - below_axis: if True, find point with kx < 0, else kx > 0
    
    Returns:
    - (kx, ky) point on Fermi surface, or None if not found
    """
    # Find closest ky index
    ky_idx = np.argmin(np.abs(ky_plot - target_ky))
    actual_ky = ky_plot[ky_idx]
    
    # Search across all bands for Fermi crossings at this ky
    for band in range(energies.shape[2]):
        E_slice = energies[:, ky_idx, band]
        
        # Find zero crossings (Fermi surface)
        sign_changes = np.diff(np.sign(E_slice))
        crossings = np.where(sign_changes != 0)[0]
        
        for crossing_idx in crossings:
            # Interpolate to get more accurate kx position
            kx1, kx2 = kx_plot[crossing_idx], kx_plot[crossing_idx + 1]
            E1, E2 = E_slice[crossing_idx], E_slice[crossing_idx + 1]
            
            if E1 != E2:
                kx_cross = kx1 - E1 * (kx2 - kx1) / (E2 - E1)
            else:
                kx_cross = (kx1 + kx2) / 2
            
            # Check if it's on the correct side of the axis
            if below_axis and kx_cross < 0:
                return (kx_cross, actual_ky)
            elif not below_axis and kx_cross > 0:
                return (kx_cross, actual_ky)
    
    return None


def plot_5f_cosine_modulated_fermi_contours(energies, weights_5f, kx_vals, ky_vals, 
                                             kz=0.0, param_set='odd_parity_paper'):
    """
    Plot Fermi surface with 5f weights modulated by |cos(ky)| for enhanced periodicity.
    
    Parameters:
    - energies: band energies [kx, ky, band]
    - weights_5f: 5f orbital weights [kx, ky, band]
    - kx_vals, ky_vals: momentum grids
    - kz: out-of-plane momentum
    - param_set: parameter set name
    """
    print("\nCreating cosine-modulated 5f-weighted Fermi surface contour plot...")
    
    # Apply cosine modulation: multiply by |cos(ky)| which peaks at ky = n*π/b
    # ky_vals are already in absolute units, so cos(ky * b) gives peaks at π/b intervals
    KX_grid, KY_grid = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    cosine_modulation = np.abs(np.cos(KY_grid * b))  # Peaks at ky = n*π/b
    
    # Apply modulation to all bands
    weights_5f_modulated = np.zeros_like(weights_5f)
    for band in range(weights_5f.shape[2]):
        weights_5f_modulated[:, :, band] = weights_5f[:, :, band] * cosine_modulation
    
    print(f"  Applied |cos(ky)| modulation")
    print(f"  Modulated 5f weight range: {weights_5f_modulated.min():.3f} to {weights_5f_modulated.max():.3f}")
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Convert k-space to units of π/a and π/b
    kx_plot = kx_vals / (np.pi/a)
    ky_plot = ky_vals / (np.pi/b)
    
    # Create meshgrid for plotting - ky on x-axis, kx on y-axis
    KY, KX = np.meshgrid(ky_plot, kx_plot, indexing='ij')
    
    # Plot each band's Fermi surface with constant thickness but varying intensity
    n_bands = energies.shape[2]
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    # Constant linewidth for all contours
    linewidth = 6.5
    
    for band in range(n_bands):
        print(f"  Plotting {band_labels[band]}...")
        
        # Extract energies and modulated weights for this band
        E_band = energies[:, :, band].T  # Transpose for correct orientation
        
        # Find Fermi surface contours (E = 0)
        contour_levels = [0.0]
        
        # Create contour plot with ky on x-axis, kx on y-axis
        contours = ax.contour(KY, KX, E_band, levels=contour_levels, 
                              linewidths=0, alpha=0)
        
        # Extract contour segments and plot with intensity based on modulated 5f weight
        from scipy.interpolate import RegularGridInterpolator
        from matplotlib.collections import LineCollection
        
        # Interpolator uses original [kx, ky] indexing with modulated weights
        weight_interp = RegularGridInterpolator((kx_plot, ky_plot), weights_5f_modulated[:, :, band],
                                               method='linear', bounds_error=False,
                                               fill_value=0)
        
        # Iterate through all contour segments
        for i_level, segments in enumerate(contours.allsegs):
            for seg_idx, segment in enumerate(segments):
                if len(segment) < 2:
                    continue
                
                # segment is a (N, 2) array - [ky, kx]
                ky_contour = segment[:, 0]
                kx_contour = segment[:, 1]
                
                # Interpolate modulated 5f weights along the contour
                weights_along_contour = weight_interp(np.column_stack([kx_contour, ky_contour]))
                
                # Create line segments for gradient coloring
                points = np.array([ky_contour, kx_contour]).T.reshape(-1, 1, 2)
                segments_for_collection = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Map weights to grayscale: black (0) = max weight, white (1) = no weight
                segment_weights = (weights_along_contour[:-1] + weights_along_contour[1:]) / 2
                gray_values = 1.0 - segment_weights
                
                # Create colors array (N, 4) for RGBA
                colors = np.zeros((len(gray_values), 4))
                colors[:, 0] = gray_values  # R
                colors[:, 1] = gray_values  # G
                colors[:, 2] = gray_values  # B
                colors[:, 3] = 1.0          # Alpha
                
                # Create LineCollection with varying colors
                lc = LineCollection(segments_for_collection, colors=colors, 
                                   linewidths=linewidth, capstyle='round', joinstyle='round')
                ax.add_collection(lc)
                
                # Add label only for first segment of first contour of each band
                if seg_idx == 0 and i_level == 0:
                    ax.plot([], [], color='gray', linewidth=linewidth, 
                           label=band_labels[band])
    
    # Style the plot
    ax.set_xlabel(r'$k_y$ (π/b)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$k_x$ (π/a)', fontsize=16, fontweight='bold')
    ax.set_title(f'UTe₂ Fermi Surface with |cos(ky)|-Modulated U 5f Weighting\n' +
                f'(Black = max 5f × |cos(ky)|, White = zero, kz={kz:.3f})', 
                fontsize=18, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.2, linewidth=1)
    ax.axvline(0, color='k', linestyle='-', alpha=0.2, linewidth=1)
    
    # Add Brillouin zone boundaries
    ax.axvline(-3, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(3, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(-1, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    # Add vertical lines at ky = n*π/b (integer multiples) to show cosine peaks
    for n in range(-3, 4):
        ax.axvline(n, color='red', linestyle=':', alpha=0.3, linewidth=0.8)
    
    # Add wavevectors p1-p6
    print("\n  Adding wavevectors...")
    
    # Wavevector definitions (in units of 2π/a, 2π/b)
    wavevectors = {
        'p1': (0.29, 0),
        'p2': (0.43, 1),
        'p3': (0.29, 2),
        'p4': (0, 2),
        'p5': (-0.14, 1),
        'p6': (0.57, 0)
    }
    
    # Convert to units of π/a, π/b (multiply by 2)
    wavevectors_pi = {k: (v[0]*2, v[1]*2) for k, v in wavevectors.items()}
    
    # Find origin points on Fermi surface
    # p1-p5 originate from ky=-2π/b (below kx axis)
    # p6 originates from ky=0 (below kx axis)
    origin_p1_p5 = find_fermi_surface_point(energies, kx_plot, ky_plot, -2.0, below_axis=True)
    origin_p6 = find_fermi_surface_point(energies, kx_plot, ky_plot, 0.0, below_axis=True)
    
    if origin_p1_p5 is None:
        print("  Warning: Could not find Fermi surface point at ky=-2π/b")
        origin_p1_p5 = (-0.5, -2.0)  # Fallback
    if origin_p6 is None:
        print("  Warning: Could not find Fermi surface point at ky=0")
        origin_p6 = (-0.5, 0.0)  # Fallback
    
    print(f"  Origin for p1-p5: ({origin_p1_p5[0]:.3f}, {origin_p1_p5[1]:.3f}) π/(a,b)")
    print(f"  Origin for p6: ({origin_p6[0]:.3f}, {origin_p6[1]:.3f}) π/(a,b)")
    
    # Colors for wavevectors
    vector_colors = {
        'p1': '#FF0000',  # Red
        'p2': '#0000FF',  # Blue
        'p3': '#FFFF00',  # Yellow
        'p4': '#00FF00',  # Green
        'p5': '#FF8800',  # Orange
        'p6': '#FF00FF'   # Magenta
    }
    
    # Label positioning offsets and alignments to avoid overlap
    label_positions = {
        'p1': {'offset': (0.15, 0.08), 'ha': 'left', 'va': 'bottom'},
        'p2': {'offset': (0.15, 0.08), 'ha': 'left', 'va': 'bottom'},
        'p3': {'offset': (0.15, 0.08), 'ha': 'left', 'va': 'bottom'},
        'p4': {'offset': (0.0, 0.12), 'ha': 'center', 'va': 'bottom'},
        'p5': {'offset': (-0.15, 0.08), 'ha': 'right', 'va': 'bottom'},
        'p6': {'offset': (0.15, 0.0), 'ha': 'left', 'va': 'center'}
    }
    
    # Plot wavevectors
    for label, (qx_pi, qy_pi) in wavevectors_pi.items():
        # Determine origin
        if label == 'p6':
            origin = origin_p6
        else:
            origin = origin_p1_p5
        
        # Calculate endpoint: origin + vector
        endpoint = (origin[0] + qx_pi, origin[1] + qy_pi)
        
        # Plot arrow (in ky, kx order for our swapped axes)
        ax.annotate('', 
                   xy=(endpoint[1], endpoint[0]),  # (ky, kx)
                   xytext=(origin[1], origin[0]),  # (ky, kx)
                   arrowprops=dict(arrowstyle='->', color=vector_colors[label], 
                                  lw=3, alpha=0.9, shrinkA=0, shrinkB=0))
        
        # Add marker at endpoint
        ax.plot(endpoint[1], endpoint[0], 'o', color=vector_colors[label], 
               markersize=8, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        
        # Add label with smart positioning (offset in ky, kx space)
        pos_info = label_positions[label]
        label_ky = endpoint[1] + pos_info['offset'][0]
        label_kx = endpoint[0] + pos_info['offset'][1]
        
        ax.text(label_ky, label_kx, label, 
               color='black', fontsize=12, fontweight='bold',
               horizontalalignment=pos_info['ha'],
               verticalalignment=pos_info['va'],
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.95),
               zorder=11)
    
    # Set axis limits
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal', adjustable='box')

    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Save plot
    os.makedirs('outputs/week16', exist_ok=True)
    save_path = f'outputs/week16/fermi_surface_5f_cosine_modulated_kz_{kz:.3f}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved cosine-modulated plot to: {save_path}")

    return fig, ax


def save_jdos_data(energies, weights_5f, kx_vals, ky_vals, kz=0.0, param_set='odd_parity_paper'):
    """
    Save data needed for JDOS calculation.
    
    Parameters:
    - energies: band energies [kx, ky, band]
    - weights_5f: 5f orbital weights [kx, ky, band]
    - kx_vals, ky_vals: momentum grids
    - kz: out-of-plane momentum
    - param_set: parameter set name
    """
    print("\nSaving JDOS data...")
    
    # Create output directory
    output_dir = 'raw_data_out/week16'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arrays
    np.save(f'{output_dir}/energies_kz_{kz:.3f}.npy', energies)
    np.save(f'{output_dir}/weights_5f_kz_{kz:.3f}.npy', weights_5f)
    np.save(f'{output_dir}/kx_vals.npy', kx_vals)
    np.save(f'{output_dir}/ky_vals.npy', ky_vals)
    
    # Save metadata
    metadata = {
        'kz': kz,
        'param_set': param_set,
        'resolution': energies.shape[0],
        'n_bands': energies.shape[2],
        'kx_range': [kx_vals.min(), kx_vals.max()],
        'ky_range': [ky_vals.min(), ky_vals.max()],
        'energy_range': [energies.min(), energies.max()],
        'weight_range': [weights_5f.min(), weights_5f.max()]
    }
    
    import json
    with open(f'{output_dir}/metadata_kz_{kz:.3f}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved to {output_dir}/")
    print(f"  - energies_kz_{kz:.3f}.npy: shape {energies.shape}")
    print(f"  - weights_5f_kz_{kz:.3f}.npy: shape {weights_5f.shape}")
    print(f"  - kx_vals.npy: {len(kx_vals)} points")
    print(f"  - ky_vals.npy: {len(ky_vals)} points")
    print(f"  - metadata_kz_{kz:.3f}.json")


def main():
    """Main function to generate 5f-weighted Fermi surface plot."""

    
    # Set default UTe2 parameters
    param_set = 'odd_parity_paper_2'
    set_parameters(param_set)

    # Calculate Fermi surface with 5f weights
    kz = 0.0  # 2D slice at kz=0
    resolution = 2001  # Good balance of speed and quality
    
    energies, weights_5f, kx_vals, ky_vals = calculate_5f_weighted_fermi_surface(
        kz=kz, resolution=resolution
    )
    
    # Save data for JDOS calculation
    save_jdos_data(energies, weights_5f, kx_vals, ky_vals, kz=kz, param_set=param_set)

    
    fig, ax = plot_5f_cosine_modulated_fermi_contours(
        energies, weights_5f, kx_vals, ky_vals, 
        kz=kz, param_set=param_set
    )



if __name__ == "__main__":
    main()
