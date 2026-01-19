#!/usr/bin/env python3
"""
UTe2 2D Fermi Surface Contour Plotter
=====================================
Creates clean 2D Fermi surface contours without gap nodes,
based on the successful approach from UTe2_fixed.py.

"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# MODEL PARAMETERS (from UTe2_fixed.py)
# =============================================================================

# Lattice constants (nm) - UTe2 orthorhombic structure
a = 0.41   
b = 0.61   
c = 1.39   

# U parameters 
muU = -0.35
DeltaU = 0.38
tU = 0.17
tpU = 0.08
tch_U = 0.01
tpch_U = 0.01
tz_U = -0.0375

# Te parameters 
muTe = -2.25
DeltaTe = -1.4
tTe = -1.5
tch_Te = 0
tz_Te = -0.05
delta = 0.13

# =============================================================================
# HAMILTONIAN FUNCTIONS (from UTe2_fixed.py)
# =============================================================================

def HU_block(kx, ky, kz):
    """2x2 Hamiltonian for U orbitals - Vectorized version"""
    diag = muU - 2*tU*np.cos(kx*a) - 2*tch_U*np.cos(ky*b)
    real_off = -DeltaU - 2*tpU*np.cos(kx*a) - 2*tpch_U*np.cos(ky*b)
    complex_amp = -4 * tz_U * np.exp(-1j * kz * c / 2) * np.cos(kx * a / 2) * np.cos(ky * b / 2)
    
    # Handle scalar
    if np.isscalar(kx):
        H = np.zeros((2,2), dtype=complex)
        H[0,0] = diag
        H[1,1] = diag
        H[0,1] = real_off + complex_amp
        H[1,0] = real_off + np.conj(complex_amp)
    else:
        # Vectorized
        shape = kx.shape
        H = np.zeros(shape + (2, 2), dtype=complex)
        H[..., 0, 0] = diag
        H[..., 1, 1] = diag
        H[..., 0, 1] = real_off + complex_amp
        H[..., 1, 0] = real_off + np.conj(complex_amp)
    return H

def HTe_block(kx, ky, kz):
    """2x2 Hamiltonian for Te orbitals - Vectorized version"""
    diag = muTe - 2*tch_Te*np.cos(kx*a)
    real_off = -DeltaTe
    complex_term1 = -tTe * np.exp(-1j * ky * b)
    complex_term2 = -2*tz_Te * np.cos(kz * c / 2) * np.cos(kx * a / 2) * np.cos(ky * b / 2)
    
    if np.isscalar(kx):
        H = np.zeros((2,2), dtype=complex)
        H[0,0] = diag
        H[1,1] = diag
        H[0,1] = real_off + complex_term1 + complex_term2
        H[1,0] = real_off + np.conj(complex_term1) + np.conj(complex_term2)
    else:
        shape = kx.shape
        H = np.zeros(shape + (2, 2), dtype=complex)
        H[..., 0, 0] = diag
        H[..., 1, 1] = diag
        H[..., 0, 1] = real_off + complex_term1 + complex_term2
        H[..., 1, 0] = real_off + np.conj(complex_term1) + np.conj(complex_term2)
    return H

def H_full(kx, ky, kz):
    """Full 4x4 Hamiltonian with U-Te hybridization - Vectorized version"""
    HU = HU_block(kx, ky, kz)
    HTe = HTe_block(kx, ky, kz)

    if np.isscalar(kx):
        Hhyb = np.eye(2) * delta
        top = np.hstack((HU, Hhyb))
        bottom = np.hstack((Hhyb.conj().T, HTe))
        H = np.vstack((top, bottom))
    else:
        shape = kx.shape
        H = np.zeros(shape + (4, 4), dtype=complex)
        H[..., 0:2, 0:2] = HU
        H[..., 2:4, 2:4] = HTe
        H[..., 0:2, 2:4] = delta * np.eye(2)
        H[..., 2:4, 0:2] = delta * np.eye(2)
    
    return H

# =============================================================================
# FERMI SURFACE CALCULATION AND PLOTTING
# =============================================================================

def compute_band_energies(kz=0, resolution=300, kx_range=(-1, 1), ky_range=(-3, 3)):
    """
    Compute band energies on a 2D slice at fixed kz.
    
    Parameters:
    -----------
    kz : float
        Fixed kz value for the slice (default: kz=0, i.e., kx-ky plane)
    resolution : int
        Grid resolution for momentum space
    kx_range : tuple
        Range for kx in units of π/a (default: -1 to 1)
    ky_range : tuple  
        Range for ky in units of π/b (default: -3 to 3, extended to capture full FS)
    
    Returns:
    --------
    kx_vals : ndarray
        kx momentum values
    ky_vals : ndarray
        ky momentum values  
    energies : ndarray, shape (nkx, nky, 4)
        Band energies at each momentum point
    """
    print(f"Computing band energies at kz = {kz:.3f}")
    print(f"Grid: {resolution} × {resolution} points")
    print(f"kx range: {kx_range[0]:.1f}π/a to {kx_range[1]:.1f}π/a")
    print(f"ky range: {ky_range[0]:.1f}π/b to {ky_range[1]:.1f}π/b")
    
    # Create momentum grid
    kx_vals = np.linspace(kx_range[0]*np.pi/a, kx_range[1]*np.pi/a, resolution)
    ky_vals = np.linspace(ky_range[0]*np.pi/b, ky_range[1]*np.pi/b, resolution)
    
    # Pre-allocate energy array
    energies = np.zeros((resolution, resolution, 4))
    
    # Compute band energies
    for i, kx in enumerate(kx_vals):
        if (i + 1) % (resolution // 10) == 0:
            print(f"  Progress: {(i+1)/resolution*100:.0f}%")
        for j, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            eigvals = np.linalg.eigvals(H)
            energies[i, j, :] = np.sort(np.real(eigvals))
    
    print("  Band energy calculation complete!")
    return kx_vals, ky_vals, energies

def plot_fermi_contours(kx_vals, ky_vals, energies, save_dir='outputs', filename='ute2_fermi_contours_clean.png'):
    """
    Plot clean Fermi surface contours using the exact method from UTe2_fixed.py.
    
    Parameters:
    -----------
    kx_vals : ndarray
        kx momentum values
    ky_vals : ndarray
        ky momentum values
    energies : ndarray, shape (nkx, nky, 4)  
        Band energies
    save_dir : str
        Output directory
    filename : str
        Output filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Creating Fermi surface contour plot...")
    
    # Create main plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Plot contours for bands that cross the Fermi level (E=0)
    bands_to_plot = [2, 3]  # Bands 3 and 4 (0-indexed) - these typically form the Fermi surface
    colors = ['red', 'blue']
    labels = ['Band 3', 'Band 4']
    
    contour_count = 0
    
    for band_idx, color, label in zip(bands_to_plot, colors, labels):
        energy_data = energies[:, :, band_idx]
        
        print(f"  Processing {label}...")
        print(f"    Energy range: {energy_data.min():.4f} to {energy_data.max():.4f} eV")
        
        # Check if band crosses Fermi level
        if energy_data.min() <= 0 <= energy_data.max():
            print(f"    ✓ {label} crosses Fermi level - extracting contours")
            
            # Create contour (use temporary figure to extract contour data)
            temp_fig, temp_ax = plt.subplots(figsize=(1, 1))
            contour = temp_ax.contour(ky_vals, kx_vals, energy_data, levels=[0.0])
            plt.close(temp_fig)
            
            # Extract and plot contours
            if len(contour.allsegs[0]) > 0:
                first_contour_plotted = False
                for i, path in enumerate(contour.allsegs[0]):
                    if len(path) > 0:
                        # Convert to π/a and π/b units for plotting
                        ky_array = path[:, 0] / (np.pi / b)  # Convert to π/b units
                        kx_array = path[:, 1] / (np.pi / a)  # Convert to π/a units
                        
                        # Plot the contour
                        if not first_contour_plotted:
                            ax.plot(ky_array, kx_array, color=color, linewidth=2.5, 
                                   label=f'{label} Fermi Surface', alpha=0.9)
                            first_contour_plotted = True
                        else:
                            ax.plot(ky_array, kx_array, color=color, linewidth=2.5, alpha=0.9)
                        
                        contour_count += 1
                        
                        # Print contour info
                        print(f"      Contour {i+1}: {len(path)} points, "
                              f"ky ∈ [{ky_array.min():.2f}, {ky_array.max():.2f}]π/b, "
                              f"kx ∈ [{kx_array.min():.2f}, {kx_array.max():.2f}]π/a")
        else:
            print(f"    ✗ {label} does not cross Fermi level")
    
    # Style the plot exactly like UTe2_fixed.py
    ax.set_xlim(-3.2, 3.2)  # Extended ky range to show full Fermi surface
    ax.set_ylim(-1.2, 1.2)  # Standard kx range
    ax.set_xlabel('ky (π/b units)', fontsize=14)
    ax.set_ylabel('kx (π/a units)', fontsize=14)
    ax.set_title('UTe2 Fermi Surface Contours (Clean)\nkz = 0 plane', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add boundary box for the main zone
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k--', alpha=0.4, linewidth=1.5)
    
    # Make it look professional
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to: {save_path}")
    
    # Also show the plot
    plt.show()
    
    print(f"✓ Successfully plotted {contour_count} Fermi surface contours")
    return fig, ax

def plot_multiple_kz_slices(kz_values=[0, 0.1, 0.2], resolution=250, save_dir='outputs'):
    """
    Plot Fermi surface contours for multiple kz values to see evolution.
    
    Parameters:
    -----------
    kz_values : list
        List of kz values to plot (in units of π/c)
    resolution : int
        Grid resolution
    save_dir : str
        Output directory
    """
    print(f"Creating Fermi surface plots for {len(kz_values)} different kz slices...")
    
    for i, kz_frac in enumerate(kz_values):
        kz = kz_frac * np.pi / c
        print(f"\n--- Slice {i+1}/{len(kz_values)}: kz = {kz_frac:.1f}π/c ---")
        
        # Compute band energies for this slice
        kx_vals, ky_vals, energies = compute_band_energies(
            kz=kz, resolution=resolution, 
            kx_range=(-1, 1), ky_range=(-3, 3)
        )
        
        # Plot contours
        filename = f'ute2_fermi_contours_kz_{kz_frac:.1f}_pi_c.png'
        plot_fermi_contours(kx_vals, ky_vals, energies, save_dir=save_dir, filename=filename)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UTe2 Clean Fermi Surface Contour Generator")
    print("=" * 70)
    print("Based on the successful contour method from UTe2_fixed.py")
    print("Creates clean 2D contours without gap node calculations")
    print()
    
    # Set output directory
    save_dir = 'outputs/fermi_contours_clean'
    
    # Option 1: Single kz=0 slice (standard Fermi surface plot)
    print("Generating main Fermi surface plot at kz = 0...")
    kx_vals, ky_vals, energies = compute_band_energies(
        kz=0, resolution=300, 
        kx_range=(-1, 1), ky_range=(-3, 3)
    )
    plot_fermi_contours(kx_vals, ky_vals, energies, 
                       save_dir=save_dir, filename='ute2_main_fermi_surface.png')
    
    # Option 2: Multiple kz slices to see 3D structure
    print("\n" + "="*50)
    print("Generating additional kz slices...")
    plot_multiple_kz_slices(
        kz_values=[0.0, 0.2, 0.4], 
        resolution=250, 
        save_dir=save_dir
    )
    
    print("\n" + "="*70)
    print("✓ All plots completed successfully!")
    print(f"✓ Output saved to: {save_dir}/")
    print("="*70)