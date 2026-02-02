#!/usr/bin/env python3
"""
UTe2 Simple Fermi Contour Plotter
=================================
Minimal script for creating clean UTe2 Fermi surface contours
"""
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator
import scipy.fft as fft

# Lattice constants (same for all parameter sets)
a, b, c = 0.41, 0.61, 1.39

# Parameter sets to test
parameter_sets = {
    'DFT': {
        'delta': 0.1,  # hybridization
        'muU': -0.35, 'DeltaU': 0.4, 'tU': 0.15, 'tpU': 0.08, 
        'tch_U': 0.01, 'tpch_U': 0, 'tz_U': -0.03,
        'muTe': -1.8, 'DeltaTe': -1.5, 'tTe': -1.5, 
        'tch_Te': 0, 'tz_Te': -0.05
    },
    'QuantumOscillation': {
        'delta': 0.1,  # using same hybridization as DFT 
        'muU': -0.17, 'DeltaU': 0.05, 'tU': 0.1, 'tpU': 0.08,
        'tch_U': 0.01, 'tpch_U': 0, 'tz_U': 0.04,
        'muTe': -1.8, 'DeltaTe': -1.5, 'tTe': -1.5,
        'tch_Te': -0.03, 'tz_Te': -0.5
    },
    'QPIFS': {
        'delta': 0.1,  # using same hybridization as DFT 
        'muU': -0.17, 'DeltaU': -0.05, 'tU': 0.1, 'tpU': 0.08,
        'tch_U': 0.01, 'tpch_U': 0, 'tz_U': 0.04,
        'muTe': -1.8, 'DeltaTe': -1.5, 'tTe': -1.5,
        'tch_Te': -0.03, 'tz_Te': -0.5
    },
    'odd_parity_paper': {
        'delta': 0.13,  
        'muU': -0.355, 'DeltaU': 0.38, 'tU': 0.17, 'tpU': 0.08,
        'tch_U': 0.015, 'tpch_U': 0.01, 'tz_U': -0.0375,
        'muTe': -2.25, 'DeltaTe': -1.4, 'tTe': -1.5,
        'tch_Te': 0, 'tz_Te': -0.05
    }
}

# Current parameter set (will be set by set_parameters function)
muU = DeltaU = tU = tpU = tch_U = tpch_U = tz_U = None
muTe = DeltaTe = tTe = tch_Te = tz_Te = delta = None

def set_parameters(param_set_name):
    """Set global parameters from the specified parameter set"""
    global muU, DeltaU, tU, tpU, tch_U, tpch_U, tz_U
    global muTe, DeltaTe, tTe, tch_Te, tz_Te, delta
    
    params = parameter_sets[param_set_name]
    muU = params['muU']
    DeltaU = params['DeltaU'] 
    tU = params['tU']
    tpU = params['tpU']
    tch_U = params['tch_U']
    tpch_U = params['tpch_U']
    tz_U = params['tz_U']
    muTe = params['muTe']
    DeltaTe = params['DeltaTe']
    tTe = params['tTe']
    tch_Te = params['tch_Te']
    tz_Te = params['tz_Te']
    delta = params['delta']
    
    print(f"Set parameters for: {param_set_name}")
    print(f"  U:  μ={muU:.2f}, Δ={DeltaU:.2f}, t={tU:.2f}, t'={tpU:.2f}, tch={tch_U:.3f}, tz={tz_U:.3f}")
    print(f"  Te: μ={muTe:.2f}, Δ={DeltaTe:.2f}, t={tTe:.2f}, tch={tch_Te:.3f}, tz={tz_Te:.2f}")
    print(f"  δ={delta:.2f}")
    print()

def H_full(kx, ky, kz):
    """Compute 4x4 UTe2 Hamiltonian"""
    # U block
    diag_U = muU - 2*tU*np.cos(kx*a) - 2*tch_U*np.cos(ky*b)
    off_U = -DeltaU - 2*tpU*np.cos(kx*a) - 2*tpch_U*np.cos(ky*b)
    complex_U = -4*tz_U*np.exp(-1j*kz*c/2)*np.cos(kx*a/2)*np.cos(ky*b/2)
    
    # Te block  
    diag_Te = muTe - 2*tch_Te*np.cos(kx*a) 
    off_Te = -DeltaTe - tTe*np.exp(-1j*ky*b) - 2*tz_Te*np.cos(kz*c/2)*np.cos(kx*a/2)*np.cos(ky*b/2)
    
    # Build 4x4 matrix
    H = np.zeros((4, 4), dtype=complex)
    H[0,0] = H[1,1] = diag_U
    H[0,1] = off_U + complex_U
    H[1,0] = off_U + np.conj(complex_U)
    H[2,2] = H[3,3] = diag_Te  
    H[2,3] = off_Te
    H[3,2] = np.conj(off_Te)
    H[0,2] = H[1,3] = H[2,0] = H[3,1] = delta
    return H

def calculate_jdos(energies, weights_5f, kx_vals, ky_vals):
    """
    Calculate JDOS using 5f-weighted spectral function and FFT autocorrelation
    
    Parameters:
    - energies: 4D array of band energies [kx, ky, band]
    - weights_5f: 4D array of 5f orbital weights [kx, ky, band]
    - kx_vals, ky_vals: momentum grid arrays
    
    Returns:
    - JDOS: 2D array of joint density of states
    - qx_vals, qy_vals: momentum transfer grids for JDOS
    """
    print(f"  Computing JDOS with no broadening (sharp Fermi surface)")
    
    # Create spectral function A(k,0) using 5f weights at exact Fermi level
    spectral_function = np.zeros((len(kx_vals), len(ky_vals)))
    
    # Sum over all bands that contribute to the Fermi surface
    energy_threshold = 0.003  # Very tight threshold around E=0
    for band_idx in range(4):
        energy_data = energies[:, :, band_idx]
        weight_data = weights_5f[:, :, band_idx]
        
        # Only include points very close to E=0 (sharp delta function)
        fermi_mask = np.abs(energy_data) < energy_threshold
        spectral_function += weight_data * fermi_mask
    
    # Normalize spectral function
    if np.max(spectral_function) > 0:
        spectral_function /= np.max(spectral_function)
    
    print(f"    Spectral function range: {spectral_function.min():.4f} to {spectral_function.max():.4f}")
    
    # Calculate JDOS using FFT-based autocorrelation 
    A_fft = fft.fft2(spectral_function)
    J_grid = fft.ifft2(np.abs(A_fft)**2)
    JDOS = np.real(fft.fftshift(J_grid))
    
    # Create momentum transfer grids
    dk_x = kx_vals[1] - kx_vals[0]
    dk_y = ky_vals[1] - ky_vals[0]
    
    # JDOS is defined on a grid from -max_k to +max_k
    qx_max = (len(kx_vals) - 1) * dk_x / 2
    qy_max = (len(ky_vals) - 1) * dk_y / 2
    
    qx_vals = np.linspace(-qx_max, qx_max, len(kx_vals))
    qy_vals = np.linspace(-qy_max, qy_max, len(ky_vals))
    
    print(f"    JDOS range: {JDOS.min():.4f} to {JDOS.max():.4f}")
    print(f"    JDOS momentum transfer ranges: qx ∈ [{qx_vals.min():.3f}, {qx_vals.max():.3f}], qy ∈ [{qy_vals.min():.3f}, {qy_vals.max():.3f}]")
    
    return JDOS, qx_vals, qy_vals

def create_fermi_contours(param_set_name, kz=0, resolution=512, plot_fermi_surface=True, plot_combined=True, plot_jdos=True):
    """Create clean Fermi surface contours using UTe2_fixed.py method"""
    print(f"Computing Fermi surface at kz = {kz:.3f} for {param_set_name} parameters")
    
    # Set parameters for this run
    set_parameters(param_set_name)
    
    # Create momentum grid matching the paper's convention
    kx_vals = np.linspace(-1 * np.pi/a, 1 * np.pi/a, resolution)
    ky_vals = np.linspace(-3 * np.pi/b, 3 * np.pi/b, resolution)
    
    # Compute band energies and 5f orbital weights
    energies = np.zeros((resolution, resolution, 4))
    weights_5f = np.zeros((resolution, resolution, 4))
    
    for i, kx in enumerate(kx_vals):
        if (i + 1) % (resolution // 10) == 0:
            print(f"  Progress: {(i+1)/resolution*100:.0f}%")
        for j, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            evals, evecs = np.linalg.eigh(H)  # Get both eigenvalues and eigenvectors
            
            # Sort by eigenvalue and rearrange eigenvectors accordingly
            sort_idx = np.argsort(np.real(evals))
            energies[i, j, :] = np.real(evals[sort_idx])
            
            # Calculate 5f orbital weight for each band
            for n in range(4):
                sorted_evec = evecs[:, sort_idx[n]]
                # 5f weight = |ψ_n(0)|² + |ψ_n(1)|² (first two components are U 5f orbitals)
                weight_5f = np.sum(np.abs(sorted_evec[:2])**2)
                weights_5f[i, j, n] = weight_5f
    
    if plot_fermi_surface:
        # Combined Fermi surface plot for both bands
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        all_segments = []
        all_weights = []
        for band_idx, label in zip([2, 3], ['Band 3', 'Band 4']):
            energy_data = energies[:, :, band_idx]
            weight_data = weights_5f[:, :, band_idx]
            print(f"  {label}: Energy range {energy_data.min():.4f} to {energy_data.max():.4f} eV")
            print(f"  {label}: 5f weight range {weight_data.min():.3f} to {weight_data.max():.3f}")
            if energy_data.min() <= 0 <= energy_data.max():
                print(f"    ✓ {label} crosses Fermi level")
                temp_fig, temp_ax = plt.subplots()
                contour = temp_ax.contour(ky_vals, kx_vals, energy_data, levels=[0.0])
                plt.close(temp_fig)
                interpolator = RegularGridInterpolator((kx_vals, ky_vals), weight_data, method='linear', bounds_error=False, fill_value=0)
                for path in contour.allsegs[0]:
                    if len(path) > 1:
                        kx_contour = path[:, 1]
                        ky_contour = path[:, 0]
                        contour_points = np.column_stack([kx_contour, ky_contour])
                        interpolated_weights = interpolator(contour_points)
                        x = ky_contour / (np.pi/b)
                        y = kx_contour / (np.pi/a)
                        for i in range(len(x)-1):
                            all_segments.append([[x[i], y[i]], [x[i+1], y[i+1]]])
                            all_weights.append((interpolated_weights[i] + interpolated_weights[i+1]) / 2)
            else:
                print(f"    ✗ {label} does not cross Fermi level")
        if all_segments:
            lc = LineCollection(all_segments, array=np.array(all_weights), cmap='YlOrRd', linewidths=4)
            ax.add_collection(lc)
        # Style combined plot
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('ky (π/b units)', fontsize=14)
        ax.set_ylabel('kx (π/a units)', fontsize=14)
        ax.set_title(f'UTe2 Combined 5f-Weighted Fermi Surface\n({param_set_name}), kz = {kz:.3f}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        # Add colorbar
        import matplotlib as mpl
        if all_weights:
            norm = mpl.colors.Normalize(vmin=min(all_weights), vmax=max(all_weights))
            sm = mpl.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.045, pad=0.04)
            cbar.set_label('5f Orbital Weight', fontsize=14)
        fig.suptitle(f'UTe2 Combined 5f-Weighted Fermi Surface ({param_set_name}), kz = {kz:.3f}', fontsize=18, y=0.98)
        os.makedirs('outputs/fermi_countours_clean', exist_ok=True)
        save_path = f'outputs/fermi_countours_clean/ute2_fermi_5f_combined_{param_set_name.lower()}_kz_{kz:.3f}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved combined plot to: {save_path}")
        print("="*60)


if __name__ == "__main__":
    print("=" * 60)
    print("UTe2 Fermi Surface Comparison")
    print("Testing Multiple Parameter Sets")
    print("=" * 60)
    
    # Run all parameter sets
    for param_set in ['odd_parity_paper']:
        print(f"\n{'='*20} {param_set} Parameters {'='*20}")
        for kz_val in np.linspace(0, np.pi/c, 10):
            create_fermi_contours(param_set, kz=kz_val, resolution=301)