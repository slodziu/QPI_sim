#!/usr/bin/env python3
"""
JDOS Calculation from Fermi Surface Data
=========================================
Calculate and plot Joint Density of States (JDOS) using FFT autocorrelation.
This version matches the approach from ute2_minimal_fermi.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json


def load_jdos_data(kz=0.0, data_dir='raw_data_out/week16'):
    """Load saved Fermi surface data."""
    print(f"Loading data from {data_dir}...")
    
    energies = np.load(f'{data_dir}/energies_kz_{kz:.3f}.npy')
    weights_5f = np.load(f'{data_dir}/weights_5f_kz_{kz:.3f}.npy')
    kx_vals = np.load(f'{data_dir}/kx_vals.npy')
    ky_vals = np.load(f'{data_dir}/ky_vals.npy')
    
    with open(f'{data_dir}/metadata_kz_{kz:.3f}.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"  Loaded energies: shape {energies.shape}")
    print(f"  Loaded weights: shape {weights_5f.shape}")
    print(f"  Metadata: {metadata['param_set']}, resolution={metadata['resolution']}")
    
    return energies, weights_5f, kx_vals, ky_vals, metadata


def calculate_jdos(energies, weights_5f, kx_vals, ky_vals, weighted=True):
    """
    Calculate JDOS using 5f-weighted spectral function and FFT autocorrelation.
    Matches the approach from ute2_minimal_fermi.py.
    
    Parameters:
    - energies: band energies [kx, ky, band]
    - weights_5f: 5f orbital weights [kx, ky, band]
    - kx_vals, ky_vals: momentum grid arrays
    - weighted: if True, weight by 5f character
    
    Returns:
    - JDOS: 2D array of joint density of states
    - qx_vals, qy_vals: momentum transfer grids
    - spectral_function: The input spectral function used
    """
    print("\nCalculating JDOS (matching ute2_minimal_fermi.py approach)...")
    
    # Create spectral function A(k,0) using 5f weights at exact Fermi level
    spectral_function = np.zeros((len(kx_vals), len(ky_vals)))
    
    # Sum over all bands that contribute to the Fermi surface
    energy_threshold = 0.003  # Very tight threshold around E=0
    for band_idx in range(4):
        energy_data = energies[:, :, band_idx]
        weight_data = weights_5f[:, :, band_idx]
        
        # Only include points very close to E=0 (sharp delta function)
        fermi_mask = np.abs(energy_data) < energy_threshold
        
        if weighted:
            spectral_function += weight_data * fermi_mask
        else:
            spectral_function += fermi_mask.astype(float)
    
    # Normalize spectral function
    if np.max(spectral_function) > 0:
        spectral_function /= np.max(spectral_function)
    
    print(f"  Spectral function range: {spectral_function.min():.4f} to {spectral_function.max():.4f}")
    
    # Apply Hanning window to reduce edge effects from FFT periodic boundary conditions
    print("  Applying Hanning window to reduce FFT edge artifacts...")
    hanning_x = np.hanning(len(kx_vals))
    hanning_y = np.hanning(len(ky_vals))
    window_2d = np.outer(hanning_x, hanning_y)
    
    # Apply window to spectral function
    windowed_spectral = spectral_function * window_2d
    print(f"  Windowed spectral function range: {windowed_spectral.min():.4f} to {windowed_spectral.max():.4f}")
    
    # Calculate JDOS using FFT-based autocorrelation with windowed data
    A_fft = np.fft.fft2(windowed_spectral)
    J_grid = np.fft.ifft2(np.abs(A_fft)**2)
    JDOS = np.real(np.fft.fftshift(J_grid))
    
    # Create momentum transfer grids
    dk_x = kx_vals[1] - kx_vals[0]
    dk_y = ky_vals[1] - ky_vals[0]
    
    # JDOS is defined on a grid from -max_k to +max_k
    qx_max = (len(kx_vals) - 1) * dk_x / 2
    qy_max = (len(ky_vals) - 1) * dk_y / 2
    
    qx_vals = np.linspace(-qx_max, qx_max, len(kx_vals))
    qy_vals = np.linspace(-qy_max, qy_max, len(ky_vals))
    
    print(f"  JDOS range: {JDOS.min():.4f} to {JDOS.max():.4f}")
    print(f"  JDOS shape: {JDOS.shape}")
    
    # Cap high intensity values to enhance visibility of other features
    print("  Capping brightest 0.5% of JDOS values to improve contrast...")
    intensity_cap = np.percentile(JDOS, 99.91)  
    
    # Count how many values are above the cap
    above_cap = np.sum(JDOS > intensity_cap)
    total_points = JDOS.size
    
    # Apply intensity cap
    JDOS_filtered = np.clip(JDOS, 0, intensity_cap)
    
    print(f"  Original JDOS max: {JDOS.max():.4f}, capped at 99.5th percentile: {intensity_cap:.4f}")
    print(f"  Values above cap: {above_cap} / {total_points} ({100*above_cap/total_points:.1f}%)")
    print(f"  Filtered JDOS range: {JDOS_filtered.min():.4f} to {JDOS_filtered.max():.4f}")
    
    return JDOS_filtered, qx_vals, qy_vals, spectral_function


def plot_jdos(jdos, qx_vals, qy_vals, kz=0.0, param_set='odd_parity_paper', use_log=False):
    """
    Plot JDOS with qy on x-axis, qx on y-axis.
    
    Parameters:
    - jdos:  JDOS array
    - qx_vals, qy_vals: momentum transfer grids
    - kz: out-of-plane momentum
    - param_set: parameter set name
    - use_log: if True, plot log scale
    """
    print("\nPlotting JDOS...")
    
    from UTe2_fixed import a, b
    
    # Convert to π/a, π/b units
    qx_plot = qx_vals / (np.pi/a)
    qy_plot = qy_vals / (np.pi/b)
    
    print(f"  q-space: qx=[{qx_plot.min():.2f}, {qx_plot.max():.2f}] π/a, qy=[{qy_plot.min():.2f}, {qy_plot.max():.2f}] π/b")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Plot JDOS with qy on x-axis, qx on y-axis
    # imshow expects [rows, columns] = [qx, qy] which plots qx on y-axis, qy on x-axis
    im = ax.imshow(jdos, extent=[qy_plot.min(), qy_plot.max(), qx_plot.min(), qx_plot.max()], 
                  origin='lower', cmap='plasma', aspect='auto')
    
    # Style
    ax.set_xlabel(r'$q_y$ (π/b)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$q_x$ (π/a)', fontsize=16, fontweight='bold')
    
    # Title
    weight_str = '5f-Weighted'
    ax.set_title(f'UTe₂ JDOS (All Bands, {weight_str})\n' +
                f'kz={kz:.3f}, {param_set}', 
                fontsize=18, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axhline(0, color='w', linestyle='-', alpha=0.7, linewidth=1)
    ax.axvline(0, color='w', linestyle='-', alpha=0.7, linewidth=1)
    
    # Set axis limits
    ax.set_xlim(qy_plot.min(), qy_plot.max())
    ax.set_ylim(qx_plot.min()*0.8, qx_plot.max()*0.8)
    ax.set_aspect('equal', adjustable='box')
    
    # Add wavevectors p1-p6 from origin (0,0)
    print("  Adding wavevectors from origin...")
    
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
    
    # Colors for wavevectors
    vector_colors = {
        'p1': '#FF0000',  # Red
        'p2': '#0000FF',  # Blue
        'p3': '#FFFF00',  # Yellow
        'p4': '#00FF00',  # Green
        'p5': '#FF8800',  # Orange
        'p6': '#FF00FF'   # Magenta
    }
    
    # Label positioning offsets and alignments
    label_positions = {
        'p1': {'offset': (0.15, 0.08), 'ha': 'left', 'va': 'bottom'},
        'p2': {'offset': (0.15, 0.08), 'ha': 'left', 'va': 'bottom'},
        'p3': {'offset': (0.15, 0.08), 'ha': 'left', 'va': 'bottom'},
        'p4': {'offset': (0.0, 0.12), 'ha': 'center', 'va': 'bottom'},
        'p5': {'offset': (-0.15, 0.08), 'ha': 'right', 'va': 'bottom'},
        'p6': {'offset': (0.15, 0.0), 'ha': 'left', 'va': 'center'}
    }
    
    # Plot wavevectors from origin
    origin = (0, 0)  # (qx, qy) = (0, 0)
    
    for label, (qx_pi, qy_pi) in wavevectors_pi.items():
        # Endpoint is just the vector itself
        endpoint = (qx_pi, qy_pi)
        
        # Plot arrow (swap to qy, qx for our axes)
        ax.annotate('', 
                   xy=(endpoint[1], endpoint[0]),  # (qy, qx)
                   xytext=(origin[1], origin[0]),  # (qy, qx)
                   arrowprops=dict(arrowstyle='->', color=vector_colors[label], 
                                  lw=3, alpha=0.9, shrinkA=0, shrinkB=0))
        
        # Add marker at endpoint
        ax.plot(endpoint[1], endpoint[0], 'o', color=vector_colors[label], 
               markersize=8, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        
        # Add label with smart positioning (offset in qy, qx space)
        pos_info = label_positions[label]
        label_qy = endpoint[1] + pos_info['offset'][0]
        label_qx = endpoint[0] + pos_info['offset'][1]
        
        ax.text(label_qy, label_qx, label, 
               color='black', fontsize=12, fontweight='bold',
               horizontalalignment=pos_info['ha'],
               verticalalignment=pos_info['va'],
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.95),
               zorder=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='JDOS (arbitrary units)', orientation='vertical',
                       fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=11)
    
    # Background
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')
    
    # Save
    os.makedirs('outputs/week16', exist_ok=True)
    save_path = f'outputs/week16/jdos_clean_5fweighted_kz_{kz:.3f}.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved JDOS plot to: {save_path}")
    
    plt.show()
    
    return fig, ax


def main():
    """Main function to calculate and plot JDOS."""
    
    # Load saved data
    kz = 0.0
    energies, weights_5f, kx_vals, ky_vals, metadata = load_jdos_data(kz=kz)
    
    # Calculate JDOS
    jdos, qx_vals, qy_vals, spectral_function = calculate_jdos(
        energies, weights_5f, kx_vals, ky_vals, weighted=True
    )
    
    # Plot JDOS
    fig, ax = plot_jdos(jdos, qx_vals, qy_vals, kz=kz,
                       param_set=metadata['param_set'], use_log=False)
    
    # Save JDOS data
    output_dir = 'raw_data_out/week16'
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/jdos_clean_kz_{kz:.3f}.npy', jdos)
    np.save(f'{output_dir}/jdos_qx_vals.npy', qx_vals)
    np.save(f'{output_dir}/jdos_qy_vals.npy', qy_vals)
    np.save(f'{output_dir}/spectral_function_kz_{kz:.3f}.npy', spectral_function)
    print(f"\nSaved JDOS data to {output_dir}/")


if __name__ == "__main__":
    main()
