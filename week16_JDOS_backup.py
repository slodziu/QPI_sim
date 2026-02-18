#!/usr/bin/env python3
"""
JDOS Calculation from Fermi Surface Data
=========================================
Calculate and plot Joint Density of States (JDOS) using FFT autocorrelation
via the Wiener-Khinchin theorem: JDOS = IFFT(|FFT(spectral)|^2)

Data indexing convention:
- energies[i_kx, i_ky, band]: kx along axis 0, ky along axis 1
- fermi_map[i_kx, i_ky]: same indexing
- jdos[i_qx, i_qy]: qx along axis 0, qy along axis 1 (from FFT of fermi_map)
- For plotting with qy on x-axis, qx on y-axis: need to transpose jdos
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json


def load_jdos_data(kz=0.0, data_dir='raw_data_out/week16'):
    """
    Load saved Fermi surface data.
    
    Parameters:
    - kz: out-of-plane momentum
    - data_dir: directory containing saved data
    
    Returns:
    - energies, weights_5f, kx_vals, ky_vals, metadata
    """
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


def norm_xcorr2d_matlab(data1, data2):
    """
    Normalized 2D cross-correlation (MATLAB approach from supervisor).
    
    Parameters:
    - data1, data2: 2D arrays of equal size
    
    Returns:
    - corr_data: Normalized cross-correlation
    """
    # Subtract mean (as in MATLAB code)
    f1 = np.fft.fft2(data1 - np.mean(data1))
    f2 = np.fft.fft2(data2 - np.mean(data2))
    
    # Cross-correlation
    unnorm = np.fft.fftshift(np.fft.ifft2(f2 * np.conj(f1)))
    
    # Normalization
    norm = np.real(np.fft.ifft2(f1 * np.conj(f1))) * np.real(np.fft.ifft2(f2 * np.conj(f2)))
    corr_data = unnorm / np.sqrt(np.max(norm))
    
    return corr_data


def calculate_jdos(energies, weights_5f=None, band=None, weighted=True, 
                   method='wiener-khinchin', zero_pad=True):
    """
    Calculate JDOS using FFT autocorrelation.
    
    Parameters:
    - energies: band energies [kx, ky, band]
    - weights_5f: 5f orbital weights [kx, ky, band] (optional)
    - band: which band to use (None = sum all bands)
    - weighted: if True, weight by 5f character
    - method: 'wiener-khinchin' (simple) or 'matlab' (normalized with mean subtraction)
    - zero_pad: if True, zero-pad to avoid circular convolution artifacts
    
    Returns:
    - jdos: JDOS map (real for wiener-khinchin, complex for matlab)
    - fermi_map: Fermi surface occupation map used
    - pad_info: dict with padding information
    """
    print(f"\nCalculating JDOS (method={method}, zero_pad={zero_pad})...")
    
    if band is not None:
        # Single band
        print(f"  Using band {band}")
        E = energies[:, :, band]
        W = weights_5f[:, :, band] if weights_5f is not None else None
    else:
        # Sum all bands (multi-band JDOS)
        print(f"  Using all {energies.shape[2]} bands")
        # For multi-band, create spectral function (like ute2_minimal_fermi.py)
        energy_threshold = 0.003  # Very tight threshold around E=0
        fermi_map_all = np.zeros(energies.shape[:2])
        
        if weighted and weights_5f is not None:
            for b in range(energies.shape[2]):
                # Only include points very close to E=0 (sharp delta function)
                fermi_mask = np.abs(energies[:, :, b]) < energy_threshold
                fermi_map_all += weights_5f[:, :, b] * fermi_mask
        else:
            for b in range(energies.shape[2]):
                fermi_mask = np.abs(energies[:, :, b]) < energy_threshold
                fermi_map_all += fermi_mask.astype(float)
        
        # Normalize spectral function
        if np.max(fermi_map_all) > 0:
            fermi_map_all /= np.max(fermi_map_all)
        
        print(f"  Spectral function range: {fermi_map_all.min():.3f} to {fermi_map_all.max():.3f}")
        
        # Apply Hanning window to reduce FFT edge artifacts
        print("  Applying Hanning window to reduce FFT edge artifacts...")
        hanning_x = np.hanning(fermi_map_all.shape[0])
        hanning_y = np.hanning(fermi_map_all.shape[1])
        window_2d = np.outer(hanning_x, hanning_y)
        fermi_map_all = fermi_map_all * window_2d
        print(f"  Windowed spectral function range: {fermi_map_all.min():.3f} to {fermi_map_all.max():.3f}")
        
        # Apply zero-padding if requested
        original_shape = fermi_map_all.shape
        if zero_pad:
            # Pad to 2x size to avoid circular convolution
            pad_x = original_shape[0]
            pad_y = original_shape[1]
            fermi_map_padded = np.pad(fermi_map_all, ((0, pad_x), (0, pad_y)), mode='constant')
            print(f"  Zero-padded: {original_shape} -> {fermi_map_padded.shape}")
        else:
            fermi_map_padded = fermi_map_all
        
        # Compute JDOS
        if method == 'matlab':
            # MATLAB supervisor's approach (normalized cross-correlation with mean subtraction)
            print("  Using MATLAB normalized cross-correlation approach")
            jdos_full = norm_xcorr2d_matlab(fermi_map_padded, fermi_map_padded)
        else:
            # Wiener-Khinchin theorem (simple autocorrelation)
            print("  Using Wiener-Khinchin theorem")
            A_fft = np.fft.fft2(fermi_map_padded)
            J_grid = np.fft.ifft2(np.abs(A_fft)**2)
            jdos_full = np.real(np.fft.fftshift(J_grid))
        
        # Extract central region if zero-padded
        if zero_pad:
            # Take central original_shape region
            start_x = original_shape[0] // 2
            start_y = original_shape[1] // 2
            end_x = start_x + original_shape[0]
            end_y = start_y + original_shape[1]
            jdos = jdos_full[start_x:end_x, start_y:end_y]
            print(f"  Extracted central region: {jdos_full.shape} -> {jdos.shape}")
        else:
            jdos = jdos_full
        
        print(f"  JDOS calculated: shape {jdos.shape}")
        if method == 'matlab':
            print(f"  JDOS magnitude range: {np.abs(jdos).min():.3e} to {np.abs(jdos).max():.3e}")
        else:
            print(f"  JDOS range: {jdos.min():.3e} to {jdos.max():.3e}")
        
        pad_info = {'zero_padded': zero_pad, 'original_shape': original_shape}
        return jdos, fermi_map_all, pad_info
    
    # Single band processing
    energy_threshold = 0.003
    fermi_mask = np.abs(E) < energy_threshold
    
    if weighted and W is not None:
        print("  Applying 5f orbital weighting")
        fermi_map = W * fermi_mask
    else:
        print("  Using binary occupation (no weighting)")
        fermi_map = fermi_mask.astype(float)
    
    # Normalize spectral function
    if np.max(fermi_map) > 0:
        fermi_map /= np.max(fermi_map)
    
    print(f"  Spectral function range: {fermi_map.min():.3f} to {fermi_map.max():.3f}")
    
    # Apply Hanning window
    print("  Applying Hanning window to reduce FFT edge artifacts...")
    hanning_x = np.hanning(fermi_map.shape[0])
    hanning_y = np.hanning(fermi_map.shape[1])
    window_2d = np.outer(hanning_x, hanning_y)
    fermi_map = fermi_map * window_2d
    print(f"  Windowed spectral function range: {fermi_map.min():.3f} to {fermi_map.max():.3f}")
    
    # Apply zero-padding if requested
    original_shape = fermi_map.shape
    if zero_pad:
        pad_x = original_shape[0]
        pad_y = original_shape[1]
        fermi_map_padded = np.pad(fermi_map, ((0, pad_x), (0, pad_y)), mode='constant')
        print(f"  Zero-padded: {original_shape} -> {fermi_map_padded.shape}")
    else:
        fermi_map_padded = fermi_map
    
    # Compute JDOS
    if method == 'matlab':
        print("  Using MATLAB normalized cross-correlation approach")
        jdos_full = norm_xcorr2d_matlab(fermi_map_padded, fermi_map_padded)
    else:
        print("  Using Wiener-Khinchin theorem")
        A_fft = np.fft.fft2(fermi_map_padded)
        J_grid = np.fft.ifft2(np.abs(A_fft)**2)
        jdos_full = np.real(np.fft.fftshift(J_grid))
    
    # Extract central region if zero-padded
    if zero_pad:
        start_x = original_shape[0] // 2
        start_y = original_shape[1] // 2
        end_x = start_x + original_shape[0]
        end_y = start_y + original_shape[1]
        jdos = jdos_full[start_x:end_x, start_y:end_y]
        print(f"  Extracted central region: {jdos_full.shape} -> {jdos.shape}")
    else:
        jdos = jdos_full
    
    print(f"  JDOS calculated: shape {jdos.shape}")
    if method == 'matlab':
        print(f"  JDOS magnitude range: {np.abs(jdos).min():.3e} to {np.abs(jdos).max():.3e}")
    else:
        print(f"  JDOS range: {jdos.min():.3e} to {jdos.max():.3e}")
    
    pad_info = {'zero_padded': zero_pad, 'original_shape': original_shape}
    return jdos, fermi_map, pad_info


def plot_jdos(jdos, kx_vals, ky_vals, kz=0.0, band=None, weighted=True, 
              param_set='odd_parity_paper', use_log=True, method='wiener-khinchin'):
    """
    Plot JDOS with qy on x-axis, qx on y-axis.
    
    Parameters:
    - jdos: JDOS array (real or complex)
    - kx_vals, ky_vals: momentum grids (original k-space)
    - kz: out-of-plane momentum
    - band: which band was used (None = all bands)
    - weighted: whether 5f weighting was applied
    - param_set: parameter set name
    - use_log: if True, plot log scale
    - method: calculation method used
    """
    print("\nPlotting JDOS...")
    
    # Take magnitude if complex (MATLAB method), otherwise already real
    if np.iscomplexobj(jdos):
        jdos_mag = np.abs(jdos)
        print("  JDOS is complex, taking magnitude")
    else:
        jdos_mag = jdos
        print("  JDOS is real")
    
    # Create q-space grid (momentum transfer)
    from UTe2_fixed import a, b
    
    # Grid spacing
    dkx = kx_vals[1] - kx_vals[0]
    dky = ky_vals[1] - ky_vals[0]
    
    # JDOS q-space: autocorrelation gives momentum transfer from -max_k to +max_k
    # where max_k is HALF the total k-space range (as in ute2_minimal_fermi.py)
    qx_max = (len(kx_vals) - 1) * dkx / 2
    qy_max = (len(ky_vals) - 1) * dky / 2
    
    qx_vals = np.linspace(-qx_max, qx_max, len(kx_vals))
    qy_vals = np.linspace(-qy_max, qy_max, len(ky_vals))
    
    print(f"  k-space: kx=[{kx_vals.min()/(np.pi/a):.2f}, {kx_vals.max()/(np.pi/a):.2f}] π/a, ky=[{ky_vals.min()/(np.pi/b):.2f}, {ky_vals.max()/(np.pi/b):.2f}] π/b")
    print(f"  q-space: qx=[{qx_vals.min()/(np.pi/a):.2f}, {qx_vals.max()/(np.pi/a):.2f}] π/a, qy=[{qy_vals.min()/(np.pi/b):.2f}, {qy_vals.max()/(np.pi/b):.2f}] π/b")
    
    # Convert to π/a, π/b units
    qx_plot = qx_vals / (np.pi/a)
    qy_plot = qy_vals / (np.pi/b)
    
    # Create meshgrid for plotting - qy on x-axis, qx on y-axis
    QY, QX = np.meshgrid(qy_plot, qx_plot, indexing='ij')
    
    # Transpose JDOS to match meshgrid indexing
    # jdos[i, j] has i=qx_index, j=qy_index (from fermi_map[kx, ky])
    # QY[i, j] = qy_plot[i], QX[i, j] = qx_plot[j] with indexing='ij'
    # So we need jdos_plot[i, j] where i=qy_index, j=qx_index
    # Therefore: jdos_plot = jdos.T
    jdos_plot = jdos_mag.T
    
    print(f"  jdos shape: {jdos_mag.shape} -> jdos_plot shape: {jdos_plot.shape}")
    print(f"  Meshgrid: QY[0,0]={QY[0,0]:.2f}, QX[0,0]={QX[0,0]:.2f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Plot JDOS
    if use_log:
        # Log scale (add small value to avoid log(0))
        plot_data = np.log10(jdos_plot + 1e-10)
        cmap = 'viridis'
        label = r'log$_{10}$(JDOS)'
    else:
        plot_data = jdos_plot
        cmap = 'hot'
        label = 'JDOS'
    
    im = ax.pcolormesh(QY, QX, plot_data, cmap=cmap, shading='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label=label)
    cbar.ax.tick_params(labelsize=11)
    
    # Style
    ax.set_xlabel(r'$q_y$ (π/b)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$q_x$ (π/a)', fontsize=16, fontweight='bold')
    
    # Title
    band_str = f'Band {band}' if band is not None else 'All Bands'
    weight_str = '5f-Weighted' if weighted else 'Unweighted'
    ax.set_title(f'UTe₂ JDOS ({band_str}, {weight_str})\n' +
                f'kz={kz:.3f}, {param_set}', 
                fontsize=18, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.axhline(0, color='white', linestyle='-', alpha=0.5, linewidth=1)
    ax.axvline(0, color='white', linestyle='-', alpha=0.5, linewidth=1)
    
    # Set axis limits based on actual q-space range
    qx_lim = qx_plot.max() * 1.05  # Add 5% padding
    qy_lim = qy_plot.max() * 1.05
    ax.set_xlim(-qy_lim, qy_lim)  # qy on x-axis
    ax.set_ylim(-qx_lim, qx_lim)  # qx on y-axis
    ax.set_aspect('equal', adjustable='box')
    
    print(f"  Plot limits: qy=[{-qy_lim:.2f}, {qy_lim:.2f}] π/b, qx=[{-qx_lim:.2f}, {qx_lim:.2f}] π/a")
    
    # Background
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')
    
    # Save
    os.makedirs('outputs/week16', exist_ok=True)
    band_suffix = f'_band{band}' if band is not None else '_allbands'
    weight_suffix = '_5fweighted' if weighted else '_unweighted'
    scale_suffix = '_log' if use_log else '_linear'
    method_suffix = f'_{method}'
    save_path = f'outputs/week16/jdos{band_suffix}{weight_suffix}{scale_suffix}{method_suffix}_kz_{kz:.3f}.png'
    
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
    
    # Calculate JDOS using both methods for comparison
    
    # Method 1: Wiener-Khinchin (simple autocorrelation) with zero-padding
    print("\n" + "="*60)
    print("METHOD 1: Wiener-Khinchin with zero-padding")
    print("="*60)
    jdos_wk, fermi_map, pad_info = calculate_jdos(
        energies, weights_5f, band=None, weighted=True,
        method='wiener-khinchin', zero_pad=True
    )
    
    # Plot log scale
    fig1, ax1 = plot_jdos(jdos_wk, kx_vals, ky_vals, kz=kz, band=None, weighted=True,
                          param_set=metadata['param_set'], use_log=False,
                          method='wiener-khinchin')
    
    # Method 2: MATLAB supervisor's approach with zero-padding
    print("\n" + "="*60)
    print("METHOD 2: MATLAB normalized cross-correlation with zero-padding")
    print("="*60)
    jdos_matlab, fermi_map2, pad_info2 = calculate_jdos(
        energies, weights_5f, band=None, weighted=True,
        method='matlab', zero_pad=True
    )
    
    # Plot log scale
    fig2, ax2 = plot_jdos(jdos_matlab, kx_vals, ky_vals, kz=kz, band=None, weighted=True,
                          param_set=metadata['param_set'], use_log=False,
                          method='matlab')
    
    # Save JDOS data
    output_dir = 'raw_data_out/week16'
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/jdos_wiener_khinchin_kz_{kz:.3f}.npy', jdos_wk)
    np.save(f'{output_dir}/jdos_matlab_kz_{kz:.3f}.npy', jdos_matlab)
    np.save(f'{output_dir}/fermi_map_5fweighted_kz_{kz:.3f}.npy', fermi_map)

if __name__ == "__main__":
    main()
