#!/usr/bin/env python3
"""
UTe2 Orbital Spectral Weights using Eigenvector Projection Method
Computes spectral weights using the correct formula:
A_orbital(k) = ∑_n |⟨orbital|ψ_n(k)⟩|² L(ω - E_n(k))
"""

import numpy as np
import matplotlib.pyplot as plt
from UTe2_fixed import *
import os
import pickle
import hashlib

def compute_orbital_spectral_weights_eigenvectors(kz, resolution=512, omega=0.0, eta=0.002):
    """
    Compute orbital spectral weights using eigenvector projection method.
    
    This uses the correct approach from Nature papers: A_orbital(k,ω) = ∑_n |⟨orbital|ψ_n(k)⟩|² L(ω - E_n(k))
    where L is a Lorentzian broadening function evaluated exactly at ω (usually Fermi level).
    
    Args:
        kz: z-component of momentum
        resolution: Number of k-points along each direction
        omega: Energy at which to evaluate spectral weight (eV) - usually 0 for Fermi level
        eta: Broadening parameter (eV) - smaller for sharper features
    
    Returns:
        U_5f_weight: U 5f orbital spectral weight matrix
        U_6d_weight: U 6d orbital spectral weight matrix  
        Te1_p_weight: Te1 p orbital spectral weight matrix
        Te2_p_weight: Te2 p orbital spectral weight matrix
        total_weight: Total spectral weight matrix
        kx_vals: kx momentum values
        ky_vals: ky momentum values
    """
    print(f"Computing orbital spectral weights using eigenvector method with {resolution}x{resolution} resolution...")
    print(f"Evaluating spectral function at ω = {omega:.3f} eV with η={eta:.4f} eV Lorentzian broadening")
    
    # Create momentum grid
    kx_vals = np.linspace(-1*np.pi/a, 1*np.pi/a, resolution)
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, resolution)
    
    # Initialize spectral weight arrays
    U_5f_weight = np.zeros((len(kx_vals), len(ky_vals)))
    U_6d_weight = np.zeros((len(kx_vals), len(ky_vals)))
    Te1_p_weight = np.zeros((len(kx_vals), len(ky_vals)))
    Te2_p_weight = np.zeros((len(kx_vals), len(ky_vals)))
    total_weight = np.zeros((len(kx_vals), len(ky_vals)))
    
    print("Computing eigenvectors and projecting orbital weights...")
    
    # Loop over k-points
    debug_count = 0
    debug_weights = []
    
    for i, kx in enumerate(kx_vals):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(kx_vals)} ({100*i/len(kx_vals):.1f}%)")
            
        for j, ky in enumerate(ky_vals):
            # Get Hamiltonian and diagonalize
            H = H_full(kx, ky, kz)
            eigenvals, eigenvecs = np.linalg.eigh(H)
            
            # Initialize spectral weights for this k-point
            weight_5f = 0.0
            weight_6d = 0.0
            weight_te1 = 0.0
            weight_te2 = 0.0
            weight_total = 0.0
            
            # Loop over bands
            for n in range(4):
                En = eigenvals[n]
                psi_n = eigenvecs[:, n]  # n-th eigenvector
                
                # Compute Lorentzian broadening at exactly ω
                # This is the key difference: evaluate L(ω - E_n) at single energy point
                energy_diff = omega - En
                lorentzian = (eta / np.pi) / (energy_diff**2 + eta**2)
                
                # Compute orbital projections: |⟨orbital|ψ_n⟩|²
                # The eigenvector components directly give orbital weights
                # Orbital indices: 0=U1_5f, 1=U2_6d, 2=Te1_p, 3=Te2_p
                proj_5f = abs(psi_n[0])**2  # |⟨U1_5f|ψ_n⟩|²
                proj_6d = abs(psi_n[1])**2  # |⟨U2_6d|ψ_n⟩|²
                proj_te1 = abs(psi_n[2])**2  # |⟨Te1_p|ψ_n⟩|²
                proj_te2 = abs(psi_n[3])**2  # |⟨Te2_p|ψ_n⟩|²
                
                # Accumulate weighted contributions
                weight_5f += proj_5f * lorentzian
                weight_6d += proj_6d * lorentzian
                weight_te1 += proj_te1 * lorentzian
                weight_te2 += proj_te2 * lorentzian
                weight_total += lorentzian
            
            # Store spectral weights
            U_5f_weight[i, j] = weight_5f
            U_6d_weight[i, j] = weight_6d
            Te1_p_weight[i, j] = weight_te1
            Te2_p_weight[i, j] = weight_te2
            total_weight[i, j] = weight_total
            
            # Debug: save some sample values
            if debug_count < 10 and (i % 50 == 0 or j % 50 == 0):
                debug_weights.append({
                    'i': i, 'j': j, 'kx': kx, 'ky': ky,
                    'weight_5f': weight_5f, 'eigenvals': eigenvals.copy()
                })
                debug_count += 1
    
    print("Orbital spectral weight computation completed")
    
    # Print debug information
    if debug_weights:
        print("\nDebug: Sample spectral weights during computation:")
        for d in debug_weights[:5]:
            print(f"  k[{d['i']},{d['j']}] = ({d['kx']:.4f}, {d['ky']:.4f}): weight_5f = {d['weight_5f']:.6f}")
            print(f"    Eigenvals: {d['eigenvals']}")
    
    return U_5f_weight, U_6d_weight, Te1_p_weight, Te2_p_weight, total_weight, kx_vals, ky_vals


def plot_orbital_spectral_weight(weight, title, ax, kx_vals, ky_vals, use_log_scale=False, cmap='gray'):
    """Helper function to plot individual orbital spectral weights."""
    print(f"Plotting {title}: min={np.min(weight):.6f}, max={np.max(weight):.6f}")
    
    if use_log_scale:
        # For log scale, handle zeros by adding small epsilon
        epsilon = 1e-10
        log_weight = np.log10(weight + epsilon)
        # Mask only truly problematic values
        masked = np.where(np.isfinite(log_weight), log_weight, np.nan)
        vmin, vmax = np.nanmin(masked), np.nanmax(masked)
        cmap_label = f'log₁₀({title})'
        print(f"  Log scale: vmin={vmin:.3f}, vmax={vmax:.3f}")
    else:
        # For linear scale, use all values including zeros
        masked = weight.copy()
        vmin, vmax = np.min(weight), np.max(weight)
        cmap_label = title
        print(f"  Linear scale: vmin={vmin:.6f}, vmax={vmax:.6f}")
    
    # Standard intensity scaling (no reversal)
    im = ax.imshow(masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                   origin='lower', cmap=cmap, alpha=0.8, 
                   vmin=vmin, vmax=vmax)
    ax.set_title(f'{title} ({"Log Scale" if use_log_scale else "Linear Scale"})')
    ax.grid(True, alpha=0.3)
    
    # Set physical axis limits and tick formatting
    ky_lim_physical = [-3*np.pi/b, 3*np.pi/b]
    kx_lim_physical = [-1*np.pi/a, 1*np.pi/a]
    
    ax.set_xlim(ky_lim_physical)
    ax.set_ylim(kx_lim_physical)
    
    # Custom tick formatting to show π/a and π/b units
    ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
    kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
    ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
    kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
    ax.set_xticks(ky_ticks)
    ax.set_yticks(kx_ticks)
    ax.set_xticklabels(ky_labels)
    ax.set_yticklabels(kx_labels)
    ax.set_xlabel('ky (π/b)')
    ax.set_ylabel('kx (π/a)')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cmap_label)
    
    return im


def test_corrected_orbital_spectral_weights():
    """Test the corrected orbital spectral weights using eigenvector projection."""
    
    print("Testing corrected orbital spectral weights using eigenvector projection...")
    
    # Configuration variables
    kz = 0.0
    resolution = 512  # High resolution for detailed features
    omega = 0.0  # Fermi level - evaluate spectral function exactly at this energy
    eta = 0.005  # Increased broadening based on debug results
    use_log_scale = False  # Use linear scaling
    
    print(f"Using eigenvector projection method with η={eta:.4f} eV broadening")
    print(f"Evaluating spectral function at exactly ω = {omega:.3f} eV (Fermi level)")
    print(f"Using {'logarithmic' if use_log_scale else 'linear'} scaling for spectral weights")
    
    # Create cache directory
    cache_dir = "outputs/ute2_fixed/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key for corrected orbital spectral weights
    cache_params = {
        'kz': kz,
        'resolution': resolution,
        'omega': omega,
        'eta': eta,
        'method': 'eigenvector_projection_delta_function'
    }
    cache_key = hashlib.md5(str(sorted(cache_params.items())).encode()).hexdigest()[:8]
    spectral_cache_file = os.path.join(cache_dir, f"corrected_orbital_spectral_weights_{cache_key}.pkl")
    
    # Get band energies for traditional Fermi surface plot
    energies = band_energies_on_slice(kz, resolution=resolution)
    
    # Check which bands cross Fermi level
    print(f"\\nAt kz = {kz}:")
    fermi_crossing_bands = []
    for i in range(4):
        E_min = np.min(energies[:,:,i])
        E_max = np.max(energies[:,:,i])
        crosses_fermi = E_min <= 0 <= E_max
        print(f"Band {i}: {E_min:.3f} to {E_max:.3f} eV - {'CROSSES FERMI' if crosses_fermi else 'no crossing'}")
        if crosses_fermi:
            fermi_crossing_bands.append(i)
    
    # Create plots with subplots for all orbitals
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
    
    # Create momentum grid for traditional Fermi surface plot
    nk_energies = energies.shape[0]
    kx_vals_energies = np.linspace(-1*np.pi/a, 1*np.pi/a, nk_energies)
    ky_vals_energies = np.linspace(-3*np.pi/b, 3*np.pi/b, nk_energies)
    
    # Plot 1: Traditional Fermi surface
    ax1.set_title(f'UTe2 Fermi Surface at kz=0', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Set physical axis limits and tick formatting
    ky_lim_physical = [-3*np.pi/b, 3*np.pi/b]
    kx_lim_physical = [-1*np.pi/a, 1*np.pi/a]
    
    ax1.set_xlim(ky_lim_physical)
    ax1.set_ylim(kx_lim_physical)
    
    # Custom tick formatting to show π/a and π/b units
    ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
    kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
    ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
    kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
    ax1.set_xticks(ky_ticks)
    ax1.set_yticks(kx_ticks)
    ax1.set_xticklabels(ky_labels)
    ax1.set_yticklabels(kx_labels)
    ax1.set_xlabel('ky (π/b)')
    ax1.set_ylabel('kx (π/a)')
    
    colors = ['#1E88E5', '#42A5F5', "#9635E5", "#FD0000"]
    
    for band in fermi_crossing_bands:
        Z = energies[:, :, band]
        cs = ax1.contour(ky_vals_energies, kx_vals_energies, Z, levels=[0], 
                        colors=[colors[band]], linewidths=3, alpha=0.8)
        ax1.plot([], [], color=colors[band], linewidth=3, label=f'Band {band+1}')
        print(f"Plotted Fermi surface for band {band}")
    
    ax1.legend()
    
    # Now compute corrected orbital spectral weights
    try:
        # Try to load from cache first
        if os.path.exists(spectral_cache_file):
            print("\\nLoading corrected orbital spectral weights from cache...")
            with open(spectral_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                U_5f_weight = cache_data['U_5f_weight']
                U_6d_weight = cache_data['U_6d_weight']
                Te1_p_weight = cache_data['Te1_p_weight'] 
                Te2_p_weight = cache_data['Te2_p_weight']
                total_weight = cache_data['total_weight']
                kx_vals = cache_data['kx_vals']
                ky_vals = cache_data['ky_vals']
        else:
            print("\nComputing corrected orbital spectral weights...")
            U_5f_weight, U_6d_weight, Te1_p_weight, Te2_p_weight, total_weight, kx_vals, ky_vals = compute_orbital_spectral_weights_eigenvectors(
                kz, resolution=resolution, omega=omega, eta=eta)
            
            # Save to cache
            print("Saving corrected orbital spectral weights to cache...")
            cache_data = {
                'U_5f_weight': U_5f_weight,
                'U_6d_weight': U_6d_weight,
                'Te1_p_weight': Te1_p_weight,
                'Te2_p_weight': Te2_p_weight,
                'total_weight': total_weight,
                'kx_vals': kx_vals,
                'ky_vals': ky_vals,
                'cache_params': cache_params
            }
            with open(spectral_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached corrected orbital spectral weights: {spectral_cache_file}")
        
        print(f"U 5f spectral weight range: {np.min(U_5f_weight):.6f} to {np.max(U_5f_weight):.6f}")
        print(f"U 6d spectral weight range: {np.min(U_6d_weight):.6f} to {np.max(U_6d_weight):.6f}")
        print(f"Te1 p spectral weight range: {np.min(Te1_p_weight):.6f} to {np.max(Te1_p_weight):.6f}")
        print(f"Te2 p spectral weight range: {np.min(Te2_p_weight):.6f} to {np.max(Te2_p_weight):.6f}")
        print(f"Total spectral weight range: {np.min(total_weight):.6f} to {np.max(total_weight):.6f}")
        
        # Plot all orbital contributions with different colormaps
        plot_orbital_spectral_weight(U_5f_weight, 'U 5f Orbital Spectral Weight', ax2, 
                                    kx_vals, ky_vals, use_log_scale, 'viridis')
        
        plot_orbital_spectral_weight(U_6d_weight, 'U 6d Orbital Spectral Weight', ax3, 
                                    kx_vals, ky_vals, use_log_scale, 'plasma')
        
        plot_orbital_spectral_weight(Te1_p_weight, 'Te1 p Orbital Spectral Weight', ax4, 
                                    kx_vals, ky_vals, use_log_scale, 'coolwarm')
        
        plot_orbital_spectral_weight(Te2_p_weight, 'Te2 p Orbital Spectral Weight', ax5, 
                                    kx_vals, ky_vals, use_log_scale, 'cividis')
        
        plot_orbital_spectral_weight(total_weight, 'Total Spectral Weight', ax6, 
                                    kx_vals, ky_vals, use_log_scale, 'gray')
        
        # Save individual standalone plots
        orbital_names = ['U_5f_corrected', 'U_6d_corrected', 'Te1_p_corrected', 'Te2_p_corrected', 'Total_corrected']
        orbital_weights = [U_5f_weight, U_6d_weight, Te1_p_weight, Te2_p_weight, total_weight]
        orbital_titles = ['U 5f Orbital (Corrected)', 'U 6d Orbital (Corrected)', 'Te1 p Orbital (Corrected)', 'Te2 p Orbital (Corrected)', 'Total (Corrected)']
        orbital_cmaps = ['viridis', 'plasma', 'coolwarm', 'cividis', 'gray']
        
        for name, weight, title, cmap in zip(orbital_names, orbital_weights, orbital_titles, orbital_cmaps):
            fig_standalone = plt.figure(figsize=(10, 8))
            ax_standalone = fig_standalone.add_subplot(111)
            
            plot_orbital_spectral_weight(weight, f'{title} Spectral Weight', ax_standalone, 
                                       kx_vals, ky_vals, use_log_scale, cmap)
            
            plt.tight_layout()
            filename = f'outputs/ute2_fixed/{name.lower()}_spectral_weight_standalone.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig_standalone)
            print(f"Saved {filename}")
        
    except Exception as e:
        print(f"Error computing corrected orbital spectral weights: {e}")
        import traceback
        traceback.print_exc()
        for ax in [ax2, ax3, ax4, ax5, ax6]:
            ax.text(0.5, 0.5, 'Error computing\\nspectral weights', 
                   transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('outputs/ute2_fixed/corrected_orbital_spectral_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\nCorrected orbital spectral weight analysis completed! Using {'logarithmic' if use_log_scale else 'linear'} scaling")
    print("Check outputs/ute2_fixed/corrected_orbital_spectral_weights.png")


if __name__ == "__main__":
    import sys
    
    # Check for diagnostic-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        print("=== UTe2 Corrected Orbital Spectral Weights Diagnostic Mode ===")
        print(f"lattice parameters: a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å")
        print()
        print("Checking Fermi surface crossing bands...")
        energies = band_energies_on_slice(0.0, resolution=256)
        for i in range(4):
            E_min = np.min(energies[:,:,i])
            E_max = np.max(energies[:,:,i])
            crosses_fermi = E_min <= 0 <= E_max
            print(f"Band {i}: {E_min:.3f} to {E_max:.3f} eV - {'CROSSES' if crosses_fermi else 'no crossing'}")
        print("Diagnostic complete. Run without 'diagnostic' argument for full analysis.")
    else:
        test_corrected_orbital_spectral_weights()