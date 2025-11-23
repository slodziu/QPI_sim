#!/usr/bin/env python3
"""
UTe2 All Orbital Spectral Weights using Green's Function Formalism
Computes and visualizes spectral weights for all orbital contributions
"""

import numpy as np
import matplotlib.pyplot as plt
from UTe2_fixed import *
import os
import pickle
import hashlib

def compute_all_orbital_spectral_weights(kz, resolution=512, omega_center=0.0, omega_width=0.02, eta=0.001, n_omega=21):
    """
    Compute all orbital spectral weights using Green's function formalism.
    
    Args:
        kz: z-component of momentum
        resolution: Number of k-points along each direction
        omega_center: Center energy for integration (eV)
        omega_width: Energy window width for integration (eV)
        eta: Broadening parameter (eV)
        n_omega: Number of energy points for integration
    
    Returns:
        U_5f_weight: U 5f orbital spectral weight matrix
        U_6d_weight: U 6d orbital spectral weight matrix  
        Te1_p_weight: Te1 p orbital spectral weight matrix
        Te2_p_weight: Te2 p orbital spectral weight matrix
        total_weight: Total spectral weight matrix
        kx_vals: kx momentum values
        ky_vals: ky momentum values
    """
    print(f"Computing all orbital spectral weights with {resolution}x{resolution} resolution...")
    print(f"Integrating over energy window: {omega_center-omega_width/2:.3f} to {omega_center+omega_width/2:.3f} eV ({n_omega} points)")
    
    # Create momentum grid (extend ky range to match projection range)
    kx_vals = np.linspace(-1*np.pi/a, 1*np.pi/a, resolution)
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, resolution)
    
    # Initialize spectral weight arrays
    U_5f_weight = np.zeros((len(kx_vals), len(ky_vals)))
    U_6d_weight = np.zeros((len(kx_vals), len(ky_vals)))
    Te1_p_weight = np.zeros((len(kx_vals), len(ky_vals)))
    Te2_p_weight = np.zeros((len(kx_vals), len(ky_vals)))
    total_weight = np.zeros((len(kx_vals), len(ky_vals)))
    
    
    # Create energy grid for integration
    omega_vals = np.linspace(omega_center - omega_width/2, omega_center + omega_width/2, n_omega)
    d_omega = omega_vals[1] - omega_vals[0] if n_omega > 1 else omega_width
    
    print("Computing Green's function spectral weights...")
    
    # Loop over k-points
    for i, kx in enumerate(kx_vals):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(kx_vals)} ({100*i/len(kx_vals):.1f}%)")
            
        for j, ky in enumerate(ky_vals):
            # Get full Hamiltonian matrix at this k-point
            H = H_full(kx, ky, kz)
            
            # Initialize integrated spectral weights for this k-point
            A_5f_integrated = 0.0
            A_6d_integrated = 0.0
            A_te1_integrated = 0.0
            A_te2_integrated = 0.0
            A_total_integrated = 0.0
            
            # Integrate over energy window
            for omega in omega_vals:
                # Compute Green's function: G₀(k,ω) = [(ω + iη)I - H(k)]⁻¹
                omega_plus_i_eta = omega + 1j * eta
                G = np.linalg.inv(omega_plus_i_eta * np.eye(4) - H)
                
                # Spectral function: A₀(k,ω) = -(1/π) * Im[G₀(k,ω)]
                A = -(1/np.pi) * np.imag(G)
                
                # Accumulate orbital contributions
                A_5f_integrated += A[0, 0] * d_omega
                A_6d_integrated += A[1, 1] * d_omega
                A_te1_integrated += A[2, 2] * d_omega
                A_te2_integrated += A[3, 3] * d_omega
                A_total_integrated += np.trace(A) * d_omega
            
            # Store integrated spectral weights
            U_5f_weight[i, j] = A_5f_integrated
            U_6d_weight[i, j] = A_6d_integrated
            Te1_p_weight[i, j] = A_te1_integrated
            Te2_p_weight[i, j] = A_te2_integrated
            total_weight[i, j] = A_total_integrated
    
    print("All orbital spectral weight computation completed")
    return U_5f_weight, U_6d_weight, Te1_p_weight, Te2_p_weight, total_weight, kx_vals, ky_vals


def plot_orbital_spectral_weight(weight, title, ax, kx_vals, ky_vals, use_log_scale=False, cmap='gray'):
    """Helper function to plot individual orbital spectral weights."""
    nonzero = weight[weight > 0]
    if len(nonzero) > 0:
        if use_log_scale:
            epsilon = np.min(nonzero) * 1e-6
            display = np.log10(weight + epsilon)
            masked = np.where(weight > 0, display, np.nan)
            vmin, vmax = np.nanmin(masked), np.nanmax(masked)
            cmap_label = f'log₁₀({title})'
        else:
            masked = np.where(weight > 0, weight, np.nan)
            vmin, vmax = np.nanmin(masked), np.nanmax(masked)
            cmap_label = title
        
        # Reverse intensity scaling for better visibility
        im = ax.imshow(masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                       origin='lower', cmap=cmap, alpha=0.8, 
                       vmin=vmax, vmax=vmin)
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
    return None


def test_all_orbital_spectral_weights():
    """Test and visualize the UTe2 Fermi surface with all orbital spectral weights."""
    
    print("Testing UTe2 Fermi surface with all orbital spectral weights...")
    
    # Configuration variables
    kz = 0.0
    resolution = 512  # Reasonable resolution for Green's function calculation
    omega_center = 0  # Center energy (Fermi level)
    omega_width = 0.01  
    eta = 0.005  # Smaller broadening parameter (eV)
    n_omega = 21  # Number of energy points for integration
    use_log_scale = False  # Use linear scaling
    
    print(f"Using Green's function formalism with η={eta:.3f} eV broadening")
    print(f"Integrating over energy window: ±{omega_width/2*1000:.1f} meV around Fermi level")
    print(f"Using {'logarithmic' if use_log_scale else 'linear'} scaling for spectral weights")
    
    # Create cache directory
    cache_dir = "outputs/ute2_fixed/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key for all orbital spectral weights
    cache_params = {
        'kz': kz,
        'resolution': resolution,
        'omega_center': omega_center,
        'omega_width': omega_width,
        'eta': eta,
        'n_omega': n_omega
    }
    cache_key = hashlib.md5(str(sorted(cache_params.items())).encode()).hexdigest()[:8]
    spectral_cache_file = os.path.join(cache_dir, f"all_orbital_spectral_weights_{cache_key}.pkl")
    
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
    
    # Now compute all orbital spectral weights
    try:
        # Try to load from cache first
        if os.path.exists(spectral_cache_file):
            print("\\nLoading all orbital spectral weights from cache...")
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
            print("\\nComputing all orbital spectral weights...")
            U_5f_weight, U_6d_weight, Te1_p_weight, Te2_p_weight, total_weight, kx_vals, ky_vals = compute_all_orbital_spectral_weights(
                kz, resolution=resolution, omega_center=omega_center, omega_width=omega_width, eta=eta, n_omega=n_omega)
            
            # Save to cache
            print("Saving all orbital spectral weights to cache...")
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
            print(f"Cached all orbital spectral weights: {spectral_cache_file}")
        
        print(f"U 5f spectral weight range: {np.min(U_5f_weight):.6f} to {np.max(U_5f_weight):.6f}")
        print(f"U 6d spectral weight range: {np.min(U_6d_weight):.6f} to {np.max(U_6d_weight):.6f}")
        print(f"Te1 p spectral weight range: {np.min(Te1_p_weight):.6f} to {np.max(Te1_p_weight):.6f}")
        print(f"Te2 p spectral weight range: {np.min(Te2_p_weight):.6f} to {np.max(Te2_p_weight):.6f}")
        print(f"Total spectral weight range: {np.min(total_weight):.6f} to {np.max(total_weight):.6f}")
        
        # Plot all orbital contributions
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
        orbital_names = ['U_5f', 'U_6d', 'Te1_p', 'Te2_p', 'Total']
        orbital_weights = [U_5f_weight, U_6d_weight, Te1_p_weight, Te2_p_weight, total_weight]
        orbital_titles = ['U 5f Orbital', 'U 6d Orbital', 'Te1 p Orbital', 'Te2 p Orbital', 'Total']
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
        print(f"Error computing orbital spectral weights: {e}")
        for ax in [ax2, ax3, ax4, ax5, ax6]:
            ax.text(0.5, 0.5, 'Error computing\\nspectral weights', 
                   transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('outputs/ute2_fixed/all_orbital_spectral_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\nAll orbital spectral weight analysis completed! Using {'logarithmic' if use_log_scale else 'linear'} scaling")
    print("Check outputs/ute2_fixed/all_orbital_spectral_weights.png")


if __name__ == "__main__":
    import sys
    
    # Check for diagnostic-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        print("=== UTe2 All Orbital Spectral Weights Diagnostic Mode ===")
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
        test_all_orbital_spectral_weights()