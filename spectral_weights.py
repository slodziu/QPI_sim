#!/usr/bin/env python3
"""
Green's function spectral weight analysis for UTe2 Fermi surface.

Uses the proper Green's function formalism:
1. Build Hamiltonian H(k) with explicit orbital degrees of freedom (U 5f, Te p)
2. Form Green's function: G₀(k,ω) = [(ω+iη)I - H(k)]⁻¹
3. Compute spectral function: A₀(k,ω) = -(1/π)Im[G₀(k,ω)]
4. Project onto orbitals: W₅f(k,ω) = Tr[P₅f A₀(k,ω)]
At ω=0 this gives the momentum map of 5f spectral weight on the Fermi surface.
"""

from UTe2_fixed import *
import matplotlib.pyplot as plt
import numpy as np
import os
import hashlib
import pickle

def compute_u5f_spectral_weights(kz, resolution=512, omega=0.0, eta=0.01):
    """
    Compute U 5f orbital spectral weights using Green's function formalism.
    
    Parameters:
    - kz: fixed kz value for 2D slice
    - resolution: k-space resolution
    - omega: frequency (typically 0.0 for Fermi surface)
    - eta: broadening parameter
    
    Returns:
    - U_5f_weight: U 5f orbital spectral weight
    - kx_vals, ky_vals: momentum grid
    """
    
    print(f"Computing U 5f spectral weights at ω={omega:.3f} eV, η={eta:.3f} eV")
    
    # Create k-space grid
    nk = resolution
    kx_vals = np.linspace(-1*np.pi/a, 1*np.pi/a, nk)
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, nk)
    
    # Initialize spectral weight array
    U_5f_weight = np.zeros((nk, nk))
    
    # Define U 5f orbital projection operator
    # Our Hamiltonian has 4 orbitals: [U1, U2, Te1, Te2]
    # U 5f orbitals: indices 0, 1
    P_5f = np.zeros((4, 4))
    P_5f[0, 0] = 1.0  # U1 orbital
    P_5f[1, 1] = 1.0  # U2 orbital
    
    print(f"Computing U 5f spectral weights on {nk}x{nk} grid...")
    
    for i, kx in enumerate(kx_vals):
        if i % (nk // 10) == 0:
            print(f"  Progress: {i}/{nk} ({100*i/nk:.1f}%)")
            
        for j, ky in enumerate(ky_vals):
            # Get Hamiltonian at this k-point
            H = H_full(kx, ky, kz)
            
            # Compute Green's function: G = [(ω + iη)I - H]^(-1)
            identity = np.eye(4)
            G_matrix = np.linalg.inv((omega + 1j*eta)*identity - H)
            
            # Compute spectral function: A = -(1/π) Im[G]
            A_matrix = -(1.0/np.pi) * np.imag(G_matrix)
            
            # Project onto U 5f orbitals: Tr[P_5f * A]
            U_5f_weight[i, j] = np.trace(P_5f @ A_matrix)
    
    print(f"U 5f spectral weight computation complete.")
    print(f"U 5f weight range: {np.min(U_5f_weight):.6f} to {np.max(U_5f_weight):.6f}")
    
    return U_5f_weight, kx_vals, ky_vals

def test_u5f_spectral_weights():
    """Test and visualize the UTe2 Fermi surface with U 5f orbital spectral weights."""
    
    print("Testing UTe2 Fermi surface with U 5f orbital spectral weights...")
    
    # Configuration variables
    kz = 0.0
    resolution = 512  # Reasonable resolution for Green's function calculation
    omega = 0.0  # At Fermi level
    eta = 0.01  # Broadening parameter in eV
    use_log_scale = True  # Use logarithmic scaling for better visibility
    
    print(f"Using Green's function formalism with η={eta:.3f} eV broadening")
    print(f"Computing at ω={omega:.3f} eV (Fermi level)")
    print(f"Using {'logarithmic' if use_log_scale else 'linear'} scaling for spectral weights")
    
    # Create cache directory
    cache_dir = "outputs/ute2_fixed/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key for U 5f spectral weights
    cache_params = {
        'kz': kz,
        'resolution': resolution,
        'omega': omega,
        'eta': eta
    }
    cache_key = hashlib.md5(str(sorted(cache_params.items())).encode()).hexdigest()[:8]
    spectral_cache_file = os.path.join(cache_dir, f"u5f_spectral_weights_{cache_key}.pkl")
    
    # Get band energies for traditional Fermi surface plot
    energies = band_energies_on_slice(kz, resolution=resolution)
    
    # Check which bands cross Fermi level
    print(f"\nAt kz = {kz}:")
    fermi_crossing_bands = []
    for i in range(4):
        E_min = np.min(energies[:,:,i])
        E_max = np.max(energies[:,:,i])
        crosses_fermi = E_min <= 0 <= E_max
        print(f"Band {i}: {E_min:.3f} to {E_max:.3f} eV - {'CROSSES FERMI' if crosses_fermi else 'no crossing'}")
        if crosses_fermi:
            fermi_crossing_bands.append(i)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
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
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    for band in fermi_crossing_bands:
        Z = energies[:, :, band]
        cs = ax1.contour(ky_vals_energies, kx_vals_energies, Z, levels=[0], 
                        colors=[colors[band]], linewidths=3, alpha=0.8)
        ax1.plot([], [], color=colors[band], linewidth=3, label=f'Band {band+1}')
        print(f"Plotted Fermi surface for band {band}")
    
    ax1.legend()
    
    # Now compute U 5f spectral weights
    try:
        # Try to load from cache first
        if os.path.exists(spectral_cache_file):
            print("\nLoading U 5f spectral weights from cache...")
            with open(spectral_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                U_5f_weight = cache_data['U_5f_weight']
                kx_vals = cache_data['kx_vals']
                ky_vals = cache_data['ky_vals']
        else:
            print("\nComputing U 5f spectral weights...")
            U_5f_weight, kx_vals, ky_vals = compute_u5f_spectral_weights(
                kz, resolution=resolution, omega=omega, eta=eta)
            
            # Save to cache
            print("Saving U 5f spectral weights to cache...")
            cache_data = {
                'U_5f_weight': U_5f_weight,
                'kx_vals': kx_vals,
                'ky_vals': ky_vals,
                'cache_params': cache_params
            }
            with open(spectral_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached U 5f spectral weights: {spectral_cache_file}")
        
        print(f"U 5f spectral weight range: {np.min(U_5f_weight):.6f} to {np.max(U_5f_weight):.6f}")
        
        # Plot 2: U 5f orbital spectral weight
        U_nonzero = U_5f_weight[U_5f_weight > 0]
        if len(U_nonzero) > 0:
            # Use logarithmic scaling
            if use_log_scale:
                epsilon = np.min(U_nonzero) * 1e-6
                U_display = np.log10(U_5f_weight + epsilon)
                U_masked = np.where(U_5f_weight > 0, U_display, np.nan)
                vmin, vmax = np.nanmin(U_masked), np.nanmax(U_masked)
                cmap_label = 'log₁₀(U 5f Spectral Weight)'
            else:
                U_masked = np.where(U_5f_weight > 0, U_5f_weight, np.nan)
                vmin, vmax = np.nanmin(U_masked), np.nanmax(U_masked)
                cmap_label = 'U 5f Spectral Weight'
            
            # Reverse intensity scaling - flip vmin/vmax for reversed grayscale
            im = ax2.imshow(U_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                           origin='lower', cmap='gray', alpha=0.8, 
                           vmin=vmax, vmax=vmin)
            ax2.set_title(f'U 5f Orbital Spectral Weight ({"Log Scale" if use_log_scale else "Linear Scale"})')
            ax2.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting
            ax2.set_xlim(ky_lim_physical)
            ax2.set_ylim(kx_lim_physical)
            ax2.set_xticks(ky_ticks)
            ax2.set_yticks(kx_ticks)
            ax2.set_xticklabels(ky_labels)
            ax2.set_yticklabels(kx_labels)
            ax2.set_xlabel('ky (π/b)')
            ax2.set_ylabel('kx (π/a)')
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label(cmap_label)
            
            # Save standalone U 5f spectral weight plot
            fig_standalone = plt.figure(figsize=(10, 8))
            ax_standalone = fig_standalone.add_subplot(111)
            
            # Reverse intensity scaling
            im_standalone = ax_standalone.imshow(U_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                               origin='lower', cmap='gray', alpha=0.8, 
                                               vmin=vmax, vmax=vmin)
            ax_standalone.set_title(f'U 5f Orbital Spectral Weight ({"Log Scale" if use_log_scale else "Linear Scale"})', fontsize=16)
            ax_standalone.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting
            ax_standalone.set_xlim(ky_lim_physical)
            ax_standalone.set_ylim(kx_lim_physical)
            ax_standalone.set_xticks(ky_ticks)
            ax_standalone.set_yticks(kx_ticks)
            ax_standalone.set_xticklabels(ky_labels)
            ax_standalone.set_yticklabels(kx_labels)
            ax_standalone.set_xlabel('ky (π/b)', fontsize=14)
            ax_standalone.set_ylabel('kx (π/a)', fontsize=14)
            
            cbar_standalone = plt.colorbar(im_standalone, ax=ax_standalone, shrink=0.8)
            cbar_standalone.set_label(cmap_label, fontsize=12)
            
            plt.tight_layout()
            plt.savefig('outputs/ute2_fixed/u5f_spectral_weight_standalone.png', dpi=300, bbox_inches='tight')
            plt.close(fig_standalone)
            print("Saved standalone U 5f spectral weight plot")
        
    except Exception as e:
        print(f"Error computing U 5f spectral weights: {e}")
        ax2.text(0.5, 0.5, 'Error computing\nU 5f spectral weights', 
                transform=ax2.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('outputs/ute2_fixed/u5f_spectral_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nU 5f spectral weight analysis completed! Using {'logarithmic' if use_log_scale else 'linear'} scaling")
    print("Check outputs/ute2_fixed/u5f_spectral_weights.png")
    """Test and visualize the UTe2 Fermi surface with Green's function spectral weights."""
    
    print("Testing UTe2 Fermi surface with Green's function spectral weights...")
    
    # Configuration variables
    kz = 0.0
    resolution = 512  # Reasonable resolution for Green's function calculation
    omega = 0.0  # At Fermi level
    eta = 0.01  # Broadening parameter in eV
    use_log_scale = True  # Use logarithmic scaling for better visibility
    
    print(f"Using Green's function formalism with η={eta:.3f} eV broadening")
    print(f"Computing at ω={omega:.3f} eV (Fermi level)")
    print(f"Using {'logarithmic' if use_log_scale else 'linear'} scaling for spectral weights")
    
    # Create cache directory
    cache_dir = "outputs/ute2_fixed/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key for Green's function spectral weights
    cache_params = {
        'kz': kz,
        'resolution': resolution,
        'omega': omega,
        'eta': eta
    }
    cache_key = hashlib.md5(str(sorted(cache_params.items())).encode()).hexdigest()[:8]
    spectral_cache_file = os.path.join(cache_dir, f"greens_spectral_weights_{cache_key}.pkl")
    
    # Get band energies for traditional Fermi surface plot
    energies = band_energies_on_slice(kz, resolution=resolution)
    
    # Check which bands cross Fermi level
    print(f"\nAt kz = {kz}:")
    fermi_crossing_bands = []
    for i in range(4):
        E_min = np.min(energies[:,:,i])
        E_max = np.max(energies[:,:,i])
        crosses_fermi = E_min <= 0 <= E_max
        print(f"Band {i}: {E_min:.3f} to {E_max:.3f} eV - {'CROSSES FERMI' if crosses_fermi else 'no crossing'}")
        if crosses_fermi:
            fermi_crossing_bands.append(i)
    
    # Create a basic Fermi surface plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
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
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    for band in fermi_crossing_bands:
        Z = energies[:, :, band]
        cs = ax1.contour(ky_vals_energies, kx_vals_energies, Z, levels=[0], 
                        colors=[colors[band]], linewidths=3, alpha=0.8)
        ax1.plot([], [], color=colors[band], linewidth=3, label=f'Band {band+1}')
        print(f"Plotted Fermi surface for band {band}")
    
    ax1.legend()
    
    # Calculate logarithmic scaling if requested
    
    # Now compute Green's function spectral weights
    try:
        # Try to load from cache first
        if os.path.exists(spectral_cache_file):
            print("\nLoading U 5f spectral weights from cache...")
            with open(spectral_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                U_5f_weight = cache_data['U_5f_weight']
                kx_vals = cache_data['kx_vals']
                ky_vals = cache_data['ky_vals']
        else:
            print("\nComputing U 5f spectral weights...")
            U_5f_weight, kx_vals, ky_vals = compute_u5f_spectral_weights(
                kz, resolution=resolution, omega=omega, eta=eta)
            
            # Save to cache
            print("Saving U 5f spectral weights to cache...")
            cache_data = {
                'U_5f_weight': U_5f_weight,
                'kx_vals': kx_vals,
                'ky_vals': ky_vals,
                'cache_params': cache_params
            }
            with open(spectral_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached Green's function spectral weights: {spectral_cache_file}")
        
        # Rename for consistency with rest of code
        U_weight = U_5f_weight  # U 5f orbital spectral weight
        Te_weight = Te_p_weight  # Te p orbital spectral weight
        
        print(f"Total spectral weight range: {np.min(total_weight):.6f} to {np.max(total_weight):.6f}")
        print(f"U orbital weight range: {np.min(U_weight):.6f} to {np.max(U_weight):.6f}")
        print(f"Te orbital weight range: {np.min(Te_weight):.6f} to {np.max(Te_weight):.6f}")
        
        # Plot 2: Total spectral weight
        total_nonzero = total_weight[total_weight > 0]
        if len(total_nonzero) > 0:
            # Use full scale with logarithmic scaling
            if use_log_scale:
                # Add small epsilon to avoid log(0), then take log
                epsilon = np.min(total_nonzero) * 1e-6
                total_display = np.log10(total_weight + epsilon)
                total_masked = np.where(total_weight > 0, total_display, np.nan)
                vmin, vmax = np.nanmin(total_masked), np.nanmax(total_masked)
                cmap_label = 'log₁₀(Spectral Weight)'
            else:
                total_masked = np.where(total_weight > 0, total_weight, np.nan)
                vmin, vmax = np.nanmin(total_masked), np.nanmax(total_masked)
                cmap_label = 'Spectral Weight'
            
            # Reverse intensity scaling - flip vmin/vmax for reversed grayscale
            im1 = ax2.imshow(total_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                           origin='lower', cmap='gray', alpha=0.8, 
                           vmin=vmax, vmax=vmin)
            ax2.set_title(f'Total Spectral Weight ({"Log Scale" if use_log_scale else "Linear Scale"})')
            ax2.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting for ax2
            ax2.set_xlim(ky_lim_physical)
            ax2.set_ylim(kx_lim_physical)
            ax2.set_xticks(ky_ticks)
            ax2.set_yticks(kx_ticks)
            ax2.set_xticklabels(ky_labels)
            ax2.set_yticklabels(kx_labels)
            ax2.set_xlabel('ky (π/b)')
            ax2.set_ylabel('kx (π/a)')
            cbar1 = plt.colorbar(im1, ax=ax2, shrink=0.8)
            cbar1.set_label(cmap_label)
            
            # Save standalone total spectral weight plot
            fig_standalone = plt.figure(figsize=(10, 8))
            ax_standalone = fig_standalone.add_subplot(111)
            
            # Reverse intensity scaling
            im_standalone = ax_standalone.imshow(total_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                               origin='lower', cmap='gray', alpha=0.8, 
                                               vmin=vmax, vmax=vmin)
            ax_standalone.set_title(f'Total Spectral Weight ({"Log Scale" if use_log_scale else "Linear Scale"})', fontsize=16)
            ax_standalone.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting
            ax_standalone.set_xlim(ky_lim_physical)
            ax_standalone.set_ylim(kx_lim_physical)
            ax_standalone.set_xticks(ky_ticks)
            ax_standalone.set_yticks(kx_ticks)
            ax_standalone.set_xticklabels(ky_labels)
            ax_standalone.set_yticklabels(kx_labels)
            ax_standalone.set_xlabel('ky (π/b)', fontsize=14)
            ax_standalone.set_ylabel('kx (π/a)', fontsize=14)
            
            cbar_standalone = plt.colorbar(im_standalone, ax=ax_standalone, shrink=0.8)
            cbar_standalone.set_label(cmap_label, fontsize=12)
            
            plt.tight_layout()
            plt.savefig('outputs/ute2_fixed/total_spectral_weight_standalone.png', dpi=300, bbox_inches='tight')
            plt.close(fig_standalone)
            print("Saved standalone total spectral weight plot")
        
        # Plot 3: Te orbital contributions
        Te_nonzero = Te_weight[Te_weight > 0]
        if len(Te_nonzero) > 0:
            # Use same scaling as U orbital to match
            if use_log_scale:
                epsilon = np.min(Te_nonzero) * 1e-6
                Te_display = np.log10(Te_weight + epsilon)
                Te_masked = np.where(Te_weight > 0, Te_display, np.nan)
                Te_vmin, Te_vmax = np.nanmin(Te_masked), np.nanmax(Te_masked)
            else:
                Te_masked = np.where(Te_weight > 0, Te_weight, np.nan)
                Te_vmin, Te_vmax = np.nanmin(Te_masked), np.nanmax(Te_masked)
            
            print(f"Te weight display range: {Te_vmin:.6f} to {Te_vmax:.6f}")
            
            im2 = ax3.imshow(Te_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                           origin='lower', cmap='plasma', alpha=0.8,
                           vmin=Te_vmin, vmax=Te_vmax)
            ax3.set_title(f'Te Orbital Spectral Weight ({"Log Scale" if use_log_scale else "Linear Scale"})')
            ax3.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting for ax3  
            ax3.set_xlim(ky_lim_physical)
            ax3.set_ylim(kx_lim_physical)
            ax3.set_xticks(ky_ticks)
            ax3.set_yticks(kx_ticks)
            ax3.set_xticklabels(ky_labels)
            ax3.set_yticklabels(kx_labels)
            ax3.set_xlabel('ky (π/b)')
            ax3.set_ylabel('kx (π/a)')
            plt.colorbar(im2, ax=ax3, shrink=0.8)
        
        # Plot 4: U orbital contributions  
        U_nonzero = U_weight[U_weight > 0]
        if len(U_nonzero) > 0:
            # Use full scale with logarithmic scaling
            if use_log_scale:
                epsilon = np.min(U_nonzero) * 1e-6
                U_display = np.log10(U_weight + epsilon)
                U_masked = np.where(U_weight > 0, U_display, np.nan)
                U_vmin, U_vmax = np.nanmin(U_masked), np.nanmax(U_masked)
            else:
                U_masked = np.where(U_weight > 0, U_weight, np.nan)
                U_vmin, U_vmax = np.nanmin(U_masked), np.nanmax(U_masked)
            
            # Reverse intensity scaling for U orbital
            im3 = ax4.imshow(U_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                           origin='lower', cmap='coolwarm_r', alpha=0.8,
                           vmin=U_vmax, vmax=U_vmin)
            ax4.set_title(f'U Orbital Spectral Weight ({"Log Scale" if use_log_scale else "Linear Scale"})')
            ax4.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting for ax4
            ax4.set_xlim(ky_lim_physical)
            ax4.set_ylim(kx_lim_physical)
            ax4.set_xticks(ky_ticks)
            ax4.set_yticks(kx_ticks)
            ax4.set_xticklabels(ky_labels)
            ax4.set_yticklabels(kx_labels)
            ax4.set_xlabel('ky (π/b)')
            ax4.set_ylabel('kx (π/a)')
            plt.colorbar(im3, ax=ax4, shrink=0.8)
            
            # Save standalone U orbital spectral weight plot
            fig_u_standalone = plt.figure(figsize=(10, 8))
            ax_u_standalone = fig_u_standalone.add_subplot(111)
            
            # Reverse intensity scaling
            im_u_standalone = ax_u_standalone.imshow(U_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                                   origin='lower', cmap='coolwarm_r', alpha=0.8,
                                                   vmin=U_vmax, vmax=U_vmin)
            ax_u_standalone.set_title(f'U Orbital Spectral Weight ({"Log Scale" if use_log_scale else "Linear Scale"})', fontsize=16)
            ax_u_standalone.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting
            ax_u_standalone.set_xlim(ky_lim_physical)
            ax_u_standalone.set_ylim(kx_lim_physical)
            ax_u_standalone.set_xticks(ky_ticks)
            ax_u_standalone.set_yticks(kx_ticks)
            ax_u_standalone.set_xticklabels(ky_labels)
            ax_u_standalone.set_yticklabels(kx_labels)
            ax_u_standalone.set_xlabel('ky (π/b)', fontsize=14)
            ax_u_standalone.set_ylabel('kx (π/a)', fontsize=14)
            
            cbar_u_standalone = plt.colorbar(im_u_standalone, ax=ax_u_standalone, shrink=0.8)
            cbar_u_standalone.set_label('log₁₀(U Orbital Weight)' if use_log_scale else 'U Orbital Weight', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('outputs/ute2_fixed/u_orbital_spectral_weight_standalone.png', dpi=300, bbox_inches='tight')
            plt.close(fig_u_standalone)
            print("Saved standalone U orbital spectral weight plot")
            
            # Create U orbital plot with scattering vectors
            print("Computing scattering vectors for U orbital plot...")
            
            # Find high intensity points in the U orbital data
            # Convert to linear scale for peak finding if using log scale
            if use_log_scale:
                U_linear = U_weight
            else:
                U_linear = U_weight
            
            # Create coordinate arrays for vector analysis
            ky_2d, kx_2d = np.meshgrid(ky_vals, kx_vals)
            
            # Find the point at approximately ky = -2π/b, kx < 0 (below kx axis)
            ky_target_1 = -2 * np.pi / b
            ky_idx_1 = np.argmin(np.abs(ky_vals - ky_target_1))
            kx_negative_mask = kx_vals < 0
            kx_negative_indices = np.where(kx_negative_mask)[0]
            
            # Find maximum intensity point near ky = -2π/b, kx < 0
            origin_intensity = U_linear[kx_negative_indices, ky_idx_1]
            origin_kx_idx = kx_negative_indices[np.argmax(origin_intensity)]
            origin_point = (ky_vals[ky_idx_1], kx_vals[origin_kx_idx])
            
            # Find the point at ky = 0, kx < 0 (below kx axis) 
            ky_center_idx = np.argmin(np.abs(ky_vals))
            origin_2_intensity = U_linear[kx_negative_indices, ky_center_idx]
            origin_2_kx_idx = kx_negative_indices[np.argmax(origin_2_intensity)]
            origin_2_point = (ky_vals[ky_center_idx], kx_vals[origin_2_kx_idx])
            
            # Find target points for 5 vectors from first origin (only to the 'right' of that point)
            # Helper function to find local maxima in specified regions
            def find_local_maximum(data, ky_indices, kx_indices):
                """Find the point with maximum intensity in specified region"""
                best_intensity = 0
                best_indices = None
                
                if hasattr(ky_indices, 'start'):  # slice object
                    ky_range = range(max(0, ky_indices.start), min(len(ky_vals), ky_indices.stop))
                else:  # array of indices
                    ky_range = ky_indices
                    
                for ky_idx in ky_range:
                    for kx_idx in kx_indices:
                        if 0 <= ky_idx < data.shape[1] and 0 <= kx_idx < data.shape[0]:
                            if data[kx_idx, ky_idx] > best_intensity:
                                best_intensity = data[kx_idx, ky_idx]
                                best_indices = (ky_idx, kx_idx)
                return best_indices
            
            # Define strategic target regions for physically meaningful scattering vectors
            target_points = []
            
            # P1: Target just above origin (small positive kx, similar ky) 
            ky_range_1 = slice(ky_idx_1-15, ky_idx_1+15)
            kx_positive_near = np.where((kx_vals > 0) & (kx_vals < 0.4*np.pi/a))[0]
            best_1 = find_local_maximum(U_linear, ky_range_1, kx_positive_near)
            if best_1:
                target_points.append((ky_vals[best_1[0]], kx_vals[best_1[1]]))
            
            # P2: Target at positive ky=0 (same as P6 endpoint, demonstrating nesting)
            kx_positive_far = np.where(kx_vals > 0.4*np.pi/a)[0]
            target_2_intensity = U_linear[kx_positive_far, ky_center_idx]
            if len(target_2_intensity) > 0:
                target_2_kx_idx = kx_positive_far[np.argmax(target_2_intensity)]
                p2_point = (ky_vals[ky_center_idx], kx_vals[target_2_kx_idx])
                target_points.append(p2_point)
            
            # P3: Target in positive ky, positive kx region
            ky_positive_near = np.where((ky_vals > 0.5*np.pi/b) & (ky_vals < 2.0*np.pi/b))[0]
            kx_positive_mid = np.where((kx_vals > 0.1*np.pi/a) & (kx_vals < 0.6*np.pi/a))[0]
            best_3 = find_local_maximum(U_linear, ky_positive_near, kx_positive_mid)
            if best_3:
                target_points.append((ky_vals[best_3[0]], kx_vals[best_3[1]]))
            
            # P4: Same as P3 but go to same kx point as origin (symmetric scattering)
            ky_positive_far = np.where((ky_vals > 1.8*np.pi/b) & (ky_vals < 2.8*np.pi/b))[0]
            # Find point at same kx as origin but in positive ky region
            origin_kx_vicinity = np.where(abs(kx_vals - origin_point[1]) < 0.1*np.pi/a)[0]
            best_4 = find_local_maximum(U_linear, ky_positive_far, origin_kx_vicinity)
            if best_4:
                target_points.append((ky_vals[best_4[0]], kx_vals[best_4[1]]))
            else:
                # Fallback: use exact same kx as origin
                target_points.append((2.0*np.pi/b, origin_point[1]))
            
            # P5: Connect first origin to second origin (inter-origin connection)
            target_points.append(origin_2_point)
            # P6: From origin 2 to same point as P2 (demonstrating Fermi surface nesting)
            target_6_point = target_points[1] if len(target_points) > 1 else (0, 0.5*np.pi/a)
            
            # Create the vector plot
            fig_vectors = plt.figure(figsize=(12, 10))
            ax_vectors = fig_vectors.add_subplot(111)
            
            # Plot the U orbital background with reversed intensity
            im_vectors = ax_vectors.imshow(U_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                         origin='lower', cmap='coolwarm_r', alpha=0.7,
                                         vmin=U_vmax, vmax=U_vmin)
            
            # Plot vectors from first origin to targets 1-5
            vector_colors = ['yellow', 'lime', 'cyan', 'magenta', 'orange']
            vector_labels = ['P₁', 'P₂', 'P₃', 'P₄', 'P₅']
            
            for i, target in enumerate(target_points[:5]):  # Only take first 5
                if target:
                    ax_vectors.annotate('', xy=target, xytext=origin_point,
                                      arrowprops=dict(arrowstyle='->', color=vector_colors[i], 
                                                    lw=3, alpha=0.9))
                    # Label the vector near the end but offset to not hide arrow tip
                    # Use different offsets for each vector to avoid overlaps
                    if i == 1:  # P2 - offset more to avoid overlap with P6
                        offset_x = -0.15 * np.pi / b  # Negative offset for P2
                        offset_y = 0.1 * np.pi / a
                    else:
                        offset_x = 0.1 * np.pi / b  # Standard offset for others
                        offset_y = 0.05 * np.pi / a
                    ax_vectors.text(target[0] + offset_x, target[1] + offset_y, vector_labels[i], 
                                  color='black', fontsize=12, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                                  ha='center', va='center')
            
            # Plot vector 6 (from second origin) - make it white with black outline for visibility
            ax_vectors.annotate('', xy=target_6_point, xytext=origin_2_point,
                              arrowprops=dict(arrowstyle='->', color='white', 
                                            lw=4, alpha=1.0))
            # Add a black outline for better visibility
            ax_vectors.annotate('', xy=target_6_point, xytext=origin_2_point,
                              arrowprops=dict(arrowstyle='->', color='black', 
                                            lw=6, alpha=0.8, zorder=1))
            ax_vectors.annotate('', xy=target_6_point, xytext=origin_2_point,
                              arrowprops=dict(arrowstyle='->', color='white', 
                                            lw=4, alpha=1.0, zorder=2))
            mid_x_6 = (origin_2_point[0] + target_6_point[0]) / 2
            mid_y_6 = (origin_2_point[1] + target_6_point[1]) / 2
            # Position P6 label near endpoint but offset opposite to P2 to avoid overlap
            offset_x_6 = 0.15 * np.pi / b  # Positive offset for P6 (opposite to P2)
            offset_y_6 = -0.1 * np.pi / a   # Negative offset in kx direction
            ax_vectors.text(target_6_point[0] + offset_x_6, target_6_point[1] + offset_y_6, 'P₆', 
                          color='black', fontsize=12, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                          ha='center', va='center')
            
            # Mark origin points
            ax_vectors.plot(origin_point[0], origin_point[1], 'ko', markersize=8, markerfacecolor='yellow', markeredgecolor='black')
            ax_vectors.plot(origin_2_point[0], origin_2_point[1], 'ko', markersize=8, markerfacecolor='white', markeredgecolor='black')
            
            ax_vectors.set_title('U Orbital Spectral Weight with Scattering Vectors', fontsize=16)
            ax_vectors.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting
            ax_vectors.set_xlim(ky_lim_physical)
            ax_vectors.set_ylim(kx_lim_physical)
            ax_vectors.set_xticks(ky_ticks)
            ax_vectors.set_yticks(kx_ticks)
            ax_vectors.set_xticklabels(ky_labels)
            ax_vectors.set_yticklabels(kx_labels)
            ax_vectors.set_xlabel('ky (π/b)', fontsize=14)
            ax_vectors.set_ylabel('kx (π/a)', fontsize=14)
            
            cbar_vectors = plt.colorbar(im_vectors, ax=ax_vectors, shrink=0.8)
            cbar_vectors.set_label('log₁₀(U Orbital Weight)' if use_log_scale else 'U Orbital Weight', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('outputs/ute2_fixed/u_orbital_with_vectors.png', dpi=300, bbox_inches='tight')
            plt.close(fig_vectors)
            print("Saved U orbital plot with scattering vectors")
            
            # Print vector information
            print(f"\nScattering Vector Analysis:")
            print(f"Origin 1: ky={origin_point[0]/(np.pi/b):.2f}π/b, kx={origin_point[1]/(np.pi/a):.2f}π/a")
            print(f"Origin 2: ky={origin_2_point[0]/(np.pi/b):.2f}π/b, kx={origin_2_point[1]/(np.pi/a):.2f}π/a")
            for i, target in enumerate(target_points[:5]):
                if target:
                    print(f"Target {i+1}: ky={target[0]/(np.pi/b):.2f}π/b, kx={target[1]/(np.pi/a):.2f}π/a")
            print(f"Target 6: ky={target_6_point[0]/(np.pi/b):.2f}π/b, kx={target_6_point[1]/(np.pi/a):.2f}π/a")
            
            # Print vectors in (kx, ky) format
            print(f"\nScattering Vectors in (kx, ky) format:")
            for i, target in enumerate(target_points[:5]):
                if target:
                    vector_kx = target[1] - origin_point[1]  # kx component
                    vector_ky = target[0] - origin_point[0]  # ky component
                    print(f"P{i+1}: ({vector_kx/(np.pi/a):.3f}π/a, {vector_ky/(np.pi/b):.3f}π/b)")
            
            # Vector 6 from different origin
            vector_6_kx = target_6_point[1] - origin_2_point[1]
            vector_6_ky = target_6_point[0] - origin_2_point[0]
            print(f"P6: ({vector_6_kx/(np.pi/a):.3f}π/a, {vector_6_ky/(np.pi/b):.3f}π/b)")
            
            # Also print in 2π/a and 2π/b units
            print(f"\nScattering Vectors in (kx, ky) format using 2π units:")
            for i, target in enumerate(target_points[:5]):
                if target:
                    vector_kx = target[1] - origin_point[1]  # kx component
                    vector_ky = target[0] - origin_point[0]  # ky component
                    print(f"P{i+1}: ({vector_kx/(2*np.pi/a):.3f}, {vector_ky/(2*np.pi/b):.3f})")
            
            # Vector 6 from different origin in 2π units
            print(f"P6: ({vector_6_kx/(2*np.pi/a):.3f}, {vector_6_ky/(2*np.pi/b):.3f})")
            
            # Create Te orbital standalone plot and vector analysis
            print("\nCreating Te orbital analysis with scattering vectors...")
            
            # Save standalone Te orbital spectral weight plot
            fig_te_standalone = plt.figure(figsize=(10, 8))
            ax_te_standalone = fig_te_standalone.add_subplot(111)
            
            im_te_standalone = ax_te_standalone.imshow(Te_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                                     origin='lower', cmap='plasma', alpha=0.8,
                                                     vmin=Te_vmin, vmax=Te_vmax)
            ax_te_standalone.set_title(f'Te Orbital Spectral Weight ({"Log Scale" if use_log_scale else "Linear Scale"})', fontsize=16)
            ax_te_standalone.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting
            ax_te_standalone.set_xlim(ky_lim_physical)
            ax_te_standalone.set_ylim(kx_lim_physical)
            ax_te_standalone.set_xticks(ky_ticks)
            ax_te_standalone.set_yticks(kx_ticks)
            ax_te_standalone.set_xticklabels(ky_labels)
            ax_te_standalone.set_yticklabels(kx_labels)
            ax_te_standalone.set_xlabel('ky (π/b)', fontsize=14)
            ax_te_standalone.set_ylabel('kx (π/a)', fontsize=14)
            
            cbar_te_standalone = plt.colorbar(im_te_standalone, ax=ax_te_standalone, shrink=0.8)
            cbar_te_standalone.set_label('log₁₀(Te Orbital Weight)' if use_log_scale else 'Te Orbital Weight', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('outputs/ute2_fixed/te_orbital_spectral_weight_standalone.png', dpi=300, bbox_inches='tight')
            plt.close(fig_te_standalone)
            print("Saved standalone Te orbital spectral weight plot")
            
            # Create Te orbital plot with scattering vectors
            print("Computing scattering vectors for Te orbital plot...")
            
            # Find high intensity points in the Te orbital data
            # Convert to linear scale for peak finding if using log scale
            if use_log_scale:
                Te_linear = Te_weight
            else:
                Te_linear = Te_weight
            
            # Find origin points for Te orbital analysis (where Te character is strong)
            # Look for high Te intensity regions
            
            # Find the point around ky = 0, kx > 0 (upper right region)
            ky_center_idx = np.argmin(np.abs(ky_vals))
            kx_positive_mask = kx_vals > 0
            kx_positive_indices = np.where(kx_positive_mask)[0]
            
            # Find maximum Te intensity point near ky = 0, kx > 0
            te_origin_intensity = Te_linear[kx_positive_indices, ky_center_idx]
            te_origin_kx_idx = kx_positive_indices[np.argmax(te_origin_intensity)]
            te_origin_point = (ky_vals[ky_center_idx], kx_vals[te_origin_kx_idx])
            
            # Find second origin around ky = -1π/b, where Te character might be strong
            ky_neg_target = -1 * np.pi / b
            ky_neg_idx = np.argmin(np.abs(ky_vals - ky_neg_target))
            te_origin_2_intensity = Te_linear[kx_positive_indices, ky_neg_idx]
            te_origin_2_kx_idx = kx_positive_indices[np.argmax(te_origin_2_intensity)]
            te_origin_2_point = (ky_vals[ky_neg_idx], kx_vals[te_origin_2_kx_idx])
            
            # Find target points for Te orbital scattering vectors
            te_target_points = []
            
            # T1: Target in negative ky, positive kx region
            ky_negative_range = np.where((ky_vals < -0.5*np.pi/b) & (ky_vals > -2.0*np.pi/b))[0]
            kx_positive_mid = np.where((kx_vals > 0.1*np.pi/a) & (kx_vals < 0.6*np.pi/a))[0]
            best_te_1 = find_local_maximum(Te_linear, ky_negative_range, kx_positive_mid)
            if best_te_1:
                te_target_points.append((ky_vals[best_te_1[0]], kx_vals[best_te_1[1]]))
            
            # T2: Target at opposite ky position (symmetry)
            ky_opposite = np.where((ky_vals > 0.5*np.pi/b) & (ky_vals < 1.5*np.pi/b))[0]
            best_te_2 = find_local_maximum(Te_linear, ky_opposite, kx_positive_mid)
            if best_te_2:
                te_target_points.append((ky_vals[best_te_2[0]], kx_vals[best_te_2[1]]))
            
            # T3: Target in far positive ky region
            ky_far_positive = np.where((ky_vals > 1.5*np.pi/b) & (ky_vals < 2.5*np.pi/b))[0]
            best_te_3 = find_local_maximum(Te_linear, ky_far_positive, kx_positive_indices)
            if best_te_3:
                te_target_points.append((ky_vals[best_te_3[0]], kx_vals[best_te_3[1]]))
            
            # T4: Target in negative kx region
            kx_negative_indices = np.where(kx_vals < 0)[0]
            ky_around_center = np.where(abs(ky_vals) < 0.5*np.pi/b)[0]
            best_te_4 = find_local_maximum(Te_linear, ky_around_center, kx_negative_indices)
            if best_te_4:
                te_target_points.append((ky_vals[best_te_4[0]], kx_vals[best_te_4[1]]))
            
            # T5: Connect to second origin
            te_target_points.append(te_origin_2_point)
            
            # Create the Te vector plot
            fig_te_vectors = plt.figure(figsize=(12, 10))
            ax_te_vectors = fig_te_vectors.add_subplot(111)
            
            # Plot the Te orbital background
            im_te_vectors = ax_te_vectors.imshow(Te_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                               origin='lower', cmap='plasma', alpha=0.7,
                                               vmin=Te_vmin, vmax=Te_vmax)
            
            # Plot vectors from first Te origin to targets 1-5
            te_vector_colors = ['yellow', 'lime', 'cyan', 'magenta', 'orange']
            te_vector_labels = ['T₁', 'T₂', 'T₃', 'T₄', 'T₅']
            
            for i, target in enumerate(te_target_points[:5]):  # Only take first 5
                if target:
                    ax_te_vectors.annotate('', xy=target, xytext=te_origin_point,
                                         arrowprops=dict(arrowstyle='->', color=te_vector_colors[i], 
                                                       lw=3, alpha=0.9))
                    # Label the vector near the end with offset to avoid overlaps
                    offset_x = 0.1 * np.pi / b
                    offset_y = 0.05 * np.pi / a
                    ax_te_vectors.text(target[0] + offset_x, target[1] + offset_y, te_vector_labels[i], 
                                     color='black', fontsize=12, fontweight='bold',
                                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                                     ha='center', va='center')
            
            # Mark Te origin points
            ax_te_vectors.plot(te_origin_point[0], te_origin_point[1], 'ko', markersize=8, markerfacecolor='yellow', markeredgecolor='black')
            ax_te_vectors.plot(te_origin_2_point[0], te_origin_2_point[1], 'ko', markersize=8, markerfacecolor='white', markeredgecolor='black')
            
            ax_te_vectors.set_title('Te Orbital Spectral Weight with Scattering Vectors', fontsize=16)
            ax_te_vectors.grid(True, alpha=0.3)
            
            # Set physical axis limits and tick formatting
            ax_te_vectors.set_xlim(ky_lim_physical)
            ax_te_vectors.set_ylim(kx_lim_physical)
            ax_te_vectors.set_xticks(ky_ticks)
            ax_te_vectors.set_yticks(kx_ticks)
            ax_te_vectors.set_xticklabels(ky_labels)
            ax_te_vectors.set_yticklabels(kx_labels)
            ax_te_vectors.set_xlabel('ky (π/b)', fontsize=14)
            ax_te_vectors.set_ylabel('kx (π/a)', fontsize=14)
            
            cbar_te_vectors = plt.colorbar(im_te_vectors, ax=ax_te_vectors, shrink=0.8)
            cbar_te_vectors.set_label('log₁₀(Te Orbital Weight)' if use_log_scale else 'Te Orbital Weight', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('outputs/ute2_fixed/te_orbital_with_vectors.png', dpi=300, bbox_inches='tight')
            plt.close(fig_te_vectors)
            print("Saved Te orbital plot with scattering vectors")
            
            # Print Te vector information
            print(f"\nTe Orbital Scattering Vector Analysis:")
            print(f"Te Origin 1: ky={te_origin_point[0]/(np.pi/b):.2f}π/b, kx={te_origin_point[1]/(np.pi/a):.2f}π/a")
            print(f"Te Origin 2: ky={te_origin_2_point[0]/(np.pi/b):.2f}π/b, kx={te_origin_2_point[1]/(np.pi/a):.2f}π/a")
            for i, target in enumerate(te_target_points[:5]):
                if target:
                    print(f"Te Target {i+1}: ky={target[0]/(np.pi/b):.2f}π/b, kx={target[1]/(np.pi/a):.2f}π/a")
            
            # Print Te vectors in (kx, ky) format
            print(f"\nTe Scattering Vectors in (kx, ky) format:")
            for i, target in enumerate(te_target_points[:5]):
                if target:
                    vector_kx = target[1] - te_origin_point[1]  # kx component
                    vector_ky = target[0] - te_origin_point[0]  # ky component
                    print(f"T{i+1}: ({vector_kx/(np.pi/a):.3f}π/a, {vector_ky/(np.pi/b):.3f}π/b)")
        
        print(f"Te orbital weight range: {np.min(Te_weight):.6f} to {np.max(Te_weight):.6f}")
        print(f"U orbital weight range: {np.min(U_weight):.6f} to {np.max(U_weight):.6f}")
        
    except Exception as e:
        print(f"Error computing spectral weights: {e}")
        ax2.text(0.5, 0.5, 'Error computing\nspectral weights', 
                transform=ax2.transAxes, ha='center', va='center')
        ax3.text(0.5, 0.5, 'Error computing\nspectral weights', 
                transform=ax3.transAxes, ha='center', va='center')
        ax4.text(0.5, 0.5, 'Error computing\nspectral weights', 
                transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('outputs/ute2_fixed/spectral_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTest completed! Using {'logarithmic' if use_log_scale else 'linear'} scaling for full spectral weight range")
    print("Check outputs/ute2_fixed/spectral_weights.png")

if __name__ == "__main__":
    import sys
    
    # Check for diagnostic-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnostic":
        print("Running diagnostic only...")
    else:
        test_u5f_spectral_weights()