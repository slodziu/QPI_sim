#!/usr/bin/env python3
"""
Test script to visualize UTe2 Fermi surface with spectral weights like the reference image.
"""

from UTe2_fixed import *
import matplotlib.pyplot as plt
import numpy as np

def test_fermi_surface_and_spectral_weights():
    """Test and visualize the UTe2 Fermi surface with spectral weights."""
    
    print("Testing UTe2 Fermi surface and spectral weights...")
    
    # Configuration variables
    kz = 0.0
    resolution = 1000  # Higher resolution for better quality
    top_intensity_percent = 40  # Control what percentage of top intensities to show
    
    print(f"Showing top {top_intensity_percent}% of spectral weight intensities")
    
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
    
    # Create momentum grid using physical k-values
    nk = energies.shape[0]
    kx_vals = np.linspace(-1*np.pi/a, 1*np.pi/a, nk)  # Match band_energies_on_slice range
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, nk)       # Match band_energies_on_slice range
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    
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
        Z = energies[:, :, band].T
        cs = ax1.contour(ky_vals, kx_vals, Z, levels=[0], 
                        colors=[colors[band]], linewidths=3, alpha=0.8)
        ax1.plot([], [], color=colors[band], linewidth=3, label=f'Band {band+1}')
        print(f"Plotted Fermi surface for band {band}")
    
    ax1.legend()
    
    # Calculate the threshold percentile (bottom threshold = 100 - top_percent)
    bottom_percentile = 100 - top_intensity_percent
    
    # Now test spectral weights
    try:
        print("\nComputing spectral weights...")
        total_weight, orbital_contributions = compute_spectral_weights(kz, energy_window=0.05, resolution=resolution)
        
        print(f"Total spectral weight range: {np.min(total_weight):.6f} to {np.max(total_weight):.6f}")
        
        # Calculate orbital contributions first for use in overlays
        Te_weight = orbital_contributions['Te1'] + orbital_contributions['Te2']
        U_weight = orbital_contributions['U1'] + orbital_contributions['U2']
        
        # Plot 2: Total spectral weight
        total_nonzero = total_weight[total_weight > 0]
        if len(total_nonzero) > 0:
            # Show only top x% of intensities
            total_threshold = np.percentile(total_nonzero, bottom_percentile)
            total_masked = np.where(total_weight >= total_threshold, total_weight, np.nan)
            
            im1 = ax2.imshow(total_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                           origin='lower', cmap='viridis', alpha=0.8, 
                           vmin=total_threshold, vmax=np.max(total_weight))
            ax2.set_title(f'Total Spectral Weight (Top {top_intensity_percent}%)')
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
            cbar1.set_label('Spectral Weight')
            
            # Save standalone total spectral weight plot
            fig_standalone = plt.figure(figsize=(10, 8))
            ax_standalone = fig_standalone.add_subplot(111)
            
            im_standalone = ax_standalone.imshow(total_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                               origin='lower', cmap='viridis', alpha=0.8, 
                                               vmin=total_threshold, vmax=np.max(total_weight))
            ax_standalone.set_title(f'Total Spectral Weight (Top {top_intensity_percent}%)', fontsize=16)
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
            cbar_standalone.set_label('Spectral Weight', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('outputs/ute2_fixed/total_spectral_weight_standalone.png', dpi=300, bbox_inches='tight')
            plt.close(fig_standalone)
            print("Saved standalone total spectral weight plot")
        
        # Plot 3: Te orbital contributions
        Te_nonzero = Te_weight[Te_weight > 0]
        if len(Te_nonzero) > 0:
            # Show only top x% of Te spectral weight intensities
            Te_threshold = np.percentile(Te_nonzero, bottom_percentile)
            Te_masked = np.where(Te_weight >= Te_threshold, Te_weight, np.nan)
            
            im2 = ax3.imshow(Te_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                           origin='lower', cmap='plasma', alpha=0.8,
                           vmin=Te_threshold, vmax=np.max(Te_weight))
            ax3.set_title(f'Te Orbital Spectral Weight (Top {top_intensity_percent}%)')
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
            # Show only top x% of U spectral weight intensities
            U_threshold = np.percentile(U_nonzero, bottom_percentile)
            U_masked = np.where(U_weight >= U_threshold, U_weight, np.nan)
            
            im3 = ax4.imshow(U_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                           origin='lower', cmap='coolwarm', alpha=0.8,
                           vmin=U_threshold, vmax=np.max(U_weight))
            ax4.set_title(f'U Orbital Spectral Weight (Top {top_intensity_percent}%)')
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
    
    print(f"\nTest completed! Showed top {top_intensity_percent}% intensities")
    print("Check outputs/ute2_fixed/spectral_weights.png")

if __name__ == "__main__":
    test_fermi_surface_and_spectral_weights()