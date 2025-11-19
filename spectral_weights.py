#!/usr/bin/env python3
"""
Test script to visualize UTe2 Fermi surface with spectral weights like the reference image.
"""

from UTe2_fixed import *
import matplotlib.pyplot as plt
import numpy as np
import os
import hashlib
import pickle

def test_fermi_surface_and_spectral_weights():
    """Test and visualize the UTe2 Fermi surface with spectral weights."""
    
    print("Testing UTe2 Fermi surface and spectral weights...")
    
    # Configuration variables
    kz = 0.0
    resolution = 1024  # Higher resolution for better quality
    use_log_scale = True  # Use logarithmic scaling for better visibility
    E_win = 0.01  # Energy window around Fermi level in eV
    print(f"Using {'logarithmic' if use_log_scale else 'linear'} scaling for spectral weights")
    
    # Create cache directory
    cache_dir = "outputs/ute2_fixed/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key for spectral weights
    cache_params = {
        'kz': kz,
        'resolution': resolution,
        'energy_window': E_win  # This will be used later
    }
    cache_key = hashlib.md5(str(sorted(cache_params.items())).encode()).hexdigest()[:8]
    spectral_cache_file = os.path.join(cache_dir, f"spectral_weights_{cache_key}.pkl")
    
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
        Z = energies[:, :, band]
        cs = ax1.contour(ky_vals, kx_vals, Z, levels=[0], 
                        colors=[colors[band]], linewidths=3, alpha=0.8)
        ax1.plot([], [], color=colors[band], linewidth=3, label=f'Band {band+1}')
        print(f"Plotted Fermi surface for band {band}")
    
    ax1.legend()
    
    # Calculate logarithmic scaling if requested
    
    # Now test spectral weights
    try:
        # Try to load from cache first
        if os.path.exists(spectral_cache_file):
            print("\nLoading spectral weights from cache...")
            with open(spectral_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                total_weight = cache_data['total_weight']
                orbital_contributions = cache_data['orbital_contributions']
        else:
            print("\nComputing spectral weights...")
            total_weight, orbital_contributions = compute_spectral_weights(kz, energy_window=E_win, resolution=resolution)
            
            # Save to cache
            print("Saving spectral weights to cache...")
            cache_data = {
                'total_weight': total_weight,
                'orbital_contributions': orbital_contributions,
                'cache_params': cache_params
            }
            with open(spectral_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached spectral weights: {spectral_cache_file}")
        
        print(f"Total spectral weight range: {np.min(total_weight):.6f} to {np.max(total_weight):.6f}")
        
        # Calculate orbital contributions first for use in overlays
        Te_weight = orbital_contributions['Te1'] + orbital_contributions['Te2']
        U_weight = orbital_contributions['U1'] + orbital_contributions['U2']
        
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
            
            im1 = ax2.imshow(total_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                           origin='lower', cmap='gray_r', alpha=0.8, 
                           vmin=vmin, vmax=vmax)
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
            
            im_standalone = ax_standalone.imshow(total_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                               origin='lower', cmap='gray_r', alpha=0.8, 
                                               vmin=vmin, vmax=vmax)
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
            # Use full scale with logarithmic scaling
            if use_log_scale:
                epsilon = np.min(Te_nonzero) * 1e-6
                Te_display = np.log10(Te_weight + epsilon)
                Te_masked = np.where(Te_weight > 0, Te_display, np.nan)
                Te_vmin, Te_vmax = np.nanmin(Te_masked), np.nanmax(Te_masked)
            else:
                Te_masked = np.where(Te_weight > 0, Te_weight, np.nan)
                Te_vmin, Te_vmax = np.nanmin(Te_masked), np.nanmax(Te_masked)
            
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
            
            im3 = ax4.imshow(U_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                           origin='lower', cmap='coolwarm', alpha=0.8,
                           vmin=U_vmin, vmax=U_vmax)
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
            
            im_u_standalone = ax_u_standalone.imshow(U_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                                   origin='lower', cmap='coolwarm', alpha=0.8,
                                                   vmin=U_vmin, vmax=U_vmax)
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
            
            # P4: Target in far positive ky region, negative kx
            ky_positive_far = np.where((ky_vals > 1.8*np.pi/b) & (ky_vals < 2.8*np.pi/b))[0]
            kx_negative_selected = kx_negative_indices[-40:-10]  # Not too deep negative
            best_4 = find_local_maximum(U_linear, ky_positive_far, kx_negative_selected)
            if best_4:
                target_points.append((ky_vals[best_4[0]], kx_vals[best_4[1]]))
            
            # P5: Target in wrap-around region (far negative ky, representing periodicity)
            ky_negative_far = np.where(ky_vals < -2.7*np.pi/b)[0]
            if len(ky_negative_far) > 0:
                kx_mixed = np.concatenate([kx_negative_indices[-20:], kx_positive_near[:10]])
                best_5 = find_local_maximum(U_linear, ky_negative_far, kx_mixed)
                if best_5:
                    target_points.append((ky_vals[best_5[0]], kx_vals[best_5[1]]))
            
            # P6: From origin 2 to same point as P2 (demonstrating Fermi surface nesting)
            target_6_point = target_points[1] if len(target_points) > 1 else (0, 0.5*np.pi/a)
            
            # Create the vector plot
            fig_vectors = plt.figure(figsize=(12, 10))
            ax_vectors = fig_vectors.add_subplot(111)
            
            # Plot the U orbital background
            im_vectors = ax_vectors.imshow(U_masked, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()], 
                                         origin='lower', cmap='coolwarm', alpha=0.7,
                                         vmin=U_vmin, vmax=U_vmax)
            
            # Plot vectors from first origin to targets 1-5
            vector_colors = ['yellow', 'lime', 'cyan', 'magenta', 'orange']
            vector_labels = ['P₁', 'P₂', 'P₃', 'P₄', 'P₅']
            
            for i, target in enumerate(target_points[:5]):  # Only take first 5
                if target:
                    ax_vectors.annotate('', xy=target, xytext=origin_point,
                                      arrowprops=dict(arrowstyle='->', color=vector_colors[i], 
                                                    lw=3, alpha=0.9))
                    # Label the vector
                    mid_x = (origin_point[0] + target[0]) / 2
                    mid_y = (origin_point[1] + target[1]) / 2
                    ax_vectors.text(mid_x, mid_y, vector_labels[i], 
                                  color=vector_colors[i], fontsize=12, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
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
            ax_vectors.text(mid_x_6, mid_y_6, 'P₆', 
                          color='white', fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.9, edgecolor='white'))
            
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
    test_fermi_surface_and_spectral_weights()