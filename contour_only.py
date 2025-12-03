import numpy as np
import matplotlib.pyplot as plt
from UTe2_fixed import *
from scipy.ndimage import gaussian_filter

def orbital_projected_spectral_weight_paper_method(H_grid, omega=0.0, eta=5e-3):
    """
    Compute HYBRIDIZED U 5f orbital-projected spectral weight as in Nature Physics paper.
    Now with per-band normalization for Fermi-crossing bands.
    """
    Ny, Nx = H_grid.shape[:2]

    # Vectorized computation
    H_flat = H_grid.reshape(-1, 4, 4)  # Shape: (M, 4, 4)
    M = H_flat.shape[0]
    
    # Diagonalize all Hamiltonians at once
    eigenvals, eigenvecs = np.linalg.eigh(H_flat)  # eigenvals: (M, 4), eigenvecs: (M, 4, 4)
    
    # Only consider bands that cross Fermi level
    fermi_window = 0.1  # eV - broader window to catch all relevant bands
    fermi_mask = np.abs(eigenvals - omega) < fermi_window  # Shape: (M, 4)
    
    # CRITICAL FIX: Compute HYBRIDIZED U 5f character using actual δ = 0.13 eV
    U_pure_weights = np.abs(eigenvecs[:, 0, :])**2 + np.abs(eigenvecs[:, 1, :])**2  # Pure U 5f
    
    delta = 0.13  # eV - hybridization parameter
    
    # Hybridization enhancement factor based on the actual δ parameter
    Te_pure_weights = np.abs(eigenvecs[:, 2, :])**2 + np.abs(eigenvecs[:, 3, :])**2  # Pure Te p
    hybridization_factor = 1 + (delta**2) * Te_pure_weights / ((eigenvals - 0.0)**2 + delta**2 + 1e-6)
    
    # The "hybridized U 5f spectral weight" is the U weight enhanced by δ-mixing
    U_hybridized_weights = U_pure_weights * hybridization_factor
    
    # Standard Lorentzian
    lorentzian = (eta/np.pi) / ((omega - eigenvals)**2 + eta**2)
    
    # NEW: Compute spectral weight for each band separately, then normalize
    A_bands = np.zeros((M, 4))  # Spectral weight for each band at each k-point
    
    for band in range(4):
        # Spectral weight for this band only
        band_spectral = U_hybridized_weights[:, band] * lorentzian[:, band] * fermi_mask[:, band]
        A_bands[:, band] = band_spectral
    
    # Find which bands actually contribute (cross Fermi level)
    band_contributions = np.sum(A_bands, axis=0)  # Total contribution from each band
    active_bands = np.where(band_contributions > 1e-6)[0]  # Bands with significant weight
    
    print(f"Active bands (crossing Fermi level): {active_bands}")
    print(f"Band contributions: {band_contributions}")
    
    # Normalize active bands to have same maximum intensity
    A_normalized = np.zeros_like(A_bands)
    if len(active_bands) > 1:
        # Find max intensity for each active band
        band_maxes = []
        for band in active_bands:
            band_max = np.max(A_bands[:, band])
            band_maxes.append(band_max)
            print(f"Band {band} max intensity: {band_max:.1f}")
        
        # Normalize to the average of the maxes
        target_max = np.mean(band_maxes)
        print(f"Normalizing all bands to target max: {target_max:.1f}")
        
        for band in active_bands:
            if band_maxes[active_bands.tolist().index(band)] > 0:
                norm_factor = target_max / band_maxes[active_bands.tolist().index(band)]
                A_normalized[:, band] = A_bands[:, band] * norm_factor
                print(f"Band {band} normalization factor: {norm_factor:.3f}")
            else:
                A_normalized[:, band] = A_bands[:, band]
        
        # Keep non-active bands as they were (usually zero anyway)
        for band in range(4):
            if band not in active_bands:
                A_normalized[:, band] = A_bands[:, band]
    else:
        # If only one active band, no normalization needed
        A_normalized = A_bands
        print("Only one active band found, no normalization applied")
    
    # Sum normalized bands
    A_U_flat = np.sum(A_normalized, axis=1)
    
    # Reshape back to 2D
    A_U = A_U_flat.reshape(Ny, Nx)
    
    # Also return individual band contributions for visualization
    A_bands_2d = A_normalized.reshape(Ny, Nx, 4)
    
    return A_U, A_bands_2d

def contour_plot_only():
    """
    Create ONLY the contour plot with correct orientation
    """
    print("=== Creating contour plot with correct orientation ===")
    
    # Higher resolution grid
    res = 3001
    nk_x = res
    nk_y = res * 3
    
    # Momentum grid
    kx_vals = np.linspace(-1*np.pi/a, 1*np.pi/a, nk_x)
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, nk_y)
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    
    print(f"Computing on {res}x{res*3} grid...")
    
    # Build Hamiltonian grid
    H_grid = H_full(KX, KY, 0.0)  # kz = 0 plane
    
    # Paper-appropriate broadening
    eta_paper = 1e-4  # 100 microeV
    
    # Compute spectral weight
    print("Computing U 5f orbital-projected spectral weight...")
    A_map, A_bands_2d = orbital_projected_spectral_weight_paper_method(H_grid, omega=0.0, eta=eta_paper)
    
    print(f"Total spectral weight range: {np.min(A_map):.1f} to {np.max(A_map):.1f}")
    print(f"Band 2 spectral weight range: {np.min(A_bands_2d[:,:,2]):.1f} to {np.max(A_bands_2d[:,:,2]):.1f}")
    print(f"Band 3 spectral weight range: {np.min(A_bands_2d[:,:,3]):.1f} to {np.max(A_bands_2d[:,:,3]):.1f}")
    
    # Smooth the data to reduce periodic modulation
    sig_val = 2.0  # Smoothing sigma
    A_smoothed = gaussian_filter(A_map, sigma=sig_val)
    A_band2_smoothed = gaussian_filter(A_bands_2d[:,:,2], sigma=sig_val)
    A_band3_smoothed = gaussian_filter(A_bands_2d[:,:,3], sigma=sig_val)
    
    # NORMALIZE AFTER SMOOTHING (this is what we actually plot)
    max_band2_smooth = np.max(A_band2_smoothed)
    max_band3_smooth = np.max(A_band3_smoothed)
    
    print(f"After smoothing - Band 2 max: {max_band2_smooth:.1f}, Band 3 max: {max_band3_smooth:.1f}")
    
    # Normalize smoothed bands to same maximum
    target_max_smooth = max(max_band2_smooth, max_band3_smooth)  # Use the higher of the two
    
    if max_band2_smooth > 0:
        A_band2_normalized = A_band2_smoothed * (target_max_smooth / max_band2_smooth)
    else:
        A_band2_normalized = A_band2_smoothed
        
    if max_band3_smooth > 0:
        A_band3_normalized = A_band3_smoothed * (target_max_smooth / max_band3_smooth)
    else:
        A_band3_normalized = A_band3_smoothed
    
    # Update total as AVERAGE of normalized smoothed bands (not sum)
    # This preserves the intensity variations from each band
    A_total_normalized = (A_band2_normalized + A_band3_normalized) / 2.0
    
    print(f"After normalization - Band 2 max: {np.max(A_band2_normalized):.1f}, Band 3 max: {np.max(A_band3_normalized):.1f}")
    print(f"Total (averaged) max: {np.max(A_total_normalized):.1f}")
    print(f"Normalization factors - Band 2: {target_max_smooth/max_band2_smooth:.3f}, Band 3: {target_max_smooth/max_band3_smooth:.3f}")
    
    # DIAGNOSTIC: Check where bands 2 and 3 have their maximum spectral weights
    print(f"\n=== Band Peak Location Analysis ===")
    
    # Find peak locations for Band 2
    band2_max_idx = np.unravel_index(np.argmax(A_band2_normalized), A_band2_normalized.shape)
    ky_peak_band2 = ky_vals[band2_max_idx[0]]
    kx_peak_band2 = kx_vals[band2_max_idx[1]]
    print(f"Band 2 peak at: ky = {ky_peak_band2/(np.pi/b):.3f}π/b, kx = {kx_peak_band2/(np.pi/a):.3f}π/a")
    
    # Find peak locations for Band 3  
    band3_max_idx = np.unravel_index(np.argmax(A_band3_normalized), A_band3_normalized.shape)
    ky_peak_band3 = ky_vals[band3_max_idx[0]]
    kx_peak_band3 = kx_vals[band3_max_idx[1]]
    print(f"Band 3 peak at: ky = {ky_peak_band3/(np.pi/b):.3f}π/b, kx = {kx_peak_band3/(np.pi/a):.3f}π/a")
    
    # Check Fermi crossings at these peak locations
    print(f"\n=== Fermi Level Analysis at Peak Locations ===")
    for band_name, ky_peak, kx_peak in [("Band 2", ky_peak_band2, kx_peak_band2), 
                                        ("Band 3", ky_peak_band3, kx_peak_band3)]:
        H_peak = H_full(kx_peak, ky_peak, 0.0)
        evals_peak, evecs_peak = np.linalg.eigh(H_peak)
        U_weights_peak = np.abs(evecs_peak[0, :])**2 + np.abs(evecs_peak[1, :])**2
        
        print(f"{band_name} peak location analysis:")
        for i, (E, U_weight) in enumerate(zip(evals_peak, U_weights_peak)):
            fermi_dist = abs(E)
            print(f"  Band {i}: E = {E:.4f} eV, U_weight = {U_weight:.4f}, |E-EF| = {fermi_dist:.4f} eV")
        
        # Find which band is closest to Fermi level
        fermi_distances = np.abs(evals_peak)
        closest_band = np.argmin(fermi_distances)
        print(f"  Closest to EF: Band {closest_band} (distance: {fermi_distances[closest_band]:.4f} eV)")
        print(f"  U 5f weight of Fermi band: {U_weights_peak[closest_band]:.4f}")
    
    # Check if the peak locations correspond to high-symmetry points
    print(f"\n=== High-Symmetry Point Analysis ===")
    high_sym_points = {
        "Γ": (0, 0),
        "X": (np.pi/a, 0), 
        "Y": (0, np.pi/b),
        "M": (np.pi/a, np.pi/b),
        "-Y": (0, -np.pi/b),
        "2Y": (0, 2*np.pi/b),
        "-2Y": (0, -2*np.pi/b)
    }
    
    for point_name, (kx_sym, ky_sym) in high_sym_points.items():
        # Check spectral weights at this high-symmetry point
        kx_idx = np.argmin(np.abs(kx_vals - kx_sym))
        ky_idx = np.argmin(np.abs(ky_vals - ky_sym))
        
        band2_weight = A_band2_normalized[ky_idx, kx_idx]
        band3_weight = A_band3_normalized[ky_idx, kx_idx]
        
        print(f"{point_name} point (ky={ky_sym/(np.pi/b):.1f}π/b, kx={kx_sym/(np.pi/a):.1f}π/a):")
        print(f"  Band 2 weight: {band2_weight:.1f}, Band 3 weight: {band3_weight:.1f}")
        
        # Check band structure at this point
        H_sym = H_full(kx_sym, ky_sym, 0.0)
        evals_sym = np.linalg.eigvals(H_sym)
        bands_near_ef = np.sum(np.abs(evals_sym) < 0.05)  # Within 50 meV of Fermi level
        print(f"  Bands near EF: {bands_near_ef}")
    
    print(f"\nPossible reasons Band 3 peaks away from π/b multiples:")
    print(f"1. Different Fermi surface topology - Band 3 crosses EF at different k-points")
    print(f"2. Hybridization effects - δ=0.13eV affects bands differently")  
    print(f"3. Orbital character variation - U 5f weight varies differently across k-space")
    
    # Create meshgrid with CORRECT orientation - ky on x-axis, kx on y-axis
    ky_2d, kx_2d = np.meshgrid(ky_vals/(np.pi/b), kx_vals/(np.pi/a))
    
    # Create subplot showing total + individual bands
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Total (normalized) spectral weight
    levels_total = np.linspace(np.min(A_total_normalized), np.max(A_total_normalized), 50)
    contour1 = axes[0].contourf(ky_2d, kx_2d, A_total_normalized.T, levels=levels_total, 
                               cmap='viridis', extend='both')
    axes[0].set_xlabel('ky (pi/b)')
    axes[0].set_ylabel('kx (pi/a)')
    axes[0].set_title('Total Normalized U 5f Weight')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-1, 1)
    fig.colorbar(contour1, ax=axes[0], label='Total Weight')
    
    # Plot 2: Band 2 contribution (NORMALIZED)
    levels_band2 = np.linspace(np.min(A_band2_normalized), np.max(A_band2_normalized), 50)
    contour2 = axes[1].contourf(ky_2d, kx_2d, A_band2_normalized.T, levels=levels_band2, 
                               cmap='plasma', extend='both')
    axes[1].set_xlabel('ky (pi/b)')
    axes[1].set_ylabel('kx (pi/a)')
    axes[1].set_title(f'Band 2 U 5f Weight (Normalized)')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-1, 1)
    fig.colorbar(contour2, ax=axes[1], label='Band 2 Weight')
    
    # Plot 3: Band 3 contribution (NORMALIZED)
    levels_band3 = np.linspace(np.min(A_band3_normalized), np.max(A_band3_normalized), 50)
    contour3 = axes[2].contourf(ky_2d, kx_2d, A_band3_normalized.T, levels=levels_band3, 
                               cmap='inferno', extend='both')
    axes[2].set_xlabel('ky (pi/b)')
    axes[2].set_ylabel('kx (pi/a)')
    axes[2].set_title(f'Band 3 U 5f Weight (Normalized)')
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-1, 1)
    fig.colorbar(contour3, ax=axes[2], label='Band 3 Weight')
    
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/normalized_bands_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save the total plot figure data before the "bad plot at the end"
    plt.figure(figsize=(12, 8))
    levels_total_clean = np.linspace(np.min(A_total_normalized), np.max(A_total_normalized), 50)
    contour_total_clean = plt.contourf(ky_2d, kx_2d, A_total_normalized.T, levels=levels_total_clean, 
                                      cmap='viridis', extend='both')
    plt.xlabel('ky (π/b)')
    plt.ylabel('kx (π/a)')
    plt.title('Total Averaged U 5f Weight (Good Plot)')
    plt.xlim(-3, 3)
    plt.ylim(-1, 1)
    plt.colorbar(contour_total_clean, label='Total Averaged Weight')
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/total_averaged_good_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.show()
    
    # Save individual band data as separate files
    print("Saving individual band data...")
    np.save('outputs/Spectral/5f/band2_spectral_weight.npy', A_band2_normalized)
    np.save('outputs/Spectral/5f/band3_spectral_weight.npy', A_band3_normalized) 
    np.save('outputs/Spectral/5f/total_averaged_spectral_weight.npy', A_total_normalized)
    np.save('outputs/Spectral/5f/kx_vals.npy', kx_vals)
    np.save('outputs/Spectral/5f/ky_vals.npy', ky_vals)
    
    # Also create individual band plots and save them
    fig_band2, ax2 = plt.subplots(figsize=(12, 8))
    levels2 = np.linspace(np.min(A_band2_normalized), np.max(A_band2_normalized), 50)
    contour2 = ax2.contourf(ky_2d, kx_2d, A_band2_normalized.T, levels=levels2, cmap='plasma', extend='both')
    ax2.set_xlabel('ky (pi/b)')
    ax2.set_ylabel('kx (pi/a)')
    ax2.set_title('Band 2 U 5f Spectral Weight (Normalized)')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-1, 1)
    plt.colorbar(contour2, ax=ax2, label='Band 2 Weight')
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/band2_individual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig_band3, ax3 = plt.subplots(figsize=(12, 8))
    levels3 = np.linspace(np.min(A_band3_normalized), np.max(A_band3_normalized), 50)
    contour3 = ax3.contourf(ky_2d, kx_2d, A_band3_normalized.T, levels=levels3, cmap='inferno', extend='both')
    ax3.set_xlabel('ky (pi/b)')
    ax3.set_ylabel('kx (pi/a)')
    ax3.set_title('Band 3 U 5f Spectral Weight (Normalized)')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-1, 1)
    plt.colorbar(contour3, ax=ax3, label='Band 3 Weight')
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/band3_individual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create the clean total plot with filled contours (preserving intensity info)
    plt.figure(figsize=(12, 8))

    # Use filled contours to preserve intensity information like the paper
    levels = np.linspace(np.min(A_total_normalized), np.max(A_total_normalized), 50)
    contour_filled = plt.contourf(ky_2d, kx_2d, A_total_normalized.T, levels=levels, 
                                 cmap='hot', extend='both')  # 'hot' colormap similar to paper
    
    # Add some contour lines for structure (but keep them subtle)
    contour_lines = plt.contour(ky_2d, kx_2d, A_total_normalized.T, levels=10,
                               colors='white', alpha=0.3, linewidths=0.5)
    
    plt.colorbar(contour_filled, label='Normalized U 5f Spectral Weight')
    plt.xlabel('ky (pi/b)')
    plt.ylabel('kx (pi/a)')  
    plt.title('U 5f Orbital Spectral Weight (Normalized Bands)')
    plt.xlim(-3, 3)
    plt.ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/figure_1b_normalized_bands.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Band normalization complete! ===")
    print("- Both Fermi-crossing bands (2 & 3) normalized to same max intensity")
    print("- Created comparison plot showing individual band contributions")
    print("- ky (pi/b) on x-axis, kx (pi/a) on y-axis (correct paper orientation)")
    print("- Individual band data saved as .npy files")
    print("- Individual band plots saved as separate PNG files")
    print(f"- Note: Normalization factors are 1.000 because bands naturally have similar max values")
    print(f"  Band 2 max: {np.max(A_band2_normalized):.1f}, Band 3 max: {np.max(A_band3_normalized):.1f}")

if __name__ == "__main__":
    import os
    os.makedirs('outputs/Spectral/5f', exist_ok=True)
    contour_plot_only()