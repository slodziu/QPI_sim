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
    
    # Compute spectral weight for each band separately
    A_bands = np.zeros((M, 4))  # Spectral weight for each band at each k-point
    
    for band in range(4):
        # Spectral weight for this band only
        band_spectral = U_hybridized_weights[:, band] * lorentzian[:, band] * fermi_mask[:, band]
        A_bands[:, band] = band_spectral
    
    # Reshape back to 2D
    A_bands_2d = A_bands.reshape(Ny, Nx, 4)
    
    return A_bands_2d

def split_ky_analysis():
    """
    Split ky into 7 smaller regions: [-3π/b, -2.5π/b], [-2.5π/b, -1.5π/b], etc.
    Calculate bands separately, normalize each region, then stitch together for better continuity
    """
    print("=== Split ky Analysis ===")
    
    # Higher resolution grid
    res = 3001  # Lower res for faster computation
    nk_x = res
    
    # Define smaller overlapping ky regions for better continuity
    ky_regions = [
        (-3*np.pi/b, -2.5*np.pi/b, "region1"),    # Region 1: -3π/b to -2.5π/b
        (-2.5*np.pi/b, -1.5*np.pi/b, "region2"),  # Region 2: -2.5π/b to -1.5π/b
        (-1.5*np.pi/b, -0.5*np.pi/b, "region3"),  # Region 3: -1.5π/b to -0.5π/b
        (-0.5*np.pi/b, 0.5*np.pi/b, "region4"),   # Region 4: -0.5π/b to 0.5π/b (center)
        (0.5*np.pi/b, 1.5*np.pi/b, "region5"),    # Region 5: 0.5π/b to 1.5π/b
        (1.5*np.pi/b, 2.5*np.pi/b, "region6"),    # Region 6: 1.5π/b to 2.5π/b
        (2.5*np.pi/b, 3*np.pi/b, "region7")       # Region 7: 2.5π/b to 3π/b
    ]
    
    # Paper-appropriate broadening
    eta_paper = 1e-4  # 100 microeV
    
    # Storage for stitched results
    all_ky_vals = []
    all_A_band2 = []
    all_A_band3 = []
    
    # Process each region separately
    for ky_min, ky_max, region_name in ky_regions:
        print(f"\nProcessing {region_name}: ky ∈ [{ky_min/(np.pi/b):.1f}π/b, {ky_max/(np.pi/b):.1f}π/b]")
        
        # Create grid for this region
        nk_y_region = res
        kx_vals = np.linspace(-1*np.pi/a, 1*np.pi/a, nk_x)
        ky_vals_region = np.linspace(ky_min, ky_max, nk_y_region)
        KX, KY = np.meshgrid(kx_vals, ky_vals_region)
        
        # Build Hamiltonian grid for this region
        H_grid = H_full(KX, KY, 0.0)  # kz = 0 plane
        
        # Compute spectral weight for this region
        A_bands_2d = orbital_projected_spectral_weight_paper_method(H_grid, omega=0.0, eta=eta_paper)
        
        # Extract bands 2 and 3
        A_band2_region = A_bands_2d[:, :, 2]
        A_band3_region = A_bands_2d[:, :, 3]
        
        # Apply smoothing
        A_band2_smooth = gaussian_filter(A_band2_region, sigma=2.0)
        A_band3_smooth = gaussian_filter(A_band3_region, sigma=2.0)
        
        # Normalize within this region (regular normalization only)
        max_band2 = np.max(A_band2_smooth)
        max_band3 = np.max(A_band3_smooth)
        
        if max_band2 > 0 and max_band3 > 0:
            # Normalize to the maximum of the two bands in this region
            target_max = max(max_band2, max_band3)
            A_band2_norm = A_band2_smooth * (target_max / max_band2)
            A_band3_norm = A_band3_smooth * (target_max / max_band3)
            
            print(f"  Band 2 max: {max_band2:.1f} -> {np.max(A_band2_norm):.1f}")
            print(f"  Band 3 max: {max_band3:.1f} -> {np.max(A_band3_norm):.1f}")
        else:
            A_band2_norm = A_band2_smooth
            A_band3_norm = A_band3_smooth
            print(f"  No significant spectral weight in this region")
        
        # Store for stitching (normalized data only)
        all_ky_vals.append(ky_vals_region)
        all_A_band2.append(A_band2_norm)
        all_A_band3.append(A_band3_norm)
    
    # Stitch the regions together
    print("\nStitching regions together...")
    ky_full = np.concatenate(all_ky_vals)
    A_band2_full = np.concatenate(all_A_band2, axis=0)
    A_band3_full = np.concatenate(all_A_band3, axis=0)
    
    # Create average of normalized bands
    A_total_full = (A_band2_full + A_band3_full) / 2.0
    
    # Apply cosine modulation AFTER stitching to avoid discontinuities
    print("Applying cosine modulation to stitched data...")
    cosine_modulation_full = 0.2 + 0.8 * np.abs(np.cos(ky_full * b))  # Range: [0.2, 1.0]
    cosine_mod_2d_full = cosine_modulation_full[:, np.newaxis]
    
    # Apply cosine to the complete stitched data
    A_band2_cos_full = A_band2_full * cosine_mod_2d_full
    A_band3_cos_full = A_band3_full * cosine_mod_2d_full  
    A_total_cos_full = A_total_full * cosine_mod_2d_full
    
    # Create meshgrid for plotting
    ky_2d, kx_2d = np.meshgrid(ky_full/(np.pi/b), kx_vals/(np.pi/a))
    
    # Create the stitched plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Total stitched
    levels_total = np.linspace(np.min(A_total_full), np.max(A_total_full), 50)
    contour1 = axes[0].contourf(ky_2d, kx_2d, A_total_full.T, levels=levels_total, 
                               cmap='viridis', extend='both')
    axes[0].set_xlabel('ky (π/b)')
    axes[0].set_ylabel('kx (π/a)')
    axes[0].set_title('Total Stitched (Region-Normalized)')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-1, 1)
    
    # Add region boundaries for all 7 regions
    region_boundaries = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    for boundary in region_boundaries:
        axes[0].axvline(boundary, color='white', linestyle='--', alpha=0.5, linewidth=1)
    fig.colorbar(contour1, ax=axes[0], label='Stitched Weight')
    
    # Plot 2: Band 2 stitched
    levels_band2 = np.linspace(np.min(A_band2_full), np.max(A_band2_full), 50)
    contour2 = axes[1].contourf(ky_2d, kx_2d, A_band2_full.T, levels=levels_band2, 
                               cmap='plasma', extend='both')
    axes[1].set_xlabel('ky (π/b)')
    axes[1].set_ylabel('kx (π/a)')
    axes[1].set_title('Band 2 Stitched (Region-Normalized)')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-1, 1)
    for boundary in region_boundaries:
        axes[1].axvline(boundary, color='white', linestyle='--', alpha=0.5, linewidth=1)
    fig.colorbar(contour2, ax=axes[1], label='Band 2 Weight')
    
    # Plot 3: Band 3 stitched  
    levels_band3 = np.linspace(np.min(A_band3_full), np.max(A_band3_full), 50)
    contour3 = axes[2].contourf(ky_2d, kx_2d, A_band3_full.T, levels=levels_band3, 
                               cmap='inferno', extend='both')
    axes[2].set_xlabel('ky (π/b)')
    axes[2].set_ylabel('kx (π/a)')
    axes[2].set_title('Band 3 Stitched (Region-Normalized)')
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-1, 1)
    for boundary in region_boundaries:
        axes[2].axvline(boundary, color='white', linestyle='--', alpha=0.5, linewidth=1)
    fig.colorbar(contour3, ax=axes[2], label='Band 3 Weight')
    
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/stitched_ky_regions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save individual plots
    print("\nSaving individual plots...")
    
    # Total plot
    plt.figure(figsize=(12, 8))
    contour_total = plt.contourf(ky_2d, kx_2d, A_total_full.T, levels=levels_total, 
                                cmap='viridis', extend='both')
    plt.xlabel('ky (π/b)')
    plt.ylabel('kx (π/a)')
    plt.title('Total Stitched U 5f Weight (Region-Normalized)')
    plt.xlim(-3, 3)
    plt.ylim(-1, 1)
    for boundary in region_boundaries:
        plt.axvline(boundary, color='white', linestyle='--', alpha=0.5, linewidth=1)
    plt.colorbar(contour_total, label='Total Weight')
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/total_stitched_individual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Band 2 plot
    plt.figure(figsize=(12, 8))
    contour_b2 = plt.contourf(ky_2d, kx_2d, A_band2_full.T, levels=levels_band2, 
                             cmap='plasma', extend='both')
    plt.xlabel('ky (π/b)')
    plt.ylabel('kx (π/a)')
    plt.title('Band 2 Stitched U 5f Weight (Region-Normalized)')
    plt.xlim(-3, 3)
    plt.ylim(-1, 1)
    for boundary in region_boundaries:
        plt.axvline(boundary, color='white', linestyle='--', alpha=0.5, linewidth=1)
    plt.colorbar(contour_b2, label='Band 2 Weight')
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/band2_stitched_individual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Band 3 plot
    plt.figure(figsize=(12, 8))
    contour_b3 = plt.contourf(ky_2d, kx_2d, A_band3_full.T, levels=levels_band3, 
                             cmap='inferno', extend='both')
    plt.xlabel('ky (π/b)')
    plt.ylabel('kx (π/a)')
    plt.title('Band 3 Stitched U 5f Weight (Region-Normalized)')
    plt.xlim(-3, 3)
    plt.ylim(-1, 1)
    for boundary in region_boundaries:
        plt.axvline(boundary, color='white', linestyle='--', alpha=0.5, linewidth=1)
    plt.colorbar(contour_b3, label='Band 3 Weight')
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/band3_stitched_individual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create cosine-modulated plots (peaks every π/b) - now using post-stitching cosine data
    print("\nCreating cosine-modulated plots (peaks every π/b)...")
    print("Using cosine modulation applied AFTER stitching to avoid discontinuities...")
    
    # Cosine-modulated total plot
    plt.figure(figsize=(12, 8))
    levels_total_cos = np.linspace(np.min(A_total_cos_full), np.max(A_total_cos_full), 50)
    contour_total_cos = plt.contourf(ky_2d, kx_2d, A_total_cos_full.T, levels=levels_total_cos, 
                                    cmap='viridis', extend='both')
    plt.xlabel('ky (π/b)')
    plt.ylabel('kx (π/a)')
    plt.title('Total Stitched × |cos(ky)| - Cosine Applied After Stitching')
    plt.xlim(-3, 3)
    plt.ylim(-1, 1)
    # Mark π/b multiples where cosine peaks
    for multiple in [-3, -2, -1, 0, 1, 2, 3]:
        plt.axvline(multiple, color='yellow', linestyle=':', alpha=0.8, linewidth=1)
    plt.colorbar(contour_total_cos, label='Post-Stitch Cosine Weight')
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/total_stitched_cosine_first.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Cosine-modulated Band 2 plot
    plt.figure(figsize=(12, 8))
    levels_band2_cos = np.linspace(np.min(A_band2_cos_full), np.max(A_band2_cos_full), 50)
    contour_b2_cos = plt.contourf(ky_2d, kx_2d, A_band2_cos_full.T, levels=levels_band2_cos, 
                                 cmap='plasma', extend='both')
    plt.xlabel('ky (π/b)')
    plt.ylabel('kx (π/a)')
    plt.title('Band 2 Stitched × |cos(ky)| - Cosine Applied After Stitching')
    plt.xlim(-3, 3)
    plt.ylim(-1, 1)
    for multiple in [-3, -2, -1, 0, 1, 2, 3]:
        plt.axvline(multiple, color='yellow', linestyle=':', alpha=0.8, linewidth=1)
    plt.colorbar(contour_b2_cos, label='Post-Stitch Band 2')
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/band2_stitched_cosine_first.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Cosine-modulated Band 3 plot
    plt.figure(figsize=(12, 8))
    levels_band3_cos = np.linspace(np.min(A_band3_cos_full), np.max(A_band3_cos_full), 50)
    contour_b3_cos = plt.contourf(ky_2d, kx_2d, A_band3_cos_full.T, levels=levels_band3_cos, 
                                 cmap='inferno', extend='both')
    plt.xlabel('ky (π/b)')
    plt.ylabel('kx (π/a)')
    plt.title('Band 3 Stitched × |cos(ky)| - Cosine Applied After Stitching')
    plt.xlim(-3, 3)
    plt.ylim(-1, 1)
    for multiple in [-3, -2, -1, 0, 1, 2, 3]:
        plt.axvline(multiple, color='yellow', linestyle=':', alpha=0.8, linewidth=1)
    plt.colorbar(contour_b3_cos, label='Post-Stitch Band 3')
    plt.tight_layout()
    plt.savefig('outputs/Spectral/5f/band3_stitched_cosine_first.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Post-stitching cosine modulation summary:")
    print(f"  Regular total max: {np.max(A_total_full):.1f}")
    print(f"  Post-stitch cosine total max: {np.max(A_total_cos_full):.1f}")
    print(f"  Cosine applied AFTER stitching to prevent discontinuities")
    
    print(f"\n=== Stitching Complete ===")
    print(f"- Created 7 smaller regions with independent normalization")
    print(f"- Region boundaries shown as white dashed lines")
    print(f"- Each region normalized to its own maximum")
    print(f"- Total grid: {A_total_full.shape}")
    print(f"- Saved individual plots for total, band 2, and band 3")
    print(f"- Created cosine-modulated versions with cosine applied AFTER stitching")
    print(f"- Yellow dotted lines mark π/b multiples where cosine modulation peaks")
    print(f"- Post-stitch cosine approach maintains continuity across regions")

if __name__ == "__main__":
    import os
    os.makedirs('outputs/Spectral/5f', exist_ok=True)
    split_ky_analysis()