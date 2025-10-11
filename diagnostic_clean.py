"""
Diagnostic script to check LDOS patterns and FFT without any smoothing.
"""

import numpy as np
import matplotlib.pyplot as plt
from qpi_G_OOP import SystemParameters, QPISimulation

def test_clean_qpi():
    """Test QPI with absolutely no smoothing to check for circular patterns."""
    
    # Create clean system
    params = SystemParameters(
        gridsize=256,
        L=30.0,
        t=0.3,
        mu=0.0,
        eta=0.1,
        V_s=2.0,
        E_min=5.0,
        E_max=15.0,
        n_frames=1,
        rotation_angle=0.0,
        disorder_strength=0.0,
        zoom_factor=1.0
    )
    
    # Single impurity at exact center
    impurity_positions = [(params.gridsize//2, params.gridsize//2)]
    
    # Create simulation
    simulation = QPISimulation(params, impurity_positions)
    
    # Test at a specific energy
    energy = 8.0
    print(f"Testing at energy E = {energy}")
    
    # Calculate LDOS and FFT
    LDOS, fft_display, peak_q = simulation.run_single_energy(energy)
    
    # Create diagnostic plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # LDOS plot
    im1 = axes[0].imshow(LDOS, cmap='RdBu_r', origin='lower',
                        extent=[0, params.L, 0, params.L])
    axes[0].set_title(f'Raw LDOS (E={energy})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Mark impurity position
    center_x = params.L/2
    center_y = params.L/2
    axes[0].plot(center_x, center_y, 'ko', markersize=8, markerfacecolor='white')
    
    # FFT plot (raw)
    k_max = np.pi * params.gridsize / params.L
    im2 = axes[1].imshow(fft_display, cmap='hot', origin='lower',
                        extent=[-k_max, k_max, -k_max, k_max])
    axes[1].set_title('Raw FFT (no processing)')
    axes[1].set_xlabel('kx')
    axes[1].set_ylabel('ky')
    plt.colorbar(im2, ax=axes[1])
    
    # Radial profile of LDOS to check circularity
    center_idx = params.gridsize // 2
    y, x = np.ogrid[:params.gridsize, :params.gridsize]
    r = np.sqrt((x - center_idx)**2 + (y - center_idx)**2)
    
    # Calculate radial average of LDOS
    max_r = int(params.gridsize // 2)
    r_bins = np.arange(0, max_r)
    radial_profile = []
    
    for r_val in r_bins:
        mask = (r >= r_val) & (r < r_val + 1)
        if np.any(mask):
            radial_profile.append(np.mean(LDOS[mask]))
        else:
            radial_profile.append(0)
    
    axes[2].plot(r_bins * params.L / params.gridsize, radial_profile, 'b-', linewidth=2)
    axes[2].set_xlabel('Distance from impurity')
    axes[2].set_ylabel('Radially averaged LDOS')
    axes[2].set_title('LDOS Radial Profile (should show oscillations)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/diagnostic_clean_qpi.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"LDOS: min={np.min(LDOS):.6f}, max={np.max(LDOS):.6f}")
    print(f"FFT: min={np.min(fft_display):.6f}, max={np.max(fft_display):.6f}")
    print(f"Peak detected at q = {peak_q:.4f}" if peak_q else "No peak detected")
    
    # Check if LDOS has proper symmetry
    center = params.gridsize // 2
    ldos_center = LDOS[center, center]
    
    # Check values at equal distances in different directions
    offset = 20
    ldos_right = LDOS[center, center + offset]
    ldos_left = LDOS[center, center - offset]
    ldos_up = LDOS[center + offset, center]
    ldos_down = LDOS[center - offset, center]
    
    print(f"\nSymmetry check (should be similar):")
    print(f"Center: {ldos_center:.6f}")
    print(f"Right:  {ldos_right:.6f}")
    print(f"Left:   {ldos_left:.6f}")
    print(f"Up:     {ldos_up:.6f}")
    print(f"Down:   {ldos_down:.6f}")
    
    symmetry_error = np.std([ldos_right, ldos_left, ldos_up, ldos_down])
    print(f"Symmetry error (std): {symmetry_error:.6f}")

if __name__ == "__main__":
    test_clean_qpi()