#!/usr/bin/env python3
"""
Same-Energy QPI Comparison Tool

Generate single-frame comparisons at the same energy for different impurity counts.
This helps verify that QPI ring radii are consistent at the same energy across
different impurity configurations, validating the simulation accuracy.
"""

import sys
import numpy as np
import copy
from qpi_G_OOP import SystemParameters, GreensFunction, ImpuritySystem, QPISimulation, QPIVisualizer
from config import RANDOM_10_IMPURITIES, RANDOM_30_IMPURITIES, setup_n_random_positions
import matplotlib.pyplot as plt

# LDOS colormap for comparison plots - can be customized
LDOS_COLORMAP = 'coolwarm'  # Perceptually uniform for comparisons

def run_comparison(E_test=50.0):
    """Run comparison at a specific energy."""
    print(f"\n{'='*70}")
    print(f"COMPARISON AT E = {E_test}")
    print(f"Expected: k_F = {np.sqrt(E_test):.2f}, 2k_F = {2*np.sqrt(E_test):.2f}")
    print(f"{'='*70}\n")
    
    results = {}
    
    for n_imp, base_config in [(10, RANDOM_10_IMPURITIES), (30, RANDOM_30_IMPURITIES)]:
        print(f"--- {n_imp} impurities ---")
        
        # Create a copy and place random impurities
        config = copy.deepcopy(base_config)
        config.gridsize = 512  # Use smaller grid for speed
        setup_n_random_positions(config, n_imp, seed=42, distributed=True)
        
        # Create system parameters
        params = SystemParameters(
            gridsize=config.gridsize,
            L=config.L,
            t=config.t,
            mu=config.mu,
            eta=config.eta,
            V_s=config.V_s,
            E_min=E_test,
            E_max=E_test,
            n_frames=1,
            rotation_angle=config.rotation_angle,
            disorder_strength=config.disorder_strength,
            zoom_factor=config.zoom_factor
        )
        
        # Create simulation with these impurities
        sim = QPISimulation(params, impurity_positions=config.impurity_positions)
        print(f"Placed {len(sim.impurities.positions)} impurities")
        
        # Run at test energy
        LDOS, fft_display, fft_complex, peak_q = sim.run_single_energy(E_test)
        
        # Extract peak from radial profile using correct k-space scaling
        k_F = np.sqrt(E_test)
        dk = 2 * np.pi / params.L  # Correct k-space pixel size
        
        # Calculate radial profile
        center = fft_display.shape[0] // 2
        y, x = np.ogrid[:fft_display.shape[0], :fft_display.shape[1]]
        r_pixel = np.sqrt((x - center)**2 + (y - center)**2)
        r_k = r_pixel * dk  # Use actual dk, not display scaling
        
        # Radial average
        max_r_k = 30  # Fixed max k for analysis
        n_bins = 200
        r_bins = np.linspace(0, max_r_k, n_bins)
        radial_profile = np.zeros(n_bins - 1)
        
        for i in range(n_bins - 1):
            mask = (r_k >= r_bins[i]) & (r_k < r_bins[i+1])
            if np.sum(mask) > 0:
                radial_profile[i] = np.mean(fft_display[mask])
        
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        # Find peak
        exclude_center = int(len(radial_profile) * 0.08)
        peak_idx = np.argmax(radial_profile[exclude_center:]) + exclude_center
        peak_k = r_centers[peak_idx]
        
        print(f"Peak at k = {peak_k:.2f} (expected 2k_F = {2*k_F:.2f})")
        print(f"Ratio: {peak_k / (2*k_F):.3f}")
        
        results[n_imp] = {
            'fft': fft_display,
            'radial': radial_profile,
            'r_centers': r_centers,
            'peak_k': peak_k,
            'LDOS': LDOS,
            'dk': dk
        }
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, n_imp in enumerate([10, 30]):
        data = results[n_imp]
        dk = data['dk']
        
        # LDOS
        ax = axes[i, 0]
        im = ax.imshow(data['LDOS'], origin='lower', cmap=LDOS_COLORMAP)
        ax.set_title(f'{n_imp} impurities: LDOS at E={E_test:.1f}')
        plt.colorbar(im, ax=ax)
        
        # Momentum space
        ax = axes[i, 1]
        fft_log = np.log10(data['fft'] + 1)
        k_F = np.sqrt(E_test)
        # Use actual k-space extent based on FFT size and dk
        k_max = dk * fft_log.shape[0] // 2
        im = ax.imshow(fft_log, origin='lower', cmap='plasma',
                      extent=[-k_max, k_max, -k_max, k_max])
        ax.set_title(f'{n_imp} imp: k-space (peak at k={data["peak_k"]:.2f})')
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        
        # Add circle at expected 2k_F
        circle = plt.Circle((0, 0), 2*k_F, fill=False, edgecolor='red', 
                           linestyle='--', linewidth=2, label=f'2k_F={2*k_F:.2f}')
        ax.add_patch(circle)
        ax.legend()
        plt.colorbar(im, ax=ax)
        
        # Radial profile
        ax = axes[i, 2]
        ax.plot(data['r_centers'], data['radial'], 'b-', linewidth=2)
        ax.axvline(2*k_F, color='red', linestyle='--', linewidth=2, 
                  label=f'Expected 2k_F={2*k_F:.2f}')
        ax.axvline(data['peak_k'], color='green', linestyle=':', linewidth=2,
                  label=f'Measured peak={data["peak_k"]:.2f}')
        ax.set_xlabel('k (momentum)')
        ax.set_ylabel('Radial average intensity')
        ax.set_title(f'{n_imp} imp: Radial profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/comparison_E{E_test:.0f}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to outputs/comparison_E{E_test:.0f}.png")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        E_test = float(sys.argv[1])
    else:
        E_test = 50.0
    
    run_comparison(E_test)
    
    print("\n" + "="*70)
    print("To test at different energies, run:")
    print("  python3 compare_same_energy.py 30.0")
    print("  python3 compare_same_energy.py 50.0")
    print("="*70)
