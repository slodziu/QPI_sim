#!/usr/bin/env python3
"""
Check what energies and expected radii correspond to each frame in the animations.
This helps diagnose why rings appear at different radii.
"""

import numpy as np
from config import RANDOM_10_IMPURITIES, RANDOM_30_IMPURITIES, FAST_PREVIEW

def analyze_config(config, name):
    """Analyze energy vs frame for a configuration."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Grid: {config.gridsize}x{config.gridsize}")
    print(f"Energy range: E_min={config.E_min}, E_max={config.E_max}")
    print(f"Number of frames: {config.n_frames}")
    print(f"\nFrame-by-frame breakdown:")
    print(f"{'Frame':<8} {'Energy':<12} {'k_F':<12} {'2k_F':<12}")
    print("-" * 50)
    
    for frame in range(config.n_frames):
        E = config.E_min + (config.E_max - config.E_min) * frame / (config.n_frames - 1)
        k_F = np.sqrt(E)
        two_k_F = 2 * k_F
        print(f"{frame:<8} {E:<12.2f} {k_F:<12.2f} {two_k_F:<12.2f}")
    
    # Also show mid-energy (snapshot)
    mid_frame = config.n_frames // 2
    E_mid = config.E_min + (config.E_max - config.E_min) * mid_frame / (config.n_frames - 1)
    k_F_mid = np.sqrt(E_mid)
    two_k_F_mid = 2 * k_F_mid
    print(f"\n*** MID-ENERGY SNAPSHOT (frame {mid_frame}) ***")
    print(f"E = {E_mid:.2f}, k_F = {k_F_mid:.2f}, expected 2k_F = {two_k_F_mid:.2f}")

if __name__ == "__main__":
    analyze_config(RANDOM_10_IMPURITIES, "10 IMPURITIES")
    analyze_config(RANDOM_30_IMPURITIES, "30 IMPURITIES")
    analyze_config(FAST_PREVIEW, "FAST PREVIEW")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The ring radius in momentum space should ALWAYS be at 2k_F = 2*sqrt(E).

If you see different radii at the SAME energy, that's a bug.
If you see different radii at DIFFERENT energies, that's expected physics!

Check which frame you're looking at in each animation - they may be
at different energies even if they look similar.
""")
