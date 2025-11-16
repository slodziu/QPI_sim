# Check if B2u gap nodes exist anywhere on or near the Fermi surface
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from UTe2_fixed import calculate_gap_magnitude, H_full

# Lattice constants
a = 0.41
b = 0.61
c = 1.39

def comprehensive_b2u_analysis():
    print("=== Comprehensive B2u Gap Node Analysis ===")
    
    # Create momentum grid
    nk = 200
    kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
    ky_vals = np.linspace(-np.pi/b, np.pi/b, nk)
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    kz = 0.0
    
    # Calculate gap and energies
    b2u_gap = calculate_gap_magnitude(KX, KY, kz, pairing_type='B2u')
    
    energies_fs = np.zeros((nk, nk, 4))
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            eigvals = np.linalg.eigvals(H)
            energies_fs[i, j, :] = np.sort(np.real(eigvals))
    
    print(f"B2u gap range: {np.min(b2u_gap):.6f} to {np.max(b2u_gap):.6f}")
    
    # Find all points where gap is very small
    small_gap_threshold = 0.05
    small_gap_points = np.where(b2u_gap < small_gap_threshold)
    print(f"Points with B2u gap < {small_gap_threshold}: {len(small_gap_points[0])}")
    
    if len(small_gap_points[0]) > 0:
        print("Small gap locations (first 10):")
        for i in range(min(10, len(small_gap_points[0]))):
            kx_idx, ky_idx = small_gap_points[0][i], small_gap_points[1][i]
            kx_val = kx_vals[kx_idx]
            ky_val = ky_vals[ky_idx]
            gap_val = b2u_gap[kx_idx, ky_idx]
            
            # Check energies at this point for bands 3 and 4
            E3 = energies_fs[kx_idx, ky_idx, 2]
            E4 = energies_fs[kx_idx, ky_idx, 3]
            
            print(f"  kx={kx_val/(np.pi/a):6.3f}π/a, ky={ky_val/(np.pi/b):6.3f}π/b, gap={gap_val:.6f}, E3={E3:.6f}, E4={E4:.6f}")
    
    # Check for points near Fermi surface
    print(f"\\n=== Fermi Surface Analysis ===")
    fermi_threshold = 0.02  # 20 meV
    
    for band in [2, 3]:  # Bands 3 and 4
        band_energies = energies_fs[:, :, band]
        fermi_points = np.where(np.abs(band_energies) < fermi_threshold)
        print(f"Band {band+1} Fermi surface points (|E| < {fermi_threshold}): {len(fermi_points[0])}")
        
        if len(fermi_points[0]) > 0:
            # Check gap values at Fermi surface points
            gaps_at_fermi = b2u_gap[fermi_points]
            print(f"  Gap range at Fermi surface: {np.min(gaps_at_fermi):.6f} to {np.max(gaps_at_fermi):.6f}")
            print(f"  Points with gap < 0.1 at Fermi surface: {np.sum(gaps_at_fermi < 0.1)}")
            print(f"  Points with gap < 0.05 at Fermi surface: {np.sum(gaps_at_fermi < 0.05)}")
            
            # Find the closest approach between small gaps and Fermi surface
            min_gap_at_fermi = np.min(gaps_at_fermi)
            min_gap_idx = np.argmin(gaps_at_fermi)
            kx_idx, ky_idx = fermi_points[0][min_gap_idx], fermi_points[1][min_gap_idx]
            
            print(f"  Smallest gap on Fermi surface: {min_gap_at_fermi:.6f}")
            print(f"    Location: kx={kx_vals[kx_idx]/(np.pi/a):.3f}π/a, ky={ky_vals[ky_idx]/(np.pi/b):.3f}π/b")
            print(f"    Energy: {band_energies[kx_idx, ky_idx]:.6f}")
    
    # Summary
    print(f"\\n=== Summary ===")
    print("B2u gap nodes in theory exist at kx = 0, ±π/a (vertical lines)")
    print("However, the Fermi surface may not intersect these lines in our momentum range.")
    print("If the smallest gap on the Fermi surface is >> 0.05, then there are no practical gap nodes.")

if __name__ == "__main__":
    comprehensive_b2u_analysis()