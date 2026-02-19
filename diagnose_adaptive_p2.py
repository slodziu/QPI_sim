#!/usr/bin/env python3
"""
Diagnostic tool to understand why p2 vector finds no k-points in adaptive code
"""

import numpy as np
import matplotlib.pyplot as plt
from UTe2_fixed import H_full, a, b, c, set_parameters
from HAEMUTe2_adaptive import fold_to_first_bz, find_fermi_surface_regions, create_dense_fs_grid

set_parameters('odd_parity_paper')

# Test parameters
kz = 0
nk_coarse = 100
refinement = 3
threshold = 0.001

# Q-vectors to test
vectors_dict = {
    'p2': (0.374, 1.000),   # Problematic
    'p5': (-0.244, 1.000),  # Works
    'p6': (0.619, 0.000)    # Works
}

print("="*70)
print("DIAGNOSTIC: Why does p2 find no k-points?")
print("="*70)

# Step 1: Find FS regions
print("\n1️⃣  Finding Fermi surface regions...")
fs_points_coarse = find_fermi_surface_regions(kz, nk_coarse, threshold)
fs_grid, fs_weights = create_dense_fs_grid(fs_points_coarse, refinement, kz, threshold)

print(f"\nFS Statistics:")
print(f"   Total FS points: {len(fs_grid)}")
print(f"   Weight range: [{fs_weights.min():.3f}, {fs_weights.max():.3f}]")
print(f"   Mean weight: {fs_weights.mean():.3f}")

# Step 2: For each q-vector, analyze k+q overlap
for vector_label, (qx_2pi, qy_2pi) in vectors_dict.items():
    print(f"\n{'='*70}")
    print(f"2️⃣  Testing vector {vector_label}: ({qx_2pi:.3f}, {qy_2pi:.3f}) × 2π/(a,b)")
    print(f"{'='*70}")
    
    qx = qx_2pi * 2*np.pi / a
    qy = qy_2pi * 2*np.pi / b
    
    print(f"   q in cartesian: ({qx:.4f}, {qy:.4f})")
    print(f"   BZ boundaries: x ∈ [{-np.pi/a:.4f}, {np.pi/a:.4f}], y ∈ [{-np.pi/b:.4f}, {np.pi/b:.4f}]")
    
    # Check a sample of FS points
    n_sample = min(100, len(fs_grid))
    sample_indices = np.random.choice(len(fs_grid), n_sample, replace=False)
    
    kpq_on_fs_count = 0
    kpq_weights_list = []
    needs_folding_count = 0
    
    for idx in sample_indices:
        kx, ky = fs_grid[idx]
        weight_k = fs_weights[idx]
        
        # Check k+q
        kx_pq_raw = kx + qx
        ky_pq_raw = ky + qy
        
        # Check if folding is needed
        if (abs(kx_pq_raw) > np.pi/a or abs(ky_pq_raw) > np.pi/b):
            needs_folding_count += 1
        
        # Fold
        kx_pq, ky_pq, kz_pq = fold_to_first_bz(kx_pq_raw, ky_pq_raw, kz)
        
        # Check if k+q is on FS
        H_kpq = H_full(kx_pq, ky_pq, kz_pq)
        eigenvals_kpq = np.linalg.eigvalsh(H_kpq)
        weight_kpq = np.exp(-np.min(np.abs(eigenvals_kpq))**2 / threshold**2)
        
        kpq_weights_list.append(weight_kpq)
        
        if weight_kpq > 0.01:
            kpq_on_fs_count += 1
    
    kpq_weights_array = np.array(kpq_weights_list)
    
    print(f"\n   Sample Results (n={n_sample}):")
    print(f"   ├─ k+q needs folding: {needs_folding_count}/{n_sample} ({100*needs_folding_count/n_sample:.1f}%)")
    print(f"   ├─ k+q on FS (weight > 0.01): {kpq_on_fs_count}/{n_sample} ({100*kpq_on_fs_count/n_sample:.1f}%)")
    print(f"   └─ k+q weight stats: min={kpq_weights_array.min():.6f}, max={kpq_weights_array.max():.6f}, mean={kpq_weights_array.mean():.6f}")
    
    # Check actual function behavior
    print(f"\n   🔍 Running find_relevant_k_points_for_q()...")
    
    # Use looser threshold for k+q check (like the fix)
    threshold_kpq = threshold * 10
    
    relevant_k = []
    relevant_weights = []
    kpq_weights = []
    
    for i, ((kx, ky), weight_k) in enumerate(zip(fs_grid, fs_weights)):
        if weight_k < 0.01:
            continue
            
        # Check if k+q is also near FS
        kx_pq, ky_pq, kz_pq = fold_to_first_bz(kx + qx, ky + qy, kz)
        
        H_kpq = H_full(kx_pq, ky_pq, kz_pq)
        eigenvals_kpq = np.linalg.eigvalsh(H_kpq)
        weight_kpq = np.exp(-np.min(np.abs(eigenvals_kpq))**2 / threshold_kpq**2)
        
        if weight_kpq > 0.01:
            relevant_k.append((kx, ky))
            relevant_weights.append(weight_k)
            kpq_weights.append(weight_kpq)
    
    print(f"   ✓ Found {len(relevant_k)} relevant k-points (with threshold_kpq={threshold_kpq:.3f})")
    
    if len(relevant_k) == 0:
        print(f"\n   ⚠️  PROBLEM: No k-points found!")
        print(f"   Let's check with looser threshold...")
        
        # Try with looser threshold
        for test_threshold in [0.005, 0.01, 0.05, 0.1]:
            count = 0
            for kx, ky in fs_grid[:500]:  # Check first 500 points
                kx_pq, ky_pq, kz_pq = fold_to_first_bz(kx + qx, ky + qy, kz)
                H_kpq = H_full(kx_pq, ky_pq, kz_pq)
                eigenvals_kpq = np.linalg.eigvalsh(H_kpq)
                weight_kpq = np.exp(-np.min(np.abs(eigenvals_kpq))**2 / test_threshold**2)
                if weight_kpq > 0.01:
                    count += 1
            print(f"      threshold={test_threshold:.3f}: {count}/500 points have k+q on FS")

# Step 3: Visualize FS and FS+q for each vector
print(f"\n{'='*70}")
print("3️⃣  Creating visualization...")
print(f"{'='*70}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (vector_label, (qx_2pi, qy_2pi)) in enumerate(vectors_dict.items()):
    ax = axes[idx]
    
    qx = qx_2pi * 2*np.pi / a
    qy = qy_2pi * 2*np.pi / b
    
    # Plot original FS
    ax.scatter(fs_grid[:, 0], fs_grid[:, 1], c=fs_weights, s=1, alpha=0.5, 
               cmap='viridis', label='FS (k)', vmin=0, vmax=1)
    
    # Plot FS shifted by q (after folding)
    fs_shifted = []
    for kx, ky in fs_grid:
        kx_pq, ky_pq, _ = fold_to_first_bz(kx + qx, ky + qy, kz)
        fs_shifted.append([kx_pq, ky_pq])
    fs_shifted = np.array(fs_shifted)
    
    ax.scatter(fs_shifted[:, 0], fs_shifted[:, 1], c='red', s=1, alpha=0.3, label='FS+q (folded)')
    
    # BZ boundary
    ax.axvline(-np.pi/a, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(np.pi/a, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-np.pi/b, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(np.pi/b, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_title(f'{vector_label}: q=({qx_2pi:.3f}, {qy_2pi:.3f})×2π/(a,b)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/haem_ute2/diagnostic_fs_overlap.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: outputs/haem_ute2/diagnostic_fs_overlap.png")

print("\n" + "="*70)
print("Diagnosis complete!")
print("="*70)

plt.show()
