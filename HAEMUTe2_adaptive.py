#!/usr/bin/env python3
"""
HAEM Signal with Adaptive FS-focused k-space sampling

Key optimization: Instead of uniform grid, identify Fermi surface regions
and sample densely only near the FS. Much faster for sparse FS systems like UTe2.

Strategy:
1. Quick low-res scan to find FS regions
2. Sample densely only in FS "patches"
3. For each q-vector, find k-points where both k AND k+q are near FS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os
import time
from UTe2_fixed import H_full, a, b, c, set_parameters

# =============================================================================
# FERMI SURFACE IDENTIFICATION
# =============================================================================

def find_fermi_surface_regions(kz=0, nk_coarse=100, threshold=0.001, weight_cutoff=0.1):
    """
    Identify regions of k-space near the Fermi surface
    
    Returns list of (kx, ky) points near FS for efficient sampling
    """
    print(f"🔍 Identifying Fermi surface regions...")
    
    kx_range = np.linspace(-np.pi/a, np.pi/a, nk_coarse)
    ky_range = np.linspace(-np.pi/b, np.pi/b, nk_coarse)
    
    fs_points = []
    
    for i, kx in enumerate(kx_range):
        if (i + 1) % 25 == 0:
            print(f"   Progress: {i+1}/{nk_coarse}")
        for ky in ky_range:
            H = H_full(kx, ky, kz)
            eigenvals = np.linalg.eigvalsh(H)
            weight = np.exp(-np.min(np.abs(eigenvals))**2 / threshold**2)
            
            if weight > weight_cutoff:
                fs_points.append((kx, ky))
    
    fs_points = np.array(fs_points)
    print(f"   ✓ Found {len(fs_points)} FS regions ({100*len(fs_points)/(nk_coarse**2):.1f}% of BZ)")
    
    return fs_points


def create_dense_fs_grid(fs_points, refinement=5, kz=0, threshold=0.001):
    """
    Create dense sampling grid around identified FS regions
    
    Args:
        fs_points: Coarse FS points from find_fermi_surface_regions
        refinement: How many subpoints per coarse point (refinement^2 total)
        
    Returns:
        fs_grid: Dense k-point array near FS
        fs_weights: Weights at each point
    """
    print(f"📐 Creating dense FS grid (refinement={refinement})...")
    
    dk_x = (2*np.pi/a) / 100  # Grid spacing from coarse scan
    dk_y = (2*np.pi/b) / 100
    
    dense_points = []
    
    for kx_center, ky_center in fs_points:
        # Create local dense grid around this FS point
        kx_local = np.linspace(kx_center - dk_x/2, kx_center + dk_x/2, refinement)
        ky_local = np.linspace(ky_center - dk_y/2, ky_center + dk_y/2, refinement)
        
        for kx in kx_local:
            for ky in ky_local:
                dense_points.append((kx, ky))
    
    dense_points = np.array(dense_points)
    
    # Compute weights
    weights = np.zeros(len(dense_points))
    print(f"   Computing weights for {len(dense_points)} points...")
    
    for i, (kx, ky) in enumerate(dense_points):
        if (i + 1) % 5000 == 0:
            print(f"   Progress: {i+1}/{len(dense_points)}")
        H = H_full(kx, ky, kz)
        eigenvals = np.linalg.eigvalsh(H)
        weights[i] = np.exp(-np.min(np.abs(eigenvals))**2 / threshold**2)
    
    # Filter out points that drifted away from FS
    mask = weights > 0.01
    fs_grid = dense_points[mask]
    fs_weights = weights[mask]
    
    print(f"   ✓ Dense grid: {len(fs_grid)} points near FS")
    print(f"   Equivalent uniform resolution: ~{int(np.sqrt(len(fs_grid) / 0.002))}×{int(np.sqrt(len(fs_grid) / 0.002))}")
    
    return fs_grid, fs_weights


def find_relevant_k_points_for_q(qx, qy, fs_grid, fs_weights, kz=0, threshold=0.001):
    """
    For a given q-vector, find k-points where BOTH k and k+q are near FS
    
    This is the minimal set needed for δN(q,E) calculation
    """
    print(f"   Finding k-points where both k and k+q on FS...")
    
    relevant_k = []
    relevant_weights = []
    kpq_weights = []
    
    for i, ((kx, ky), weight_k) in enumerate(zip(fs_grid, fs_weights)):
        if weight_k < 0.01:
            continue
            
        # Check if k+q is also near FS
        kx_pq = kx + qx
        ky_pq = ky + qy
        
        H_kpq = H_full(kx_pq, ky_pq, kz)
        eigenvals_kpq = np.linalg.eigvalsh(H_kpq)
        weight_kpq = np.exp(-np.min(np.abs(eigenvals_kpq))**2 / threshold**2)
        
        if weight_kpq > 0.01:
            relevant_k.append((kx, ky))
            relevant_weights.append(weight_k)
            kpq_weights.append(weight_kpq)
    
    if len(relevant_k) == 0:
        print(f"   ⚠️  No k-points found where both k and k+q on FS!")
        return None, None, None
    
    print(f"   ✓ Found {len(relevant_k)} relevant k-points")
    
    return np.array(relevant_k), np.array(relevant_weights), np.array(kpq_weights)


# =============================================================================
# OPTIMIZED HAEM CALCULATION
# =============================================================================

def compute_delta_N_adaptive(qx, qy, qz, energy, pairing_type, T_matrix, 
                             fs_grid, fs_weights, eta=1e-5):
    """
    Compute δN using only FS-relevant k-points
    """
    from HAEMUTe2 import construct_BdG_hamiltonian_vectorized, compute_green_function_vectorized
    
    # Find k-points where both k and k+q on FS
    rel_k, rel_weights, kpq_weights = find_relevant_k_points_for_q(qx, qy, fs_grid, fs_weights, qz)
    
    if rel_k is None:
        return 0.0
    
    # Compute Green's functions only at relevant points
    kx_arr = rel_k[:, 0].reshape(-1, 1)
    ky_arr = rel_k[:, 1].reshape(-1, 1)
    
    G_k_array = compute_green_function_vectorized(kx_arr, ky_arr, qz, energy, pairing_type, eta)
    G_kpq_array = compute_green_function_vectorized(kx_arr + qx, ky_arr + qy, qz, energy, pairing_type, eta)
    
    # Compute matrix elements
    n_points = len(rel_k)
    matrix_elements = np.zeros(n_points, dtype=complex)
    
    for i in range(n_points):
        product = G_k_array[i, 0] @ T_matrix @ G_kpq_array[i, 0]
        electron_block = product[:4, :4]
        matrix_elements[i] = electron_block[0, 0]
    
    # Weighted sum
    combined_weights = rel_weights * kpq_weights
    delta_N_sum = np.sum(matrix_elements * combined_weights)
    total_weight = np.sum(combined_weights)
    
    if total_weight > 0:
        delta_N_sum /= total_weight
    
    return np.imag(delta_N_sum) / np.pi


def compute_haem_adaptive(vectors_dict, pairing_types=['B2u', 'B3u'],
                         energy_range=(1e-6, 3e-4), n_energies=20,
                         kz=0, V_imp=0.03, eta=5e-5,
                         nk_coarse=100, refinement=5):
    """
    Adaptive HAEM calculation focusing on FS regions
    """
    from HAEMUTe2 import compute_t_matrix
    
    print("🚀 Adaptive HAEM Calculation")
    print("="*70)
    
    # Step 1: Identify FS regions (do this once)
    fs_points_coarse = find_fermi_surface_regions(kz, nk_coarse)
    fs_grid, fs_weights = create_dense_fs_grid(fs_points_coarse, refinement, kz)
    
    energies = np.linspace(energy_range[0], energy_range[1], n_energies)
    results = {}
    
    for pairing_type in pairing_types:
        print(f"\n📊 Computing {pairing_type}...")
        start_time = time.time()
        
        # Pre-compute T-matrices
        print(f"   Pre-computing T-matrices...")
        T_matrices_pos = []
        T_matrices_neg = []
        
        for i, energy in enumerate(energies):
            if (i + 1) % 5 == 0:
                print(f"      Progress: {i+1}/{n_energies}")
            # T-matrix uses standard approach (could also optimize this)
            T_pos = compute_t_matrix(energy, pairing_type, V_imp, eta, nk=100, kz=kz)
            T_neg = compute_t_matrix(-energy, pairing_type, V_imp, eta, nk=100, kz=kz)
            T_matrices_pos.append(T_pos)
            T_matrices_neg.append(T_neg)
        
        # Compute HAEM for each vector
        vector_results = {}
        
        for vector_label, (qx_2pi, qy_2pi) in vectors_dict.items():
            print(f"\n   Vector {vector_label}: ({qx_2pi:.2f}, {qy_2pi:.2f}) × 2π/(a,b)")
            
            qx = qx_2pi * 2*np.pi / a
            qy = qy_2pi * 2*np.pi / b
            
            haem_values = []
            
            for i, energy in enumerate(energies):
                if abs(energy) < 1e-8:
                    haem_values.append(0.0)
                    continue
                
                delta_N_pos = compute_delta_N_adaptive(qx, qy, kz, energy, pairing_type,
                                                       T_matrices_pos[i], fs_grid, fs_weights, eta)
                delta_N_neg = compute_delta_N_adaptive(qx, qy, kz, -energy, pairing_type,
                                                       T_matrices_neg[i], fs_grid, fs_weights, eta)
                
                haem_values.append(delta_N_pos - delta_N_neg)
                
                if (i + 1) % 5 == 0:
                    print(f"      Energy {i+1}/{n_energies}: ρ⁻ = {haem_values[-1]:.3e}")
            
            vector_results[vector_label] = np.array(haem_values)
        
        results[pairing_type] = {
            'energies': energies * 1e6,
            'vectors': vector_results
        }
        
        elapsed = time.time() - start_time
        print(f"   ✓ Completed in {elapsed:.1f}s")
    
    return results


def main():
    """Test adaptive sampling"""
    set_parameters('odd_parity_paper')
    
    vectors_dict = {
        'p2': (0.43, 1.0),
        'p5': (-0.244, 1.000),
        'p6': (0.619, 0.000)
    }
    
    print("Testing adaptive FS-focused sampling")
    print("="*70)
    
    results = compute_haem_adaptive(
        vectors_dict=vectors_dict,
        pairing_types=['B2u'],  # Test with one first
        energy_range=(1e-6, 3e-4),
        n_energies=10,  # Fewer for testing
        kz=0,
        V_imp=0.03,
        eta=5e-5,
        nk_coarse=100,  # Coarse FS identification
        refinement=3    # Dense sampling factor
    )
    
    # Quick plot
    from HAEMUTe2 import plot_haem_simple_overlay
    plot_haem_simple_overlay(results, vectors_dict, save_dir='outputs/haem_ute2')
    
    print("\n✅ Adaptive calculation complete!")


if __name__ == "__main__":
    main()
