#!/usr/bin/env python3
"""
Visualize the difference between uniform and adaptive sampling strategies

Shows why adaptive sampling is more efficient for sparse Fermi surfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from UTe2_fixed import H_full, a, b, c, set_parameters
import os

def visualize_sampling_strategies(qx_2pi=0.43, qy_2pi=1.0, kz=0):
    """
    Compare uniform vs adaptive sampling for a specific q-vector
    """
    set_parameters('odd_parity_paper')
    
    qx = qx_2pi * 2*np.pi / a
    qy = qy_2pi * 2*np.pi / b
    
    # High-res reference for FS visualization
    nk_ref = 200
    kx_ref = np.linspace(-np.pi/a, np.pi/a, nk_ref)
    ky_ref = np.linspace(-np.pi/b, np.pi/b, nk_ref)
    KX_ref, KY_ref = np.meshgrid(kx_ref, ky_ref, indexing='ij')
    
    print("Computing reference Fermi surface...")
    weights_ref = np.zeros_like(KX_ref)
    for i in range(nk_ref):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{nk_ref}")
        for j in range(nk_ref):
            H = H_full(KX_ref[i, j], KY_ref[i, j], kz)
            eigs = np.linalg.eigvalsh(H)
            weights_ref[i, j] = np.exp(-np.min(np.abs(eigs))**2 / 0.001**2)
    
    # Convert to 2π/(a,b) units
    KX_units = KX_ref / (2*np.pi/a)
    KY_units = KY_ref / (2*np.pi/b)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ========== STRATEGY 1: UNIFORM SAMPLING ==========
    
    # Row 1: Uniform 70×70 grid
    nk_uniform = 70
    kx_uniform = np.linspace(-np.pi/a, np.pi/a, nk_uniform)
    ky_uniform = np.linspace(-np.pi/b, np.pi/b, nk_uniform)
    KX_uniform, KY_uniform = np.meshgrid(kx_uniform, ky_uniform, indexing='ij')
    
    ax = axes[0, 0]
    ax.contourf(KX_units, KY_units, weights_ref, levels=20, cmap='Greys', alpha=0.3)
    ax.contour(KX_units, KY_units, weights_ref, levels=[0.1], colors='red', linewidths=2, label='Fermi Surface')
    ax.plot(KX_uniform / (2*np.pi/a), KY_uniform / (2*np.pi/b), 'b.', markersize=1, alpha=0.5)
    ax.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
    ax.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
    ax.set_title(f'UNIFORM: Sample entire BZ\n{nk_uniform}×{nk_uniform} = {nk_uniform**2} points', 
                fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Check which points are on FS
    on_fs_uniform = 0
    for kx, ky in zip(KX_uniform.flatten(), KY_uniform.flatten()):
        H = H_full(kx, ky, kz)
        eigs = np.linalg.eigvalsh(H)
        w = np.exp(-np.min(np.abs(eigs))**2 / 0.001**2)
        if w > 0.1:
            on_fs_uniform += 1
    
    ax = axes[0, 1]
    ax.contourf(KX_units, KY_units, weights_ref, levels=20, cmap='Greys', alpha=0.3)
    ax.contour(KX_units, KY_units, weights_ref, levels=[0.1], colors='red', linewidths=2)
    
    # Show k-points on FS
    for kx, ky in zip(KX_uniform.flatten(), KY_uniform.flatten()):
        H = H_full(kx, ky, kz)
        eigs = np.linalg.eigvalsh(H)
        w = np.exp(-np.min(np.abs(eigs))**2 / 0.001**2)
        if w > 0.1:
            ax.plot(kx / (2*np.pi/a), ky / (2*np.pi/b), 'go', markersize=8)
    
    ax.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
    ax.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
    ax.set_title(f'k-points on FS\n{on_fs_uniform} points ({100*on_fs_uniform/nk_uniform**2:.1f}% useful)', 
                fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Check k+q points
    on_fs_kpq = 0
    overlap_uniform = 0
    for kx, ky in zip(KX_uniform.flatten(), KY_uniform.flatten()):
        H_k = H_full(kx, ky, kz)
        eigs_k = np.linalg.eigvalsh(H_k)
        w_k = np.exp(-np.min(np.abs(eigs_k))**2 / 0.001**2)
        
        H_kpq = H_full(kx + qx, ky + qy, kz)
        eigs_kpq = np.linalg.eigvalsh(H_kpq)
        w_kpq = np.exp(-np.min(np.abs(eigs_kpq))**2 / 0.001**2)
        
        if w_kpq > 0.1:
            on_fs_kpq += 1
        if w_k > 0.1 and w_kpq > 0.1:
            overlap_uniform += 1
    
    ax = axes[0, 2]
    ax.contourf(KX_units, KY_units, weights_ref, levels=20, cmap='Greys', alpha=0.3)
    ax.contour(KX_units, KY_units, weights_ref, levels=[0.1], colors='red', linewidths=2)
    
    # Show overlap points (both k and k+q on FS)
    for kx, ky in zip(KX_uniform.flatten(), KY_uniform.flatten()):
        H_k = H_full(kx, ky, kz)
        eigs_k = np.linalg.eigvalsh(H_k)
        w_k = np.exp(-np.min(np.abs(eigs_k))**2 / 0.001**2)
        
        H_kpq = H_full(kx + qx, ky + qy, kz)
        eigs_kpq = np.linalg.eigvalsh(H_kpq)
        w_kpq = np.exp(-np.min(np.abs(eigs_kpq))**2 / 0.001**2)
        
        if w_k > 0.1 and w_kpq > 0.1:
            ax.plot(kx / (2*np.pi/a), ky / (2*np.pi/b), 'ro', markersize=10)
    
    ax.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
    ax.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
    ax.set_title(f'HAEM signal from overlap\n{overlap_uniform} points contribute!', 
                fontsize=13, fontweight='bold', color='red' if overlap_uniform < 5 else 'green')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # ========== STRATEGY 2: ADAPTIVE SAMPLING ==========
    
    # Row 2: Adaptive - only sample near FS
    print("\nFinding FS regions...")
    fs_points = []
    nk_coarse = 100
    kx_coarse = np.linspace(-np.pi/a, np.pi/a, nk_coarse)
    ky_coarse = np.linspace(-np.pi/b, np.pi/b, nk_coarse)
    
    for kx in kx_coarse:
        for ky in ky_coarse:
            H = H_full(kx, ky, kz)
            eigs = np.linalg.eigvalsh(H)
            w = np.exp(-np.min(np.abs(eigs))**2 / 0.001**2)
            if w > 0.1:
                fs_points.append((kx, ky))
    
    fs_points = np.array(fs_points)
    
    # Dense sampling around FS points
    print(f"Creating dense grid around {len(fs_points)} FS regions...")
    dense_points = []
    refinement = 9  # 9×9 = 81 subpoints per FS point
    dk_x = (2*np.pi/a) / nk_coarse
    dk_y = (2*np.pi/b) / nk_coarse
    
    for kx_c, ky_c in fs_points:
        kx_local = np.linspace(kx_c - dk_x/2, kx_c + dk_x/2, refinement)
        ky_local = np.linspace(ky_c - dk_y/2, ky_c + dk_y/2, refinement)
        for kx in kx_local:
            for ky in ky_local:
                dense_points.append((kx, ky))
    
    dense_points = np.array(dense_points)
    
    ax = axes[1, 0]
    ax.contourf(KX_units, KY_units, weights_ref, levels=20, cmap='Greys', alpha=0.3)
    ax.contour(KX_units, KY_units, weights_ref, levels=[0.1], colors='red', linewidths=2)
    ax.plot(dense_points[:, 0] / (2*np.pi/a), dense_points[:, 1] / (2*np.pi/b), 
           'b.', markersize=1, alpha=0.5)
    ax.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
    ax.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
    ax.set_title(f'ADAPTIVE: Sample only near FS\n{len(dense_points)} points (was {nk_uniform**2})', 
                fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Check adaptive FS coverage
    on_fs_adaptive = 0
    for kx, ky in dense_points:
        H = H_full(kx, ky, kz)
        eigs = np.linalg.eigvalsh(H)
        w = np.exp(-np.min(np.abs(eigs))**2 / 0.001**2)
        if w > 0.1:
            on_fs_adaptive += 1
    
    ax = axes[1, 1]
    ax.contourf(KX_units, KY_units, weights_ref, levels=20, cmap='Greys', alpha=0.3)
    ax.contour(KX_units, KY_units, weights_ref, levels=[0.1], colors='red', linewidths=2)
    
    for kx, ky in dense_points:
        H = H_full(kx, ky, kz)
        eigs = np.linalg.eigvalsh(H)
        w = np.exp(-np.min(np.abs(eigs))**2 / 0.001**2)
        if w > 0.1:
            ax.plot(kx / (2*np.pi/a), ky / (2*np.pi/b), 'go', markersize=4, alpha=0.8)
    
    ax.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
    ax.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
    ax.set_title(f'k-points on FS\n{on_fs_adaptive} points ({100*on_fs_adaptive/len(dense_points):.1f}% useful)', 
                fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Check adaptive overlap
    overlap_adaptive = 0
    for kx, ky in dense_points:
        H_k = H_full(kx, ky, kz)
        eigs_k = np.linalg.eigvalsh(H_k)
        w_k = np.exp(-np.min(np.abs(eigs_k))**2 / 0.001**2)
        
        H_kpq = H_full(kx + qx, ky + qy, kz)
        eigs_kpq = np.linalg.eigvalsh(H_kpq)
        w_kpq = np.exp(-np.min(np.abs(eigs_kpq))**2 / 0.001**2)
        
        if w_k > 0.1 and w_kpq > 0.1:
            overlap_adaptive += 1
    
    ax = axes[1, 2]
    ax.contourf(KX_units, KY_units, weights_ref, levels=20, cmap='Greys', alpha=0.3)
    ax.contour(KX_units, KY_units, weights_ref, levels=[0.1], colors='red', linewidths=2)
    
    for kx, ky in dense_points:
        H_k = H_full(kx, ky, kz)
        eigs_k = np.linalg.eigvalsh(H_k)
        w_k = np.exp(-np.min(np.abs(eigs_k))**2 / 0.001**2)
        
        H_kpq = H_full(kx + qx, ky + qy, kz)
        eigs_kpq = np.linalg.eigvalsh(H_kpq)
        w_kpq = np.exp(-np.min(np.abs(eigs_kpq))**2 / 0.001**2)
        
        if w_k > 0.1 and w_kpq > 0.1:
            ax.plot(kx / (2*np.pi/a), ky / (2*np.pi/b), 'ro', markersize=6, alpha=0.8)
    
    ax.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
    ax.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
    ax.set_title(f'HAEM signal from overlap\n{overlap_adaptive} points contribute!', 
                fontsize=13, fontweight='bold', color='green')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('outputs/haem_ute2', exist_ok=True)
    filename = 'outputs/haem_ute2/sampling_strategy_comparison.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n📊 Saved: {filename}")
    
    # Print comparison
    print("\n" + "="*70)
    print("SAMPLING STRATEGY COMPARISON")
    print("="*70)
    print(f"\nVector: q = ({qx_2pi:.2f}, {qy_2pi:.2f}) × 2π/(a,b)")
    print(f"\nUNIFORM {nk_uniform}×{nk_uniform}:")
    print(f"  Total points: {nk_uniform**2}")
    print(f"  k on FS: {on_fs_uniform} ({100*on_fs_uniform/nk_uniform**2:.2f}%)")
    print(f"  Overlap: {overlap_uniform} points")
    print(f"  Efficiency: {100*overlap_uniform/nk_uniform**2:.3f}% of points used")
    
    print(f"\nADAPTIVE (coarse={nk_coarse}, refinement={refinement}):")
    print(f"  Total points: {len(dense_points)}")
    print(f"  k on FS: {on_fs_adaptive} ({100*on_fs_adaptive/len(dense_points):.2f}%)")
    print(f"  Overlap: {overlap_adaptive} points")
    print(f"  Efficiency: {100*overlap_adaptive/len(dense_points):.2f}% of points used")
    
    print(f"\nSPEEDUP:")
    speedup = nk_uniform**2 / len(dense_points)
    quality = overlap_adaptive / max(overlap_uniform, 1)
    print(f"  Points reduced by: {speedup:.1f}×")
    print(f"  Overlap increased by: {quality:.1f}×")
    print(f"  Effective speedup: ~{speedup:.1f}× with {quality:.1f}× better sampling")
    
    plt.show()


if __name__ == "__main__":
    visualize_sampling_strategies(qx_2pi=0.43, qy_2pi=1.0)
