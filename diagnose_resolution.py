#!/usr/bin/env python3
"""
Diagnostic tool to check if k-space resolution is adequate for HAEM calculation

This visualizes:
1. Fermi surface weight distribution
2. Which k-points contribute to each q-vector
3. Recommended minimum resolution
"""

import numpy as np
import matplotlib.pyplot as plt
from UTe2_fixed import H_full, a, b, c, set_parameters
import os

def compute_fermi_surface_density(nk=200, kz=0, threshold=0.001):
    """
    Compute high-resolution Fermi surface weight map
    
    Args:
        nk: k-space resolution (higher = more accurate)
        kz: Out-of-plane momentum
        threshold: Energy threshold for FS proximity
        
    Returns:
        KX, KY, weights: k-space grids and FS weights
    """
    kx_range = np.linspace(-np.pi/a, np.pi/a, nk)
    ky_range = np.linspace(-np.pi/b, np.pi/b, nk)
    KX, KY = np.meshgrid(kx_range, ky_range, indexing='ij')
    
    weights = np.zeros_like(KX)
    
    print(f"Computing Fermi surface map with {nk}×{nk} resolution...")
    for i in range(nk):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{nk}")
        for j in range(nk):
            H = H_full(KX[i, j], KY[i, j], kz)
            eigenvals = np.linalg.eigvalsh(H)
            weights[i, j] = np.exp(-np.min(np.abs(eigenvals))**2 / threshold**2)
    
    return KX, KY, weights


def analyze_qvector_coverage(qx_2pi, qy_2pi, nk_test, kz=0, threshold=0.001):
    """
    Check how many k-points near the Fermi surface are captured for a given q-vector
    
    Args:
        qx_2pi, qy_2pi: q-vector in units of 2π/(a,b)
        nk_test: Resolution to test
        kz: Out-of-plane momentum
        threshold: FS proximity threshold
        
    Returns:
        n_fs_points: Number of k-points on FS
        n_kpq_fs_points: Number of k+q points on FS
        overlap: Number of points where both k and k+q are on FS
    """
    qx = qx_2pi * 2*np.pi / a
    qy = qy_2pi * 2*np.pi / b
    
    kx_range = np.linspace(-np.pi/a, np.pi/a, nk_test)
    ky_range = np.linspace(-np.pi/b, np.pi/b, nk_test)
    KX, KY = np.meshgrid(kx_range, ky_range, indexing='ij')
    
    # Check k-points on FS
    weights_k = np.zeros_like(KX)
    weights_kpq = np.zeros_like(KX)
    
    for i in range(nk_test):
        for j in range(nk_test):
            H_k = H_full(KX[i, j], KY[i, j], kz)
            eigenvals_k = np.linalg.eigvalsh(H_k)
            weights_k[i, j] = np.exp(-np.min(np.abs(eigenvals_k))**2 / threshold**2)
            
            H_kpq = H_full(KX[i, j] + qx, KY[i, j] + qy, kz)
            eigenvals_kpq = np.linalg.eigvalsh(H_kpq)
            weights_kpq[i, j] = np.exp(-np.min(np.abs(eigenvals_kpq))**2 / threshold**2)
    
    # Count significant contributions (weight > 0.1)
    n_fs_points = np.sum(weights_k > 0.1)
    n_kpq_fs_points = np.sum(weights_kpq > 0.1)
    overlap = np.sum((weights_k > 0.1) & (weights_kpq > 0.1))
    
    return n_fs_points, n_kpq_fs_points, overlap, weights_k, weights_kpq


def plot_fermi_surface_diagnostic(vectors_dict, nk_high=200, nk_test=70, kz=0, save_dir='outputs/haem_ute2'):
    """
    Create diagnostic plots showing FS coverage for different resolutions
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get high-resolution FS map
    KX, KY, weights_high = compute_fermi_surface_density(nk_high, kz)
    
    # Convert to 2π/(a,b) units for plotting
    KX_units = KX / (2*np.pi / a)
    KY_units = KY / (2*np.pi / b)
    
    # Create figure
    n_vectors = len(vectors_dict)
    fig, axes = plt.subplots(2, n_vectors + 1, figsize=(6*(n_vectors+1), 10))
    
    # Plot 1: High-res Fermi surface
    ax = axes[0, 0]
    im = ax.contourf(KX_units, KY_units, weights_high, levels=20, cmap='hot')
    ax.contour(KX_units, KY_units, weights_high, levels=[0.1, 0.5, 0.9], colors='cyan', linewidths=2)
    ax.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
    ax.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
    ax.set_title(f'Fermi Surface\n(nk={nk_high})', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='FS Weight')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Lower-res sampling
    ax = axes[1, 0]
    kx_test = np.linspace(-np.pi/a, np.pi/a, nk_test)
    ky_test = np.linspace(-np.pi/b, np.pi/b, nk_test)
    KX_test, KY_test = np.meshgrid(kx_test, ky_test, indexing='ij')
    KX_test_units = KX_test / (2*np.pi / a)
    KY_test_units = KY_test / (2*np.pi / b)
    
    ax.contourf(KX_units, KY_units, weights_high, levels=20, cmap='hot', alpha=0.3)
    ax.plot(KX_test_units.flatten(), KY_test_units.flatten(), 'b.', markersize=2, alpha=0.5)
    ax.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
    ax.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
    ax.set_title(f'Sampling Grid\n(nk={nk_test})', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Analyze each q-vector
    print("\n" + "="*70)
    print("RESOLUTION ANALYSIS")
    print("="*70)
    
    for idx, (label, (qx_2pi, qy_2pi)) in enumerate(vectors_dict.items(), 1):
        print(f"\n{label}: q = ({qx_2pi:+.3f}, {qy_2pi:+.3f}) × 2π/(a,b)")
        
        n_fs, n_kpq_fs, overlap, weights_k, weights_kpq = analyze_qvector_coverage(
            qx_2pi, qy_2pi, nk_test, kz
        )
        
        total_points = nk_test * nk_test
        print(f"  Grid: {nk_test}×{nk_test} = {total_points} points")
        print(f"  k-points on FS: {n_fs} ({100*n_fs/total_points:.1f}%)")
        print(f"  k+q points on FS: {n_kpq_fs} ({100*n_kpq_fs/total_points:.1f}%)")
        print(f"  Overlap (both on FS): {overlap} ({100*overlap/total_points:.1f}%)")
        
        if overlap < 10:
            print(f"  ⚠️  WARNING: Very low overlap! Increase resolution!")
            recommended = int(nk_test * 2)
            print(f"  Recommended: nk ≥ {recommended}")
        elif overlap < 50:
            print(f"  ⚠️  Low overlap. Consider higher resolution.")
            recommended = int(nk_test * 1.5)
            print(f"  Recommended: nk ≥ {recommended}")
        else:
            print(f"  ✓ Resolution adequate")
        
        # Plot k-space regions
        ax_k = axes[0, idx]
        ax_k.contourf(KX_units, KY_units, weights_high, levels=20, cmap='Greys', alpha=0.3)
        im_k = ax_k.contourf(KX_test_units, KY_test_units, weights_k, levels=20, cmap='Reds', alpha=0.7)
        ax_k.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
        ax_k.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
        ax_k.set_title(f'{label}: FS weight at k\n({n_fs} points)', fontsize=12)
        ax_k.set_aspect('equal')
        plt.colorbar(im_k, ax=ax_k, label='Weight')
        ax_k.grid(True, alpha=0.3)
        
        ax_kpq = axes[1, idx]
        ax_kpq.contourf(KX_units, KY_units, weights_high, levels=20, cmap='Greys', alpha=0.3)
        im_kpq = ax_kpq.contourf(KX_test_units, KY_test_units, weights_kpq, levels=20, cmap='Blues', alpha=0.7)
        ax_kpq.set_xlabel(r'$k_x$ [2π/a]', fontsize=12)
        ax_kpq.set_ylabel(r'$k_y$ [2π/b]', fontsize=12)
        ax_kpq.set_title(f'{label}: FS weight at k+q\n({n_kpq_fs} points)', fontsize=12)
        ax_kpq.set_aspect('equal')
        plt.colorbar(im_kpq, ax=ax_kpq, label='Weight')
        ax_kpq.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(save_dir, f'resolution_diagnostic_nk{nk_test}.png')
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n📊 Diagnostic plot saved: {filename}")
    
    plt.show()
    
    return fig


def estimate_minimum_resolution(vectors_dict, target_overlap=50, kz=0):
    """
    Estimate minimum resolution needed to get adequate FS coverage
    
    Args:
        vectors_dict: Dictionary of q-vectors
        target_overlap: Minimum number of overlapping points needed
        kz: Out-of-plane momentum
        
    Returns:
        recommendations: Dictionary with recommended resolutions
    """
    print("\n" + "="*70)
    print("TESTING DIFFERENT RESOLUTIONS")
    print("="*70)
    
    test_resolutions = [30, 50, 70, 100, 120, 150]
    recommendations = {}
    
    for label, (qx_2pi, qy_2pi) in vectors_dict.items():
        print(f"\n{label}: q = ({qx_2pi:+.3f}, {qy_2pi:+.3f})")
        
        for nk in test_resolutions:
            n_fs, n_kpq_fs, overlap, _, _ = analyze_qvector_coverage(qx_2pi, qy_2pi, nk, kz)
            
            status = "✓" if overlap >= target_overlap else "✗"
            print(f"  nk={nk:3d}: overlap={overlap:4d} points ({100*overlap/(nk*nk):4.1f}%) {status}")
            
            if label not in recommendations and overlap >= target_overlap:
                recommendations[label] = nk
        
        if label not in recommendations:
            recommendations[label] = max(test_resolutions) + 50
            print(f"  ⚠️  Need nk > {max(test_resolutions)}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    max_recommended = max(recommendations.values())
    print(f"\nMinimum resolution to cover all vectors: nk = {max_recommended}")
    print("\nPer-vector recommendations:")
    for label, nk in recommendations.items():
        print(f"  {label}: nk ≥ {nk}")
    
    return recommendations


def main():
    """Run resolution diagnostic"""
    set_parameters('odd_parity_paper')
    
    vectors_dict = {
        'p2': (0.43, 1.0),
        'p5': (-0.244, 1.000),
        'p6': (0.619, 0.000)
    }
    
    print("🔍 HAEM Resolution Diagnostic")
    print("="*70)
    print("\nThis tool helps determine the minimum k-space resolution needed")
    print("to properly capture the Fermi surface for HAEM calculations.")
    print("\nVectors to analyze:")
    for label, (qx, qy) in vectors_dict.items():
        print(f"  {label}: ({qx:+.3f}, {qy:+.3f}) × 2π/(a,b)")
    
    # Test current resolution
    current_nk = 70
    print(f"\nTesting current resolution: {current_nk}×{current_nk}")
    
    plot_fermi_surface_diagnostic(vectors_dict, nk_high=200, nk_test=current_nk, kz=0)
    
    # Find optimal resolution
    recommendations = estimate_minimum_resolution(vectors_dict, target_overlap=50, kz=0)
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nBased on this analysis:")
    print(f"- Current resolution ({current_nk}×{current_nk}) may be inadequate for some vectors")
    print(f"- Recommended minimum: {max(recommendations.values())}×{max(recommendations.values())}")
    print("\nUpdate HAEMUTe2.py with these values:")
    print(f"  nk_t = {max(recommendations.values())}  # T-matrix")
    print(f"  nk_ldos = {max(recommendations.values())}  # LDOS")


if __name__ == "__main__":
    main()
