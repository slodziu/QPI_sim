#!/usr/bin/env python3
"""
Visualize gap structure with q-vectors and integration regions
Shows the signed gap function to understand which pockets have opposite signs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from UTe2_fixed import H_full, a, b, c, set_parameters

def d_vector_component(kx, ky, kz, pairing_type, C1=0.0003, C2=0.0003, C3=0.0003):
    """
    Return the dominant d-vector component for visualization
    At kz=0: B2u → dz = C3*sin(kx*a), B3u → dz = C3*sin(ky*b)
    """
    if pairing_type == 'B2u':
        # B2u: dx = C1*sin(kz*c), dz = C3*sin(kx*a)
        # At kz=0: dx=0, so dz dominates
        return C3 * np.sin(kx * a)
    elif pairing_type == 'B3u':
        # B3u: dy = C2*sin(kz*c), dz = C3*sin(ky*b)
        # At kz=0: dy=0, so dz dominates
        return C3 * np.sin(ky * b)
    else:
        raise ValueError(f"Unknown pairing: {pairing_type}")


def plot_gap_with_vectors(vectors_dict, pairing_types=['B2u', 'B3u'], 
                          q_radius=0.05, kz=0, save_dir='outputs/haem_ute2'):
    """
    Plot gap structure in k-space with q-vectors and integration circles
    
    Args:
        vectors_dict: {label: (qx_2pi, qy_2pi)} q-vectors in units of 2π/(a,b)
        pairing_types: Pairing symmetries to plot
        q_radius: Integration radius (in units of 2π/a)
        kz: Out-of-plane momentum
        save_dir: Output directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("🎨 Creating gap structure visualization...")
    
    # k-space grid
    nk = 200
    kx_range = np.linspace(-2*np.pi/a, 2*np.pi/a, nk)
    ky_range = np.linspace(-2*np.pi/b, 2*np.pi/b, nk)
    KX, KY = np.meshgrid(kx_range, ky_range, indexing='ij')
    
    # Compute Fermi surface for overlay
    print("   Computing Fermi surface...")
    FS_weights = np.zeros_like(KX)
    for i in range(nk):
        for j in range(nk):
            H = H_full(KX[i,j], KY[i,j], kz)
            eigenvals = np.linalg.eigvalsh(H)
            FS_weights[i,j] = np.exp(-np.min(np.abs(eigenvals))**2 / 0.001**2)
    
    # Create figure
    n_types = len(pairing_types)
    fig, axes = plt.subplots(1, n_types, figsize=(8*n_types, 7))
    
    if n_types == 1:
        axes = [axes]
    
    for idx, pairing_type in enumerate(pairing_types):
        ax = axes[idx]
        
        print(f"   Plotting {pairing_type}...")
        
        # Compute gap function
        gap_map = d_vector_component(KX, KY, kz, pairing_type)
        
        # Plot gap with diverging colormap (red=positive, blue=negative)
        vmax = np.max(np.abs(gap_map))
        im = ax.contourf(KX*a/(2*np.pi), KY*b/(2*np.pi), gap_map, 
                        levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        alpha=0.8)
        
        # Overlay Fermi surface contours
        ax.contour(KX*a/(2*np.pi), KY*b/(2*np.pi), FS_weights,
                  levels=[0.1], colors='black', linewidths=2, 
                  linestyles='-', alpha=0.7)
        
        # Overlay nodal lines (where gap = 0)
        ax.contour(KX*a/(2*np.pi), KY*b/(2*np.pi), gap_map,
                  levels=[0], colors='green', linewidths=3,
                  linestyles='--', alpha=0.9)
        
        # Plot q-vectors and integration circles
        colors_vec = {'p2': '#FF6B6B', 'p5': '#4ECDC4', 'p6': '#FFD93D'}
        
        for vector_label, (qx_2pi, qy_2pi) in vectors_dict.items():
            color = colors_vec.get(vector_label, 'white')
            
            # Plot q-vector arrow from origin
            ax.arrow(0, 0, qx_2pi, qy_2pi, head_width=0.08, head_length=0.08,
                    fc=color, ec=color, linewidth=3, length_includes_head=True,
                    alpha=0.8, zorder=10)
            
            # Integration circle (always shown, but auto-crops to BZ)
            circle = Circle((qx_2pi, qy_2pi), q_radius, 
                          fill=False, edgecolor=color, linewidth=2.5,
                          linestyle=':', alpha=0.7, zorder=10)
            ax.add_patch(circle)
            
            # Check if near BZ edge (will use partial circle)
            near_edge = (abs(qx_2pi) > 0.85 or abs(qy_2pi) > 0.85 or 
                        abs(abs(qx_2pi) - 2.0) < 0.15 or abs(abs(qy_2pi) - 2.0) < 0.15)
            
            if near_edge:
                label_text = f"{vector_label}\n(partial integral)"
            else:
                label_text = f"{vector_label}\n(full integral)"
            
            # Label
            ax.text(qx_2pi, qy_2pi + 0.15, label_text,
                   ha='center', va='bottom', fontsize=11, fontweight='bold',
                   color=color, bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', edgecolor=color, alpha=0.8))
        
        # Formatting
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 2.3)
        ax.set_xlabel(r'$q_x$ / (2π/a)', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'$q_y$ / (2π/b)', fontsize=14, fontweight='bold')
        ax.set_title(f'{pairing_type}: Gap d-vector component\n' + 
                    r'(red: $\Delta > 0$, blue: $\Delta < 0$, green: nodes)' + '\n' +
                    r'Dotted circles: integration regions (auto-crop to 1st BZ)',
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label=r'Gap $d_z(k)$ (eV)')
        cbar.ax.tick_params(labelsize=11)
        
        # Mark BZ boundaries
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axhline(y=-1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=-1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.text(0.95, 1.05, 'BZ edge', ha='right', va='bottom', 
               fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save
    plot_file = os.path.join(save_dir, 'gap_structure_with_vectors.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Gap visualization saved: {plot_file}")
    
    plt.show()
    
    return fig, axes


def plot_gap_sign_analysis(vectors_dict, pairing_types=['B2u', 'B3u'],
                           kz=0, save_dir='outputs/haem_ute2'):
    """
    Analyze gap signs at k and k+q for each vector
    Shows why certain vectors produce sign changes
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n🔍 Gap Sign Analysis:")
    print("="*70)
    
    # Sample k-points on Fermi surface
    nk_sample = 100
    kx_range = np.linspace(-np.pi/a, np.pi/a, nk_sample)
    ky_range = np.linspace(-np.pi/b, np.pi/b, nk_sample)
    
    # Find FS points
    fs_points = []
    for kx in kx_range:
        for ky in ky_range:
            H = H_full(kx, ky, kz)
            eigenvals = np.linalg.eigvalsh(H)
            weight = np.exp(-np.min(np.abs(eigenvals))**2 / 0.001**2)
            if weight > 0.1:
                fs_points.append((kx, ky))
    
    fs_points = np.array(fs_points)
    print(f"Found {len(fs_points)} Fermi surface points\n")
    
    for pairing_type in pairing_types:
        print(f"{pairing_type}:")
        print("-" * 70)
        
        for vector_label, (qx_2pi, qy_2pi) in vectors_dict.items():
            qx = qx_2pi * 2*np.pi / a
            qy = qy_2pi * 2*np.pi / b
            
            # Sample gap at k and k+q
            gap_k = []
            gap_kpq = []
            
            for kx, ky in fs_points[:1000]:  # Sample subset
                gap_k.append(d_vector_component(kx, ky, kz, pairing_type))
                
                # k+q with folding
                kx_pq = kx + qx
                ky_pq = ky + qy
                # Fold to first BZ
                kx_pq = ((kx_pq + np.pi/a) % (2*np.pi/a)) - np.pi/a
                ky_pq = ((ky_pq + np.pi/b) % (2*np.pi/b)) - np.pi/b
                
                gap_kpq.append(d_vector_component(kx_pq, ky_pq, kz, pairing_type))
            
            gap_k = np.array(gap_k)
            gap_kpq = np.array(gap_kpq)
            
            # Check for opposite signs
            opposite_signs = (gap_k * gap_kpq < 0)
            frac_opposite = np.sum(opposite_signs) / len(gap_k) * 100
            
            print(f"  {vector_label}: q=({qx_2pi:.3f}, {qy_2pi:.3f})")
            print(f"    Opposite sign fraction: {frac_opposite:.1f}%")
            
            if frac_opposite > 30:
                print(f"    → Expect HAEM sign change ✓")
            else:
                print(f"    → Expect NO sign change (same-sign scattering)")
        
        print()


def main():
    """Create gap visualization with q-vectors"""
    set_parameters('odd_parity_paper')
    
    vectors_dict = {
        'p2': (0.374, 1.000),
        'p5': (-0.244, 1.000),
        'p6': (0.619, 0.000)
    }
    
    print("Gap Structure Visualization")
    print("="*70)
    
    # Plot gap maps with vectors
    plot_gap_with_vectors(
        vectors_dict=vectors_dict,
        pairing_types=['B2u', 'B3u'],
        q_radius=0.05,
        kz=0
    )
    
    # Analyze gap signs
    plot_gap_sign_analysis(
        vectors_dict=vectors_dict,
        pairing_types=['B2u', 'B3u'],
        kz=0
    )
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
