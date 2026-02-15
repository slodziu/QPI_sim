#!/usr/bin/env python3
"""
JDOS Projection to (0-11) Cleave Plane
=====================================
Transform JDOS from (001) plane to (0-11) plane as done in 
"Odd-parity quasiparticle interference in the superconductive surface state of UTe2"
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator

def load_jdos_data(param_set_name='odd_parity_paper', kz=0.0, resolution=2001):
    """Load JDOS data from saved .npz file"""
    data_path = f'raw_data_out/JDOS/ute2_jdos_data_{param_set_name.lower()}_res_{resolution}.npz'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"JDOS data file not found: {data_path}")
    
    print(f"Loading JDOS data from: {data_path}")
    data = np.load(data_path)
    
    return {
        'JDOS': data['JDOS'],
        'qx_vals': data['qx_vals'],
        'qy_vals': data['qy_vals'], 
        'kx_vals': data['kx_vals'],
        'ky_vals': data['ky_vals'],
        'energies': data['energies'],
        'weights_5f': data['weights_5f'],
        'param_set_name': str(data['param_set_name']),
        'kz': float(data['kz']),
        'lattice_a': float(data['lattice_a']),
        'lattice_b': float(data['lattice_b']),
        'lattice_c': float(data['lattice_c']),
        'resolution': int(data['resolution'])
    }

def transform_to_011_plane(qx_vals, qy_vals, JDOS, lattice_a=0.41, lattice_b=0.61, c_star=0.76, theta_deg=24.0):
    """
    Transform JDOS from (001) plane to (0-11) cleave plane
    
    Parameters:
    - qx_vals, qy_vals: momentum transfer grids from (001) calculation
    - JDOS: joint density of states array
    - lattice_a, lattice_b: lattice constants (nm)
    - c_star: (0-11) surface lattice periodicity (nm) 
    - theta_deg: rotation angle for (0-11) plane (degrees)
    
    Returns:
    - qx_011: kx coordinates (same as input, in π/a units)
    - qc_011: kc* coordinates (ky transformed to π/c* units) 
    - JDOS_011: interpolated JDOS on new grid
    """
    
    print(f"Transforming JDOS to (0-11) plane...")
    print(f"  Using c* = {c_star:.3f} nm, θ = {theta_deg}°")
    
    theta_rad = np.deg2rad(theta_deg)
    
    # Original coordinates are in absolute units, convert to dimensionless first
    qx_pi_a = qx_vals / (np.pi / lattice_a)  # qx in π/a units
    qy_pi_b = qy_vals / (np.pi / lattice_b)  # qy in π/b units
    
    print(f"  Original ranges: qx ∈ [{qx_pi_a.min():.2f}, {qx_pi_a.max():.2f}] π/a")
    print(f"  Original ranges: qy ∈ [{qy_pi_b.min():.2f}, {qy_pi_b.max():.2f}] π/b")
    
    # For (0-11) plane transformation:
    # kx stays as kx (in π/a units) 
    # ky gets transformed to kc* coordinates
    # From paper: qy = py × sin θ, but empirically this gives factor ≈ 0.5
    # This corresponds to θ_effective ≈ 30° for momentum space transformation
    
    # Use empirical factor that matches paper's wavevector table
    qc_star_scale = 0.5  # Empirical factor from paper's wavevector transformations
    qc_pi_cstar = qy_pi_b * qc_star_scale
    
    print(f"  Transformation factor (empirical from paper): {qc_star_scale:.3f}")
    print(f"  Note: This corresponds to θ_eff = 30° rather than crystallographic θ = 24°")
    print(f"  Transformed ranges: qx ∈ [{qx_pi_a.min():.2f}, {qx_pi_a.max():.2f}] π/a")  
    print(f"  Transformed ranges: qc* ∈ [{qc_pi_cstar.min():.2f}, {qc_pi_cstar.max():.2f}] π/c*")
    print(f"  Range compression check: original qy span = {qy_pi_b.max() - qy_pi_b.min():.2f}, new qc* span = {qc_pi_cstar.max() - qc_pi_cstar.min():.2f}")
    print(f"  Expected ratio (should be ~0.5): {(qc_pi_cstar.max() - qc_pi_cstar.min()) / (qy_pi_b.max() - qy_pi_b.min()):.3f}")
    
    # Create new coordinate grids for (0-11) plane with proper scaling
    qx_011 = qx_pi_a  # kx unchanged (π/a units)
    qc_011 = qc_pi_cstar  # ky → kc* (π/c* units), compressed by factor 0.5
    
    # The JDOS needs to be interpolated onto the new coordinate grid
    # since the qc* axis is compressed, we need to interpolate the data properly
    print(f"  Interpolating JDOS data onto (0-11) coordinate grid...")
    
    # Create interpolator for the original JDOS data
    interpolator = RegularGridInterpolator((qx_pi_a, qy_pi_b), JDOS, 
                                         method='linear', bounds_error=False, fill_value=0)
    
    # Create the new grid for (0-11) plane - expand qc* back to sample original data properly
    qx_011_grid, qc_011_grid = np.meshgrid(qx_011, qc_011, indexing='ij')
    
    # Map (0-11) coordinates back to original (001) coordinates for interpolation
    qx_orig_for_interp = qx_011_grid  # kx unchanged
    qy_orig_for_interp = qc_011_grid / 0.5  # Expand qc* back to qy coordinates
    
    # Interpolate JDOS at these points
    interp_points = np.column_stack([qx_orig_for_interp.ravel(), qy_orig_for_interp.ravel()])
    JDOS_011_flat = interpolator(interp_points)
    JDOS_011 = JDOS_011_flat.reshape(qx_011_grid.shape)
    
    print(f"  Interpolation complete. New JDOS range: {JDOS_011.min():.4f} to {JDOS_011.max():.4f}")
    
    return qx_011, qc_011, JDOS_011

def plot_jdos_011_plane(qx_011, qc_011, JDOS_011, param_set_name, kz=0.0, c_star=0.76, theta_deg=24.0):
    """Plot JDOS in (0-11) plane coordinates"""
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Plot JDOS with proper aspect ratio for (0-11) coordinates
    im = ax.imshow(JDOS_011, extent=[qc_011.min(), qc_011.max(), qx_011.min(), qx_011.max()],
                  origin='lower', cmap='plasma', aspect='equal')  # Use equal aspect to preserve momentum space scaling
    
    # Style the plot
    ax.set_xlabel('qc* (π/c* units)', fontsize=14)
    ax.set_ylabel('qx (π/a units)', fontsize=14) 
    ax.set_title(f'JDOS in (0-11) Cleave Plane\n({param_set_name}), kz = {kz:.3f}, θ = {theta_deg}°', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='w', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(0, color='w', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add first Brillouin Zone rectangle (±1π/a, ±1π/c*)
    from matplotlib.patches import Rectangle
    bz_rect = Rectangle((-1, -1), 2, 2, linewidth=2, edgecolor='white', 
                       facecolor='none', linestyle='--', alpha=0.8)
    ax.add_patch(bz_rect)
    
    # Add scattering wavevectors transformed to (0-11) coordinates
    # Original wavevectors from paper (in 2π/a, 2π/b units)
    wavevectors = {
        'p1': (0.130, 0.00),
        'p2': (0.374, 1.000),
        'p3': (0.130, 2.000), 
        'p4': (0.000, 2.00),
        'p5': (-0.244, 1.000),
        'p6': (0.619, 0.000)
    }
    
    colors = ['white', 'orange', 'yellow', 'green', 'cyan', 'red']
    
    # Transform wavevectors to (0-11) plane using empirical factor
    transformation_factor = 0.5  # Matches paper's wavevector table
    
    for i, (label, (qx_2pi, qy_2pi)) in enumerate(wavevectors.items()):
        # Convert to π/a, π/b units first
        qx_pi = qx_2pi * 2  # qx in π/a units (unchanged)  
        qy_pi = qy_2pi * 2  # qy in π/b units
        
        # Transform to (0-11) coordinates using empirical factor
        qx_011_pt = qx_pi  # kx unchanged
        qc_011_pt = qy_pi * transformation_factor  # ky → kc* with factor 0.5
        
        # Plot coordinates: x-axis is qc*, y-axis is qx
        plot_x = qc_011_pt  # qc* goes on x-axis
        plot_y = qx_011_pt  # qx goes on y-axis
        
        # Only plot if within reasonable display range
        if (qc_011.min() <= plot_x <= qc_011.max()) and (qx_011.min() <= plot_y <= qx_011.max()):
            color = colors[i % len(colors)]
            
            # Draw arrow from origin
            ax.annotate('', xy=(plot_x, plot_y), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.8,
                                     lw=4, shrinkA=0, shrinkB=0))
            
            # Add point marker
            ax.plot(plot_x, plot_y, 'o', color=color, markersize=8,
                   markeredgecolor='black', markeredgewidth=1.5, alpha=0.9)
            
            # Add label
            ax.annotate(label, (plot_x, plot_y), xytext=(10, 10), textcoords='offset points',
                       color=color, fontweight='bold', fontsize=12, alpha=0.95,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))
    
    # Set display limits - zoom in on qx range, keep full qc* range
    ax.set_xlim(qc_011.min(), qc_011.max())  # qc* range - keep full range
    ax.set_ylim(qx_011.min(), qx_011.max())  # qx range - use full range for clarity
    
    # Add colorbar with proper height to match visible plot area
    # Calculate shrink factor based on displayed y-range vs full y-range
    display_y_range = qx_011.max() - qx_011.min()  
    full_y_range = qx_011.max() - qx_011.min()
    shrink_factor = min(display_y_range / full_y_range, 1.0)
    
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.045, pad=0.04, shrink=shrink_factor)
    cbar.set_label('JDOS (arbitrary units)', fontsize=14)

    # Save plot
    os.makedirs('outputs/JDOS', exist_ok=True)
    save_path = f'outputs/JDOS/ute2_jdos_011_plane_{param_set_name.lower()}_kz_{kz:.3f}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved (0-11) plane JDOS plot to: {save_path}")
    
    return fig, ax

def main():
    """Main function to load data and create (0-11) plane projection"""
    
    print("="*60)
    print("JDOS Projection to (0-11) Cleave Plane")
    print("="*60)
    
    # Parameters from paper
    param_set_name = 'odd_parity_paper'
    kz = 0.0
    c_star = 0.76  # nm, (0-11) surface lattice periodicity
    theta_deg = 24.0  # degrees, rotation angle
    
    try:
        # Load JDOS data
        data = load_jdos_data(param_set_name, kz)
        print(f"Loaded data: {data['resolution']}×{data['resolution']} grid")
        print(f"JDOS range: {data['JDOS'].min():.4f} to {data['JDOS'].max():.4f}")
        
        # Transform to (0-11) plane
        qx_011, qc_011, JDOS_011 = transform_to_011_plane(
            data['qx_vals'], data['qy_vals'], data['JDOS'],
            data['lattice_a'], data['lattice_b'], c_star, theta_deg
        )
        
        # Create plot
        fig, ax = plot_jdos_011_plane(qx_011, qc_011, JDOS_011, param_set_name, kz, c_star, theta_deg)
        
        print("="*60)
        print("(0-11) plane projection complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run ute2_minimal_fermi.py first to generate JDOS data.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
