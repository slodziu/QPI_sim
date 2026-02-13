# 2D Fermi Surface visualization for UTe2 at kz=0
import numpy as np
import matplotlib.pyplot as plt
import os

# Import Hamiltonian functions and parameters from UTe2_fixed.py
from UTe2_fixed import (
    HU_block, HTe_block, H_full, 
    a, b, c,  # lattice constants
    muU, DeltaU, tU, tpU, tch_U, tpch_U, tz_U,  # U parameters
    muTe, DeltaTe, tTe, tch_Te, tz_Te,  # Te parameters  
    delta,  # hybridization parameter
    verify_hamiltonian_hermiticity as verify_hermiticity_main
)

def verify_hamiltonian_hermiticity():
    """Verify that the Hamiltonian is Hermitian using the main implementation"""
    print("\n=== 2DFS Using UTe2_fixed Implementation ===")
    print("Running Hermiticity check from UTe2_fixed.py...")
    return verify_hermiticity_main()

def calculate_band_energies(kz=0.0, resolution=300):
    """Calculate band energies on 2D slice at fixed kz"""
    print(f"Calculating band energies at kz = {kz:.3f} with resolution {resolution}...")
    
    # Create momentum grid with specified extents
    # kx: -1 to 1 in units of π/a (y-axis)
    # ky: -3 to 3 in units of π/b (x-axis)
    kx_vals = np.linspace(-np.pi/a, np.pi/a, resolution)  # -1π/a to 1π/a
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, resolution)  # -3π/b to 3π/b
    
    # Create meshgrid (match UTe2_fixed.py exactly)
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    
    # Pre-allocate energy array
    energies = np.zeros((resolution, resolution, 4))
    
    # Calculate Hamiltonian and eigenvalues for each point
    print("Computing eigenvalues...")
    for i in range(resolution):
        for j in range(resolution):
            kx = KX[i, j]
            ky = KY[i, j]
            H = H_full(kx, ky, kz)
            eigenvals = np.linalg.eigvals(H)
            energies[i, j, :] = np.sort(np.real(eigenvals))
    
    return energies, KX, KY

def plot_fermi_surface_2d(energies, KX, KY, save_dir='outputs', filename='fermi_surface_2d_kz0'):
    """Plot 2D Fermi surface contours"""
    print("Creating 2D Fermi surface plot...")
    
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Colors for different bands
    colors = ['#1E88E5', '#42A5F5', '#9635E5', '#FD0000']
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    # Convert momentum arrays to units of π/a and π/b for plotting
    KY_plot = KY / (np.pi/b)  # ky in units of π/b (x-axis)
    KX_plot = KX / (np.pi/a)  # kx in units of π/a (y-axis)
    
    # Plot Fermi surface contours (E = 0)
    EF = 0.0
    contours_plotted = 0
    legend_handles = []
    
    for band in range(4):
        try:
            # Check energy range for this band (use transpose like UTe2_fixed.py)
            Z = energies[:, :, band].T  # CRITICAL: Transpose to match UTe2_fixed.py
            E_min = np.min(Z)
            E_max = np.max(Z)
            print(f"  {band_labels[band]}: E ∈ [{E_min:.3f}, {E_max:.3f}] eV")
            
            # Only plot if Fermi level crosses this band
            if E_min <= EF <= E_max:
                # Create contours at Fermi level (ky=x-axis, kx=y-axis)
                contour_set = ax.contour(KY_plot, KX_plot, Z, 
                                       levels=[EF], colors=[colors[band]], linewidths=2.5)
                
                # Check if contours were found (compatible with both old and new matplotlib)
                if hasattr(contour_set, 'collections'):
                    collections = contour_set.collections
                else:
                    collections = [contour_set]
                
                if len(collections) > 0:
                    # Check if any paths exist
                    has_paths = any(len(collection.get_paths()) > 0 for collection in collections)
                    if has_paths:
                        print(f"  ✓ Found Fermi surface for {band_labels[band]}")
                        contours_plotted += 1
                        
                        # Create legend entry
                        from matplotlib.lines import Line2D
                        legend_handles.append(Line2D([0], [0], color=colors[band], 
                                                   linewidth=2.5, label=band_labels[band]))
                    else:
                        print(f"  ✗ No contour paths for {band_labels[band]}")
                else:
                    print(f"  ✗ No contour collections for {band_labels[band]}")
            else:
                print(f"  ✗ No Fermi crossing in {band_labels[band]} (E_F not in range)")
                
        except Exception as e:
            print(f"  ✗ Error plotting {band_labels[band]}: {e}")
    
    if contours_plotted == 0:
        print("⚠️  No Fermi surface contours found!")
        # Show which bands might have crossings
        for band in range(4):
            Z = energies[:, :, band].T  # Use same transpose for consistency
            E_min = np.min(Z)
            E_max = np.max(Z)
            if E_min <= 0 <= E_max:
                print(f"    {band_labels[band]} spans Fermi level: E ∈ [{E_min:.3f}, {E_max:.3f}] eV")
    else:
        print(f"✓ Successfully plotted {contours_plotted} band(s)")
    
    # Set up the plot 
    ax.set_xlim(-3, 3)  # ky in π/b units
    ax.set_ylim(-1, 1)  # kx in π/a units
    ax.set_xlabel('ky (π/b units)', fontsize=14)
    ax.set_ylabel('kx (π/a units)', fontsize=14)
    ax.set_title('UTe2 Fermi Surface at kz = 0', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add reference lines at zone boundaries
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Add zone boundary box for reference
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k:', alpha=0.4, linewidth=1)
    
    # Add legend if we have contours
    if len(legend_handles) > 0:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=12, framealpha=0.8)
    
    # Add text annotation with energy information
    info_text = "Energy ranges at kz=0:\n"
    for band in range(4):
        Z = energies[:, :, band].T  # Use same transpose as above
        E_min = np.min(Z)
        E_max = np.max(Z)
        crosses_fermi = "✓" if E_min <= 0 <= E_max else "✗"
        info_text += f"{band_labels[band]}: [{E_min:.2f}, {E_max:.2f}] eV {crosses_fermi}\n"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    save_path = os.path.join(save_dir, f'{filename}.png')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Fermi surface plot saved to: {save_path}")
    
    plt.show()
    return fig, ax

def main():
    """Main execution function"""
    print("="*60)
    print("UTe2 2D Fermi Surface Visualization at kz = 0")
    print("Using corrected Hamiltonian from UTe2_fixed.py")
    print("="*60)

    # Parameters
    kz = 0.0  # Fixed kz value
    resolution = 300  # Grid resolution (match UTe2_fixed.py)
    
    print(f"\nModel parameters:")
    print(f"  Lattice: a = {a} nm, b = {b} nm, c = {c} nm")
    print(f"  U sector: μU = {muU}, ΔU = {DeltaU}, tU = {tU}")
    print(f"  Te sector: μTe = {muTe}, ΔTe = {DeltaTe}, tTe = {tTe}")
    print(f"  Hybridization: δ = {delta}")
    print(f"  Grid: {resolution}×{resolution} points")
    print(f"  Extents: kx ∈ [-1, 1]×π/a, ky ∈ [-3, 3]×π/b")
    
    # Calculate band energies
    energies, KX, KY = calculate_band_energies(kz=kz, resolution=resolution)
    
    # Plot Fermi surface
    fig, ax = plot_fermi_surface_2d(energies, KX, KY)
    
    print("="*60)
    print("Fermi surface calculation completed!")
    print("="*60)

if __name__ == "__main__":
    main()
