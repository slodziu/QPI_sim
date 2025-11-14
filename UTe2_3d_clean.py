# UTe2 model with 3D Fermi surface visualization
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from scipy.interpolate import griddata
except ImportError:
    print("Warning: scipy not available, will use simpler surface plotting")

# Lattice constants (nm)
a = 0.41
b = 0.61
c = 1.39

# Parameters (eV) from the paper
muU = -0.355
DeltaU = 0.38
tU = 0.17
tpU = 0.08
tch_U = 0.015
tpch_U = 0.01
tz_U = -0.0375

# Te parameters (updated to match paper exactly)
muTe = -2.25
DeltaTe = -1.4
tTe = -1.5  # hopping along Te(2) chain in b direction
tch_Te = 0  # hopping between chains in a direction (paper shows 0)
tz_Te = -0.05  # hopping between chains along c axis
delta = 0.13

# momentum grid for kx-ky plane (in nm^-1, will be converted to π/a, π/b units for plotting)
nk = 201  # Reduced for faster computation
# Extend ky range to cover -3π/b to 3π/b
kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, nk)  # Extended range
KX, KY = np.meshgrid(kx_vals, ky_vals)

def HU_block(kx, ky, kz):
    diag = muU - 2*tU*np.cos(kx*a) - 2*tch_U*np.cos(ky*b)
    real_off = -DeltaU - 2*tpU*np.cos(kx*a) - 2*tpch_U*np.cos(ky*b)
    complex_amp = -4 * tz_U * np.exp(-1j * kz * c / 2) * np.cos(kx * a / 2) * np.cos(ky * b / 2)
    H = np.zeros((2,2), dtype=complex)
    H[0,0] = diag
    H[1,1] = diag
    H[0,1] = real_off + complex_amp
    H[1,0] = real_off + np.conj(complex_amp)
    return H

def HTe_block(kx, ky, kz):
    # Diagonal elements: μTe (no chain hopping since tch_Te = 0 in paper)
    diag = muTe
    # Off-diagonal elements from paper:
    # -ΔTe - tTe*exp(-iky*b) - tz_Te*cos(kz*c/2)*cos(kx*a/2)*cos(ky*b/2)
    real_off = -DeltaTe
    complex_term1 = -tTe * np.exp(-1j * ky * b)  # hopping along b direction
    complex_term2 = -tz_Te * np.cos(kz * c / 2) * np.cos(kx * a / 2) * np.cos(ky * b / 2)
    
    H = np.zeros((2,2), dtype=complex)
    H[0,0] = diag
    H[1,1] = diag
    H[0,1] = real_off + complex_term1 + complex_term2
    H[1,0] = real_off + np.conj(complex_term1) + complex_term2  # ensures Hermiticity
    return H

def H_full(kx, ky, kz):
    HU = HU_block(kx, ky, kz)
    HTe = HTe_block(kx, ky, kz)
    Hhyb = np.eye(2) * delta
    top = np.hstack((HU, Hhyb))
    bottom = np.hstack((Hhyb.conj().T, HTe))
    H = np.vstack((top, bottom))
    return H

def verify_model_parameters():
    print("\\n=== Model Verification ===")
    print("Checking parameters against paper values...")
    
    # Paper values (from the Methods section)
    paper_params = {
        'μU': -0.355, 'ΔU': 0.38, 'tU': 0.17, "t'U": 0.08, 
        'tch,U': 0.015, "t'ch,U": 0.01, 'tz,U': -0.0375,
        'μTe': -2.25, 'ΔTe': -1.4, 'tTe': -1.5, 'tch,Te': 0, 'tz,Te': -0.05,
        'δ': 0.13
    }
    
    # Our values
    our_params = {
        'μU': muU, 'ΔU': DeltaU, 'tU': tU, "t'U": tpU,
        'tch,U': tch_U, "t'ch,U": tpch_U, 'tz,U': tz_U,
        'μTe': muTe, 'ΔTe': DeltaTe, 'tTe': tTe, 'tch,Te': tch_Te, 'tz,Te': tz_Te,
        'δ': delta
    }
    
    all_match = True
    for param, paper_val in paper_params.items():
        our_val = our_params[param]
        match = abs(our_val - paper_val) < 1e-6
        status = "✓" if match else "✗"
        print(f"  {param:8}: Paper={paper_val:8.4f}, Ours={our_val:8.4f} {status}")
        if not match:
            all_match = False
    
    print(f"\\nHybridization matrix H_U-Te: δ * I = {delta} * I")
    Hhyb = np.eye(2) * delta
    print("H_U-Te =")
    print(Hhyb)
    
    print(f"\\nOverall parameter match: {'✓ ALL CORRECT' if all_match else '✗ SOME MISMATCHES'}")
    print("=========================\\n")
    
    return all_match

def band_energies_on_slice(kz):
    nk = kx_vals.size
    energies = np.zeros((nk, nk, 4))
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            eigs = np.linalg.eigvalsh(H)
            energies[j, i, :] = np.sort(eigs)
    return energies

def plot_fs_2d(energies, kz_label, save_dir='outputs/ute2_updated_hamiltonian'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Colors for U and Te bands
    colors = ['#1E88E5', '#42A5F5', '#E53935', '#FF7043']  # Blue for U, Red for Te
    band_labels = ['U1', 'U2', 'Te1', 'Te2']
    
    # Convert coordinate grids to π/a, π/b units and SWAP kx <-> ky
    KY_units = KX / (np.pi/a)  # Now showing ky on x-axis
    KX_units = KY / (np.pi/b)  # Now showing kx on y-axis
    
    # Check energy ranges for each band
    print(f"\nEnergy ranges at kz={kz_label/(np.pi/c):.2f}π/c:")
    for band in range(4):
        Z = energies[:, :, band].T
        print(f"  {band_labels[band]}: {Z.min():.3f} to {Z.max():.3f} eV")
    
    plt.figure(figsize=(8,6))
    
    # Plot all bands at Fermi level (E=0) only
    EF = 0.0
    bands_plotted = 0
    
    for band in range(4):
        Z = energies[:, :, band].T
        
        # Only plot if band crosses the Fermi level
        if Z.min() <= EF <= Z.max():
            cs = plt.contour(KY_units, KX_units, Z, levels=[EF], 
                           linewidths=2.5, colors=[colors[band]])
            if len(cs.allsegs[0]) > 0:
                seg = cs.allsegs[0][0]
                if len(seg) > 0:
                    mid = seg.shape[0]//2
                    xlbl, ylbl = seg[mid,0], seg[mid,1]
                    plt.text(xlbl, ylbl, f"{band_labels[band]} (0.00eV)", fontsize=9, 
                           color=colors[band], weight='bold', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                    bands_plotted += 1
                    print(f"  Plotted {band_labels[band]} Fermi surface at E=0.000 eV")
        else:
            print(f"  {band_labels[band]} does not cross Fermi level (range: {Z.min():.3f} to {Z.max():.3f} eV)")
    
    if bands_plotted == 0:
        print("  Warning: No bands cross the Fermi level at this kz!")
    
    plt.xlabel(r'$k_y$ (π/b)', fontsize=12)  # ky on x-axis
    plt.ylabel(r'$k_x$ (π/a)', fontsize=12)  # kx on y-axis
    plt.title(f'Fermi surface (E=0) at kz={kz_label/(np.pi/c):.2f}π/c', fontsize=14)
    plt.xlim(-3, 3)  # Extended range for ky
    plt.ylim(-1, 1)   # Standard range for kx
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=band_labels[i]) for i in range(4)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    filename = f'fermi_surface_kz_{kz_label:.3f}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {os.path.join(save_dir, filename)}")

def plot_3d_fermi_surface(save_dir='outputs/ute2_updated_hamiltonian'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Create multiple kz slices for 3D plot
    nkz = 50  # Number of kz slices
    kz_vals = np.linspace(-np.pi/c, np.pi/c, nkz)
    
    print("Computing Fermi surfaces for 3D plot...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different bands - U bands (0,1) vs Te bands (2,3)
    colors = ['#1E88E5', '#42A5F5', '#E53935', '#FF7043']  # Blue for U, Red for Te
    band_labels = ['U Band 1', 'U Band 2', 'Te Band 1', 'Te Band 2']
    
    # Use Fermi level (E=0) for all bands
    EF = 0.0
    print(f"  3D plot will show all bands at Fermi level E={EF:.3f} eV")
    
    # Store contour data for surface creation
    all_surfaces = {band: [] for band in range(4)}
    
    for i, kz in enumerate(kz_vals):
        energies = band_energies_on_slice(kz)
        
        for band in range(4):
            Z = energies[:, :, band].T
            
            # Create a separate figure for contour extraction
            fig_temp, ax_temp = plt.subplots()
            cs = ax_temp.contour(KX, KY, Z, levels=[EF])
            
            # Extract contour paths and filter out edge artifacts
            if hasattr(cs, 'allsegs') and len(cs.allsegs) > 0:
                for path_collection in cs.allsegs[0]:
                    if len(path_collection) > 10:  # Need enough points
                        vertices = path_collection
                        
                        # Filter out contours that touch the edges (likely artifacts)
                        x_vals = vertices[:, 0] / (np.pi/a)  # Convert to π/a units
                        y_vals = vertices[:, 1] / (np.pi/b)  # Convert to π/b units
                        
                        # Check if contour touches boundaries (with small margin)
                        margin = 0.05
                        touches_edge = (
                            np.any(np.abs(x_vals) > (1 - margin)) or  # kx edges
                            np.any(np.abs(y_vals) > (3 - margin))     # ky edges
                        )
                        
                        if not touches_edge and len(vertices) > 15:
                            # Close the contour if needed
                            if not np.allclose(vertices[0], vertices[-1]):
                                vertices = np.vstack([vertices, vertices[0:1]])
                            
                            # Convert to proper units with axis swap
                            y_coords = vertices[:, 0] / (np.pi/a)  # kx data -> y-axis
                            x_coords = vertices[:, 1] / (np.pi/b)  # ky data -> x-axis
                            z_coords = np.full_like(x_coords, kz / (np.pi/c))
                            
                            # Store surface data
                            surface_data = np.column_stack([x_coords, y_coords, z_coords])
                            all_surfaces[band].append(surface_data)
            
            plt.close(fig_temp)  # Close the temporary figure
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{nkz} kz slices")
    
    # Create surfaces from collected data
    for band in range(4):
        if len(all_surfaces[band]) > 3:  # Need multiple slices for surface
            print(f"  Creating surface for {band_labels[band]}...")
            
            # Combine all contour data for this band
            try:
                all_points = np.vstack(all_surfaces[band])
                
                if len(all_points) > 50:  # Need enough points for surface
                    # Create triangulated surfaces between adjacent z-levels
                    for j in range(len(all_surfaces[band]) - 1):
                        if len(all_surfaces[band][j]) > 10 and len(all_surfaces[band][j+1]) > 10:
                            # Get two adjacent contours
                            contour1 = all_surfaces[band][j]
                            contour2 = all_surfaces[band][j+1]
                            
                            # Create surface between them
                            min_len = min(len(contour1), len(contour2))
                            if min_len > 10:
                                # Resample both contours to same length
                                indices1 = np.linspace(0, len(contour1)-1, min_len).astype(int)
                                indices2 = np.linspace(0, len(contour2)-1, min_len).astype(int)
                                
                                c1_sampled = contour1[indices1]
                                c2_sampled = contour2[indices2]
                                
                                # Create surface strips
                                for k in range(min_len - 1):
                                    # Create quadrilateral surface patch
                                    vertices = [
                                        c1_sampled[k],
                                        c1_sampled[k+1], 
                                        c2_sampled[k+1],
                                        c2_sampled[k]
                                    ]
                                    
                                    # Plot as triangulated surface
                                    x_surf = [v[0] for v in vertices] + [vertices[0][0]]
                                    y_surf = [v[1] for v in vertices] + [vertices[0][1]]
                                    z_surf = [v[2] for v in vertices] + [vertices[0][2]]
                                    
                                    ax.plot_trisurf(x_surf[:-1], y_surf[:-1], z_surf[:-1],
                                                  color=colors[band], alpha=0.7, 
                                                  linewidth=0.1, antialiased=True)
            except Exception as e:
                print(f"    Surface creation failed for {band_labels[band]}, using fallback lines")
                # Fallback to line plot
                for surface in all_surfaces[band][:8]:  # Plot some contours as lines
                    if len(surface) > 5:
                        ax.plot(surface[:, 0], surface[:, 1], surface[:, 2], 
                               color=colors[band], alpha=0.8, linewidth=1.0)
    
    # Set labels and title with proper k-space units
    ax.set_xlabel(r'$k_y$ (π/b)', fontsize=12)  # ky on x-axis
    ax.set_ylabel(r'$k_x$ (π/a)', fontsize=12)  # kx on y-axis
    ax.set_zlabel(r'$k_z$ (π/c)', fontsize=12)
    ax.set_title('3D Fermi Surface of UTe₂', fontsize=14, pad=20)
    
    # Set axis limits in units of π/a, π/b, π/c
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    # Improve view angle
    ax.view_init(elev=15, azim=45)
    
    # Add legend with U/Te distinction
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=3, alpha=0.8, 
                             label=band_labels[i]) for i in range(4)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Make background cleaner
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.2)
    ax.yaxis.pane.set_alpha(0.2)
    ax.zaxis.pane.set_alpha(0.2)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'fermi_surface_3d.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved 3D plot to {filepath}")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Verify model parameters first
    verify_model_parameters()
    
    # Compute for kz=0 slice only
    print("Computing 2D Fermi surface at kz=0...")
    kz0 = 0.0
    energies_kz0 = band_energies_on_slice(kz0)
    
    # Generate only kz=0 2D plot
    plot_fs_2d(energies_kz0, kz0)
    
    # Generate 3D plot
    plot_3d_fermi_surface()
    
    # Save energies to file
    import os
    output_dir = 'outputs/ute2_updated_hamiltonian'
    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(output_dir, 'UTe2_FS_energies.npz')
    np.savez(data_file, kx_vals=kx_vals, ky_vals=ky_vals,
             kz0=kz0, energies_kz0=energies_kz0)
    
    print(f"Saved energies to {data_file}")
    print("\nAll plots generated successfully!")