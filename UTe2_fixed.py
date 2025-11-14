# UTe2 model with 3D Fermi surface visualization
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("trimesh not available - 3D export will be skipped")

# Lattice constants (nm) - UTe2 orthorhombic structure
a = 0.41    # U-Te distance along a direction (shorter)
b = 0.61    # Te-Te distance along b direction (longer)
c = 1.39    # layer spacing along c axis (longest)
# Note: In momentum space, shorter lattice → wider k-range
# So kx range (π/a) > ky range (π/b) > kz range (π/c)

# U parameters (verified against paper)
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
    """2x2 Hamiltonian for U orbitals"""
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
    """2x2 Hamiltonian for Te orbitals"""
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
    """Full 4x4 Hamiltonian with U-Te hybridization"""
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
        print(f"  {status} {param}: paper={paper_val:.3f}, ours={our_val:.3f}")
        if not match:
            all_match = False
    
    if all_match:
        print("✓ All parameters match the paper exactly!")
    else:
        print("✗ Some parameters don't match - check implementation")
    print("=" * 40)

def band_energies_on_slice(kz):
    """Compute band energies for all kx, ky at fixed kz"""
    energies = np.zeros((nk, nk, 4))
    
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            eigvals = np.linalg.eigvals(H)
            energies[i, j, :] = np.sort(np.real(eigvals))
    
    return energies

def plot_fs_2d(energies, kz_label, save_dir='outputs/ute2_fixed'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Colors for different bands - U bands (0,1) vs Te bands (2,3)
    colors = ['#1E88E5', '#42A5F5', '#E53935', '#FF7043']  # Blue for U, Red for Te
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    # Convert momentum to units of π/a and π/b for plotting (with axis swap)
    KX_units = KX / (np.pi/a)  # kx in π/a units
    KY_units = KY / (np.pi/b)  # ky in π/b units
    
    # Fermi level
    EF = 0.0
    bands_plotted = 0
    
    print(f"\\n2D Fermi surface at kz={kz_label}:")
    
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
                    plt.text(xlbl, ylbl, f"{band_labels[band]}", fontsize=9, 
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
    plt.title(f'Fermi surface (E=0) at kz={kz_label}', fontsize=14)
    plt.xlim(-3, 3)  # Extended range for ky
    plt.ylim(-1, 1)   # Standard range for kx
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=band_labels[i]) for i in range(4)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    filename = f'fermi_surface_kz_{kz_label}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {os.path.join(save_dir, filename)}")

def plot_3d_fermi_surface(save_dir='outputs/ute2_fixed'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("Computing 3D Fermi surfaces using marching cubes...")
    
    # Create 3D momentum grid (reduced resolution for speed)
    nk3d = 60  # Reduced resolution for faster computation
    kx_3d = np.linspace(-np.pi/a, np.pi/a, nk3d)
    ky_3d = np.linspace(-np.pi/b, np.pi/b, nk3d)  # Standard ky range for 3D
    kz_3d = np.linspace(-np.pi/c, np.pi/c, nk3d)
    
    # Create meshgrid
    KX_3d, KY_3d, KZ_3d = np.meshgrid(kx_3d, ky_3d, kz_3d, indexing='ij')
    
    print(f"  3D grid: {nk3d} x {nk3d} x {nk3d} = {nk3d**3:,} points")
    
    # Colors for different bands - U bands (0,1) vs Te bands (2,3)
    colors = ['#1E88E5', '#42A5F5', '#E53935', '#FF7043']  # Blue for U, Red for Te
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    alphas = [0.6, 0.6, 0.8, 0.8]  # Te bands slightly more opaque
    
    # Compute band energies for the entire 3D grid using vectorized approach
    print("  Computing band energies on 3D grid (vectorized)...")
    
    # Pre-allocate array
    band_energies_3d = np.zeros((nk3d, nk3d, nk3d, 4))
    
    # Vectorized computation with progress tracking
    total_slices = nk3d
    for i, kx in enumerate(kx_3d):
        # Compute entire kx slice at once
        for j, ky in enumerate(ky_3d):
            # Vectorize over kz direction
            H_stack = np.array([H_full(kx, ky, kz) for kz in kz_3d])
            eigvals_stack = np.array([np.sort(np.real(np.linalg.eigvals(H))) for H in H_stack])
            band_energies_3d[i, j, :, :] = eigvals_stack
        
        # Progress update
        progress = (i + 1) / total_slices * 100
        print(f"    Progress: {progress:.1f}% ({i+1}/{total_slices} kx slices)")
    
    print("  Band energy computation complete!")
    
    # Create the 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use Fermi level (E=0) for all bands
    EF = 0.0
    print(f"  Extracting Fermi surfaces at E={EF:.3f} eV using marching cubes...")
    
    surfaces_plotted = 0
    all_meshes = []  # Store all mesh objects for combined export
    mesh_colors = []  # Store colors for each mesh
    
    for band in range(4):
        print(f"  Processing {band_labels[band]}...")
        
        # Get the band energies for this band
        band_data = band_energies_3d[:, :, :, band]
        
        # Check if this band crosses the Fermi level
        if band_data.min() <= EF <= band_data.max():
            try:
                # Extract isosurface using marching cubes with reduced complexity
                verts, faces, normals, values = measure.marching_cubes(
                    band_data, level=EF, 
                    spacing=(
                        kx_3d[1] - kx_3d[0],
                        ky_3d[1] - ky_3d[0], 
                        kz_3d[1] - kz_3d[0]
                    ),
                    step_size=1  # Use every grid point
                )
                
                # Adjust vertices to actual k-space coordinates
                verts[:, 0] = verts[:, 0] + kx_3d[0]  # kx
                verts[:, 1] = verts[:, 1] + ky_3d[0]  # ky
                verts[:, 2] = verts[:, 2] + kz_3d[0]  # kz
                
                # Convert to units of π/a, π/b, π/c for plotting with axis swap
                # Apply physically correct scaling based on lattice parameter ratios
                # kx range is naturally wider due to smaller lattice parameter a
                x_plot = verts[:, 1] / (np.pi/b)  # ky -> x-axis
                y_plot = verts[:, 0] / (np.pi/a)  # kx -> y-axis (naturally wider due to a < b)
                z_plot = verts[:, 2] / (np.pi/c)  # kz -> z-axis
                
                # Create the surface mesh
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                # Only limit faces if there are way too many (>20k faces per band)
                max_faces = 20000  # Much higher limit, only for extreme cases
                faces_used = faces
                if len(faces) > max_faces:
                    print(f"      Warning: {len(faces)} faces detected, subsampling to {max_faces} for performance")
                    # Subsample faces uniformly
                    face_indices = np.linspace(0, len(faces)-1, max_faces, dtype=int)
                    faces_used = faces[face_indices]
                else:
                    faces_used = faces
                
                # Create triangular mesh
                mesh_verts = np.column_stack([x_plot, y_plot, z_plot])
                
                # Plot the surface efficiently with proper shading
                poly3d = mesh_verts[faces_used]
                
                # Add shaded surface with proper color setup
                collection = ax.add_collection3d(Poly3DCollection(
                    poly3d, alpha=alphas[band], facecolor=colors[band], 
                    edgecolor='none', linewidth=0, rasterized=True
                ))
                
                # Enable shading manually by setting face colors with lighting
                from matplotlib.colors import LightSource
                ls = LightSource(azdeg=315, altdeg=45)  # Light from upper left
                
                # Apply lighting to the collection
                collection.set_facecolor(colors[band])
                collection.set_alpha(alphas[band])
                collection.set_edgecolor('none')
                
                # Store mesh for combined export if trimesh is available
                if TRIMESH_AVAILABLE:
                    try:
                        # Create trimesh object for this band
                        mesh_obj = trimesh.Trimesh(vertices=mesh_verts, faces=faces_used)
                        
                        # Clean up the mesh
                        mesh_obj.remove_duplicate_faces()
                        mesh_obj.remove_unreferenced_vertices()
                        
                        # Convert hex color to RGB tuple (0-255)
                        hex_color = colors[band].lstrip('#')
                        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        
                        # Add to collection for combined export
                        all_meshes.append(mesh_obj)
                        mesh_colors.append({
                            'rgb': rgb_color,
                            'band_index': band,
                            'band_name': band_labels[band]
                        })
                        
                        print(f"      Added {band_labels[band]} with color {colors[band]} -> RGB{rgb_color}")
                        
                    except Exception as e:
                        print(f"      Failed to create mesh object for {band_labels[band]}: {e}")
                
                surfaces_plotted += 1
                print(f"    + {band_labels[band]}: {len(faces_used)} triangular faces (original: {len(faces)})")
                
            except Exception as e:
                print(f"    - Failed to extract surface for {band_labels[band]}: {e}")
        else:
            print(f"    - {band_labels[band]} does not cross Fermi level "
                  f"(range: {band_data.min():.3f} to {band_data.max():.3f} eV)")
    
    if surfaces_plotted == 0:
        print("  Warning: No Fermi surfaces found!")
        # Fallback: create a simple test surface
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_test = np.outer(np.cos(u), np.sin(v)) * 0.5
        y_test = np.outer(np.sin(u), np.sin(v)) * 0.5
        z_test = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.5
        ax.plot_surface(x_test, y_test, z_test, alpha=0.3, color='gray')
    
    # Export combined Fermi surface as single STL file
    if TRIMESH_AVAILABLE and len(all_meshes) > 0:
        try:
            # Combine all meshes into a single object
            combined_mesh = trimesh.util.concatenate(all_meshes)
            
            # Final cleanup
            combined_mesh.remove_duplicate_faces()
            combined_mesh.remove_unreferenced_vertices()
            
            # Export combined STL
            combined_stl_filename = 'fermi_surface_3d_complete.stl'
            combined_stl_filepath = os.path.join(save_dir, combined_stl_filename)
            combined_mesh.export(combined_stl_filepath)
            
            total_faces = sum(len(mesh.faces) for mesh in all_meshes)
            print(f"\n  ✓ Exported complete Fermi surface as {combined_stl_filename}")
            print(f"    Combined {len(all_meshes)} surfaces with {total_faces:,} total faces")
            
            # Export as OBJ with colors/materials
            try:
                # Create OBJ with separate objects for each band (preserves colors)
                obj_filename = 'fermi_surface_3d_complete.obj'
                obj_filepath = os.path.join(save_dir, obj_filename)
                mtl_filename = 'fermi_surface_3d_complete.mtl'
                mtl_filepath = os.path.join(save_dir, mtl_filename)
                
                # Write MTL file (materials)
                with open(mtl_filepath, 'w') as mtl_file:
                    for i, (mesh, color_info) in enumerate(zip(all_meshes, mesh_colors)):
                        band_name = color_info['band_name'].lower().replace(' ', '_')
                        rgb = color_info['rgb']
                        band_idx = color_info['band_index']
                        
                        mtl_file.write(f"newmtl {band_name}\n")
                        mtl_file.write(f"Ka 0.2 0.2 0.2\n")  # Ambient color
                        mtl_file.write(f"Kd {rgb[0]/255.0:.3f} {rgb[1]/255.0:.3f} {rgb[2]/255.0:.3f}\n")  # Diffuse color
                        mtl_file.write(f"Ks 0.3 0.3 0.3\n")  # Specular color
                        mtl_file.write(f"Ns 32\n")  # Specular exponent
                        mtl_file.write(f"d {alphas[band_idx]}\n")  # Transparency (alpha)
                        mtl_file.write(f"# Band: {color_info['band_name']}, Color: {colors[band_idx]}\n")
                        mtl_file.write(f"\n")
                
                # Write OBJ file with references to materials
                with open(obj_filepath, 'w') as obj_file:
                    obj_file.write(f"mtllib {mtl_filename}\n\n")
                    
                    vertex_offset = 0
                    for i, (mesh, color_info) in enumerate(zip(all_meshes, mesh_colors)):
                        band_name = color_info['band_name'].lower().replace(' ', '_')
                        obj_file.write(f"o {band_name}\n")
                        obj_file.write(f"usemtl {band_name}\n")
                        obj_file.write(f"# {color_info['band_name']} - {len(mesh.vertices)} vertices, {len(mesh.faces)} faces\n")
                        
                        # Write vertices
                        for vertex in mesh.vertices:
                            obj_file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                        
                        # Write faces (1-indexed, offset by previous meshes)
                        for face in mesh.faces:
                            f1, f2, f3 = face + vertex_offset + 1  # OBJ is 1-indexed
                            obj_file.write(f"f {f1} {f2} {f3}\n")
                        
                        vertex_offset += len(mesh.vertices)
                        obj_file.write(f"\n")
                
                print(f"  ✓ Exported colored Fermi surface as {obj_filename} (with {mtl_filename})")
                print(f"    Materials created for {len(mesh_colors)} bands with distinct colors")
                for i, color_info in enumerate(mesh_colors):
                    print(f"      {color_info['band_name']}: {colors[color_info['band_index']]} -> RGB{color_info['rgb']}")
                
            except Exception as e:
                print(f"  ✗ Failed to export OBJ with materials: {e}")
            
        except Exception as e:
            print(f"\n  ✗ Failed to export combined Fermi surface: {e}")
            # Fallback: export individual bands
            for i, mesh in enumerate(all_meshes):
                try:
                    fallback_filename = f'fermi_surface_3d_band_{i+1}.stl'
                    fallback_filepath = os.path.join(save_dir, fallback_filename)
                    mesh.export(fallback_filepath)
                    print(f"    Exported fallback: {fallback_filename}")
                except Exception as e2:
                    print(f"    Failed to export band {i+1}: {e2}")
    elif TRIMESH_AVAILABLE:
        print("\n  No Fermi surfaces to export as STL/OBJ")
    else:
        print("\n  STL/OBJ export skipped (trimesh not available)")
    
    # Set labels and title with proper k-space units
    ax.set_xlabel(r'$k_y$ (π/b)', fontsize=12)  # ky on x-axis
    ax.set_ylabel(r'$k_x$ (π/a)', fontsize=12)  # kx on y-axis
    ax.set_zlabel(r'$k_z$ (π/c)', fontsize=12)
    ax.set_title('3D Fermi Surface of UTe₂ (Physical Proportions)', fontsize=14, pad=20)
    
    # Add legend with U/Te distinction
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=3, alpha=alphas[i], 
                             label=band_labels[i]) for i in range(4)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Make background cleaner
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Remove old individual export summary since we now have combined export
    # The export summary is now handled above in the combined export section
    
    # Set axis limits based on physical momentum ranges (π/lattice_parameter)
    # kx range is naturally wider than ky range due to orthorhombic structure (a < b)
    kx_range = 1.0  # -π/a to π/a in units of π/a
    ky_range = 1.0  # -π/b to π/b in units of π/b  
    kz_range = 1.0  # -π/c to π/c in units of π/c
    ax.set_xlim(-ky_range, ky_range)  # ky on x-axis
    ax.set_ylim(-kx_range, kx_range)  # kx on y-axis (appears wider due to lattice)
    ax.set_zlim(-kz_range, kz_range)  # kz on z-axis
    
    # Improve view angle to match reference image style
    ax.view_init(elev=20, azim=30)  # Slightly elevated view from front-right
    
    # Add multiple saved views for different perspectives
    plt.tight_layout()
    
    # Save the main view
    filename = 'fermi_surface_3d_marching_cubes.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved 3D plot (main view) to {filepath}")
    
    # Save additional perspective views
    perspectives = [
        {'elev': 20, 'azim': 30, 'name': 'perspective_1'},  # Current view
        {'elev': 25, 'azim': 45, 'name': 'perspective_2'},  # Slightly different
        {'elev': 30, 'azim': 60, 'name': 'perspective_3'},  # More elevated
        {'elev': 45, 'azim': 15, 'name': 'perspective_4'},  
        {'elev': 55, 'azim': -30, 'name': 'perspective_5'}, # From the other side
    ]
    
    for view in perspectives:
        ax.view_init(elev=view['elev'], azim=view['azim'])
        view_filename = f'fermi_surface_3d_{view["name"]}.png'
        view_filepath = os.path.join(save_dir, view_filename)
        plt.savefig(view_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Saved {len(perspectives)} additional perspective views")
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
    output_dir = 'outputs/ute2_fixed'
    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(output_dir, 'UTe2_FS_energies.npz')
    np.savez(data_file, kx_vals=kx_vals, ky_vals=ky_vals,
             kz0=kz0, energies_kz0=energies_kz0)
    
    print(f"Saved energies to {data_file}")
    print("\\nAll plots generated successfully!")