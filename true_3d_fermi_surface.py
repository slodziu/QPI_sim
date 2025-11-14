"""
True 3D Fermi surface visualization using isosurfaces in kx-ky-kz space.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional
import os

# Try to import scikit-image for marching cubes, fallback if not available
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Using alternative 3D visualization method.")


def create_true_3d_fermi_surface(model, mu: float = 0.0, k_range: float = np.pi, 
                                 resolution: int = 50, save_path: Optional[str] = None):
    """
    Create a true 3D Fermi surface visualization using isosurfaces.
    
    Args:
        model: TightBindingModel instance with get_3d_band_structure method
        mu: Chemical potential (Fermi level)
        k_range: Range of k-space to plot (-k_range to +k_range) 
        resolution: Number of points in each k-direction
        save_path: Path to save the figure (if None, shows interactively)
    """
    print(f"Generating TRUE 3D Fermi surface for {model.__class__.__name__}")
    print(f"Chemical potential μ = {mu}")
    print(f"k-space range: ±{k_range:.2f}")
    print(f"3D Resolution: {resolution}×{resolution}×{resolution}")
    
    # Create k-space grid that includes high-symmetry points
    # Use endpoint=True and odd resolution to ensure k=0 is included
    if resolution % 2 == 0:
        resolution += 1  # Make odd to include k=0 exactly
        print(f"Adjusted resolution to {resolution} (odd) to include Gamma point")
    
    kx = np.linspace(-k_range, k_range, resolution)
    ky = np.linspace(-k_range, k_range, resolution)
    kz = np.linspace(-k_range, k_range, resolution)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Get 3D band structure
    if hasattr(model, 'get_3d_band_structure'):
        eigenvalues, _ = model.get_3d_band_structure(KX, KY, KZ)
    else:
        # Fallback for models without 3D support
        print(f"Warning: {model.__class__.__name__} doesn't support 3D. Using 2D slices.")
        return create_stacked_2d_fermi_surface(model, mu, k_range, resolution, save_path)
    
    n_bands = eigenvalues.shape[0]
    
    # Create figure with subplots for different viewing angles
    fig = plt.figure(figsize=(20, 7*n_bands))
    
    for band_idx in range(n_bands):
        energy_3d = eigenvalues[band_idx]
        
        print(f"Band {band_idx+1}: Energy range [{energy_3d.min():.2f}, {energy_3d.max():.2f}]")
        
        # Check if Fermi surface exists in this band
        if energy_3d.min() <= mu <= energy_3d.max():
            print(f"Band {band_idx+1}: Fermi surface found!")
            
            # Create multiple views of the 3D Fermi surface
            for view_idx, (elev, azim, title_suffix) in enumerate([
                (20, 45, "Perspective View"),
                (0, 0, "Front View (yz-plane)"),
                (90, 0, "Top View (xy-plane)")
            ]):
                ax = fig.add_subplot(n_bands, 3, band_idx*3 + view_idx + 1, projection='3d')
                
                try:
                    if HAS_SKIMAGE:
                        # Use marching cubes to find isosurface where E(k) = μ
                        verts, faces, normals, values = measure.marching_cubes(
                            energy_3d, level=mu, spacing=(
                                kx[1]-kx[0], ky[1]-ky[0], kz[1]-kz[0]
                            )
                        )
                        
                        # Transform vertices to actual k-space coordinates
                        verts[:, 0] = verts[:, 0] + kx.min()
                        verts[:, 1] = verts[:, 1] + ky.min() 
                        verts[:, 2] = verts[:, 2] + kz.min()
                        
                        # Create 3D surface mesh
                        mesh = Poly3DCollection(verts[faces], alpha=0.7, linewidths=0.1)
                        
                        # Color based on position for better 3D visualization
                        face_centers = verts[faces].mean(axis=1)
                        colors = plt.cm.coolwarm((face_centers[:, 2] - kz.min()) / (kz.max() - kz.min()))
                        mesh.set_facecolors(colors)
                        mesh.set_edgecolors('black')
                        
                        ax.add_collection3d(mesh)
                        
                        print(f"Band {band_idx+1}: Generated {len(faces)} triangular faces")
                    else:
                        # Alternative method: Create volumetric 3D surfaces without marching cubes
                        print("Creating volumetric 3D Fermi surface...")
                        
                        # Method 1: Find isosurface points and create 3D scatter surface
                        tolerance = 0.02 * (energy_3d.max() - energy_3d.min())  # Adaptive tolerance
                        fermi_mask = np.abs(energy_3d - mu) <= tolerance
                        
                        if np.sum(fermi_mask) > 500:  # Enough points for a good surface
                            # Extract coordinates of points near Fermi surface
                            fermi_indices = np.where(fermi_mask)
                            fermi_kx = KX[fermi_indices]
                            fermi_ky = KY[fermi_indices]
                            fermi_kz = KZ[fermi_indices]
                            
                            # Create a dense 3D surface using scatter with varying opacity
                            ax.scatter(fermi_kx, fermi_ky, fermi_kz, 
                                     c=fermi_kz, cmap='plasma', 
                                     alpha=0.8, s=2, depthshade=True)
                            print(f"Band {band_idx+1}: Generated volumetric surface with {len(fermi_kx)} points")
                            
                        else:
                            # Method 2: Enhanced contour slicing with surface-like appearance
                            n_slices = 40  # More slices for smoother appearance
                            kz_slices = np.linspace(kz.min(), kz.max(), n_slices)
                            
                            # Use transparency gradient to create 3D depth effect
                            alphas = np.exp(-2 * np.abs(np.arange(n_slices) - n_slices/2) / (n_slices/4))
                            colors_slice = plt.cm.plasma(np.linspace(0, 1, n_slices))
                            
                            surface_found = False
                            for i, kz_slice in enumerate(kz_slices):
                                kz_idx = np.argmin(np.abs(kz - kz_slice))
                                energy_slice = energy_3d[:, :, kz_idx]
                                
                                if energy_slice.min() <= mu <= energy_slice.max():
                                    kx_2d, ky_2d = np.meshgrid(kx, ky)
                                    
                                    try:
                                        # Create filled contours for surface-like appearance
                                        contours = ax.contour(kx_2d, ky_2d, energy_slice, 
                                                            levels=[mu], colors=[colors_slice[i]], 
                                                            linewidths=4, alpha=alphas[i])
                                        
                                        if hasattr(contours, 'collections'):
                                            for collection in contours.collections:
                                                for path in collection.get_paths():
                                                    vertices = path.vertices
                                                    if len(vertices) > 2:  # Valid contour
                                                        # Create filled polygons to simulate surface
                                                        z_vals = np.full(len(vertices), kz_slice)
                                                        ax.plot(vertices[:, 0], vertices[:, 1], z_vals,
                                                               color=colors_slice[i], linewidth=4, 
                                                               alpha=alphas[i]*1.2, solid_capstyle='round')
                                                        surface_found = True
                                    except Exception:
                                        continue
                            
                            if surface_found:
                                print(f"Band {band_idx+1}: Generated enhanced surface with {n_slices} slices")
                            else:
                                print(f"Band {band_idx+1}: No Fermi surface found")
                    
                except Exception as e:
                    print(f"Warning: Could not generate 3D Fermi surface for band {band_idx+1}: {e}")
                    # Final fallback: show some sample contour lines
                    for kz_slice in np.linspace(kz.min(), kz.max(), 5):
                        kz_idx = np.argmin(np.abs(kz - kz_slice))
                        energy_slice = energy_3d[:, :, kz_idx]
                        kx_2d, ky_2d = np.meshgrid(kx, ky)
                        
                        if energy_slice.min() <= mu <= energy_slice.max():
                            try:
                                contours = ax.contour(kx_2d, ky_2d, energy_slice, 
                                                    levels=[mu], colors='red', alpha=0.6)
                                # Project contours to the kz_slice level
                                if hasattr(contours, 'collections'):
                                    for collection in contours.collections:
                                        for path in collection.get_paths():
                                            vertices = path.vertices
                                            if len(vertices) > 0:
                                                ax.plot(vertices[:, 0], vertices[:, 1], 
                                                       kz_slice, 'red', linewidth=2, alpha=0.8)
                            except Exception as e:
                                print(f"Warning: Fallback contour failed: {e}")
                                continue
                
                # Set equal aspect ratio and labels
                ax.set_xlim(kx.min(), kx.max())
                ax.set_ylim(ky.min(), ky.max())
                ax.set_zlim(kz.min(), kz.max())
                
                ax.set_xlabel('kₓ (1/a)', fontsize=12)
                ax.set_ylabel('kᵧ (1/a)', fontsize=12)
                ax.set_zlabel('kᵤ (1/a)', fontsize=12)
                ax.set_title(f'Band {band_idx+1}: 3D Fermi Surface\n{title_suffix} (μ={mu})', 
                           fontsize=14, fontweight='bold')
                
                # Set viewing angle
                ax.view_init(elev=elev, azim=azim)
                
                # Add wireframe for the Brillouin zone boundaries
                # Simple cubic BZ boundaries
                bz_lim = k_range
                bz_corners = np.array([
                    [-bz_lim, -bz_lim, -bz_lim], [bz_lim, -bz_lim, -bz_lim],
                    [bz_lim, bz_lim, -bz_lim], [-bz_lim, bz_lim, -bz_lim],
                    [-bz_lim, -bz_lim, bz_lim], [bz_lim, -bz_lim, bz_lim],
                    [bz_lim, bz_lim, bz_lim], [-bz_lim, bz_lim, bz_lim]
                ])
                
                # Draw BZ edges
                bz_edges = [
                    [0,1], [1,2], [2,3], [3,0],  # bottom face
                    [4,5], [5,6], [6,7], [7,4],  # top face
                    [0,4], [1,5], [2,6], [3,7]   # vertical edges
                ]
                
                for edge in bz_edges:
                    points = bz_corners[edge]
                    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.3, linewidth=1)
        
        else:
            # No Fermi surface in this band
            for view_idx in range(3):
                ax = fig.add_subplot(n_bands, 3, band_idx*3 + view_idx + 1, projection='3d')
                ax.text(0, 0, 0, f'No Fermi Surface\nμ={mu} outside\nband range\n[{energy_3d.min():.1f}, {energy_3d.max():.1f}]', 
                       ha='center', va='center', fontsize=12, color='red')
                ax.set_xlim(-k_range, k_range)
                ax.set_ylim(-k_range, k_range)  
                ax.set_zlim(-k_range, k_range)
                ax.set_xlabel('kₓ (1/a)')
                ax.set_ylabel('kᵧ (1/a)')
                ax.set_zlabel('kᵤ (1/a)')
                ax.set_title(f'Band {band_idx+1}: No Fermi Surface')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ True 3D Fermi surface saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def create_stacked_2d_fermi_surface(model, mu: float = 0.0, k_range: float = np.pi,
                                   resolution: int = 50, save_path: Optional[str] = None):
    """
    Fallback: Create stacked 2D Fermi surfaces for models without 3D support.
    """
    print(f"Creating stacked 2D Fermi surfaces for {model.__class__.__name__}")
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create several kz slices
    n_slices = 10
    kz_values = np.linspace(-k_range, k_range, n_slices)
    colors = plt.cm.viridis(np.linspace(0, 1, n_slices))
    
    kx = np.linspace(-k_range, k_range, resolution)
    ky = np.linspace(-k_range, k_range, resolution)
    KX, KY = np.meshgrid(kx, ky)
    
    for i, kz in enumerate(kz_values):
        # For models that support kz_slice parameter
        if hasattr(model, 'kz_slice'):
            model.kz_slice = kz
        
        eigenvalues, _ = model.get_band_structure(KX, KY)
        energy_surface = eigenvalues[0]
        
        if energy_surface.min() <= mu <= energy_surface.max():
            try:
                contours = ax.contour(KX, KY, energy_surface, levels=[mu], 
                                    colors=[colors[i]], linewidths=3, alpha=0.8)
                
                # Project each contour to its kz level
                if hasattr(contours, 'collections'):
                    for collection in contours.collections:
                        for path in collection.get_paths():
                            vertices = path.vertices
                            if len(vertices) > 0:
                                ax.plot(vertices[:, 0], vertices[:, 1], 
                                       np.full(len(vertices), kz), color=colors[i], linewidth=2)
            except Exception as e:
                print(f"Warning: Could not plot contour at kz={kz:.3f}: {e}")
                continue
    
    ax.set_xlabel('kₓ (1/a)', fontsize=12)
    ax.set_ylabel('kᵧ (1/a)', fontsize=12) 
    ax.set_zlabel('kᵤ (1/a)', fontsize=12)
    ax.set_title(f'Stacked 2D Fermi Surfaces\n{model.__class__.__name__} (μ={mu})', 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Stacked 2D Fermi surface saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_true_3d_fermi_surface(model, config_name: str, mu: float = 0.0):
    """
    Complete 3D Fermi surface visualization workflow.
    
    Args:
        model: TightBindingModel instance
        config_name: Name of configuration for output file naming
        mu: Chemical potential
    """
    print("="*70)
    print("TRUE 3D FERMI SURFACE VISUALIZATION")
    print("="*70)
    
    # Create outputs directory
    outputs_base = "outputs"
    config_output_dir = os.path.join(outputs_base, config_name)
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Generate output filename
    true_3d_path = os.path.join(config_output_dir, f"true_3d_fermi_surface_{config_name}.png")
    
    # Create true 3D Fermi surface plot
    create_true_3d_fermi_surface(model, mu=mu, save_path=true_3d_path, resolution=50)
    
    print("="*70)
    print("TRUE 3D FERMI SURFACE VISUALIZATION COMPLETED")
    print("="*70)
    print(f"✓ True 3D Fermi surface: {true_3d_path}")
    print("="*70)