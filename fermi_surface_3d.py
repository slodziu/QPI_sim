"""
3D Fermi surface visualization for tight-binding models.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
import os


def create_3d_fermi_surface(model, mu: float = 0.0, k_range: float = 2*np.pi, 
                           resolution: int = 100, save_path: Optional[str] = None):
    """
    Create a 3D Fermi surface visualization for a tight-binding model.
    Shows the actual Fermi surface (where E(k) = μ) as 3D isosurfaces.
    
    Args:
        model: TightBindingModel instance
        mu: Chemical potential (Fermi level)
        k_range: Range of k-space to plot (-k_range to +k_range)
        resolution: Number of points in each k-direction
        save_path: Path to save the figure (if None, shows interactively)
    """
    print(f"Generating 3D Fermi surface for {model.__class__.__name__}")
    print(f"Chemical potential μ = {mu}")
    print(f"k-space range: ±{k_range:.2f}")
    print(f"Resolution: {resolution}×{resolution}")
    
    # Create k-space grid
    kx = np.linspace(-k_range, k_range, resolution)
    ky = np.linspace(-k_range, k_range, resolution)
    KX, KY = np.meshgrid(kx, ky)
    
    # Get band structure
    eigenvalues, eigenvectors = model.get_band_structure(KX, KY)
    n_bands = eigenvalues.shape[0]
    
    # Create figure with both energy landscape and pure Fermi surface views
    fig = plt.figure(figsize=(20, 8*n_bands))
    
    for band_idx in range(n_bands):
        # Left subplot: Energy landscape with Fermi level
        ax1 = fig.add_subplot(n_bands, 2, 2*band_idx + 1, projection='3d')
        
        # Right subplot: Pure Fermi surface
        ax2 = fig.add_subplot(n_bands, 2, 2*band_idx + 2, projection='3d')
        
        # Get energy surface for this band
        energy_surface = eigenvalues[band_idx]
        
        # --- LEFT PLOT: Energy landscape with Fermi level ---
        # Plot the energy surface with transparency
        surf = ax1.plot_surface(KX, KY, energy_surface, alpha=0.4, cmap='coolwarm', 
                              linewidth=0, antialiased=True, vmin=energy_surface.min(), 
                              vmax=energy_surface.max())
        
        # Add Fermi level plane
        fermi_plane = np.full_like(KX, mu)
        ax1.plot_surface(KX, KY, fermi_plane, alpha=0.6, color='red', 
                       linewidth=0, antialiased=True)
        
        # Highlight Fermi surface contours at the intersection
        if energy_surface.min() <= mu <= energy_surface.max():
            contours = ax1.contour(KX, KY, energy_surface, levels=[mu], 
                                 colors='gold', linewidths=4, alpha=1.0)
            # Project Fermi surface onto the Fermi plane
            for collection in contours.collections:
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) > 0:
                        ax1.plot(vertices[:, 0], vertices[:, 1], mu, 'gold', linewidth=6, alpha=0.9)
        
        ax1.set_xlabel('kₓ (1/a)', fontsize=12)
        ax1.set_ylabel('kᵧ (1/a)', fontsize=12)
        ax1.set_zlabel('Energy E', fontsize=12)
        ax1.set_title(f'Band {band_idx+1}: Energy Landscape\nμ = {mu}', fontsize=14, fontweight='bold')
        ax1.view_init(elev=25, azim=45)
        
        # --- RIGHT PLOT: Pure Fermi surface ---
        # Only show the Fermi surface itself
        if energy_surface.min() <= mu <= energy_surface.max():
            # Create 3D Fermi surface using marching cubes concept
            # For 2D, we use contour lines extruded in a "pseudo-3D" way
            contours = ax2.contour(KX, KY, energy_surface, levels=[mu], 
                                 colors='gold', linewidths=3, alpha=1.0)
            
            # Create a "3D" representation by plotting multiple offset contours
            z_offsets = np.linspace(-0.2, 0.2, 5)  # Small offsets to create 3D effect
            colors = plt.cm.plasma(np.linspace(0, 1, len(z_offsets)))
            
            for i, z_offset in enumerate(z_offsets):
                mu_level = mu + z_offset
                if energy_surface.min() <= mu_level <= energy_surface.max():
                    cont = ax2.contour(KX, KY, energy_surface, levels=[mu_level], 
                                     colors=[colors[i]], linewidths=2, alpha=0.7)
            
            # Main Fermi surface in bright color
            ax2.contour(KX, KY, energy_surface, levels=[mu], 
                       colors='red', linewidths=5, alpha=1.0)
            
            # Add some visual enhancement - filled contours at Fermi level
            fermi_mask = np.abs(energy_surface - mu) < 0.1  # Thin shell around Fermi level
            kx_fermi = KX[fermi_mask]
            ky_fermi = KY[fermi_mask]
            if len(kx_fermi) > 0:
                ax2.scatter(kx_fermi, ky_fermi, mu, c='red', s=10, alpha=0.6)
        
        else:
            # No Fermi surface in this energy range
            ax2.text(0, 0, mu, f'No Fermi surface\nμ={mu} outside\nband range', 
                    ha='center', va='center', fontsize=12, color='red')
        
        ax2.set_xlabel('kₓ (1/a)', fontsize=12)
        ax2.set_ylabel('kᵧ (1/a)', fontsize=12) 
        ax2.set_zlabel('Energy E', fontsize=12)
        ax2.set_title(f'Band {band_idx+1}: Fermi Surface Only\nμ = {mu}', fontsize=14, fontweight='bold')
        ax2.view_init(elev=25, azim=45)
        
        # Set consistent z-limits for both plots
        z_margin = max(abs(energy_surface.max()), abs(energy_surface.min())) * 0.1
        ax1.set_zlim(energy_surface.min() - z_margin, energy_surface.max() + z_margin)
        ax2.set_zlim(mu - 0.5, mu + 0.5)  # Focus on Fermi level region
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ 3D Fermi surface saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def create_fermi_contour_plots(model, mu: float = 0.0, k_range: float = 2*np.pi, 
                              resolution: int = 200, save_path: Optional[str] = None):
    """
    Create 2D contour plots showing Fermi surface cross-sections with enhanced visibility.
    
    Args:
        model: TightBindingModel instance
        mu: Chemical potential (Fermi level)
        k_range: Range of k-space to plot (-k_range to +k_range)
        resolution: Number of points in each k-direction
        save_path: Path to save the figure (if None, shows interactively)
    """
    print(f"Generating 2D Fermi contour plots for {model.__class__.__name__}")
    
    # Create k-space grid
    kx = np.linspace(-k_range, k_range, resolution)
    ky = np.linspace(-k_range, k_range, resolution)
    KX, KY = np.meshgrid(kx, ky)
    
    # Get band structure
    eigenvalues, eigenvectors = model.get_band_structure(KX, KY)
    n_bands = eigenvalues.shape[0]
    
    # Create figure with subplots for each band
    fig, axes = plt.subplots(1, n_bands, figsize=(8*n_bands, 6))
    if n_bands == 1:
        axes = [axes]
    
    for band_idx, ax in enumerate(axes):
        energy_surface = eigenvalues[band_idx]
        
        # Create filled contour plot with more levels for better detail
        n_levels = 30
        levels = np.linspace(energy_surface.min(), energy_surface.max(), n_levels)
        contourf = ax.contourf(KX, KY, energy_surface, levels=levels, cmap='coolwarm', alpha=0.8)
        
        # Add fine energy contour lines
        fine_levels = np.linspace(energy_surface.min(), energy_surface.max(), 15)
        ax.contour(KX, KY, energy_surface, levels=fine_levels, 
                  colors='black', linewidths=0.5, alpha=0.3)
        
        # Add Fermi surface contour with very thick line
        if energy_surface.min() <= mu <= energy_surface.max():
            fermi_contour = ax.contour(KX, KY, energy_surface, levels=[mu], 
                                      colors='gold', linewidths=4, linestyles='-')
            ax.clabel(fermi_contour, fmt=f'μ={mu}', fontsize=12, inline=True, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Add multiple nearby contours to emphasize the Fermi surface region
            nearby_levels = [mu - 0.1, mu + 0.1]
            for level in nearby_levels:
                if energy_surface.min() <= level <= energy_surface.max():
                    ax.contour(KX, KY, energy_surface, levels=[level], 
                              colors='orange', linewidths=2, alpha=0.7, linestyles='--')
        else:
            # Indicate no Fermi surface in this band
            ax.text(0, 0, f'No Fermi surface\nμ={mu} outside band\n[{energy_surface.min():.1f}, {energy_surface.max():.1f}]', 
                   ha='center', va='center', fontsize=12, color='red', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        # Labels and formatting
        ax.set_xlabel('kₓ (1/a)', fontsize=14)
        ax.set_ylabel('kᵧ (1/a)', fontsize=14)
        ax.set_title(f'Band {band_idx+1}: Fermi Surface\nμ = {mu}, E ∈ [{energy_surface.min():.1f}, {energy_surface.max():.1f}]', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar with better positioning
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.8, aspect=30, label='Energy E')
        cbar.ax.tick_params(labelsize=12)
        
        # Highlight Fermi level on colorbar
        if energy_surface.min() <= mu <= energy_surface.max():
            cbar.ax.axhline(y=mu, color='gold', linewidth=3, alpha=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ 2D Fermi contour plots saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_tight_binding_fermi_surface(model, config_name: str, mu: float = 0.0):
    """
    Complete Fermi surface visualization workflow for tight-binding models.
    
    Args:
        model: TightBindingModel instance
        config_name: Name of configuration for output file naming
        mu: Chemical potential
    """
    print("="*70)
    print("TIGHT-BINDING FERMI SURFACE VISUALIZATION")
    print("="*70)
    
    # Create outputs directory
    outputs_base = "outputs"
    config_output_dir = os.path.join(outputs_base, config_name)
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Generate output filenames
    fermi_3d_path = os.path.join(config_output_dir, f"fermi_surface_3d_{config_name}.png")
    fermi_2d_path = os.path.join(config_output_dir, f"fermi_contours_2d_{config_name}.png")
    
    # Create 3D Fermi surface plot
    create_3d_fermi_surface(model, mu=mu, save_path=fermi_3d_path)
    
    # Create 2D contour plots  
    create_fermi_contour_plots(model, mu=mu, save_path=fermi_2d_path)
    
    print("="*70)
    print("FERMI SURFACE VISUALIZATION COMPLETED")
    print("="*70)
    print(f"✓ 3D Fermi surface: {fermi_3d_path}")
    print(f"✓ 2D contour plots: {fermi_2d_path}")
    print("="*70)