#!/usr/bin/env python3
"""
Create a side-by-side comparison of 3D Fermi surface plots:
- Left: No gap nodes (alt orientation, perspective 3)
- Center: B2u gap nodes (alt orientation, perspective 3) 
- Right: B3u gap nodes (alt orientation, perspective 3)
Total width constrained to 8 inches.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
import hashlib
from contextlib import redirect_stdout
from io import StringIO

def create_3d_comparison(save_dir='outputs/ute2_fixed'):
    """Create side-by-side 3D Fermi surface comparison plot."""
    
    print("Creating 3D Fermi surface comparison plot...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Import what we need from UTe2_fixed
    from UTe2_fixed import (a, b, c, calculate_gap_magnitude)
    import pickle
    
    # Load cached 3D data using the existing cache file
    cache_dir = os.path.join('outputs', 'global_cache')
    cache_file = os.path.join(cache_dir, 'band_energies_3d_40d54448.pkl')  # Use known cache file
    
    if os.path.exists(cache_file):
        print(f"  Loading cached band energies from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            band_energies_3d = cache_data['band_energies_3d']
            kx_3d = cache_data['kx_3d']
            ky_3d = cache_data['ky_3d']
            kz_3d = cache_data['kz_3d']
            print(f"   Successfully loaded cached 3D band energies ({band_energies_3d.shape})")
        except Exception as e:
            print(f"   Failed to load cache file: {e}")
            return
    else:
        print("  No cached data found. Run UTe2_fixed.py first to generate 3D data.")
        return
    
    # Create our comparison plot - make it taller, less wide
    fig = plt.figure(figsize=(9, 6), dpi=300)
    
    # Configuration for each subplot
    configs = [
        {'title': 'B1u', 'show_gap_nodes': False, 'gap_node_pairing': None},
        {'title': 'B2u', 'show_gap_nodes': True, 'gap_node_pairing': 'B2u'},
        {'title': 'B3u', 'show_gap_nodes': True, 'gap_node_pairing': 'B3u'}
    ]
    
    # Colors and settings (copy from UTe2_fixed)
    colors = ['#1E88E5', '#42A5F5', "#9635E5", "#FD0000"]
    alphas = [0.6, 0.6, 0.8, 0.8]
    EF = 0.0
    
    # Create each subplot
    for i, config in enumerate(configs):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        print(f"  Creating subplot {i+1}: {config['title']}")
        
        # Plot Fermi surfaces
        for band in range(4):
            band_data = band_energies_3d[:, :, :, band]
            
            if band_data.min() <= EF <= band_data.max():
                try:
                    from skimage import measure
                    from scipy.ndimage import gaussian_filter
                    
                    band_data_smooth = gaussian_filter(band_data, sigma=0.5)
                    optimal_step = 2 if band_data.shape[0] > 100 else 1
                    
                    verts, faces, normals, values = measure.marching_cubes(
                        band_data_smooth, level=EF,
                        spacing=(kx_3d[1] - kx_3d[0], ky_3d[1] - ky_3d[0], kz_3d[1] - kz_3d[0]),
                        step_size=optimal_step,
                        gradient_direction='ascent'
                    )
                    
                    verts[:, 0] = verts[:, 0] + kx_3d[0]
                    verts[:, 1] = verts[:, 1] + ky_3d[0]
                    verts[:, 2] = verts[:, 2] + kz_3d[0]
                    
                    # Alt orientation like UTe2_fixed
                    x_plot = verts[:, 1] / (np.pi/b)  # ky -> x-axis
                    y_plot = verts[:, 2] / (np.pi/c)  # kz -> y-axis  
                    z_plot = verts[:, 0] / (np.pi/a)  # kx -> z-axis
                    
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    mesh_verts = np.column_stack([x_plot, y_plot, z_plot])
                    
                    max_faces = 30000
                    faces_used = faces
                    if len(faces) > max_faces:
                        faces_used = faces
                    
                    poly3d = mesh_verts[faces_used]
                    collection = ax.add_collection3d(Poly3DCollection(
                        poly3d, alpha=alphas[band], facecolor=colors[band], 
                        edgecolor='none', linewidth=0, rasterized=True
                    ))
                    
                except Exception as e:
                    print(f"    Warning: Band {band} failed: {e}")
        
        # Add gap nodes using UTe2_fixed's exact method
        if config['show_gap_nodes']:
            try:
                pairing_type = config['gap_node_pairing']
                print(f"    Adding {pairing_type} gap nodes using UTe2_fixed method...")
                
                # Create 3D meshgrids for gap calculation
                KX, KY, KZ = np.meshgrid(kx_3d, ky_3d, kz_3d, indexing='ij')
                gap_magnitude = calculate_gap_magnitude(KX, KY, KZ, pairing_type=pairing_type)
                
                # Use exact thresholds from UTe2_fixed
                if pairing_type == 'B2u':
                    gap_threshold = 0.05 * np.max(gap_magnitude)
                    fermi_threshold = 0.010  # 10 meV
                elif pairing_type == 'B3u':
                    gap_threshold = 0.04 * np.max(gap_magnitude)
                    fermi_threshold = 0.008  # 8 meV
                else:
                    gap_threshold = 0.05 * np.max(gap_magnitude)
                    fermi_threshold = 0.010  # 10 meV
                
                # Exact copy of UTe2_fixed slice-based detection
                all_nodes_kx = []
                all_nodes_ky = []
                all_nodes_kz = []
                all_nodes_gap = []
                all_nodes_band = []
                
                nk3d = len(kz_3d)
                
                # Find critical kz indices for B2u
                critical_kz_indices = []
                if pairing_type == 'B2u':
                    idx_0 = np.argmin(np.abs(kz_3d - 0.0))
                    idx_plus = np.argmin(np.abs(kz_3d - np.pi/c))
                    idx_minus = np.argmin(np.abs(kz_3d - (-np.pi/c)))
                    critical_kz_indices = [idx_0, idx_plus, idx_minus]
                
                for band in [2, 3]:  # Bands 3 and 4
                    band_nodes_found = 0
                    
                    for iz, kz_val in enumerate(kz_3d):
                        # Extract 2D slice at this kz
                        energy_slice = band_energies_3d[:, :, iz, band]
                        gap_slice = gap_magnitude[:, :, iz]
                        
                        # Check if this is a critical kz plane for B2u
                        is_critical_kz = (pairing_type == 'B2u' and iz in critical_kz_indices)
                        effective_fermi_threshold = fermi_threshold * 2.0 if is_critical_kz else fermi_threshold
                        
                        # Check if Fermi surface crosses this slice
                        if energy_slice.min() > effective_fermi_threshold or energy_slice.max() < -effective_fermi_threshold:
                            continue
                        
                        # Find intersections in this 2D slice
                        fermi_mask = np.abs(energy_slice) < effective_fermi_threshold
                        gap_mask = gap_slice < gap_threshold
                        intersection = fermi_mask & gap_mask
                        
                        if np.any(intersection):
                            ix_slice, iy_slice = np.where(intersection)
                            
                            # Apply 2D clustering within this slice
                            selected_2d = []
                            min_pixel_dist = 8
                            
                            for idx in range(len(ix_slice)):
                                ix_pt, iy_pt = ix_slice[idx], iy_slice[idx]
                                
                                # Check distance to already selected points in this slice
                                too_close = False
                                for (prev_ix, prev_iy) in selected_2d:
                                    if np.sqrt((ix_pt - prev_ix)**2 + (iy_pt - prev_iy)**2) < min_pixel_dist:
                                        too_close = True
                                        break
                                
                                if not too_close:
                                    selected_2d.append((ix_pt, iy_pt))
                                    
                                    # Store this node
                                    all_nodes_kx.append(kx_3d[ix_pt])
                                    all_nodes_ky.append(ky_3d[iy_pt])
                                    all_nodes_kz.append(kz_val)
                                    all_nodes_gap.append(gap_slice[ix_pt, iy_pt])
                                    all_nodes_band.append(band)
                                    band_nodes_found += 1
                
                # 3D clustering like UTe2_fixed
                if len(all_nodes_kx) > 0:
                    all_nodes_kx = np.array(all_nodes_kx)
                    all_nodes_ky = np.array(all_nodes_ky)
                    all_nodes_kz = np.array(all_nodes_kz)
                    
                    # Clustering parameters from UTe2_fixed
                    min_distance_3d = 0.20 if pairing_type == 'B2u' else 0.4
                    
                    selected_indices = []
                    used = np.zeros(len(all_nodes_kx), dtype=bool)
                    
                    for node_idx in range(len(all_nodes_kx)):
                        if used[node_idx]:
                            continue
                        
                        cluster = [node_idx]
                        for j in range(node_idx+1, len(all_nodes_kx)):
                            if used[j]:
                                continue
                            dist = np.sqrt((all_nodes_kx[node_idx] - all_nodes_kx[j])**2 +
                                          (all_nodes_ky[node_idx] - all_nodes_ky[j])**2 +
                                          (all_nodes_kz[node_idx] - all_nodes_kz[j])**2)
                            if dist < min_distance_3d:
                                cluster.append(j)
                        
                        # Pick node with smallest gap from cluster
                        best_idx = cluster[0]
                        best_gap = all_nodes_gap[cluster[0]]
                        for idx in cluster:
                            if all_nodes_gap[idx] < best_gap:
                                best_gap = all_nodes_gap[idx]
                                best_idx = idx
                        
                        # Mark cluster as used
                        for idx in cluster:
                            used[idx] = True
                        
                        selected_indices.append(best_idx)
                    
                    # Extract final nodes and convert to plot coordinates
                    if len(selected_indices) > 0:
                        final_kx = all_nodes_kx[selected_indices]
                        final_ky = all_nodes_ky[selected_indices]
                        final_kz = all_nodes_kz[selected_indices]
                        
                        # Convert to alt orientation plot coordinates
                        x_nodes = final_ky / (np.pi/b)  # ky -> x-axis
                        y_nodes = final_kz / (np.pi/c)  # kz -> y-axis
                        z_nodes = final_kx / (np.pi/a)  # kx -> z-axis
                        
                        # Plot with exact UTe2_fixed style
                        for node_idx in range(len(x_nodes)):
                            ax.scatter([x_nodes[node_idx]], [y_nodes[node_idx]], [z_nodes[node_idx]],
                                     c='yellow', marker='o', s=120, alpha=1.0,
                                     edgecolors='black', linewidth=2.5,
                                     zorder=1000, depthshade=False)
                        
                        print(f"    Plotted {len(selected_indices)} gap nodes")
                              
            except Exception as e:
                print(f"    Warning: Gap nodes failed: {e}")
        
        # Set up axes labels very close to the axes (alt orientation)
        ax.set_xlabel(r'$k_y / (\pi/b)$', fontsize=11, labelpad=0)
        ax.set_ylabel(r'$k_z / (\pi/c)$', fontsize=11, labelpad=0) 
        ax.set_zlabel(r'$k_x / (\pi/a)$', fontsize=11, labelpad=0)
        
        # Strategic tick placement to avoid redundancy but keep grid lines
        # x-axis (ky): show all ticks
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_xticklabels(['-1', '0', '1'], fontsize=10)
        
        # y-axis (kz): show all to maintain grid lines
        ax.set_yticks([-1.0, 0.0, 1.0])
        ax.set_yticklabels(['', '0', ''], fontsize=10)  # Only label center
        
        # z-axis (kx): show all ticks including 0
        ax.set_zticks([-1.0, 0.0, 1.0])
        ax.set_zticklabels(['-1', '0', '1'], fontsize=10)  # Show all labels
        
        # Fix tick positioning to be closer to axes
        ax.tick_params(axis='x', pad=-5)
        ax.tick_params(axis='y', pad=-5) 
        ax.tick_params(axis='z', pad=-5)
        
        # Set viewing angle to match UTe2_fixed perspective 3 exactly
        ax.view_init(elev=30, azim=60)  # This is perspective 3 from UTe2_fixed
        
        # Set title
        ax.set_title(config['title'], fontsize=12, pad=10)
        
        # Add panel label (a), b), c)) - debug with print
        panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
        print(f"    Adding panel label {panel_labels[i]} to subplot {i+1}")
        if i < len(panel_labels):
            ax.text2D(0.02, 0.98, panel_labels[i], transform=ax.transAxes, 
                     fontsize=12, fontweight='bold', va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'),
                     zorder=1000)
        
        # Set limits and aspect ratio like UTe2_fixed
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1) 
        ax.set_zlim(-1, 1)
        
        # Set proper aspect ratio for alt orientation (matches UTe2_fixed)
        aspect_x = c / b  # ky (x-axis) relative to kz
        aspect_y = 1.0    # kz (y-axis) is reference
        aspect_z = c / a  # kx (z-axis) relative to kz (tallest because a is smallest)
        ax.set_box_aspect([aspect_x, aspect_y, aspect_z])
        
        # Make background cleaner
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
    # Add legend with proper positioning closer to plots
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FD0000', lw=4, alpha=0.8, label='Electron-like'),
        Line2D([0], [0], color='#9635E5', lw=4, alpha=0.8, label='Hole-like'),
        Line2D([0], [0], marker='o', color='yellow', markersize=8, 
              markeredgecolor='black', markeredgewidth=2, linestyle='None', label='Nodes')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.92))  # Moved closer to plots
    
    # Adjust layout with larger margins to prevent axis title clipping
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.2, top=0.85, wspace=0.05)
    
    # Save the comparison plot - use more padding to prevent clipping
    comparison_filename = os.path.join(save_dir, '3d_fermi_surface_comparison.png')
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight', pad_inches=0.5,
                facecolor='white', edgecolor='none')
    print(f"  Saved comparison plot: {comparison_filename}")
    
    plt.show()

if __name__ == "__main__":
    create_3d_comparison()